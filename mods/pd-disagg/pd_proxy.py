# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated prefilling proxy for the cluster-config P/D stack.

Originally vendored from the vLLM fork's
examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py
(branch cuda13.3-aarch64-gb10).  Two additions so it drops into the
cluster's systemd health-gate and the route manager's uniform /v1/models
probe without special-casing:

  GET /health        -> {"status": "ok"}
  GET /v1/models     -> returns --model as the single entry

Store-only request flow (2026-08-21, MooncakeStoreConnector as the sole
KV path — supersedes both the Nixl-style kv_transfer_params extraction and
the P2P MooncakeConnector transfer-id / bootstrap-server contract):

  1. P phase: POST a copy of the request with stream=false, max_tokens=1
     (pop stream_options; clamp max_completion_tokens; drop min_tokens as
     today).  Response body is discarded; only the status code matters.
     P computes the prompt KV and puts chunks into the shared store pool
     (MooncakeStoreConnector kv_producer).  On non-2xx: existing instance-
     removal behavior.
  2. D phase: POST the original request (model rewritten to self.model),
     streamed.  D's scheduler does hash-keyed lookup into the store pool
     (MooncakeStoreConnector kv_consumer) — no kv_transfer_params needed,
     no bootstrap server, no engine-id resolution.

The two-phase flow has no transfer hints in either direction: P puts
chunks into the pool at its own pace; D resolves them by hash lookup
natively.  This retires the resolve_prefill_engine / _engine_id_cache /
--bootstrap-port P2P-specific machinery.
"""

import argparse
import asyncio
import ipaddress
import itertools
import json
import logging
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable

import aiohttp
import requests
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class SchedulingPolicy(ABC):
    @abstractmethod
    async def schedule(self, instances: list[str]) -> str:
        raise NotImplementedError("Scheduling Proxy is not set.")

    async def release(self, instance: str) -> None:
        pass

    def initialize(self, instances: list[str]) -> None:
        pass


class Proxy:
    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        scheduling_policy: SchedulingPolicy,
        custom_create_completion: Callable[[Request], StreamingResponse] | None = None,
        custom_create_chat_completion: Callable[[Request], StreamingResponse]
        | None = None,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.scheduling_policy = scheduling_policy
        self.scheduling_policy.initialize(prefill_instances)
        self.scheduling_policy.initialize(decode_instances)
        self.custom_create_completion = custom_create_completion
        self.custom_create_chat_completion = custom_create_chat_completion
        self.router = APIRouter()
        self.setup_routes()
        self._removed_instances: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._readd_task: asyncio.Task | None = None

    def setup_routes(self):
        self.router.post(
            "/v1/completions", dependencies=[Depends(self.validate_json_request)]
        )(
            self.custom_create_completion
            if self.custom_create_completion
            else self.create_completion
        )
        self.router.post(
            "/v1/chat/completions", dependencies=[Depends(self.validate_json_request)]
        )(
            self.custom_create_chat_completion
            if self.custom_create_chat_completion
            else self.create_chat_completion
        )
        self.router.get("/status", response_class=JSONResponse)(self.get_status)
        self.router.post(
            "/instances/add", dependencies=[Depends(self.api_key_authenticate)]
        )(self.add_instance_endpoint)
        self.router.get("/health", response_class=JSONResponse)(self.get_health)
        self.router.get("/v1/models", response_class=JSONResponse)(self.get_models)

    async def validate_json_request(self, raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail="Unsupported Media Type: Only 'application/json' is allowed",
            )

    def api_key_authenticate(self, x_api_key: str = Header(...)):
        expected_api_key = os.environ.get("ADMIN_API_KEY")
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error.",
            )
        if x_api_key != expected_api_key:
            logger.warning("Unauthorized access attempt with API Key: %s", x_api_key)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid API Key.",
            )

    async def validate_instance(self, instance: str) -> bool:
        url = f"http://{instance}/v1/models"
        try:
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as client:
                logger.info("Verifying %s ...", instance)
                async with client.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            if model_cur == self.model:
                                logger.info("Instance: %s could be added.", instance)
                                return True
                            else:
                                logger.warning(
                                    "Mismatch model %s : %s != %s",
                                    instance,
                                    model_cur,
                                    self.model,
                                )
                                return False
                        else:
                            return False
                    else:
                        return False
        except aiohttp.ClientError as e:
            logger.error(str(e))
            return False
        except Exception as e:
            logger.error(str(e))
            return False

    async def add_instance_endpoint(self, request: Request):
        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in ["prefill", "decode"]:
                raise HTTPException(status_code=400, detail="Invalid instance type.")
            if not instance or ":" not in instance:
                raise HTTPException(status_code=400, detail="Invalid instance format.")
            host, port_str = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    raise HTTPException(status_code=400, detail="Invalid port number.")
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail="Invalid instance address."
                ) from e

            is_valid = await self.validate_instance(instance)
            if not is_valid:
                raise HTTPException(
                    status_code=400, detail="Instance validation failed."
                )

            async with self._lock:
                if instance_type == "prefill":
                    if instance not in self.prefill_instances:
                        self.prefill_instances.append(instance)
                        self.prefill_cycler = itertools.cycle(self.prefill_instances)
                        self.scheduling_policy.initialize([instance])
                    else:
                        raise HTTPException(
                            status_code=400, detail="Instance already exists."
                        )
                else:
                    if instance not in self.decode_instances:
                        self.decode_instances.append(instance)
                        self.decode_cycler = itertools.cycle(self.decode_instances)
                        self.scheduling_policy.initialize([instance])
                    else:
                        raise HTTPException(
                            status_code=400, detail="Instance already exists."
                        )

            return JSONResponse(
                content={"message": f"Added {instance} to {instance_type}_instances."}
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def forward_request(self, url, data, use_chunked=True, request_id=None):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
            if request_id is not None:
                headers["X-Request-Id"] = request_id
            try:
                async with session.post(
                    url=url, json=data, headers=headers
                ) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:
                        if use_chunked:
                            async for chunk_bytes in response.content.iter_chunked(
                                1024
                            ):
                                yield chunk_bytes
                        else:
                            content = await response.read()
                            yield content
                    else:
                        error_content = await response.text()
                        try:
                            error_content = json.loads(error_content)
                        except json.JSONDecodeError:
                            error_content = error_content
                        logger.error(
                            "Request failed with status %s: %s",
                            response.status,
                            error_content,
                        )
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError occurred: %s", str(e))
                raise HTTPException(
                    status_code=502,
                    detail="Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    async def schedule(self, instances: list[str]) -> str:
        return await self.scheduling_policy.schedule(instances)

    async def get_status(self):
        status = {
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
        }
        return status

    async def get_health(self):
        return {"status": "ok"}

    async def get_models(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "cluster",
                }
            ],
        }

    async def run_prefill_phase(self, request: dict, api: str) -> str:
        """Run prefill on a P node (max_tokens=1, stream=false).

        P computes the prompt KV and puts chunks into the shared store pool
        (MooncakeStoreConnector kv_producer). The response body is discarded;
        only the status code matters. No kv_transfer_params — the store
        connector handles chunking natively through the pool.
        """
        prefill_request = request.copy()
        prefill_request["stream"] = False
        prefill_request.pop("stream_options", None)
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        prefill_request.pop("min_tokens", None)
        prefill_request.pop("min_completion_tokens", None)

        prefill_instance = await self.schedule(self.prefill_instances)
        try:
            url = f"http://{prefill_instance}{api}"
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            }
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
                try:
                    async with session.post(
                        url=url, json=prefill_request, headers=headers
                    ) as response:
                        if not (200 <= response.status < 300):
                            error_content = await response.text()
                            logger.error(
                                "Prefill %s failed with status %s: %s",
                                prefill_instance,
                                response.status,
                                error_content,
                            )
                            await self.remove_instance_endpoint("prefill", prefill_instance)
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"Prefill failed with status {response.status}: "
                                f"{error_content}",
                            )
                        await response.release()
                except aiohttp.ClientError as e:
                    logger.error("Prefill ClientError: %s", str(e))
                    await self.remove_instance_endpoint("prefill", prefill_instance)
                    raise HTTPException(
                        status_code=502,
                        detail="Bad Gateway: Error communicating with prefill instance.",
                    ) from e
        finally:
            await self.scheduling_policy.release(prefill_instance)

        return prefill_instance

    async def create_completion(self, raw_request: Request):
        return await self._create_disagg(raw_request, "/v1/completions")

    async def create_chat_completion(self, raw_request: Request):
        return await self._create_disagg(raw_request, "/v1/chat/completions")

    async def _create_disagg(self, raw_request: Request, api: str):
        try:
            request = await raw_request.json()
            request["model"] = self.model
            request_id = str(uuid.uuid4())

            # Prefill stage: P computes the KV cache and puts chunks into
            # the shared store pool. No kv_transfer_params — D resolves
            # them by hash lookup natively.
            await self.run_prefill_phase(request, api)

            # Decode stage: D looks up KV chunks from the pool via the
            # store connector (hash-keyed), no transfer hints needed.
            decode_instance = await self.schedule(self.decode_instances)
            decode_request = request.copy()

            async def tracked_forward():
                try:
                    async for chunk in self.forward_request(
                        f"http://{decode_instance}{api}",
                        decode_request,
                        request_id=request_id,
                    ):
                        yield chunk
                except HTTPException:
                    await self.remove_instance_endpoint("decode", decode_instance)
                    raise
                finally:
                    await self.scheduling_policy.release(decode_instance)

            response = StreamingResponse(content=tracked_forward())
            return response
        except HTTPException:
            raise
        except Exception:
            exc_info = sys.exc_info()
            error_messages = [str(e) for e in exc_info if e]
            print("Error occurred in disagg proxy server")
            print(error_messages)
            return StreamingResponse(
                content=iter(error_messages), media_type="text/event-stream"
            )

    async def remove_instance_endpoint(self, instance_type, instance):
        async with self._lock:
            removed_key = f"{instance_type}:{instance}"
            self._removed_instances[removed_key] = {
                "removed_at": time.monotonic(),
                "instance": instance,
                "instance_type": instance_type,
                "attempts": 0,
            }
            # Recreate cycler without the removed instance
            if instance_type == "decode":
                self.decode_instances = [
                    inst for inst in self.decode_instances if inst != instance
                ]
                self.decode_cycler = itertools.cycle(self.decode_instances)
            if instance_type == "prefill":
                self.prefill_instances = [
                    inst for inst in self.prefill_instances if inst != instance
                ]
                self.prefill_cycler = itertools.cycle(self.prefill_instances)

        # Start a background re-add loop if one is not already running
        if self._readd_task is None or self._readd_task.done():
            self._readd_task = asyncio.create_task(self._readd_loop())

    async def _readd_loop(self):
        while True:
            await asyncio.sleep(10)
            await self._maybe_readd_removed_instances()
            async with self._lock:
                if not self._removed_instances:
                    break

    async def _maybe_readd_removed_instances(self):
        now = time.monotonic()
        candidates: list[tuple[str, dict]] = []
        to_evict: list[str] = []

        async with self._lock:
            for key, info in list(self._removed_instances.items()):
                age = now - info["removed_at"]
                if age > 10:
                    info["attempts"] += 1
                    if info["attempts"] > 6:
                        to_evict.append(key)
                    else:
                        candidates.append((key, info))

        # Evict instances that have been unhealthy for too long
        if to_evict:
            async with self._lock:
                for key in to_evict:
                    self._removed_instances.pop(key, None)

        # Check health of candidates outside the lock to avoid blocking I/O
        healthy: list[tuple[str, dict]] = []
        for key, info in candidates:
            if await self.validate_instance(info["instance"]):
                healthy.append((key, info))

        # Re-add healthy instances
        if healthy:
            async with self._lock:
                for key, info in healthy:
                    instance = info["instance"]
                    inst_type = info["instance_type"]
                    if inst_type == "prefill":
                        if instance not in self.prefill_instances:
                            self.prefill_instances.append(instance)
                            self.prefill_cycler = itertools.cycle(
                                self.prefill_instances
                            )
                    else:
                        if instance not in self.decode_instances:
                            self.decode_instances.append(instance)
                            self.decode_cycler = itertools.cycle(
                                self.decode_instances
                            )
                    self._removed_instances.pop(key, None)


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    def __init__(self):
        super().__init__()
        self._index = 0
        self._lock = asyncio.Lock()

    async def schedule(self, instances: list[str]) -> str:
        async with self._lock:
            inst = instances[self._index % len(instances)]
            self._index += 1
            return inst


class LeastLoadedSchedulingPolicy(SchedulingPolicy):
    def __init__(self):
        super().__init__()
        self.in_flight: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def initialize(self, instances: list[str]) -> None:
        for inst in instances:
            self.in_flight.setdefault(inst, 0)

    async def schedule(self, instances: list[str]) -> str:
        async with self._lock:
            inst = min(instances, key=lambda x: self.in_flight.get(x, 0))
            self.in_flight[inst] = self.in_flight.get(inst, 0) + 1
            return inst

    async def release(self, instance: str) -> None:
        async with self._lock:
            self.in_flight[instance] = max(0, self.in_flight.get(instance, 0) - 1)


class ProxyServer:
    def __init__(
        self,
        args: argparse.Namespace,
        scheduling_policy: SchedulingPolicy | None = None,
        create_completion: Callable[[Request], StreamingResponse] | None = None,
        create_chat_completion: Callable[[Request], StreamingResponse]
        | None = None,
    ):
        self.validate_parsed_serve_args(args)
        self.port = args.port
        if scheduling_policy is None:
            if args.scheduling_policy == "least_loaded":
                scheduling_policy = LeastLoadedSchedulingPolicy()
            else:
                scheduling_policy = RoundRobinSchedulingPolicy()
        self.proxy_instance = Proxy(
            prefill_instances=[] if args.prefill is None else args.prefill,
            decode_instances=[] if args.decode is None else args.decode,
            model=args.model,
            scheduling_policy=scheduling_policy,
            custom_create_completion=create_completion,
            custom_create_chat_completion=create_chat_completion,
        )
    def validate_parsed_serve_args(self, args: argparse.Namespace):
        if not args.prefill:
            raise ValueError("Please specify at least one prefill node.")
        if not args.decode:
            raise ValueError("Please specify at least one decode node.")
        self.validate_instances(args.prefill)
        self.validate_instances(args.decode)
        self.verify_model_config(
            args.prefill, args.model,
            timeout=args.verify_timeout,
            retry_interval=args.verify_retry_interval,
            role="prefill",
        )
        self.verify_model_config(
            args.decode, args.model,
            timeout=args.verify_timeout,
            retry_interval=args.verify_retry_interval,
            role="decode",
        )

    def validate_instances(self, instances: list):
        for instance in instances:
            if len(instance.split(":")) != 2:
                raise ValueError(f"Invalid instance format: {instance}")
            host, port = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port)
                if not (0 < port < 65536):
                    raise ValueError(f"Invalid port number in instance: {instance}")
            except Exception as e:
                raise ValueError(f"Invalid instance {instance}: {str(e)}") from e

    def verify_model_config(
        self,
        instances: list,
        model: str,
        timeout: int = 1800,
        retry_interval: int = 5,
        role: str = "",
    ) -> None:
        """Verify that each instance serves the expected model.

        Retries with backoff so the proxy stays alive while the prefill/decode
        vLLM instances finish loading (a large model can take 10-20 min).  Logs
        a "waiting for ..." message on each retry instead of crash-looping.
        """
        model_suffix = model.split("/")[-1]
        deadline = time.monotonic() + timeout
        prefix = f"[verify:{role}] " if role else ""

        for instance in instances:
            attempt = 0
            while True:
                attempt += 1
                try:
                    response = requests.get(
                        f"http://{instance}/v1/models",
                        timeout=10,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            model_cur_suffix = model_cur.split("/")[-1]
                            if model_cur_suffix != model_suffix:
                                raise ValueError(
                                    f"{instance} serves a different model: "
                                    f"{model_cur} != {model}"
                                )
                            logger.info(
                                "%sVerified %s (model=%s)", prefix, instance, model_cur
                            )
                            break
                        else:
                            raise ValueError(f"Empty model list from {instance}!")
                    else:
                        raise ValueError(
                            f"HTTP {response.status_code} from {instance}!"
                        )
                except (requests.RequestException, ValueError) as e:
                    if time.monotonic() >= deadline:
                        if isinstance(e, ValueError) and "serves a different model" in str(e):
                            raise
                        raise ValueError(
                            f"Timed out after {timeout}s waiting for {instance}: {e}"
                        ) from e
                    logger.info(
                        "%sWaiting for %s (%s, attempt %d, retry in %ds)",
                        prefix, instance, e, attempt, retry_interval,
                    )
                    time.sleep(retry_interval)

    def run_server(self):
        app = FastAPI()
        app.include_router(self.proxy_instance.router)
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, loop="uvloop")
        server = uvicorn.Server(config)
        server.run()


def parse_args():
    parser = argparse.ArgumentParser("vLLM disaggregated proxy server.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")

    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )

    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )

    parser.add_argument(
        "--scheduling-policy",
        type=str,
        choices=["round_robin", "least_loaded"],
        default="round_robin",
        help="Scheduling policy for prefill/decode instances. "
        "least_loaded routes to the instance with the fewest in-flight requests.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number",
    )

    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for prefill/decode backends to become available "
        "before giving up (default 1800 = 30 min). The proxy retries with "
        "backoff instead of crash-looping while vLLM instances load.",
    )

    parser.add_argument(
        "--verify-retry-interval",
        type=int,
        default=5,
        help="Seconds between retry attempts when waiting for backends (default 5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    proxy_server = ProxyServer(args=args)
    proxy_server.run_server()