# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any
import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size


logger = init_logger(__name__)

from ._protocol import (
    EngineId, MooncakeConnectorMetadata, MooncakeXferMetadata,
    MooncakeXferResponse, MooncakeXferResponseStatus, PullReqMeta, ReqId,
)

class _WorkerReceiveMixin:
    async def receive_kv_from_single_worker(
            self,
            worker_addr: str,
            pull_metas: dict[ReqId, PullReqMeta],
        ):
            req_ids = set(pull_metas)
            if self.host_staging:
                # Advertise the pinned receive window's geometry in place of the
                # real GPU cache: the producer's destination arithmetic then lands
                # in the window, where it registers with the engine. Per-request
                # block ids are replaced with window slot ids assigned by
                # _staging_assign_slots; the producer's logical->kernel expansion
                # maps them onto window slots exactly as it would real blocks.
                advertised_bases = self._staging_window_bases
                req_blocks = {
                    req_id: (
                        pull_meta.transfer_id,
                        self._staging_slot_block_ids(pull_meta),
                    )
                    for req_id, pull_meta in pull_metas.items()
                }
            else:
                advertised_bases = self.kv_caches_base_addr
                req_blocks = {
                    req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                    for req_id, pull_meta in pull_metas.items()
                }
            if self.host_staging and self.staging_chunk:
                req_chunk_totals = {
                    rid: [
                        len([b for b in g if b != NULL_BLOCK_ID])
                        for g in pm.local_block_ids
                    ]
                    for rid, pm in pull_metas.items()
                }
                req_chunk_offsets = {
                    rid: [pm.chunk_offsets.get(g, 0) for g in range(len(pm.local_block_ids))]
                    for rid, pm in pull_metas.items()
                }
                req_chunk_is_last = {
                    rid: pm.chunk_is_last for rid, pm in pull_metas.items()
                }
            else:
                req_chunk_offsets = {}
                req_chunk_totals = {}
                req_chunk_is_last = {}

            metadata = MooncakeXferMetadata(
                remote_hostname=self.hostname,
                remote_port=self.rpc_port,
                remote_tp_size=self.tp_size,
                remote_tp_rank=self.tp_rank,
                req_blocks=req_blocks,
                kv_caches_base_addr=advertised_bases,
                block_lens=self.block_len_per_layer,
                kv_block_lens=self.kv_block_len_per_layer,
                registered_layer_names=self.registered_layer_names,
                registered_layer_indices=self.registered_layer_indices,
                registered_group_indices=self.registered_group_indices,
                req_chunk_offsets=req_chunk_offsets,
                req_chunk_totals=req_chunk_totals,
                req_chunk_is_last=req_chunk_is_last,
            )
    
            encoded_data = self._encoder.encode(metadata)
            logger.debug(
                "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
            )
            logger.debug(
                "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
            )
    
            # Send query for the request.
            try:
                with make_zmq_socket(
                    self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
                ) as sock:
                    # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                    sock.setsockopt(
                        zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                    )
                    await sock.send(encoded_data)
                    while True:
                        ret_msg = await sock.recv()
                        response = self._xfer_resp_decoder.decode(ret_msg)
                        if response.status == MooncakeXferResponseStatus.ERROR:
                            logger.error(
                                "Error happens during transferring kvcache for %s: %s",
                                req_ids,
                                response.err_msg,
                            )
                            self.xfer_stats.record_failed_recv()
                            return
                        self.process_pulling_result(response, pull_metas)
                        if response.status == MooncakeXferResponseStatus.FINISH:
                            break
            except zmq.ContextTerminated:
                logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
            except Exception as e:
                logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
                self.xfer_stats.record_failed_recv()
                return
    def process_pulling_result(
            self,
            response: MooncakeXferResponse,
            pull_metas: dict[ReqId, PullReqMeta],
        ):
            ok_reqs: list[ReqId] = response.ok_reqs or []
    
            for req_id in ok_reqs:
                pull_meta = pull_metas[req_id]
                # No race because we are in async loop.
                pull_meta.pull_tasks_count -= 1
                if pull_meta.pull_tasks_count == 0:
                    if self.host_staging and self.staging_chunk and pull_meta.staging_slots is not None:
                        # Chunked staging: mark round done; driver owns H2D copy and addition to finished_recving_reqs.
                        pull_meta.round_done = True
                    elif self.host_staging and pull_meta.staging_slots:
                        # Legacy host_staging (non-chunked): existing H2D+add path.
                        try:
                            self._staging_h2d_copy_request(pull_meta)
                        except Exception:
                            logger.exception(
                                "host_staging: H2D completion copy failed for "
                                "req %s; NOT marking it received.",
                                req_id,
                            )
                            self.xfer_stats.record_failed_recv()
                            continue
                        self.finished_recving_reqs.add(pull_meta.d_req_id)
                    else:
                        self.finished_recving_reqs.add(pull_meta.d_req_id)
    
            if ok_reqs:
                logger.debug("pulling kv_caches for %s finished", ok_reqs)
    
            if response.err_reqs:
                logger.error(
                    "pulling kv_caches for %s failed: %s",
                    response.err_reqs,
                    response.err_msg,
                )
    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
            url = remote_bootstrap_addr + "/query"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    data: dict = response.json()
                    for _, dp_entry in data.items():
                        remote_engine_id = dp_entry["engine_id"]
                        self._remote_agents[remote_engine_id] = {
                            int(tp_rank): {
                                int(pp_rank): worker_addr
                                for pp_rank, worker_addr in tp_entry.items()
                            }
                            for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                        }
                        self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
            except Exception as e:
                logger.error(
                    "Failed to connect to bootstrap server %s: %s",
                    remote_bootstrap_addr,
                    e,
                )
    
            # Always notify others regardless of connection success or failure.
            self._pending_bootstrap_queries[remote_bootstrap_addr].set()
            del self._pending_bootstrap_queries[remote_bootstrap_addr]
    def receive_kv(
            self,
            remote_engine_id: EngineId,
            pull_metas: dict[ReqId, PullReqMeta],
        assign_slots: bool = True,
        ) -> list[asyncio.Task] | None:
            if assign_slots and self.host_staging:
                # Serialized in _start_load_kv: only one pull batch occupies the
                # receive window at a time. Assign window slots before any
                # metadata goes out.
                overflow = self._staging_assign_slots(pull_metas)
                for req_id in overflow:
                    logger.error(
                        "host_staging: req %s exceeds the recv window capacity "
                        "(%d blocks); it will fail on the producer's timeout.",
                        req_id,
                        self._staging_window_blocks,
                    )
                pull_metas = {
                    req_id: meta
                    for req_id, meta in pull_metas.items()
                    if meta.staging_slots is not None
                }
                if not pull_metas:
                    return []
            remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
                self._tp_size[remote_engine_id]
            )
            worker_addrs: list[str] = []
            selected_remote_pp: dict[int, list[int]] = {}
            for remote_tp_rank in remote_tp_ranks:
                pp_to_addr = self._remote_agents[remote_engine_id][remote_tp_rank]
                if self.pp_size == len(pp_to_addr) and self.pp_rank in pp_to_addr:
                    pp_ranks = [self.pp_rank]
                else:
                    pp_ranks = sorted(pp_to_addr)
                selected_remote_pp[remote_tp_rank] = pp_ranks
                worker_addrs.extend(pp_to_addr[pp_rank] for pp_rank in pp_ranks)
    
            count = len(worker_addrs)
            logger.debug(
                "Receiving Mooncake KV for engine %s from producer TP ranks %s "
                "and PP ranks %s",
                remote_engine_id,
                remote_tp_ranks,
                selected_remote_pp,
            )
            for pull_meta in pull_metas.values():
                pull_meta.pull_tasks_count = count
            tasks = [
                asyncio.create_task(
                    self.receive_kv_from_single_worker(worker_addr, pull_metas)
                )
                for worker_addr in worker_addrs
            ]
            return tasks if self.host_staging else None
    async def _ensure_remote_agents(
            self, engine_id: EngineId, pull_metas: dict[ReqId, PullReqMeta]
        ) -> bool:
            """Wait for bootstrap queries and return whether engine_id is mapped."""
            remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
            if remote_bootstrap_addr not in self._pending_bootstrap_queries:
                self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
                await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
            else:
                await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

            if engine_id not in self._remote_agents:
                logger.error(
                    "Failed to find remote engine_id %s from bootstrap server %s",
                    engine_id,
                    remote_bootstrap_addr,
                )
                return False
            return True

    async def handle_new_engine_id(
            self,
            remote_engine_id: EngineId,
            pull_metas: dict[ReqId, PullReqMeta],
        ) -> list[asyncio.Task] | None:
            if await self._ensure_remote_agents(remote_engine_id, pull_metas):
                return self.receive_kv(remote_engine_id, pull_metas)
            return None
    def _mark_recv_failed(self, req_id: ReqId, pull_meta: PullReqMeta) -> None:
        """Mark request failed: add all non-NULL blocks to _recv_failed_block_ids and finish req."""
        for group in pull_meta.local_block_ids:
            for b in group:
                if b != NULL_BLOCK_ID:
                    self._recv_failed_block_ids.add(b)
        self.finished_recving_reqs.add(req_id)
        logger.error(
            "host_staging: req %s failed after %d retries; releasing to recompute",
            req_id,
            pull_meta.chunk_retries,
        )

    def _chunk_retry(self, pull_meta: PullReqMeta) -> bool:
        """Decrement/check retry budget for one request chunk."""
        if pull_meta.chunk_retries >= self.staging_chunk_retries:
            return False
        pull_meta.chunk_retries += 1
        return True

    async def _receive_one_round(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ) -> bool:
        """Send one chunked round of metadata, await worker tasks, return success."""
        for pm in pull_metas.values():
            pm.round_done = False
        tasks = self.receive_kv(remote_engine_id, pull_metas, assign_slots=False)
        if not tasks:
            return False
        await asyncio.gather(*tasks, return_exceptions=True)
        return all(pm.round_done for pm in pull_metas.values())

    async def _start_load_kv_staged_rounds(
        self,
        reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]],
    ) -> None:
        """Drive multi-round KV transfers sequentially per request."""
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if not await self._ensure_remote_agents(remote_engine_id, pull_metas):
                for rid, pm in pull_metas.items():
                    self._mark_recv_failed(rid, pm)
                continue

            for req_id, pull_meta in list(pull_metas.items()):
                if not pull_meta.group_cursor:
                    pull_meta.group_cursor = {
                        g: 0 for g in range(len(pull_meta.local_block_ids))
                    }
                round_idx = 0
                # req_chunk_offsets and chunk_is_last are set inside the loop
                # below via _staging_assign_round_slots; we just need to ensure
                # the initial values reflect an unstarted request.
                pull_meta.chunk_is_last = False
                while True:
                    fits = self._staging_assign_round_slots({req_id: pull_meta})
                    if not fits:
                        self._mark_recv_failed(req_id, pull_meta)
                        break
                    pull_meta.chunk_is_last = self._round_is_last(pull_meta)
                    round_idx += 1
                    slots_count = sum(len(c) for _, c in pull_meta.staging_slots.values()) if pull_meta.staging_slots else 0
                    logger.info(
                        "host_staging: req %s round %d, %d slots, last=%s",
                        req_id,
                        round_idx,
                        slots_count,
                        pull_meta.chunk_is_last,
                    )

                    ok = await self._receive_one_round(remote_engine_id, {req_id: pull_meta})
                    if not ok:
                        if self._chunk_retry(pull_meta):
                            continue
                        self._mark_recv_failed(req_id, pull_meta)
                        break

                    try:
                        self._staging_h2d_copy_request(pull_meta)
                    except Exception:
                        logger.exception(
                            "host_staging: H2D copy failed for req %s round %d",
                            req_id,
                            round_idx,
                        )
                        self._mark_recv_failed(req_id, pull_meta)
                        break

                    if pull_meta.staging_slots:
                        for g, (_, chunk) in pull_meta.staging_slots.items():
                            pull_meta.group_cursor[g] += len(chunk)

                    if pull_meta.chunk_is_last:
                        self.finished_recving_reqs.add(req_id)
                        break

    async def _start_load_kv(
            self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
        ):
            if self.host_staging:
                # Serialize pull batches so the receive window's slot space is
                # owned by one batch at a time (slots reset per batch). The lock
                # is held until the batch's transfers complete because
                # receive_kv only spawns the per-worker tasks.
                #
                # BOUNDED (2026-10-13): the batch await MUST NOT be unbounded.
                # A batch whose transfer never resolves (e.g. the producer side
                # dropped an over-capacity request and freed its blocks only on
                # its own 480s timeout, or a peer vanished) would otherwise hold
                # _recv_staging_lock forever -> every later start_load_kv queues
                # behind it -> the worker parks in get_finished
                # (_worker_setup.py:332) and decode wedges (requests hang with
                # /v1/models still 200). Bound it with the same budget the
                # producer uses (VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT); on expiry
                # release the lock and log loudly so the scheduler's error path
                # can fail those reqs instead of hanging the engine. This block
                # handles BOTH staging_chunk=True (multi-round) and False
                # (legacy single-round) paths — both must mark failed reqs on
                # timeout so they enter recompute via _recv_failed_block_ids.
                if self._recv_staging_lock is None:
                    self._recv_staging_lock = asyncio.Lock()
                async with self._recv_staging_lock:
                    budget = float(envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT) + 30.0
                    try:
                        if self.staging_chunk:
                            await asyncio.wait_for(
                                self._start_load_kv_staged_rounds(reqs_to_recv),
                                timeout=budget,
                            )
                        else:
                            await asyncio.wait_for(
                                self._start_load_kv_inner(reqs_to_recv, await_tasks=True),
                                timeout=budget,
                            )
                    except asyncio.TimeoutError:
                        logger.error(
                            "host_staging: recv batch did not complete within "
                            "%.0fs (%d engine(s)); releasing the staging window "
                            "lock. Marking reqs failed for local recompute.",
                            budget,
                            len(reqs_to_recv),
                        )
                        self.xfer_stats.record_failed_recv()
                        for metas in reqs_to_recv.values():
                            for rid, pm in metas.items():
                                self._mark_recv_failed(rid, pm)
                return
            await self._start_load_kv_inner(reqs_to_recv, await_tasks=False)
    async def _start_load_kv_inner(
            self,
            reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]],
            await_tasks: bool,
        ):
            for remote_engine_id, pull_metas in reqs_to_recv.items():
                if remote_engine_id not in self._remote_agents:
                    if await_tasks:
                        # host_staging: keep the window lock held across bootstrap
                        # discovery AND the ensuing transfer.
                        tasks = await self.handle_new_engine_id(
                            remote_engine_id, pull_metas
                        )
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)
                    else:
                        asyncio.create_task(
                            self.handle_new_engine_id(remote_engine_id, pull_metas)
                        )
                else:
                    tasks = self.receive_kv(remote_engine_id, pull_metas)
                    if await_tasks and tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
            if not self.is_kv_producer and metadata.reqs_to_recv:
                asyncio.run_coroutine_threadsafe(
                    self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
                )
    
            if not self.is_kv_consumer and (
                metadata.reqs_to_send or metadata.reqs_not_processed
            ):
                asyncio.run_coroutine_threadsafe(
                    self.record_send_reqs(metadata), self.sender_loop
                )
