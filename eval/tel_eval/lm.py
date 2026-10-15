"""Telemetry-capturing LM backends for lm-evaluation-harness.

Subclasses the harness's ``local-completions`` / ``local-chat-completions``
backends (no upstream edits) and records, per model call: wall-clock latency,
the API ``usage`` token counts, ``reasoning_content`` / ``content``, and any
base64 input images (decoded to files).  Records are written to the JSONL sink
in ``tel_eval.sink``.

Registered aliases:
  * ``local-completions-telemetry``
  * ``local-chat-completions-telemetry``

Both backends share every method through ``_TelemetryMixin``; only the base
``_create_payload`` / ``parse_generations`` / ``parse_logprobs`` differ, and
those are inherited from the harness parent classes via normal polymorphism.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import aiohttp
import requests
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import (
    LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER,
    JsonChatStr,
    eval_logger,
)
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI

from . import sink

# Transport-level failures that mean "the upstream is unreachable right now",
# not "the request was rejected". Retried by the plugin (see ``amodel_call``)
# for up to ``outage_retry_seconds`` before re-raising. HTTP 4xx/5xx are NOT
# included: ``aiohttp.ClientResponseError`` / ``requests.HTTPError`` fall
# through and fail fast, exactly as before.
_OUTAGE_EXC = (
    aiohttp.ClientConnectorError,
    aiohttp.ClientOSError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ServerTimeoutError,
    asyncio.TimeoutError,
    ConnectionError,
)
_OUTAGE_EXC_SYNC = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

# Retry backoff and outage-row cadence.
_OUTAGE_MAX_BACKOFF_S = 30.0
_OUTAGE_EMIT_INTERVAL_S = 30.0

_USAGE_CACHE_KEYS = (
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_write_input_tokens",
)


def _extract_usage(outputs: Any) -> dict[str, Any]:
    """Pull token counts out of an OpenAI-compatible response dict."""
    out: dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
    }
    if not isinstance(outputs, dict):
        return out
    usage = outputs.get("usage")
    if not isinstance(usage, dict):
        return out
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    out["total_tokens"] = usage.get("total_tokens")
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        out["cache_read_tokens"] = details.get("cached_tokens")
    for key in _USAGE_CACHE_KEYS:
        if usage.get(key) is None:
            continue
        if "read" in key or "cache_read" in key:
            out["cache_read_tokens"] = usage.get(key)
        elif "creation" in key or "write" in key:
            out["cache_write_tokens"] = usage.get(key)
    return out


def _reasoning_and_content(outputs: Any) -> tuple[str | None, str | None]:
    if not isinstance(outputs, dict):
        return None, None
    choices = outputs.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, None
    message = choices[0].get("message")
    if isinstance(message, dict):
        return message.get("reasoning_content"), message.get("content")
    return None, choices[0].get("text")


def _decode_images(payload: dict[str, Any], request_hash: str) -> list[dict[str, Any]]:
    """Decode any ``data:image/...;base64,...`` parts from a chat payload to files."""
    img_dir = sink.images_dir()
    if img_dir is None:
        return []
    root = sink.results_root()
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    found: list[dict[str, Any]] = []
    index = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            image_url = part.get("image_url") if isinstance(part, dict) else None
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url.startswith("data:image/"):
                continue
            try:
                header, b64 = url.split(",", 1)
                mime = header.split(";", 1)[0].removeprefix("data:")
                ext = mime.split("/", 1)[1] if "/" in mime else "png"
                raw = base64.b64decode(b64)
            except (ValueError, TypeError):
                continue
            img_dir.mkdir(parents=True, exist_ok=True)
            name = f"{request_hash}_{index}.{ext}"
            (img_dir / name).write_bytes(raw)
            # Store the path relative to the results root so the console's
            # image route (rooted at the results dir) can resolve it.
            if root is not None:
                try:
                    rel_path = (img_dir / name).resolve().relative_to(root.resolve())
                except ValueError:
                    rel_path = Path("images") / name
            else:
                rel_path = Path("images") / name
            found.append(
                {"path": str(rel_path), "index": index, "mime": mime, "bytes": len(raw)}
            )
            index += 1
    return found


def _prompt_text_from_payload(payload: dict[str, Any]) -> str:
    """First text content of a chat payload, or the raw prompt string."""
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text") or "")
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        return str(prompt[0])
    return ""


def _hash_payload(payload: dict[str, Any]) -> str:
    """sha256 of the request we actually send (stable, sorted keys)."""
    try:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        return ""


def _harness_prompt_hash(messages: Any) -> str:
    """Reproduce the harness's ``prompt_hash = sha256(arguments[0])``.

    ``arguments[0]`` is the per-request context the harness passes to
    ``generate_until``/``loglikelihood``: a ``JsonChatStr`` (chat), a plain
    context string (completions), or a token-id list (tokenized completions,
    unrecoverable). Mirrors ``lm_eval/utils.hash_string`` resolved through
    ``JsonChatStr.encode``.
    """
    try:
        first = messages[0] if isinstance(messages, (list, tuple)) and messages else messages
    except (TypeError, IndexError):
        return ""
    if isinstance(first, JsonChatStr):
        return hashlib.sha256(first.prompt.encode("utf-8")).hexdigest()
    if isinstance(first, str):
        return hashlib.sha256(first.encode("utf-8")).hexdigest()
    return ""


class _TelemetryMixin:
    """Shared capture logic layered over the harness API backends."""

    def __init__(self, *args, outage_retry_seconds: float = 600.0, **kwargs):
        # ``outage_retry_seconds`` is popped here (never forwarded to the
        # harness parent, which would reject the unknown kwarg). It bounds how
        # long a single request rides a *connection* outage — see
        # ``_retry_on_outage`` for why the harness's own tenacity budget is
        # too short (seconds) to survive a gateway/LiteLLM restart.
        super().__init__(*args, **kwargs)
        self._outage_retry_seconds = max(0.0, float(outage_retry_seconds))

    def _emit_outage(self, *, payload: dict[str, Any], messages: Any, exc: BaseException,
                     remaining_s: float, t_start: float) -> None:
        """Record a low-rate ``status:"outage"`` row so /live can show the stall.

        Distinct from the terminal ``status:"error"`` row emitted by
        ``_emit`` on final give-up: this one carries no prompt completion and
        is throttled (one per ``_OUTAGE_EMIT_INTERVAL_S``) so a long outage
        does not flood the sidecar.
        """
        now = time.perf_counter()
        sink.record(
            {
                "request_hash": _hash_payload(payload),
                "prompt_hash": _harness_prompt_hash(messages),
                "ordinal": sink.next_ordinal(),
                "model": getattr(self, "model", None),
                "prompt_text": _prompt_text_from_payload(payload),
                "t_request_start": t_start,
                "t_end": now,
                "wall_ms": (now - t_start) * 1000.0,
                "ttfb_ms": None,
                "ttft_ms": None,
                "stream": False,
                "status": "outage",
                "error": repr(exc),
                "remaining_s": round(remaining_s),
                "reasoning_content": None,
                "content": None,
                "images": [],
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
            }
        )

    @staticmethod
    def _backoff(delay: float, remaining: float) -> float:
        """Full-jitter exponential backoff, capped by the remaining budget."""
        delay = min(delay * 2.0, _OUTAGE_MAX_BACKOFF_S)
        return max(0.0, min(random.uniform(0.0, delay), remaining))

    def _emit(
        self,
        *,
        payload: dict[str, Any],
        messages: Any,
        outputs: Any,
        t_start: float,
        t_end: float,
        error: str | None,
    ) -> None:
        request_hash = _hash_payload(payload)
        wall_ms = (t_end - t_start) * 1000.0
        reasoning, content = _reasoning_and_content(outputs)
        sink.record(
            {
                "request_hash": request_hash,
                "prompt_hash": _harness_prompt_hash(messages),
                "ordinal": sink.next_ordinal(),
                "model": getattr(self, "model", None),
                "prompt_text": _prompt_text_from_payload(payload),
                "t_request_start": t_start,
                "t_end": t_end,
                "wall_ms": wall_ms,
                "ttfb_ms": wall_ms,
                "ttft_ms": None,
                "stream": False,
                "status": "error" if error else "ok",
                "error": error,
                "reasoning_content": reasoning,
                "content": content,
                "images": _decode_images(payload, request_hash),
                **_extract_usage(outputs),
            }
        )

    def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=copy.deepcopy(gen_kwargs),
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )
        t_start = time.perf_counter()
        outputs = None
        err = None
        try:
            outputs = self._model_call_with_outage_retry(
                payload=payload,
                messages=messages,
                generate=generate,
                gen_kwargs=gen_kwargs,
                t_start=t_start,
                **kwargs,
            )
            return outputs
        except BaseException as exc:  # noqa: BLE001 - record then re-raise
            err = repr(exc)
            raise
        finally:
            self._emit(
                payload=payload,
                messages=messages,
                outputs=outputs,
                t_start=t_start,
                t_end=time.perf_counter(),
                error=err,
            )

    def _model_call_with_outage_retry(
        self, *, payload, messages, generate, gen_kwargs, t_start, **kwargs
    ):
        """Sync ``super().model_call`` wrapped in the bounded outage retry.

        The parent transport is ``requests.post`` (raised as
        ``requests.exceptions.*``), so a connection outage surfaces as
        ``_OUTAGE_EXC_SYNC``. Graduated backoff + periodic ``status:"outage"``
        rows mirror the async path; any non-outage exception (incl. HTTP 4xx/
        5xx) propagates unchanged.
        """
        deadline = time.monotonic() + self._outage_retry_seconds
        last_emit = -_OUTAGE_EMIT_INTERVAL_S
        delay = 1.0
        while True:
            try:
                return super().model_call(  # type: ignore[misc]
                    messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs
                )
            except _OUTAGE_EXC_SYNC as exc:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    raise
                if now - last_emit >= _OUTAGE_EMIT_INTERVAL_S:
                    last_emit = now
                    self._emit_outage(
                        payload=payload, messages=messages, exc=exc,
                        remaining_s=remaining, t_start=t_start,
                    )
                time.sleep(self._backoff(delay, remaining))
                delay = min(delay * 2.0, _OUTAGE_MAX_BACKOFF_S)

    async def amodel_call(
        self,
        session,
        sem,
        messages,
        *,
        generate=True,
        cache_keys=None,
        ctxlens=None,
        gen_kwargs=None,
        **kwargs,
    ):
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        outputs = None
        err = None
        t_start = time.perf_counter()
        acquired = await sem.acquire()
        try:
            outputs = await self._amodel_fetch_with_outage_retry(
                session=session, payload=payload, messages=messages, t_start=t_start
            )
            tmp_answers = (
                self.parse_generations(outputs=outputs)
                if generate
                else self.parse_logprobs(outputs=outputs, tokens=messages, ctxlens=ctxlens)
            )
            answers = []
            for a in tmp_answers:
                if a is None:
                    eval_logger.warning(
                        "API returned null content. Content filled with "
                        f"`LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = {LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER}`."
                    )
                    answers.append(LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER)
                else:
                    answers.append(a)
            if cache_keys:
                for res, cache in zip(answers, cache_keys, strict=False):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        except BaseException as exc:  # noqa: BLE001 - record then re-raise
            err = repr(exc)
            raise
        finally:
            if acquired:
                sem.release()
            self._emit(
                payload=payload,
                messages=messages,
                outputs=outputs,
                t_start=t_start,
                t_end=time.perf_counter(),
                error=err,
            )

    async def _amodel_fetch_with_outage_retry(self, *, session, payload, messages, t_start):
        """POST ``payload``, riding a connection outage within the budget.

        HTTP 4xx/5xx raise ``aiohttp.ClientResponseError`` (NOT in
        ``_OUTAGE_EXC``) and propagate immediately, preserving the previous
        fail-fast semantics. Only transport-level failures are retried, with
        full-jitter exponential backoff and one throttled ``status:"outage"``
        row per ``_OUTAGE_EMIT_INTERVAL_S``. On budget exhaustion the last
        exception is re-raised so ``_emit`` records exactly one terminal
        ``status:"error"`` row.
        """
        deadline = time.monotonic() + self._outage_retry_seconds
        last_emit = -_OUTAGE_EMIT_INTERVAL_S
        delay = 1.0
        while True:
            try:
                async with session.post(
                    self.base_url, json=payload, headers=self.header
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        eval_logger.warning(
                            f"API request failed! Status code: {response.status}, "
                            f"Response text: {error_text}. Retrying..."
                        )
                    response.raise_for_status()
                    return await response.json()
            except _OUTAGE_EXC as exc:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    raise
                if now - last_emit >= _OUTAGE_EMIT_INTERVAL_S:
                    last_emit = now
                    self._emit_outage(
                        payload=payload, messages=messages, exc=exc,
                        remaining_s=remaining, t_start=t_start,
                    )
                await asyncio.sleep(self._backoff(delay, remaining))
                delay = min(delay * 2.0, _OUTAGE_MAX_BACKOFF_S)


@register_model("local-completions-telemetry")
class TelemetryCompletions(_TelemetryMixin, LocalCompletionsAPI):
    """``local-completions`` + per-request telemetry."""


@register_model("local-chat-completions-telemetry")
class TelemetryChatCompletion(_TelemetryMixin, LocalChatCompletion):
    """``local-chat-completions`` + per-request telemetry."""


__all__ = ["TelemetryCompletions", "TelemetryChatCompletion"]
