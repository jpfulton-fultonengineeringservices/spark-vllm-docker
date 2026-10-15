"""tel_eval — eval rig telemetry capture plugin for lm-evaluation-harness.

Loaded by the harness via ``--plugins tel_eval.lm`` (see ``cli.py``), this
package registers drop-in LM backends that behave exactly like the harness's
``local-completions`` / ``local-chat-completions`` backends but additionally
record per-request telemetry (wall-clock latency, API ``usage`` token counts,
``reasoning_content``, and any base64 input images) to a JSONL sidecar plus an
images directory.

Correlation key
---------------
The harness stores ``prompt_hash = sha256(arguments[0])`` per logged sample
(``lm_eval/evaluator.py``).  ``arguments[0]`` is:
  * chat path  -> ``JsonChatStr(json_chat)`` whose ``.encode`` delegates to the
    wrapped JSON string, so ``prompt_hash = sha256(json_chat.encode())``.
  * completions -> the raw context *string* (tokenization happens inside the
    model), so ``prompt_hash = sha256(context.encode())``.

This plugin recomputes the same digest as ``request_hash`` when it can (chat is
exact; completions decodes the token ids back to text through the model's own
tokenizer) and also stores the human-readable ``prompt_text`` so the ingester
can fall back to an exact prompt-text match.
"""

from __future__ import annotations

from . import lm as _lm  # noqa: F401  (import triggers @register_model)

__all__ = ["_lm"]
