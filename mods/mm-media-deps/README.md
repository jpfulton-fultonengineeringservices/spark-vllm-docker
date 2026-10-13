# mods/mm-media-deps

Boot-time multimodal media-decoding dependencies for the pulled
`vllm-node-b12x` image.

## Problem

The `eugr/spark-vllm-b12x` image ships the Omni-capable vLLM fork but **not**
the media libraries its multimodal loaders require
(`vllm/multimodal/media/{audio,video}.py`):

| Modality | Needs | In base image? | Symptom without it |
|---|---|---|---|
| image | PIL | ✅ | — (works) |
| audio | `av` (PyAV), `soundfile` | ❌ | `audio_url`/`input_audio` → HTTP 500 `Please install vllm[audio] for audio support` |
| video | `opencv` (default, ✅), `av`/`torchcodec` (alt) | partial | default opencv works; `pyav`/`torchcodec` backends unavailable |
| (both) | system `ffmpeg` | ❌ | only `torchcodec` needs it (dlopen) |

## Fix

`run.sh` installs `av`, `soundfile`, `torchcodec` (+ best-effort system
`ffmpeg`) into the image at launch, idempotently (no-op once importable).
It is a **boot-time mod** rather than a Dockerfile layer because the base is a
*pulled* image: a source rebuild would recompile vLLM from scratch, and this
matches the fork's existing dep-injection pattern (`mods/mooncake-worker`,
`mods/lmcache-offload`).

**Torch safety:** `torchcodec` is installed `--no-deps`. A bare
`pip install torchcodec` pulls the PyPI default torch and clobbers the image's
custom **torch 2.13.0+cu13.3** build; `--no-deps` avoids that (torch/torchvision
already satisfy it in-image). Verified: torch stays `2.13.0+cu13.3.glibc239`.

## Wiring

Add `mods/mm-media-deps` to the `mods:` list of the recipes that serve a
multimodal model (currently `pd-disagg-mimo-uncensored-exl3-4x` and its
`-lmcache` twin, plus any other MiMo-Omni recipe).

## Verify

```
docker exec vllm_node python3 -c "import av, soundfile; from torchcodec.decoders import VideoDecoder; print('ok')"
```
Then an `audio_url` request through the router should return 200 (no more
`Please install vllm[audio]` 500).
