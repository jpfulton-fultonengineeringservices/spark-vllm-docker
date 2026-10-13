#!/bin/bash
# [mm-media-deps] Multimodal media-decoding deps for vLLM served by the pulled
# vllm-node-b12x image. vllm/multimodal/media/{audio,video}.py need `av`
# (PyAV) + `soundfile` (audio) and `av`/`torchcodec` (video alt backends);
# without them audio_url/input_audio return HTTP 500 and pyav/torchcodec video
# is unavailable (image already has PIL for images and cv2 for opencv video).
# torchcodec installed --no-deps so it cannot clobber the image's custom
# torch 2.13.0+cu13.3. Idempotent: no-op once importable.
set -euo pipefail
PREFIX="[mm-media-deps]"
if python3 -c "import av, soundfile, torchcodec" 2>/dev/null; then
  echo "$PREFIX already importable; nothing to do"; exit 0
fi
echo "$PREFIX installing av + soundfile + torchcodec (--break-system-packages)"
python3 -m pip install --break-system-packages --quiet av soundfile
python3 -m pip install --break-system-packages --quiet --no-deps torchcodec
if command -v apt-get >/dev/null 2>&1 && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "$PREFIX installing system ffmpeg (best-effort, torchcodec only)"
  (apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg) >/dev/null 2>&1 || echo "$PREFIX warn: ffmpeg apt failed; av/opencv unaffected" >&2
fi
python3 -c "import av, soundfile; from torchcodec.decoders import VideoDecoder" || { echo "$PREFIX ERROR: media deps import failed" >&2; exit 1; }
python3 -c "import av,soundfile,torch; print(f'$PREFIX av={av.__version__} soundfile={soundfile.__version__} torch={torch.__version__}')"
echo "$PREFIX multimodal media deps ready (audio + video + image)"
