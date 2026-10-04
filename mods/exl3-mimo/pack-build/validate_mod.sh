#!/bin/bash
set -e
ROOT=/usr/local/lib/python3.12/dist-packages
for f in layers/quantization/exl3.py models/mimo_v2.py; do
  d=/tmp/site/vllm/model_executor/$(dirname "$f")
  mkdir -p "$d"
  cp "$ROOT/vllm/model_executor/$f" "$d/"
done
cp "$ROOT/vllm/model_executor/models/mimo_v2.py" /tmp/mimo.orig.py
export PYTHON_ROOT=/tmp/site
bash /mod/run.sh 2>&1 | tail -8
echo "=== mimo_v2.py diff ==="
diff /tmp/mimo.orig.py /tmp/site/vllm/model_executor/models/mimo_v2.py || true
echo "=== syntax ==="
python3 -c "import ast; ast.parse(open('/tmp/site/vllm/model_executor/models/mimo_v2.py').read()); print('mimo_v2.py OK')"
python3 -c "import ast; ast.parse(open('/tmp/site/vllm/model_executor/layers/quantization/exl3.py').read()); print('exl3.py OK')"
echo "=== idempotent re-run ==="
export PYTHON_ROOT=/tmp/site
bash /mod/run.sh 2>&1 | tail -6
