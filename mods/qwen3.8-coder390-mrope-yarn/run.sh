#!/bin/bash
set -euo pipefail

# Patch the installed image, not the separate vLLM submodule.
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
python3 - "$PYTHON_ROOT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "vllm/model_executor/layers/rotary_embedding/mrope.py"
source = path.read_text()
old = '''        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        super().__init__(
            head_size,
            rotary_dim,
            self.cache_max_position_num,
'''
new = '''        # Unscaled Qwen2.5-VL needs extra video positions. YaRN instead uses
        # the original context for its correction range and scales its own cache.
        self.cache_max_position_num = (
            max_position_embeddings * 4
            if self.scaling_factor is None
            else max_position_embeddings
        )
        super().__init__(
            head_size,
            rotary_dim,
            self.cache_max_position_num,
'''
if source.count(old) == 1 and source.count(new) == 0:
    result = source.replace(old, new, 1)
    compile(result, str(path), "exec")
    path.write_text(result)
    print(f"[mrope-yarn] patched {path}")
elif source.count(old) == 0 and source.count(new) == 1:
    compile(source, str(path), "exec")
    print(f"[mrope-yarn] already patched {path}")
else:
    raise SystemExit(f"[mrope-yarn] incompatible or ambiguous source at {path}")
PY
