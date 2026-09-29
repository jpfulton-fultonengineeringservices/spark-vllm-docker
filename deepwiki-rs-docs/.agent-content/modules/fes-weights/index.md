# fes-weights

*module · agent map*

Prepares and verifies staged model weights so vLLM can serve fully offline from local storage rather than querying the Hugging Face hub — essential for air-gapped or network-constrained Spark clusters. Consists of weight verification via shard/index checks and offline hub cache…

Weight verification is implemented in verify.py: it loads safetensors index, validates shard file presence and counts, checks total size parity within a tolerance, and enforces existence of hf_quant_config.json for NVFP4 checkpoints. The script outputs a JSON summary on success…

**Location:** [Home](../../index.md) › **fes-weights**

## Diagrams

- [Flowchart](flowchart.md)
- [Sequence](sequence.md)

## Source

- [`mods/fes-weights/verify.py`](../../../../.litho/tree/repo/mods/fes-weights/verify.py)
- [`mods/fes-weights/run.sh`](../../../../.litho/tree/repo/mods/fes-weights/run.sh)
- [`mods/fes-weights/hf-download.sh`](../../../../.litho/tree/repo/mods/fes-weights/hf-download.sh)

## Interaction

- The module exposes a shell entrypoint (run.sh) that verifies mounted weights and sets up the Hugging Face hub-cache layout for offline vLLM inference. Environment variables like FES_WEIGHTS_DIR, FES_HUB_MODEL control behavior. The Python verifier (verify.py) is called by the…
