# MRoPE YaRN cache correction

Apply this mod when using YaRN with MRoPE on the `vllm-node-b12x` image. The
image's MRoPE implementation multiplies the native context by four before
passing it to YaRN. That changes the YaRN correction frequencies and causes
YaRN to allocate four times the intended cache. The patch retains the fourfold
video-position cache for **unscaled** MRoPE only; scaled MRoPE passes the native
context to YaRN, which applies its scaling factor itself.

The source fixture was extracted from the installed `vllm_node` image on
gx10-node1, image ID `sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02`.
The installed `mrope.py` SHA-256 is
`7a9383944326ece13f659d051ec1433a0b5e2a14b4a95e903b3c30f26bdfc6a2`.
The mod checks its exact source anchor, compiles before writing, fails closed on
image drift, and is repeatable within a fresh container.

For a native context of 262144 and YaRN factor 4, the model's
`rope_parameters.original_max_position_embeddings` remains 262144, while the
effective `text_config.max_position_embeddings` and `--max-model-len` should
be 1048576. The MTP recipe must apply this mod and pass both values in its
`--hf-overrides`; the cache then has exactly 1048576 rows.

The CPU behavior test needs PyTorch and can be run with
`python3 tests/test_qwen38_mrope_yarn_mod.py`. The test uses the captured image
classes and the actual mod, without a vLLM installation, GPU, or model weights.
