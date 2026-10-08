from abc import ABC
from dataclasses import dataclass
from typing import Any

@dataclass
class InferParams:
    # Avoid reconstruct path during GEMM. Forces use of low-bsz GEMM/GEMV kernels.
    # Also disables MGEMM path.
    no_reconstruct: bool = False

    # Bitrate threshold for enabling MGEMM.
    mgemm_K_threshold: int = 0

    # Width threshold for enabling MGEMM regardless of bitrate.
    mgemm_n_threshold: int = 0

    # The remaining attributes are set from environment variables in __init__.
    mgemm_K_env: bool = False
    moe_cpu_offload: int = 0
    draft_moe_cpu_offload: int = 0
    moe_cpu_offload_assigned: dict[Any, Any] = ...
    moe_cpu_split: int = 0
    moe_cpu_component: str = "text"
    moe_cpu_threads: Any = None
    draft_moe_cpu_threads: Any = None
    vision_pinned: bool = False
    ngram_stream_from_disk: bool = False

    def __init__(self) -> None: ...

    def use_mgemm(
        self,
        K: int,
        out_features: int,
        mul1: bool = False,
        device: Any = None,
    ) -> bool: ...


class NullConfig:
    infer_params: InferParams

    def __init__(self) -> None: ...


class Config(ABC):
    arch_string: Any
    load_isq: bool

    directory: str
    model_classes: dict[str, Any]
    config_dict: dict[str, Any]
    stc: Any
    infer_params: InferParams

    def __init__(
        self,
        directory: str,
        model_classes: dict[str, Any],
        layer_map: list[int] | str | None = None,
        **kwargs: Any,
    ) -> None: ...

    @staticmethod
    def from_directory(directory: str, **kwargs: Any) -> Config: ...

    def get_tensor_name_fixes(self) -> dict[str, Any]: ...

    def default_max_position_embeddings(self) -> int: ...

    def read_cfg(self, *args: Any) -> Any: ...

    def assert_cfg(
        self,
        expected_type: type | list[type],
        keys: str | list[str],
        expected_value: Any = ...,
        optional: bool = False,
    ) -> None: ...

    def read_rope_settings_default(
        self,
        rope_style: Any,
        default_rope_theta: float = 10000.0,
        default_partial_rotary_factor: float = 1.0,
        config_dict: dict[str, Any] | None = None,
        theta_key: str | list[str] | None = None,
        override_type: str | None = None,
        override_head_dim: int | None = None,
        yarn_mscale_ratio: bool = False,
    ) -> Any: ...

    def override_dynamic_seq_len(self, new_max_position_embeddings: int) -> None: ...
