from __future__ import annotations

from typing import Any

from exllamav3.model.config import Config

# `Module` is the un-stubbed base class. Declared minimally so the model tree
# (`model.modules`, per-module `load`/`forward`, ...) and `Linear`'s shared
# surface are typed where `exl3pack` touches them.
class Module:
    key: str
    device: Any
    modules: list[Module]
    def load(
        self,
        device: Any,
        source: Any = None,
        keep_source_weights: bool = False,
        **kwargs: Any,
    ) -> None: ...
    def unload(self) -> None: ...
    def prepare_for_device(self, state: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Any, **kwargs: Any) -> Any: ...


class Linear(Module):
    key: str
    qmap: str | None
    device: Any
    weight: Any
    in_features: int
    out_features: int
    caps: dict[str, Any]
    quant_type: str | None
    q_priority: int
    select_hq_bits: int
    # Backing kernel wrapper; carries swap_cpu()/unswap_cpu()/get_weight_tensor()/unload().
    inner: Any

    def weights_numel(self) -> int: ...
    def get_weight_tensor(self) -> Any: ...

    def __init__(
        self,
        config: Config | None,
        key: str,
        in_features: int,
        out_features: int,
        qmap: str | None = None,
        alt_key: str | list[str] | None = None,
        qbits_key: str = "bits",
        fkey: str | None = None,
        frange: tuple[int, int] | None = None,
        frange_dim: int = 0,
        fidx: int | None = None,
        finterleaved: bool = False,
        fdequant: Any = None,
        caps: dict[str, Any] | None = None,
        softcap: float = 0.0,
        pad_to: int = 128,
        trim_padded_out: bool = False,
        full_in_features: int | None = None,
        full_out_features: int | None = None,
        first_in_feature: int | None = None,
        first_out_feature: int | None = None,
        out_dtype: Any = None,
        allow_input_padding: bool = False,
        pre_scale: float = 1.0,
        post_scale: float = 1.0,
        weight_scale: float = 1.0,
        transposed_load: bool = True,
        transpose_fused_weights: bool = True,
        ftranspose_after_load: bool = True,
        select_hq_bits: int = 0,
        qgroup: str | None = None,
    ) -> None: ...

    def load(
        self,
        device: Any,
        source: Any = None,
        keep_source_weights: bool = False,
        **kwargs: Any,
    ) -> None: ...

    def unload(self) -> None: ...
