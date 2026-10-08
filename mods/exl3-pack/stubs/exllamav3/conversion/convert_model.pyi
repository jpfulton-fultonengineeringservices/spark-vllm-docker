from collections import deque
from typing import Any

import torch

def monkeypatch_triton_autotuner_thread_safety() -> None: ...
def check_system() -> None: ...
def save_dict(filename: str, dict_: dict[str, Any], args: Any) -> None: ...
def load_dict(filename: str, args: Any) -> dict[str, Any]: ...
def load_tensor(filename: str, args: Any) -> list[torch.Tensor]: ...
def save_tensor(tensor: list[torch.Tensor], filename: str, args: Any) -> None: ...
def prepare_env(args: Any) -> None: ...
def prepare(args: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool, str | None]: ...
def get_base_model(args: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, bool]: ...
def prepare_state(
    args: dict[str, Any], job_state: dict[str, Any], config: Any, model: Any, tokenizer: Any | None
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]: ...
def get_state_error(x: Any, ref: Any) -> tuple[float, float, float]: ...
def unpack_sym(t: Any, n: int) -> Any: ...
def get_H_data(args: Any, linear: Any, capture_H: Any, state: Any) -> Any: ...
def make_quant_args(
    args: Any, idx: int, K: float, devices: Any, device_ratios: Any = None
) -> Any: ...
def print_quantized_linear(
    config: Any, linear: Any, quant_args: Any, proxy_err: Any, time_str: str = ""
) -> None: ...
def group_quant_linears(
    linears: Any, strategy: Any, capture_H: Any, max_cat_cols: int = 32768, max_stack: int = 16
) -> Any: ...
def group_label(group: Any) -> str: ...
def _tile_split_devices(numel: int, devices: Any, device_ratios: Any) -> Any: ...
def quantize_linears_single(
    args: Any,
    linears: Any,
    config: Any,
    strategy: Any,
    idx: int,
    devices: Any,
    device_ratios: Any,
    capture_H: Any,
    state: Any,
) -> None: ...
def quantize_linears_parallel(
    args: Any,
    linears: Any,
    config: Any,
    strategy: Any,
    idx: int,
    devices: Any,
    device_ratios: Any,
    capture_H: Any,
    state: Any,
) -> Any: ...
def check_bad_rows(bad_rows: Any, num_rows: int, max_fraction: float = 0.1) -> None: ...
def calibration_row_shards(num_rows: int, devices: Any, device_ratios: Any) -> Any: ...
def load_parallel_calib_modules(
    replica_models: Any, idx: int, devices: Any, load_slice: Any, source: Any = None
) -> Any: ...
def run_row_workers(title: Any, num_rows: int, workers: Any, progress_count: Any) -> Any: ...
def capture_module_parallel(
    model: Any,
    modules: Any,
    devices: Any,
    device_ratios: Any,
    state: Any,
    original_input_ids: Any,
    get_preserve: Any,
    put_preserve: Any,
    slicing: Any,
    current_slice: Any,
    title: Any,
    bad_rows: Any,
) -> Any: ...
def advance_state_parallel(
    model: Any,
    modules: Any,
    devices: Any,
    device_ratios: Any,
    state: Any,
    original_input_ids: Any,
    get_preserve: Any,
    put_preserve: Any,
    ref_states: Any,
    have_linears: Any,
    is_last_module: Any,
    title: Any,
    bad_rows: Any,
) -> Any: ...
def image_dump(args: Any, linears: Any) -> None: ...
def host_rss_str() -> str: ...
def feedback_module(
    state: Any,
    module: Any,
    config: Any,
    final_bpw: Any,
    error: Any,
    cos_error: Any,
    sqnr_: Any,
    module_time: Any,
) -> None: ...
def feedback_eta(idx: int, model: Any, module_time: Any) -> None: ...
def clear_temp_files(args: Any) -> None: ...
def main(args: Any, job_state: Any) -> Any: ...

timed_blocks: int
eta_window: deque[Any]

col_default: str
col_red: str
curr_progress: int
group: Any
max_progress: int
num_ref_states: int
parser: Any
progress_lock: Any
