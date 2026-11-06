# Extracted unchanged from image sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02
# vllm/model_executor/model_loader/weight_utils.py (2026-11-06).
from __future__ import annotations
from dataclasses import dataclass
import os

def _matches_weight_name_prefixes(
    weight_name: str, weight_name_prefixes: Sequence[str]
) -> bool:
    return weight_name.startswith(tuple(weight_name_prefixes))

@dataclass(frozen=True)
class _InstantTensorCpuFallback:
    position: int
    name: str
    filename: str

@dataclass(frozen=True)
class _InstantTensorSelection:
    gpu_tensor_count: int
    selected_tensor_count: int
    cpu_fallbacks: tuple[_InstantTensorCpuFallback, ...]

def _restrict_instanttensor_to_selected_ranges(
    instant_open: Any,
    *,
    indexed_tensor_files: dict[str, str] | None,
    weight_name_prefixes: Sequence[str] | None,
    max_tensor_size: int | None = None,
    excluded_tensor_names: set[str] | None = None,
) -> _InstantTensorSelection:
    """Restrict an InstantTensor loader to requested checkpoint ranges.

    Args:
        instant_open: Unopened InstantTensor safetensors loader whose metadata
            can be restricted before I/O starts.
        indexed_tensor_files: Mapping from tensor names to their canonical
            checkpoint files. ``None`` accepts tensors from every input file.
        weight_name_prefixes: Model parameter prefixes selected for loading.
            ``None`` or an empty sequence accepts every prefix.
        max_tensor_size: Largest tensor payload, in bytes, retained in the GPU
            staging path. Larger selected tensors are assigned to CPU loading.
        excluded_tensor_names: Tensor names already emitted by an earlier
            loading pass. Excluded names are removed before InstantTensor opens
            its GPU staging context.

    Returns:
        Selected GPU tensor count, total selected tensor count, and CPU
        fallbacks with their positions in checkpoint order.
    """
    required_attrs = (
        "filename",
        "ordered_tensor_metadatas",
        "tensor_offsets",
        "tensor_sizes",
        "total_tensor_size",
        "tensor_name_to_index",
        "loader_handle",
        "_determine_buffer_size",
    )
    missing = [name for name in required_attrs if not hasattr(instant_open, name)]
    if missing:
        raise RuntimeError(
            "Installed InstantTensor does not expose the metadata layout needed "
            f"for index-aware loading (missing: {', '.join(missing)})"
        )
    if instant_open.loader_handle is not None:
        raise RuntimeError("InstantTensor selection must be applied before opening I/O")

    filenames = list(instant_open.filename)
    metadata = list(instant_open.ordered_tensor_metadatas)
    offsets = list(instant_open.tensor_offsets)
    selected_filenames: list[str] = []
    selected_metadata: list[tuple[str, dict[str, Any]]] = []
    selected_offsets: list[tuple[int, int]] = []
    cpu_fallbacks: list[_InstantTensorCpuFallback] = []
    selected_position = 0
    excluded_selected_count = 0
    metadata_pos = 0
    offset_pos = 0

    for filename in filenames:
        filename_abs = os.path.abspath(filename)
        with safe_open(filename, framework="pt") as physical_file:
            physical_names = list(physical_file.offset_keys())
        tensor_count = len(physical_names)
        file_metadata = metadata[metadata_pos : metadata_pos + tensor_count]
        file_offsets = offsets[offset_pos : offset_pos + tensor_count + 1]
        if len(file_metadata) != tensor_count or len(file_offsets) != tensor_count + 1:
            raise RuntimeError(
                "InstantTensor metadata/offset count does not match safetensors "
                f"header for {filename}"
            )
        if [name for name, _ in file_metadata] != physical_names:
            raise RuntimeError(
                "InstantTensor tensor order does not match safetensors offset order "
                f"for {filename}"
            )

        keep = []
        for item_index, name in enumerate(physical_names):
            indexed_here = indexed_tensor_files is None or (
                indexed_tensor_files.get(name) == filename_abs
            )
            prefix_matches = not weight_name_prefixes or _matches_weight_name_prefixes(
                name, weight_name_prefixes
            )
            otherwise_selected = indexed_here and prefix_matches
            excluded_here = (
                otherwise_selected
                and excluded_tensor_names is not None
                and name in excluded_tensor_names
            )
            excluded_selected_count += int(excluded_here)
            selected_here = otherwise_selected and not excluded_here
            item_data_offsets = file_metadata[item_index][1]["data_offsets"]
            tensor_size = int(item_data_offsets[1]) - int(item_data_offsets[0])
            use_cpu_fallback = (
                selected_here
                and max_tensor_size is not None
                and tensor_size > max_tensor_size
            )
            keep.append(selected_here and not use_cpu_fallback)
            if selected_here:
                if use_cpu_fallback:
                    cpu_fallbacks.append(
                        _InstantTensorCpuFallback(
                            position=selected_position,
                            name=name,
                            filename=filename,
                        )
                    )
                selected_position += 1

        run_start = 0
        while run_start < tensor_count:
            while run_start < tensor_count and not keep[run_start]:
                run_start += 1
            if run_start == tensor_count:
                break
            run_end = run_start + 1
            while run_end < tensor_count and keep[run_end]:
                run_end += 1

            logical_file_index = len(selected_filenames)
            selected_filenames.append(filename)
            selected_metadata.extend(file_metadata[run_start:run_end])
            selected_offsets.extend(
                (logical_file_index, int(file_offsets[index][1]))
                for index in range(run_start, run_end)
            )
            selected_offsets.append((logical_file_index, int(file_offsets[run_end][1])))
            run_start = run_end

        metadata_pos += tensor_count
        offset_pos += tensor_count + 1

    if metadata_pos != len(metadata) or offset_pos != len(offsets):
        raise RuntimeError(
            "InstantTensor layout contains unaccounted metadata or offsets"
        )
    if not selected_metadata and not cpu_fallbacks and excluded_selected_count == 0:
        raise RuntimeError("InstantTensor index/prefix selection matched no tensors")

    selected_names = [name for name, _ in selected_metadata]
    if len(selected_names) != len(set(selected_names)):
        raise RuntimeError(
            "InstantTensor index-aware selection still contains duplicate tensor names"
        )
    fallback_names = [fallback.name for fallback in cpu_fallbacks]
    if len(fallback_names) != len(set(fallback_names)):
        raise RuntimeError(
            "InstantTensor CPU fallback selection contains duplicate tensor names"
        )
    overlap = set(selected_names).intersection(fallback_names)
    if overlap:
        raise RuntimeError(
            f"InstantTensor GPU and CPU selections overlap: {sorted(overlap)[:3]}"
        )
    selected_sizes = [
        int(item["data_offsets"][1]) - int(item["data_offsets"][0])
        for _, item in selected_metadata
    ]
    if selected_metadata:
        instant_open.filename = selected_filenames
        instant_open.ordered_tensor_metadatas = selected_metadata
        instant_open.tensor_name_to_index = {
            name: index for index, name in enumerate(selected_names)
        }
        instant_open.tensor_offsets = selected_offsets
        instant_open.tensor_sizes = selected_sizes
        instant_open.total_tensor_size = sum(selected_sizes)
        instant_open._determine_buffer_size(None)

    return _InstantTensorSelection(
        gpu_tensor_count=len(selected_metadata),
        selected_tensor_count=selected_position,
        cpu_fallbacks=tuple(cpu_fallbacks),
    )


def instanttensor_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    weight_name_prefixes: Sequence[str] | None = None,
    *,
    indexed_tensor_files: dict[str, str] | None = None,
    priority_weight_name_prefixes: Sequence[str] | None = None,
    small_checkpoint_max_bytes: int | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over weights in model safetensor files with InstantTensor.

    When an index is available, restrict the physical I/O layout before
    opening the loader. This avoids loading stale tensors from base shards
    when a composed checkpoint remaps their names to overlay shards.
    """
    try:
        import instanttensor
    except ImportError as e:
        raise ImportError(
            "Please install instanttensor via `pip install vllm[instanttensor]`"
        ) from e

    if not current_platform.is_cuda():
        raise ValueError("InstantTensor requires NVIDIA GPUs")

    if small_checkpoint_max_bytes is not None and (
        isinstance(small_checkpoint_max_bytes, bool) or small_checkpoint_max_bytes <= 0
    ):
        raise ValueError(
            "InstantTensor small-checkpoint threshold must be a positive integer"
        )

    if priority_weight_name_prefixes:
        if not all(
            isinstance(prefix, str) and prefix
            for prefix in priority_weight_name_prefixes
        ):
            raise ValueError(
                "InstantTensor priority weight prefixes must be non-empty strings"
            )
        priority_names_by_file: dict[str, set[str]] = defaultdict(set)
        tensor_files: Iterable[tuple[str, str]]
        if indexed_tensor_files is not None:
            tensor_files = indexed_tensor_files.items()
        else:
            # The target and every speculative-model source share one loader
            # configuration. Header discovery lets an unindexed draft ignore
            # target-only priorities without disabling them for the target.
            discovered_tensor_files: list[tuple[str, str]] = []
            for filename in hf_weights_files:
                absolute_filename = os.path.abspath(filename)
                with safe_open(filename, framework="pt", device="cpu") as reader:
                    discovered_tensor_files.extend(
                        (name, absolute_filename) for name in reader.offset_keys()
                    )
            tensor_files = discovered_tensor_files

        for name, filename in tensor_files:
            if not name.startswith(tuple(priority_weight_name_prefixes)):
                continue
            if weight_name_prefixes and not _matches_weight_name_prefixes(
                name, weight_name_prefixes
            ):
                continue
            priority_names_by_file[os.path.abspath(filename)].add(name)
        if not priority_names_by_file:
            logger.info_once(
                "InstantTensor priority prefixes %s match no tensors in this "
                "checkpoint source; all selected tensors use the normal "
                "loading schedule.",
                tuple(priority_weight_name_prefixes),
            )
        else:
            logger.info_once(
                "InstantTensor loads %d tensors from %d/%d checkpoint shards "
                "through CPU safetensors before GPU staging because they match "
                "priority prefixes %s.",
                sum(len(names) for names in priority_names_by_file.values()),
                len(priority_names_by_file),
                len(hf_weights_files),
                tuple(priority_weight_name_prefixes),
            )
        priority_names: set[str] = set()
        for filename in hf_weights_files:
            names = priority_names_by_file.get(os.path.abspath(filename))
            if not names:
                continue
            with safe_open(filename, framework="pt", device="cpu") as reader:
                physical_names = list(reader.offset_keys())
                missing_names = names.difference(physical_names)
                if missing_names:
                    raise RuntimeError(
                        "Safetensors index maps priority tensors to a shard "
                        "that does not contain them: "
                        f"{sorted(missing_names)[:3]} in {filename}"
                    )
                for name in physical_names:
                    if name not in names:
                        continue
                    priority_names.add(name)
                    yield name, reader.get_tensor(name)
        expected_priority_names = set().union(*priority_names_by_file.values())
        if priority_names != expected_priority_names:
            missing_names = expected_priority_names.difference(priority_names)
            raise RuntimeError(
                "InstantTensor priority tensors reference checkpoint shards "
                f"outside the selected file set: {sorted(missing_names)[:3]}"
            )
    else:
        priority_names = set()

    if (
        small_checkpoint_max_bytes is not None
        and indexed_tensor_files is None
        and not priority_names
    ):
        checkpoint_size = _get_checkpoints_size_bytes(hf_weights_files)
        if checkpoint_size <= small_checkpoint_max_bytes:
            logger.info_once(
                "Loading the %.2f GiB unindexed checkpoint through CPU "
                "safetensors because it does not exceed the configured %.2f GiB "
                "InstantTensor small-checkpoint threshold.",
                checkpoint_size / 1024**3,
                small_checkpoint_max_bytes / 1024**3,
            )
            yield from safetensors_weights_iterator(
                hf_weights_files,
                use_tqdm_on_load,
                weight_name_prefixes=weight_name_prefixes,
            )
            return

    try:
        world_group = get_world_group()
    except AssertionError:
        # Entering here only in unit tests where the world group is not initialized.
        process_group = None
    else:
        process_group = world_group.device_group if world_group.world_size > 1 else None

    device = current_platform.current_device()
    copy_setting = os.getenv("INSTANTTENSOR_COPY", "1")
    if copy_setting not in {"0", "1"}:
        raise ValueError(f"INSTANTTENSOR_COPY must be 0 or 1, got {copy_setting!r}")
    copy_tensors = copy_setting == "1"

    configured_buffer_size = os.getenv("INSTANTTENSOR_BUFFER_SIZE")
    max_gpu_tensor_size: int | None = None
    if configured_buffer_size is not None:
        try:
            max_gpu_tensor_size = int(configured_buffer_size)
        except ValueError as e:
            raise ValueError(
                "INSTANTTENSOR_BUFFER_SIZE must be an integer number of bytes"
            ) from e
        if max_gpu_tensor_size <= 0:
            raise ValueError("INSTANTTENSOR_BUFFER_SIZE must be greater than zero")

    restrict_before_io = (
        indexed_tensor_files is not None
        or bool(weight_name_prefixes)
        or max_gpu_tensor_size is not None
        or bool(priority_names)
    )
    open_kwargs: dict[str, Any] = {}
    if restrict_before_io:
        open_kwargs["load_now"] = False

    # copy=True is the safe default. copy=False exposes views into
    # InstantTensor's reusable buffer for synchronous consumers.
    instant_open = instanttensor.safe_open(
        list(hf_weights_files),
        framework="pt",
        device=device,
        process_group=process_group,
        copy=copy_tensors,
        **open_kwargs,
    )
    selection: _InstantTensorSelection | None = None
    if restrict_before_io:
        selection = _restrict_instanttensor_to_selected_ranges(
            instant_open,
            indexed_tensor_files=indexed_tensor_files,
            weight_name_prefixes=weight_name_prefixes,
            max_tensor_size=max_gpu_tensor_size,
            excluded_tensor_names=priority_names,
        )
        if selection.selected_tensor_count == 0:
            return

    cpu_fallbacks = selection.cpu_fallbacks if selection is not None else ()
    if not cpu_fallbacks:
        with instant_open as f:
            # Track bytes so the bar reports load throughput (GB/s).
            pbar = tqdm(
                total=f.total_tensor_size,
                desc="Loading safetensors using InstantTensor loader",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
                position=tqdm._get_free_pos(),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                mininterval=1.0,
            )
            try:
                tensor: torch.Tensor | None = None
                for name, tensor in f.tensors():
                    pbar.update(tensor.numel() * tensor.element_size())
                    if weight_name_prefixes and not _matches_weight_name_prefixes(
                        name, weight_name_prefixes
                    ):
                        continue
                    if not copy_tensors:
                        # Parameter loaders consume borrowed tensors synchronously.
                        # A deferred loader must own the tensor before advancing.
                        tensor._vllm_instanttensor_borrowed = True
                    yield name, tensor
                # A generator frame retains its final DLPack-backed view. Drop it
                # before the InstantTensor context releases the staging ring.
                tensor = None
            finally:
                pbar.close()
        return

    assert max_gpu_tensor_size is not None
    assert selection is not None
    logger.info_once(
        "Loading %d tensors larger than the %d-byte InstantTensor buffer "
        "through CPU safetensors",
        len(cpu_fallbacks),
        max_gpu_tensor_size,
    )
    fallback_by_position = {fallback.position: fallback for fallback in cpu_fallbacks}
    with ExitStack() as stack:
        fallback_readers: dict[str, Any] = {}
        for cpu_fallback in cpu_fallbacks:
            if cpu_fallback.filename not in fallback_readers:
                fallback_readers[cpu_fallback.filename] = stack.enter_context(
                    safe_open(cpu_fallback.filename, framework="pt", device="cpu")
                )

        pbar = None
        gpu_tensors: Any = iter(())
        if selection.gpu_tensor_count:
            instant_reader = stack.enter_context(instant_open)
            pbar = tqdm(
                total=instant_reader.total_tensor_size,
                desc="Loading safetensors using InstantTensor loader",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
                position=tqdm._get_free_pos(),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                mininterval=1.0,
            )
            gpu_tensors = iter(instant_reader.tensors())

        try:
            gpu_tensor: torch.Tensor | None = None
            for position in range(selection.selected_tensor_count):
                scheduled_fallback = fallback_by_position.get(position)
                if scheduled_fallback is not None:
                    yield (
                        scheduled_fallback.name,
                        fallback_readers[scheduled_fallback.filename].get_tensor(
                            scheduled_fallback.name
                        ),
                    )
                    continue
                try:
                    name, gpu_tensor = next(gpu_tensors)
                except StopIteration as e:
                    raise RuntimeError(
                        "InstantTensor produced fewer tensors than the selected "
                        "checkpoint schedule"
                    ) from e
                if pbar is not None:
                    pbar.update(gpu_tensor.numel() * gpu_tensor.element_size())
                if not copy_tensors:
                    gpu_tensor._vllm_instanttensor_borrowed = True
                yield name, gpu_tensor

            if next(gpu_tensors, None) is not None:
                raise RuntimeError(
                    "InstantTensor produced more tensors than the selected "
                    "checkpoint schedule"
                )
            gpu_tensor = None
        finally:
            if pbar is not None:
                pbar.close()
