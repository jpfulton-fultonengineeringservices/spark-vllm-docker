# Extracted unchanged from InstantTensor 0.2.2 in image
# sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02
from __future__ import annotations

import time
import warnings

class safe_open:
    def _finalize_buffer_size(self, buffer_size):
            if buffer_size is None:
                # make sure any two contiguous tensors will not be overlapped with each other in the buffer
                buffer_size_for_tensors = recommended_buffer_size_for_tensors(self.tensor_sizes)
                buffer_size_for_io = required_buffer_size_for_io(
                    self.chunk_size, self.io_depth, self.world_size,
                )
                self.buffer_size = max(buffer_size_for_tensors, buffer_size_for_io)
            else:
                self.buffer_size = buffer_size
                min_buffer_size = max(self.tensor_sizes, default=0)
                if self.buffer_size < min_buffer_size:
                    warnings.warn(
                        f"Enlarge buffer size from {self.buffer_size} to {min_buffer_size} to match the largest tensor size.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    self.buffer_size = min_buffer_size
    
                max_buffer_size = max(
                    self.total_tensor_size,
                    required_buffer_size_for_io(
                        self.chunk_size, self.io_depth, self.world_size,
                    ),
                )
                if self.buffer_size > max_buffer_size:
                    warnings.warn(
                        f"Shrink buffer size from {self.buffer_size} to {max_buffer_size} to avoid memory waste.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    self.buffer_size = max_buffer_size
    
            if self.buffer_size > self._device_memory_budget:
                raise RuntimeError(
                    f"buffer_size ({self.buffer_size} B) exceeds device memory "
                    f"budget ({self._device_memory_budget} B)"
                )
    
    def _open(self):
            self.open_time = time.perf_counter()
            nccl_communicator = self.process_group._get_backend(self.device)._comm_ptr() if self.process_group is not None else 0
            self.loader_handle = _C.open(
                self.filename, self.device_idx, nccl_communicator, self.buffer_size, 
                self.chunk_size, self.concurrency, self.io_depth, self.backend.value, self.tensor_offsets, self.tensor_keep)

    def __enter__(self) -> 'safe_open':
            if self.loader_handle is None:
                self._open()
            self.enter_time = time.perf_counter()
            return self
    
    def tensors(self) -> Generator[tuple[str, torch.Tensor], None, None]:
            """Iterate over all tensors in the safetensors file(s).
    
            Yields ``(name, tensor)`` pairs. With ``copy=True`` (default) tensors
            own their memory; with ``copy=False`` they are zero-copy views into
            the ring buffer (see ``safe_open``).
    
            Note:
                Synchronizes the current CUDA stream to ensure data transfer
                completion.
            """
            if self._invalidated:
                raise RuntimeError("tensors() called after safe_open context exited")
            if self.iterated:
                raise RuntimeError("tensors() can only be called once")
            self.iterated = True
            for tensor_index, (name, metadata) in enumerate(self.ordered_tensor_metadatas):
                stream = torch.cuda.current_stream()
                stream.synchronize()
                shape = metadata["shape"]
                safetensors_dtype = metadata["dtype"]
                torch_dtype = safetensors_to_torch_dtype.get(safetensors_dtype, None)
                if torch_dtype is None:
                    raise ValueError(f"Unsupported safetensors dtype: {safetensors_dtype}")
    
                tensor_size = get_tensor_size(shape, torch_dtype)
                dl_tensor = _C.get_dl_tensor(self.loader_handle, tensor_index, tensor_size) # always returns int8 tensor
                tensor_int8 = torch.from_dlpack(dl_tensor)
    
                required_alignment = torch.empty((), dtype=torch_dtype).element_size()
                is_aligned = tensor_int8.data_ptr() % required_alignment == 0
                if self.copy or not is_aligned:
                    if not self.copy:
                        warnings.warn(
                            f"Tensor {name} address {tensor_int8.data_ptr():#x} is not aligned "
                            f"to dtype {torch_dtype} size {required_alignment}B; falling back "
                            "to copy=True for this tensor",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    tensor_int8 = tensor_int8.clone()
                    # PyTorch's CUDA allocator is expected to return 512-byte-aligned
                    # storage. This is an implementation detail; 512 is divisible by
                    # every supported element size, and the check below remains authoritative.
                
                tensor = tensor_int8.view(torch_dtype).view(torch.Size(shape))
    
                if tensor.data_ptr() % tensor.element_size() != 0:
                    raise ValueError(f"Tensor {name} address {tensor.data_ptr():#x} is not aligned to dtype {torch_dtype} size {tensor.element_size()}B")
                yield name, tensor

def env_buffer_size():
    ret = os.environ.get("INSTANTTENSOR_BUFFER_SIZE")
    return int(ret) if ret is not None else None
