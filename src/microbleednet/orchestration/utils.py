import gc

import torch


def release_gpu_memory():
    """Reclaim unreferenced GPU memory back to the CUDA driver.

    Runs a garbage collection so any just-dropped tensors are finalized, then
    returns the caching allocator's now-free blocks to the driver. A no-op-ish
    call off CUDA. Call this after dropping the last reference to a model or
    tensor that is no longer needed, so its memory is freed promptly rather than
    at the end of the enclosing scope.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def delete_model(model):
    del model
    release_gpu_memory()
