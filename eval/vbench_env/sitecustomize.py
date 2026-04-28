# Loaded automatically when this directory is first on PYTHONPATH (see
# run_vbench_custom.sh). VBench motion_smoothness uses torch.load on older
# checkpoints; PyTorch 2.6+ defaults weights_only=True, which rejects them.
try:
    import torch

    _torch_load = torch.load

    def _load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _torch_load(*args, **kwargs)

    torch.load = _load  # type: ignore[method-assign]
except Exception:
    pass
