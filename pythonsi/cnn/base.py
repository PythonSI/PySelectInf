from .CPU import CPUModel


def InferenceModel(model, device="cpu", img_shape=None):
    """Create the CPU or PyTorch-CUDA CNN affine inference backend."""
    if device == "cuda":
        from .CUDA import CUDAModel

        return CUDAModel(model, img_shape=img_shape)
    return CPUModel(model, img_shape=img_shape)
