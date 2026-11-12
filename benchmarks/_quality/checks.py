"""Explicit model rejections, distinct from infrastructure and provenance failures."""

from __future__ import annotations


class ModelCheckError(ValueError):
    """A loaded model violates the fixed quality or prompt contract."""


def cuda_model(native):
    import torch

    parameters = list(native.parameters())
    if native.device.type != "cuda" or any(p.device.type != "cuda" for p in parameters):
        raise ModelCheckError("GPU quality model left CUDA; CPU fallback is forbidden")
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ModelCheckError("GPU quality requires observed float32 weights")
    if type(list(native)[-1]).__name__ != "Normalize":
        raise ModelCheckError("GPU quality reference must end in Normalize")
