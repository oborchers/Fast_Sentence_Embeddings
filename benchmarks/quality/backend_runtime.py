"""Separate actual model code identity from the common fse comparison reference."""

from __future__ import annotations

from compare_runtime import backend_identity

KINDS = {"fse": "fse", "model2vec": "model2vec", "sentence_transformers": "st"}


def model_runtime(model, environment: dict) -> dict:
    inner = getattr(model, "model", model)
    package = type(inner).__module__.split(".", maxsplit=1)[0]
    if package not in KINDS:
        raise ValueError(f"cannot identify quality backend package {package}")
    kind = KINDS[package]
    return {
        **backend_identity(kind),
        "backend_kind": kind,
        "comparison_identity_version": 3,
        "fse_reference_runtime_fingerprint": environment["fse_reference_runtime_fingerprint"],
    }
