"""Validate inherited LlamaIndex options before framework conversion."""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.rate_limiter import BaseRateLimiter
from llama_index.core.storage.kvstore.types import BaseKVStore


def _framework_integer(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    if name == "embed_batch_size" and value > 2048:
        raise ValueError("embed_batch_size must not exceed 2048")


def framework_options(options: dict[str, object]) -> None:
    """Validate original inherited fields before Pydantic converts them."""
    unknown = options.keys() - BaseEmbedding.model_fields.keys()
    if unknown:
        raise TypeError(f"Unknown BaseEmbedding options: {sorted(unknown)}")
    for name in ("embed_batch_size", "num_workers"):
        if name not in options or (name == "num_workers" and options[name] is None):
            continue
        _framework_integer(name, options[name])
    for name, expected in (
        ("callback_manager", CallbackManager),
        ("embeddings_cache", BaseKVStore),
        ("rate_limiter", BaseRateLimiter),
    ):
        value = options.get(name)
        if value is not None and not isinstance(value, expected):
            raise TypeError(f"{name} must be a {expected.__name__} or None")
