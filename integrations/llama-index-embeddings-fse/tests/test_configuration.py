"""Check inherited options before Pydantic can coerce or replace them."""

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.rate_limiter import BaseRateLimiter
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore
from llama_index.embeddings.fse import FseEmbedding

from fse import Encoder


@pytest.mark.parametrize("name", ["embed_batch_size", "num_workers"])
@pytest.mark.parametrize("value", [True, False, "2", 2.0, [], {}, {2}])
def test_invalid_integer_types_before_loading(monkeypatch, name, value):
    def unexpected_load(*_args, **_kwargs):
        pytest.fail("invalid configuration reached the loader")

    monkeypatch.setattr(Encoder, "from_pretrained", unexpected_load)
    with pytest.raises(TypeError, match=name):
        FseEmbedding(**{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [("embed_batch_size", 0), ("embed_batch_size", 2049), ("num_workers", 0), ("num_workers", -1)],
)
def test_invalid_integer_values(name, value):
    with pytest.raises(ValueError, match=name):
        FseEmbedding(**{name: value})


@pytest.mark.parametrize("name", ["callback_manager", "rate_limiter", "embeddings_cache"])
@pytest.mark.parametrize("value", [True, False, 1, "cache", [], {}, object()])
def test_invalid_framework_objects_before_loading(monkeypatch, name, value):
    def unexpected_load(*_args, **_kwargs):
        pytest.fail("invalid configuration reached the loader")

    monkeypatch.setattr(Encoder, "from_pretrained", unexpected_load)
    with pytest.raises(TypeError, match=name):
        FseEmbedding(**{name: value})


class RecordingLimiter(BaseRateLimiter):
    def __init__(self):
        self.calls = []

    def acquire(self, num_tokens=0):
        assert num_tokens == 0
        self.calls.append("sync")

    async def async_acquire(self, num_tokens=0):
        assert num_tokens == 0
        self.calls.append("async")


async def test_framework_objects_preserve_behavior(encoder):
    callbacks = CallbackManager([])
    limiter = RecordingLimiter()
    cache = SimpleKVStore()
    model = FseEmbedding(
        encoder=encoder,
        embed_batch_size=2048,
        num_workers=1,
        callback_manager=callbacks,
        rate_limiter=limiter,
        embeddings_cache=cache,
    )
    assert model.callback_manager is callbacks
    assert model.rate_limiter is limiter
    assert model.embeddings_cache is cache
    assert model.get_text_embedding("hello") == encoder.encode(["hello"]).tolist()[0]
    assert await model.aget_text_embedding("world") == encoder.encode(["world"]).tolist()[0]
    # Cache hits bypass both limiter and encoder in LlamaIndex.
    assert model.get_text_embedding("hello") == await model.aget_text_embedding("hello")
    assert limiter.calls == ["sync", "async"]


def test_optional_framework_defaults(encoder):
    model = FseEmbedding(
        encoder=encoder,
        callback_manager=None,
        rate_limiter=None,
        embeddings_cache=None,
        num_workers=None,
        embed_batch_size=1,
    )
    assert isinstance(model.callback_manager, CallbackManager)
    assert model.get_text_embedding("hello") == encoder.encode(["hello"]).tolist()[0]


def test_inherited_field_inventory():
    # New upstream fields require an explicit validation decision here.
    assert set(BaseEmbedding.model_fields) == {
        "model_name",
        "embed_batch_size",
        "callback_manager",
        "num_workers",
        "embeddings_cache",
        "rate_limiter",
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [("encoder_fingerprint", 1), ("encoder_injected", 1), ("embed_batch_size", None)],
)
def test_remaining_constructor_types(encoder, name, value):
    with pytest.raises(TypeError, match=name):
        FseEmbedding(encoder=encoder, **{name: value})
