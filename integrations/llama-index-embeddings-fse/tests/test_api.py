from threading import get_ident

import pytest
from llama_index.core.schema import TextNode
from llama_index.embeddings.fse import FseEmbedding

from fse import Encoder


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_public_api(encoder, dtype):
    model = FseEmbedding(encoder=encoder, dtype=dtype, workers=1, embed_batch_size=2)
    texts = ["hello world", "", "unknown", "world", "hello"]
    expected = encoder.encode(texts, dtype=dtype, workers=1).tolist()
    assert model.get_text_embedding_batch(texts) == expected
    assert model.get_query_embedding("hello") == expected[-1]
    assert model.get_text_embedding_batch([]) == []
    nodes = model([TextNode(text="hello"), TextNode(text="world")])
    assert [node.embedding for node in nodes] == [expected[-1], expected[-2]]


async def test_async_public_api(encoder, monkeypatch):
    model = FseEmbedding(encoder=encoder, embed_batch_size=1)
    loop_thread = get_ident()
    sync = FseEmbedding._get_text_embeddings
    threads = []

    def record(self, texts):
        threads.append(get_ident())
        return sync(self, texts)

    monkeypatch.setattr(FseEmbedding, "_get_text_embeddings", record)
    expected = encoder.encode(["hello", "world"]).tolist()
    assert await model.aget_text_embedding_batch(["hello", "world"]) == expected
    assert await model.aget_query_embedding("hello") == expected[0]
    assert await model.aget_text_embedding("world") == expected[1]
    assert all(thread != loop_thread for thread in threads)


def test_validation_and_loading(encoder, monkeypatch):
    calls = []
    monkeypatch.setattr(
        Encoder, "from_pretrained", lambda *a, **kw: calls.append((a, kw)) or encoder
    )
    FseEmbedding("local", load_kwargs={"normalize": False})
    assert calls == [(("local",), {"normalize": False})]
    with pytest.raises(ValueError, match="dtype"):
        FseEmbedding(encoder=encoder, dtype="int8")
    with pytest.raises(ValueError, match="embed_batch_size"):
        FseEmbedding(encoder=encoder, embed_batch_size=0)
    with pytest.raises(ValueError, match="load_kwargs"):
        FseEmbedding(encoder=encoder, load_kwargs={"normalize": False})


@pytest.mark.network
def test_potion():
    model = FseEmbedding(workers=1)
    assert len(model.get_query_embedding("hello world")) == 256
