import asyncio
from threading import get_ident

import pytest
from langchain_fse import FseEmbeddings
from langchain_tests.integration_tests.embeddings import EmbeddingsIntegrationTests

from fse import Encoder


class TestStandardEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self):
        return FseEmbeddings

    @property
    def embedding_model_params(self):
        import numpy as np

        return {
            "encoder": Encoder.from_arrays(np.array([[1, 2]], dtype=np.float32), words=["hello"])
        }


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_local_vectors(encoder, dtype):
    model = FseEmbeddings(encoder=encoder, dtype=dtype, workers=1)
    texts = ["hello world", "", "unknown", "world"]
    assert model.embed_documents(texts) == encoder.encode(texts, dtype=dtype, workers=1).tolist()
    assert model.embed_query("hello") == encoder.encode(["hello"], dtype=dtype).tolist()[0]
    assert model.embed_documents([]) == []


async def test_async_runs_off_loop(encoder, monkeypatch):
    model = FseEmbeddings(encoder=encoder)
    loop_thread = get_ident()
    sync = model.embed_documents
    threads = []

    def record(texts):
        threads.append(get_ident())
        return sync(texts)

    monkeypatch.setattr(model, "embed_documents", record)
    expected = sync(["hello"])
    assert await model.aembed_documents(["hello"]) == expected
    assert await model.aembed_query("hello") == expected[0]
    assert all(thread != loop_thread for thread in threads)
    await asyncio.sleep(0)


def test_validation_and_loading(encoder, monkeypatch):
    calls = []
    monkeypatch.setattr(
        Encoder, "from_pretrained", lambda *a, **kw: calls.append((a, kw)) or encoder
    )
    FseEmbeddings("local", load_kwargs={"normalize": False})
    assert calls == [(("local",), {"normalize": False})]
    with pytest.raises(ValueError, match="dtype"):
        FseEmbeddings(encoder=encoder, dtype="int8")
    with pytest.raises(ValueError, match="load_kwargs"):
        FseEmbeddings(encoder=encoder, load_kwargs={"normalize": False})
    with pytest.raises((TypeError, ValueError)):
        FseEmbeddings(encoder=encoder).embed_documents([3])


@pytest.mark.network
def test_potion():
    model = FseEmbeddings(workers=1)
    assert len(model.embed_query("hello world")) == 256
