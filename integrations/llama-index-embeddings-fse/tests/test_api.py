import pickle
from threading import get_ident

import numpy as np
import pytest
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.ingestion.pipeline import get_transformation_hash, run_transformations
from llama_index.core.schema import TextNode
from llama_index.embeddings.fse import FseEmbedding

from fse import Encoder


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_public_api(encoder, dtype):
    model = FseEmbedding(encoder=encoder, dtype=dtype, workers=1, embed_batch_size=2)
    assert model.class_name() == "FseEmbedding"
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
    with pytest.raises(TypeError, match="Unknown BaseEmbedding"):
        FseEmbedding(encoder=encoder, misspelled_option=True)
    with pytest.raises(ValueError, match="dtype"):
        FseEmbedding(encoder=encoder, dtype="int8")
    with pytest.raises(ValueError, match="embed_batch_size"):
        FseEmbedding(encoder=encoder, embed_batch_size=0)
    with pytest.raises(ValueError, match="load_kwargs"):
        FseEmbedding(encoder=encoder, load_kwargs={"normalize": False})


@pytest.mark.network
@pytest.mark.enable_socket
def test_potion():
    model = FseEmbedding(workers=1)
    assert len(model.get_query_embedding("hello world")) == 256


def test_local_artifact(encoder, tmp_path):
    path = tmp_path / "encoder.safetensors"
    encoder.save(path)
    model = FseEmbedding(str(path))
    assert (
        model.get_text_embedding_batch(["hello", "world"])
        == encoder.encode(["hello", "world"]).tolist()
    )


def test_ingestion_cache_identity(encoder):
    other = Encoder.from_arrays(
        np.array([[2, 0, 1], [1, 2, 0]], dtype=np.float32), words=["hello", "world"]
    )
    first = FseEmbedding(encoder=encoder)
    second = FseEmbedding(encoder=other)
    half = FseEmbedding(encoder=encoder, dtype="float16")
    nodes = [TextNode(text="hello")]
    identities = {get_transformation_hash(nodes, model) for model in [first, second, half]}
    assert len(identities) == 3
    cache = IngestionCache()
    for model in [first, second, half]:
        result = run_transformations([TextNode(text="hello")], [model], cache=cache)
        assert result[0].embedding == model.get_text_embedding("hello")
    with pytest.raises(ValueError, match="injected encoder"):
        pickle.dumps(first)
    with pytest.raises(ValueError, match="injected encoder"):
        FseEmbedding.from_dict(first.to_dict())


def test_process_ingestion_keeps_options(encoder, tmp_path, monkeypatch):
    monkeypatch.setattr("llama_index.core.ingestion.pipeline.multiprocessing.cpu_count", lambda: 2)
    path = tmp_path / "encoder.safetensors"
    encoder.save(path)
    model = FseEmbedding(
        str(path),
        dtype="float16",
        workers=1,
        embed_batch_size=1,
        load_kwargs={"mmap": True},
    )
    restored = FseEmbedding.from_json(model.to_json())
    assert restored.get_query_embedding("hello") == model.get_query_embedding("hello")
    clone = pickle.loads(pickle.dumps(model))
    assert clone.dtype == "float16"
    assert clone.workers == 1
    assert clone.load_kwargs == {"mmap": True}
    assert clone.encoder_fingerprint == encoder.fingerprint
    pipeline = IngestionPipeline(transformations=[model])
    result = pipeline.run(nodes=[TextNode(text="hello"), TextNode(text="world")], num_workers=2)
    assert [node.embedding for node in result] == encoder.encode(
        ["hello", "world"], dtype="float16", workers=1
    ).tolist()
    data = model.to_dict()
    data["encoder_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        FseEmbedding.from_dict(data)


def test_hub_cache_path_json(encoder, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        Encoder, "from_pretrained", lambda *a, **kw: calls.append((a, kw)) or encoder
    )
    model = FseEmbedding("test/model", load_kwargs={"cache_dir": tmp_path})
    restored = FseEmbedding.from_json(model.to_json())
    assert restored.get_query_embedding("hello") == model.get_query_embedding("hello")
    assert calls == [
        (("test/model",), {"cache_dir": tmp_path}),
        (("test/model",), {"cache_dir": str(tmp_path)}),
    ]
