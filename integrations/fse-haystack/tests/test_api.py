from threading import get_ident

import pytest
from haystack import Document, Pipeline
from haystack_integrations.components.embedders.fse import FseDocumentEmbedder, FseTextEmbedder

from fse import Encoder


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_local_vectors(encoder, dtype):
    model = FseDocumentEmbedder(encoder=encoder, dtype=dtype, workers=1, batch_size=2)
    docs = [
        Document(content=text, meta={"label": i})
        for i, text in enumerate(["hello world", "", "unknown", "world", "hello"])
    ]
    rows = encoder.encode([doc.content for doc in docs], dtype=dtype, workers=1).tolist()
    result = model.run(docs)["documents"]
    assert [doc.embedding for doc in result] == rows
    assert all(doc.embedding is None for doc in docs)
    assert [doc.meta for doc in result] == [doc.meta for doc in docs]
    assert [doc.id for doc in result] == [doc.id for doc in docs]
    assert model.run([]) == {"documents": []}
    assert FseTextEmbedder(encoder=encoder, dtype=dtype).run("hello")["embedding"] == rows[-1]


def test_pipeline_and_serialization(encoder, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        Encoder, "from_pretrained", lambda *a, **kw: calls.append((a, kw)) or encoder
    )
    pipeline = Pipeline()
    pipeline.add_component(
        "embed",
        FseTextEmbedder("test/model", load_kwargs={"normalize": False, "cache_dir": tmp_path}),
    )
    restored = Pipeline.loads(pipeline.dumps())
    assert (
        restored.run({"embed": {"text": "hello"}})["embed"]["embedding"]
        == encoder.encode(["hello"]).tolist()[0]
    )
    assert calls == [(("test/model",), {"normalize": False, "cache_dir": str(tmp_path)})]
    restored.run({"embed": {"text": "world"}})
    assert len(calls) == 1


async def test_async_off_loop(encoder, monkeypatch):
    threads = []
    loop_thread = get_ident()
    model = FseTextEmbedder(encoder=encoder)
    sync = model.run

    def record(text):
        threads.append(get_ident())
        return sync(text)

    monkeypatch.setattr(model, "run", record)
    result = await model.run_async("hello")
    assert result["embedding"] == encoder.encode(["hello"]).tolist()[0]
    assert threads[0] != loop_thread
    docs = [Document(content="hello")]
    assert (await FseDocumentEmbedder(encoder=encoder).run_async(docs))["documents"][
        0
    ].embedding == result["embedding"]


def test_validation(encoder):
    with pytest.raises(RuntimeError, match="warm_up"):
        FseTextEmbedder().run("hello")
    with pytest.raises(ValueError, match="Save the encoder"):
        FseTextEmbedder(encoder=encoder).to_dict()
    with pytest.raises(ValueError, match="tokens"):
        FseTextEmbedder(load_kwargs={"token": "not-a-secret"}).to_dict()
    with pytest.raises(ValueError, match="text content"):
        FseDocumentEmbedder(encoder=encoder).run([Document()])
    with pytest.raises(TypeError, match="Documents"):
        FseDocumentEmbedder(encoder=encoder).run(["hello"])
    for batch in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="batch_size"):
            FseDocumentEmbedder(batch_size=batch)


@pytest.mark.network
@pytest.mark.enable_socket
def test_potion_retrieval():
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "retrieval.py"))


def test_local_artifact(encoder, tmp_path):
    path = tmp_path / "encoder.safetensors"
    encoder.save(path)
    model = FseTextEmbedder(model=str(path))
    model.warm_up()
    assert model.run("hello")["embedding"] == encoder.encode(["hello"]).tolist()[0]
    pipeline = Pipeline()
    pipeline.add_component("embed", model)
    clone = Pipeline.loads(pipeline.dumps())
    assert clone.run({"embed": {"text": "hello"}})["embed"] == model.run("hello")


async def test_cold_async_pipeline_loads_off_loop(encoder, monkeypatch):
    threads = []
    loop_thread = get_ident()

    def load(*_args, **_kwargs):
        threads.append(get_ident())
        return encoder

    monkeypatch.setattr(Encoder, "from_pretrained", load)
    pipeline = Pipeline()
    pipeline.add_component("embed", FseTextEmbedder("local"))
    result = await pipeline.run_async({"embed": {"text": "hello"}})
    assert result["embed"]["embedding"] == encoder.encode(["hello"]).tolist()[0]
    assert threads
    assert all(thread != loop_thread for thread in threads)
