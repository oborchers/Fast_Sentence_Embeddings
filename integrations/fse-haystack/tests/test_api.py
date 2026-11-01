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


def test_pipeline_and_serialization(encoder, monkeypatch):
    calls = []
    monkeypatch.setattr(
        Encoder, "from_pretrained", lambda *a, **kw: calls.append((a, kw)) or encoder
    )
    pipeline = Pipeline()
    pipeline.add_component("embed", FseTextEmbedder("local", load_kwargs={"normalize": False}))
    restored = Pipeline.loads(pipeline.dumps())
    assert (
        restored.run({"embed": {"text": "hello"}})["embed"]["embedding"]
        == encoder.encode(["hello"]).tolist()[0]
    )
    assert calls == [(("local",), {"normalize": False})]
    restored.run({"embed": {"text": "world"}})
    assert len(calls) == 1


async def test_async_off_loop(encoder, monkeypatch):
    threads = []
    loop_thread = get_ident()
    sync = FseTextEmbedder.run

    def record(self, text):
        threads.append(get_ident())
        return sync(self, text)

    monkeypatch.setattr(FseTextEmbedder, "run", record)
    result = await FseTextEmbedder(encoder=encoder).run_async("hello")
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
def test_potion():
    model = FseTextEmbedder(workers=1)
    model.warm_up()
    assert len(model.run("hello world")["embedding"]) == 256


def test_local_artifact(encoder, tmp_path):
    path = tmp_path / "encoder.safetensors"
    encoder.save(path)
    model = FseTextEmbedder(model=str(path), load_kwargs={"local_files_only": True})
    model.warm_up()
    assert model.run("hello")["embedding"] == encoder.encode(["hello"]).tolist()[0]
    clone = FseTextEmbedder.from_dict(model.to_dict())
    clone.warm_up()
    assert clone.run("hello") == model.run("hello")
