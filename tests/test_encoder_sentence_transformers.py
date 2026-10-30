"""sentence-transformers models without sentence-transformers: what the reader reads, the
format of its encoders, and every refusal, on folders written by hand in ST's layout.

The comparison with sentence-transformers itself is
tests/test_encoder_sentence_transformers_parity.py.
"""

import json

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._format import Source
from support import converted as cv
from support import sentence_transformers as st
from support import wordpiece as wp

pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")
TOKENS = wp.vocabulary()
UNK = TOKENS.index("[UNK]")
CARD = "---\nlicense: apache-2.0\nlibrary_name: sentence-transformers\n---\n# card\n"


@pytest.fixture
def model(tmp_path):
    folder = st.write_model(tmp_path / "model")
    (folder / "README.md").write_text(CARD)
    return folder


def _modules(folder, *modules):
    (folder / "modules.json").write_text(json.dumps(list(modules)))


def _static(path="0_StaticEmbedding", name="0", type_=st.LEGACY, **extra):
    return {"idx": 0, "name": name, "path": path, "type": type_, **extra}


def test_a_static_embedding_is_converted_with_its_rules(model, tmp_path):
    encoder = Encoder.from_pretrained(model)
    spec = encoder._content.tokenizer
    assert encoder.tokenizer == "wordpiece@1"
    assert (spec.unknown, spec.max_tokens, spec.max_chars) == ("keep", None, None)
    assert encoder.steps == ()
    assert encoder.output_dim == st.DIM
    assert encoder._content.source == Source("sentence-transformers", None, None)
    assert encoder._content.license == "apache-2.0"
    assert "unknown ids kept" in repr(encoder)
    texts = ["zzzq", "", "hello [CLS] world [SEP]", "[UNK] [PAD]"]
    ids = [i.tolist() for i in cv.ids_of(encoder, texts)]
    assert ids == wp.hf_ids(st.tokenizer(), texts, unk_id=None)
    assert ids[0] == [UNK]  # an unknown word pools the row of [UNK]
    rows = encoder.encode(texts)
    assert rows[0].tobytes() == st.table(len(TOKENS))[UNK].tobytes()
    assert not rows[1].any()
    encoder.save(tmp_path / "e.safetensors")
    loaded = Encoder.load(tmp_path / "e.safetensors")
    assert loaded.fingerprint == encoder.fingerprint
    assert loaded.encode(texts).tobytes() == rows.tobytes()


def test_long_texts_are_read_whole(model):
    """No length rule: tokens after 512 and after 4,096 count (the Model2Vec reader would
    cut at 512), and empty texts between long ones stay empty."""
    encoder = Encoder.from_pretrained(model)
    words = [t for t in TOKENS[5:] if not t.startswith(("##", "[", "<")) and t.isalpha()]
    head = " ".join(words[i % 50] for i in range(4096))
    tails = [words[60], words[61]]
    texts = [head + " " + tails[0], "", head + " " + tails[1], "", head[:3000]]
    ids = cv.ids_of(encoder, texts)
    assert len(ids[0]) > 4096
    assert ids[0][-1] == TOKENS.index(tails[0])
    assert ids[2][-1] == TOKENS.index(tails[1])
    rows = encoder.encode(texts)
    assert not rows[1].any()
    assert not rows[3].any()
    assert rows[0].tobytes() != rows[2].tobytes()
    table = st.table(len(TOKENS)).astype(np.float64)
    assert np.allclose(rows[0], table[ids[0]].mean(axis=0), atol=1e-5)


@pytest.mark.parametrize("kept", [4, st.DIM, None])
def test_truncate_dim_keeps_the_first_dimensions(model, kept):
    full = Encoder.from_pretrained(model)
    cut = Encoder.from_pretrained(model, truncate_dim=kept)
    texts = ["hello world", "zzz", "", "the cat sat on the mat"]
    width = st.DIM if kept is None else kept
    assert cut.output_dim == width
    assert (
        cut.encode(texts).tobytes() == np.ascontiguousarray(full.encode(texts)[:, :width]).tobytes()
    )
    dims = None if width == st.DIM else (st.DIM, width)
    assert cut._content.source == Source("sentence-transformers", None, None, dims)


def test_truncate_dim_of_the_config_and_its_overrides(model):
    config = json.loads((model / "config_sentence_transformers.json").read_text(encoding="utf-8"))
    (model / "config_sentence_transformers.json").write_text(
        json.dumps(config | {"truncate_dim": 4})
    )
    assert Encoder.from_pretrained(model).output_dim == 4
    assert Encoder.from_pretrained(model, truncate_dim="model").output_dim == 4
    assert Encoder.from_pretrained(model, truncate_dim=None).output_dim == st.DIM
    assert Encoder.from_pretrained(model, truncate_dim=st.DIM).output_dim == st.DIM
    assert Encoder.from_pretrained(model, truncate_dim=8).output_dim == 8
    with pytest.raises(ValueError, match="more than the 16 dimensions"):
        Encoder.from_pretrained(model, truncate_dim=17)
    (model / "config_sentence_transformers.json").write_text(
        json.dumps(config | {"truncate_dim": 99})
    )
    with pytest.raises(ValueError, match="states more than the 16"):
        Encoder.from_pretrained(model)


def test_normalize_is_fse_s_own_after_the_cut(model):
    plain = Encoder.from_pretrained(model, truncate_dim=4)
    normal = Encoder.from_pretrained(model, truncate_dim=4, normalize=True)
    assert normal.steps == ("normalize@1",)
    texts = ["hello world", "zzz", ""]
    raw, rows = plain.encode(texts), normal.encode(texts)
    norms = np.linalg.norm(raw[:2], axis=1, keepdims=True)
    assert np.allclose(rows[:2], raw[:2] / norms, atol=1e-6)
    assert not rows[2].any()


def test_a_plain_tokenizer_truncation_is_the_token_limit(model):
    path = model / "0_StaticEmbedding" / "tokenizer.json"
    st.tokenizer(truncation={"max_length": 3}).save(str(path))
    encoder = Encoder.from_pretrained(model)
    assert encoder._content.tokenizer.max_tokens == 3
    texts = ["hello world the cat sat", "zzz zzz zzz zzz"]
    assert [i.tolist() for i in cv.ids_of(encoder, texts)] == wp.hf_ids(
        st.tokenizer(truncation={"max_length": 3}), texts, unk_id=None
    )
    whole = Encoder.from_pretrained(model, truncate=False)
    assert whole._content.tokenizer.max_tokens is None


@pytest.mark.parametrize("spelling", [st.LEGACY, st.V5])
@pytest.mark.parametrize("path", ["0_StaticEmbedding", "", "."])
@pytest.mark.parametrize("tensor", ["embedding.weight", "embeddings"])
def test_the_layouts_of_the_versions(tmp_path, spelling, path, tensor):
    rows = st.table(len(TOKENS))
    folder = st.write_model(tmp_path / "m", rows, tensors={tensor: rows})
    if path != "0_StaticEmbedding":
        for name in ("model.safetensors", "tokenizer.json"):
            (folder / "0_StaticEmbedding" / name).rename(folder / name)
    _modules(folder, _static(path, type_=spelling))
    want = Encoder.from_pretrained(st.write_model(tmp_path / "ref", rows))
    assert Encoder.from_pretrained(folder).fingerprint == want.fingerprint


def _normalize(path="1_Normalize", spelling="sentence_transformers.models.Normalize"):
    return {"idx": 1, "name": "1", "path": path, "type": spelling}


def _dense():
    return {"idx": 1, "name": "1", "path": "1_Dense", "type": "sentence_transformers.models.Dense"}


@pytest.mark.parametrize(
    "spelling",
    [
        "sentence_transformers.models.Normalize",
        "sentence_transformers.sentence_transformer.modules.normalize.Normalize",
        "sentence_transformers.base.modules.normalize.Normalize",
    ],
)
def test_a_normalize_module_is_refused(model, spelling):
    _modules(model, _static(), _normalize(spelling=spelling))
    with pytest.raises(UnsupportedArtifactError, match=r"Normalize module .* max\(norm, 1e-12\)"):
        Encoder.from_pretrained(model)


@pytest.mark.parametrize(
    ("activation", "match"),
    [
        ("torch.nn.modules.linear.Identity", "Dense module .* does not fold"),
        ("torch.nn.modules.activation.Tanh", "no static embedding model"),
        (None, "no static embedding model"),
    ],
)
def test_a_dense_module_is_refused(model, activation, match):
    _modules(model, _static(), _dense())
    if activation is not None:
        (model / "1_Dense").mkdir()
        (model / "1_Dense" / "config.json").write_text(
            json.dumps({"in_features": 16, "out_features": 4, "activation_function": activation})
        )
    with pytest.raises(UnsupportedArtifactError, match=match):
        Encoder.from_pretrained(model)


@pytest.mark.parametrize(
    "types",
    [
        ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling"],
        ["sentence_transformers.models.Pooling"],
        [st.LEGACY, st.LEGACY],
        ["sentence_transformers.models.Normalize", st.LEGACY],
        ["modeling_custom.Static"],
        [st.LEGACY, "sentence_transformers.models.Pooling"],
    ],
)
def test_models_of_other_modules_are_no_static_models(model, types):
    _modules(
        model, *[_static(type_=t, name=str(k), idx=k) | {"idx": k} for k, t in enumerate(types)]
    )
    with pytest.raises(
        UnsupportedArtifactError, match=r"no static embedding model: its modules are"
    ):
        Encoder.from_pretrained(model)


@pytest.mark.parametrize(
    ("module", "error", "match"),
    [
        (_static(name="0.a"), CorruptArtifactError, "name of a module"),
        (_static(name=""), CorruptArtifactError, "name of a module"),
        (_static(name=0), CorruptArtifactError, "name of a module"),
        (_static(path=".."), CorruptArtifactError, "path of a module"),
        (_static(path="/tmp"), CorruptArtifactError, "path of a module"),
        (_static(path="a/b"), CorruptArtifactError, "path of a module"),
        (_static(path="a\\b"), CorruptArtifactError, "path of a module"),
        (_static(kwargs=["task"]), UnsupportedArtifactError, "kwargs"),
        ({"name": "0", "path": "", "type": st.LEGACY}, CorruptArtifactError, "holds exactly"),
    ],
)
def test_the_module_entry_is_checked(model, module, error, match):
    _modules(model, module)
    with pytest.raises(error, match=match):
        Encoder.from_pretrained(model)


def test_modules_json_is_a_list_of_modules(model):
    for value in ({}, [], [1], [{"path": ""}]):
        (model / "modules.json").write_text(json.dumps(value))
        with pytest.raises(CorruptArtifactError, match="must be a list of modules"):
            Encoder.from_pretrained(model)


@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        ({"model_type": "SparseEncoder"}, UnsupportedArtifactError, "not a SentenceTransformer"),
        ({"prompts": {"query": "query: "}}, UnsupportedArtifactError, r"prompts \['query'\]"),
        (
            {"prompts": {"q": "query: "}, "default_prompt_name": "q"},
            UnsupportedArtifactError,
            "prompts",
        ),
        ({"default_prompt_name": "q"}, CorruptArtifactError, "is no prompt"),
        ({"prompts": {"q": 1}}, CorruptArtifactError, "map names to strings"),
        ({"pooling": "max"}, UnsupportedArtifactError, r"does not know: \['pooling'\]"),
        ({"truncate_dim": 0}, CorruptArtifactError, "truncate_dim must be"),
        ({"truncate_dim": True}, CorruptArtifactError, "truncate_dim must be"),
        ([], CorruptArtifactError, "must be a JSON object"),
    ],
)
def test_the_config_is_checked(model, config, error, match):
    (model / "config_sentence_transformers.json").write_text(json.dumps(config))
    with pytest.raises(error, match=match):
        Encoder.from_pretrained(model)


def test_empty_prompts_and_provenance_are_accepted(model):
    config = {
        "__version__": {"sentence_transformers": "6.2.0"},
        "model_type": "SentenceTransformer",
        "prompts": {"query": "", "document": ""},
        "default_prompt_name": "query",
        "similarity_fn_name": "dot",
        "requirements": {"transformers": ">=5"},
    }
    (model / "config_sentence_transformers.json").write_text(json.dumps(config))
    Encoder.from_pretrained(model)
    (model / "config_sentence_transformers.json").unlink()  # optional
    Encoder.from_pretrained(model)


def test_the_table_is_checked(tmp_path):
    rows = st.table(len(TOKENS))
    cases = [
        ({"embedding.weight": rows[:-1]}, CorruptArtifactError, "rows, but the tokenizer has"),
        ({"embedding.weight": rows, "x": rows}, CorruptArtifactError, "exactly one tensor"),
        ({"weight": rows}, CorruptArtifactError, "exactly one tensor"),
        ({"embeddings": rows, "mapping": rows[:, 0]}, UnsupportedArtifactError, "quantization"),
        ({"embedding.weight": rows.astype(np.float16)}, UnsupportedArtifactError, "float16"),
        ({"embedding.weight": rows[0]}, CorruptArtifactError, "must be a matrix"),
    ]
    for k, (tensors, error, match) in enumerate(cases):
        folder = st.write_model(tmp_path / str(k), tensors=tensors)
        with pytest.raises(error, match=match):
            Encoder.from_pretrained(folder)


def test_a_pickle_or_a_missing_table_is_refused(model):
    (model / "0_StaticEmbedding" / "model.safetensors").unlink()
    with pytest.raises(CorruptArtifactError, match=r"no file 0_StaticEmbedding/model\.safetensors"):
        Encoder.from_pretrained(model)
    (model / "0_StaticEmbedding" / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(UnsupportedArtifactError, match="a pickle file"):
        Encoder.from_pretrained(model)


@pytest.mark.parametrize(
    "truncation",
    [
        {"max_length": 3, "direction": "Left", "strategy": "LongestFirst", "stride": 0},
        {"max_length": 3, "direction": "Right", "strategy": "LongestFirst", "stride": 1},
        {"max_length": 3, "direction": "Right", "strategy": "OnlySecond", "stride": 0},
        {"max_length": 0, "direction": "Right", "strategy": "LongestFirst", "stride": 0},
    ],
)
def test_another_tokenizer_truncation_is_refused(model, truncation):
    path = model / "0_StaticEmbedding" / "tokenizer.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["truncation"] = truncation
    path.write_text(json.dumps(document))
    with pytest.raises((UnsupportedArtifactError, CorruptArtifactError), match="truncation"):
        Encoder.from_pretrained(model)


def test_the_arguments_are_checked(model, tmp_path):
    with pytest.raises(ValueError, match="max_length is a length rule of Model2Vec"):
        Encoder.from_pretrained(model, max_length=8)
    with pytest.raises(ValueError, match="mmap is for fse encoders"):
        Encoder.from_pretrained(model, mmap=True)
    with pytest.raises(ValueError, match="validation error"):
        Encoder.from_pretrained(model, truncate_dim=0)
    encoder = Encoder.from_pretrained(model)
    encoder.save(tmp_path / "e.safetensors")
    for options in ({"truncate_dim": 4}, {"normalize": True}):
        with pytest.raises(ValueError, match="holds its own rules"):
            Encoder.from_pretrained(tmp_path / "e.safetensors", **options)
    pytest.importorskip("model2vec")
    import support.model2vec as mv

    model2vec = mv.save_model(tmp_path / "m2v")
    for options in ({"truncate_dim": 4}, {"normalize": False}):
        with pytest.raises(ValueError, match="holds its own rules"):
            Encoder.from_pretrained(model2vec, **options)


def test_detection_without_modules_json(tmp_path):
    folder = tmp_path / "st"
    folder.mkdir()
    (folder / "config_sentence_transformers.json").write_text("{}")
    with pytest.raises(UnsupportedArtifactError, match=r"no modules\.json"):
        Encoder.from_pretrained(folder)
