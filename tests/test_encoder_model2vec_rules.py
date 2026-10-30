"""A model read under the rules of model2vec 0.10.0: its three folder layouts and their
priority, the config of each, the card fields, dimensionality and subfolder. The folders
are tiny, written by model2vec or by hand in the layout of sentence-transformers."""

import json
import shutil

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._encoder._model2vec_rules import (
    LAYOUTS,
    UNSET,
    card_fields,
    find_rules,
    reduced,
    subfolder_of,
)
from fse._encoder._pretrained import LocalFiles
from support import sentence_transformers as stm
from support import static_model as sm

model2vec = pytest.importorskip("model2vec")
pytest.importorskip("safetensors")


def _rules(folder, *, normalize=None, max_length=UNSET, dimensionality=None, subfolder=None):
    files = subfolder_of(LocalFiles(folder), subfolder)
    return find_rules(
        files, normalize=normalize, max_length=max_length, dimensionality=dimensionality
    )


@pytest.mark.parametrize(
    ("layout", "config"),
    [
        ("model2vec", None),
        ("st-root", None),
        ("st-nested", None),
        ("st-root", {"normalize": True, "max_length": 7}),
        ("st-nested", {"normalize": True, "max_length": 7}),
    ],
)
def test_every_layout_gives_the_attributes_of_model2vec(tmp_path, layout, config):
    folder = sm.model_folder(tmp_path / "m", layout, config=config)
    rules = _rules(folder)
    want = model2vec.StaticModel.from_pretrained(folder)
    assert list(rules.tokens) == list(want.tokens)
    assert np.array_equal(rules.word_vectors.stored_vectors, want.embedding)
    assert rules.parameters.unk_id == want.unk_token_id
    assert rules.median_token_length == want.median_token_length
    assert (rules.normalize, rules.max_length) == (want.normalize, want.max_length)
    assert rules.config == {k: v for k, v in want.config.items() if k in rules.config}
    assert (rules.base_model_name, rules.language) == (want.base_model_name, want.language)


def test_a_layout_is_taken_in_the_order_of_model2vec(tmp_path):
    """The first layout whose three files exist wins, whatever the others hold."""
    folder = sm.model_folder(tmp_path / "m", "model2vec")
    nested = sm.model_folder(tmp_path / "n", "st-nested", config={"normalize": False})
    shutil.copytree(nested / "0_StaticEmbedding", folder / "0_StaticEmbedding")
    shutil.copy(nested / "config_sentence_transformers.json", folder)
    rules = _rules(folder)
    assert rules.normalize is True  # the config.json of layout 1, not the other config
    assert "embedding_dtype" in rules.config
    # without the table of layout 1 the layout is no longer complete: the nested one is read
    (folder / "model.safetensors").unlink()
    rules = _rules(folder)
    assert rules.normalize is False
    assert "embedding_dtype" not in rules.config
    assert [layout.sentence_transformers for layout in LAYOUTS] == [False, True, True]


def test_the_config_of_a_sentence_transformers_layout_gives_normalize_and_max_length(tmp_path):
    for config in ({"normalize": True}, {"max_length": None}, {"max_length": 9, "x": [1]}):
        folder = sm.model_folder(
            tmp_path / f"m{len(list(tmp_path.iterdir()))}", "st-root", config=config
        )
        rules = _rules(folder)
        want = model2vec.StaticModel.from_pretrained(folder)
        assert (rules.normalize, rules.max_length) == (want.normalize, want.max_length)
    rules = _rules(folder, normalize=False, max_length=3)
    assert (rules.normalize, rules.max_length) == (False, 3)
    assert _rules(folder, max_length=None).max_length is None


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"normalize": 1}, "normalize must be true or false"),
        ({"max_length": 0}, "max_length must be a positive int or null"),
        ({"max_length": True}, "max_length must be a positive int or null"),
        ({"max_length": "5"}, "max_length must be a positive int or null"),
    ],
)
def test_a_wrong_config_of_a_sentence_transformers_layout_raises(tmp_path, config, match):
    folder = sm.model_folder(tmp_path / "m", "st-root", config=config)
    with pytest.raises(CorruptArtifactError, match=match):
        _rules(folder)
    (folder / "config_sentence_transformers.json").write_text("[1]")
    with pytest.raises(CorruptArtifactError, match="must be a JSON object"):
        _rules(folder)


def test_a_model2vec_config_is_checked_as_the_encoder_checks_it(tmp_path):
    folder = sm.model_folder(tmp_path / "m")
    config = json.loads((folder / "config.json").read_text())
    (folder / "config.json").write_text(json.dumps({**config, "surprise": 1}))
    with pytest.raises(UnsupportedArtifactError, match="keys this fse does not know"):
        _rules(folder)


def test_a_table_is_read_under_the_name_model2vec_reads(tmp_path):
    from safetensors.numpy import load_file, save_file

    for layout, wrong in (("st-root", "embeddings"), ("model2vec", "embedding.weight")):
        folder = sm.model_folder(tmp_path / layout, layout)
        path = folder / "model.safetensors"
        (name,) = load_file(path)
        save_file({wrong: load_file(path)[name]}, path)
        with pytest.raises(CorruptArtifactError, match="must hold exactly"):
            _rules(folder)


def test_no_layout_raises_and_an_fse_folder_gives_none(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(UnsupportedArtifactError, match="none of the folder layouts"):
        _rules(tmp_path / "empty")
    folder = sm.model_folder(tmp_path / "m")
    (folder / "tokenizer.json").unlink()
    with pytest.raises(UnsupportedArtifactError, match="none of the folder layouts"):
        _rules(folder)
    from fse._encoder import Encoder

    source = sm.model_folder(tmp_path / "src")
    fse_folder = Encoder.from_pretrained(source).save_pretrained(tmp_path / "fse")
    assert _rules(fse_folder) is None


def test_a_module_other_than_a_static_embedding_or_a_normalize_is_refused(tmp_path):
    folder = sm.model_folder(tmp_path / "m", "st-nested")
    static = json.loads((folder / "modules.json").read_text())[0]
    for kind, accepted in (("Normalize", True), ("Dense", False), ("Pooling", False)):
        module = {
            "idx": 1,
            "name": "1",
            "path": "1_x",
            "type": f"sentence_transformers.models.{kind}",
        }
        (folder / "modules.json").write_text(json.dumps([static, module]))
        if accepted:
            assert _rules(folder).normalize is False  # model2vec ignores the module
        else:
            with pytest.raises(UnsupportedArtifactError, match="no static embedding model"):
                _rules(folder)
    (folder / "modules.json").unlink()  # model2vec does not need the file
    assert _rules(folder).normalize is False
    (folder / "modules.json").write_text("{}")
    with pytest.raises(CorruptArtifactError, match="must be a list of modules"):
        _rules(folder)


@pytest.mark.parametrize("layout", sm.LAYOUTS)
def test_dimensionality_keeps_the_first_columns(tmp_path, layout):
    folder = sm.model_folder(tmp_path / "m", layout)
    rules = _rules(folder, dimensionality=5)
    want = model2vec.StaticModel.from_pretrained(folder, dimensionality=5)
    table = rules.word_vectors.stored_vectors
    assert np.array_equal(table, want.embedding)
    assert table.flags.c_contiguous
    assert not table.flags.writeable
    assert rules.source.dims == (stm.DIM, 5)
    whole = _rules(folder, dimensionality=stm.DIM)
    assert whole.source.dims is None
    assert whole.word_vectors.dim == stm.DIM
    with pytest.raises(ValueError, match="greater than the 16 dimensions"):
        _rules(folder, dimensionality=17)


def test_reduced_copies_only_when_it_cuts():
    table = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert reduced(table, None)[0] is table
    assert reduced(table, None)[1] is None
    assert reduced(table, 4)[0] is table
    cut, kept = reduced(table, 2)
    assert kept == 2
    assert cut.base is None
    assert np.array_equal(cut, table[:, :2])


def test_a_subfolder_holds_the_model(tmp_path):
    for layout in sm.LAYOUTS:
        folder = tmp_path / layout
        inner = sm.model_folder(tmp_path / f"{layout}-inner", layout)
        shutil.copytree(inner, folder / "a" / "b")
        (folder / "README.md").write_text("---\nlanguage: de\n---\n")
        rules = _rules(folder, subfolder="a/b")
        want = model2vec.StaticModel.from_pretrained(folder, subfolder="a/b")
        assert list(rules.tokens) == list(want.tokens)
        assert rules.language is None  # the card of the subfolder, as model2vec reads it
        assert (rules.normalize, rules.max_length) == (want.normalize, want.max_length)
        with pytest.raises(UnsupportedArtifactError, match="none of the folder layouts"):
            _rules(folder)


@pytest.mark.parametrize("subfolder", ["", "/a", "a/", "a//b", "../a", "a/..", "a\\b", ".", "a b"])
def test_a_subfolder_is_a_path_of_plain_names(tmp_path, subfolder):
    with pytest.raises(ValueError, match="plain names"):
        _rules(tmp_path, subfolder=subfolder)


FRONT_MATTERS = [
    "base_model: baai/bge-base-en-v1.5\nlanguage:\n- en\n- multilingual\nlicense: mit",
    "language: [en, de]\nbase_model:\n  - a/b\n  - c/d",
    "language: en  # the language\nbase_model: 'x/y'",
    'base_model: "x/y"\nlanguage: "zh-Hans"',
    "base_model: ~\nlanguage: null",
    "license: mit\nbase_model_relation: finetune\nwidget:\n- text: hello\n  language: fr",
    "language:\n\n# a comment\n- en\n\n- fr",
    "library_name: model2vec",
    # YAML 1.1 words PyYAML (and so huggingface_hub and model2vec) reads as strings here.
    "language: y\nbase_model: n",
    "base_model: &anchor x/y\nlanguage: [en,]",
    '"language": en\n"base_model": a/b',
]


@pytest.mark.parametrize("front", FRONT_MATTERS)
def test_the_card_fields_are_those_model2vec_reads(tmp_path, front):
    from model2vec.modelcards import get_metadata_from_readme

    path = tmp_path / "README.md"
    path.write_text(f"---\n{front}\n---\n# card\n", encoding="utf-8")
    want = get_metadata_from_readme(path)
    got = card_fields(path.read_bytes(), path)
    assert got == {"base_model": want.get("base_model"), "language": want.get("language")}


def test_a_card_without_front_matter_has_no_fields(tmp_path):
    path = tmp_path / "README.md"
    assert card_fields(None, path) == {"base_model": None, "language": None}
    assert card_fields(b"# only a title\n", path) == {"base_model": None, "language": None}


@pytest.mark.parametrize(
    ("front", "match"),
    [
        ("language: en\nlanguage: de", "language 2 times"),
        ("language: en\n- de", "cannot read as YAML: expected <block end>"),
        ("language:\n- en\nlanguage: [", "cannot read as YAML"),
        ("language: []", "list fse cannot read"),
        ("language: [en", "cannot read as YAML"),
        ("language: ''", "language fse cannot read: ''"),
        ('language: ""', "language fse cannot read: ''"),
        ('language: "日本語"', "language fse cannot read: '日本語'"),
        ("language: 日本語", "language fse cannot read: '日本語'"),
        ('language: "a b"', "language fse cannot read: 'a b'"),
        ("language: [en, null]", "language fse cannot read: None"),
        ("language: yes", "language fse cannot read"),
        ("language:\n- no", "language fse cannot read"),
        ("base_model: 12", "base_model fse cannot read"),
        ("base_model: a b", "base_model fse cannot read"),
        ("base_model: 'a'b'", "cannot read as YAML"),
        ("base_model: {a: b}", "base_model fse cannot read: a mapping"),
        ('base_model: "a b"', "base_model fse cannot read: 'a b'"),
        ('base_model: "a\\nb"', "base_model fse cannot read: 'a"),
        ("base_model: ''", "base_model fse cannot read: ''"),
        ('base_model: x\n"base_model": y', "base_model 2 times"),
        ("base_model: x\nlanguage: !!int abc", "cannot read as YAML"),
        ("base_model: x\nfoo: [", "cannot read as YAML"),
        ("language: en\n<<: {a: b}", "merges keys"),
    ],
)
def test_a_card_field_in_another_form_raises(tmp_path, front, match):
    path = tmp_path / "README.md"
    with pytest.raises(CorruptArtifactError, match=match):
        card_fields(f"---\n{front}\n---\n".encode(), path)
