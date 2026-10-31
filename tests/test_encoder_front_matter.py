"""The front matter of a model card: which block is read, that every YAML error is a
CorruptArtifactError naming the file, and that only the values of the keys fse reads, and
only scalars and flat lists of scalars, are ever constructed."""

import re
import time

import pytest
import yaml

from fse._container import CorruptArtifactError
from fse._encoder._front_matter import front_matter, front_matter_text, strings

KEYS = ("license", "base_model", "language")
ID = re.compile(r"[a-z]+")


def _read(tmp_path, front, keys=KEYS):
    path = tmp_path / "README.md"
    return front_matter(f"---\n{front}\n---\n# card\n".encode(), path, keys)


def _raises(tmp_path, front, match=None, keys=KEYS):
    path = tmp_path / "README.md"
    with pytest.raises(CorruptArtifactError, match=match) as info:
        front_matter(f"---\n{front}\n---\n".encode(), path, keys)
    assert str(info.value).startswith(str(path))  # the file first
    return info.value


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("---\na: b\n---\nbody", "a: b"),
        ("---\na: b\n...\nbody", "a: b"),
        ("---   \na: b\n---  \n", "a: b"),
        ("---\r\na: b\r\n---\r\n", "a: b"),
        ("---\n---\n", ""),
        ("---\na: b\n", None),  # not closed: no front matter
        ("a: b\n---\n", None),
        (" ---\na: b\n---\n", None),
        ("", None),
    ],
)
def test_the_front_matter_block(text, want):
    assert front_matter_text(text) == want


def test_a_card_without_front_matter_has_no_values(tmp_path):
    path = tmp_path / "README.md"
    assert front_matter(b"# a title\nlicense: mit\n", path, KEYS) == dict.fromkeys(KEYS)
    assert front_matter(b"---\nlicense: mit\n", path, KEYS) == dict.fromkeys(KEYS)


@pytest.mark.parametrize("front", ["", "# only a comment", "~", "null", "\n\n"])
def test_an_empty_or_null_front_matter_has_no_values(tmp_path, front):
    assert _read(tmp_path, front) == dict.fromkeys(KEYS)


def test_the_values_of_the_keys_fse_reads(tmp_path):
    front = "license: mit\nbase_model:\n- a\n- b\nlanguage: [en, de]\nother: 1"
    assert _read(tmp_path, front) == {
        "license": "mit",
        "base_model": ["a", "b"],
        "language": ["en", "de"],
    }
    assert _read(tmp_path, front, ("license",)) == {"license": "mit"}
    assert _read(tmp_path, "license: 1.5\nbase_model: [1, null, yes]")["base_model"] == [
        1,
        None,
        True,
    ]


def test_crlf_line_ends(tmp_path):
    path = tmp_path / "README.md"
    data = b"---\r\nlicense: mit\r\nlanguage:\r\n- en\r\n- de\r\n---\r\n# card\r\n"
    assert front_matter(data, path, KEYS)["language"] == ["en", "de"]


def test_a_front_matter_of_another_form_than_a_mapping_raises(tmp_path):
    _raises(tmp_path, "- mit", "must be a YAML mapping, not list")
    _raises(tmp_path, "mit", "must be a YAML mapping, not scalar")
    _raises(tmp_path, "[a, b]", "must be a YAML mapping, not list")


@pytest.mark.parametrize("merge", ["<<: {a: b}", "<<: [{a: b}]", "x: &x {a: b}\n<<: *x"])
def test_a_merge_on_the_root_raises(tmp_path, merge):
    _raises(tmp_path, f"license: mit\n{merge}", r"merges keys \(<<\)")


def test_a_merge_below_the_root_is_not_read(tmp_path):
    assert _read(tmp_path, "license: mit\nx:\n  <<: {a: b}")["license"] == "mit"


@pytest.mark.parametrize(
    "front",
    [
        "license: a\nlicense: b",
        'license: a\n"license": b',
        "license: a\n'license': b",
        "license: a\n? license\n: b",
        "license: a\n!!str license: b",
        "license: a\nlicense:",
    ],
)
def test_a_key_twice_raises_whichever_way_it_is_written(tmp_path, front):
    _raises(tmp_path, front, "license 2 times")


def test_only_the_keys_fse_reads_are_checked_for_duplicates(tmp_path):
    assert _read(tmp_path, "license: a\nx: 1\nx: 2")["license"] == "a"
    assert _read(tmp_path, "license: a\n1: x\n1: y")["license"] == "a"
    assert _read(tmp_path, "{license: a}")["license"] == "a"
    assert _read(tmp_path, '"license": a')["license"] == "a"


def test_a_nested_key_of_the_same_name_is_not_read(tmp_path):
    assert _read(tmp_path, "widget:\n- language: fr\n  license: x")["language"] is None


@pytest.mark.parametrize(
    "front",
    [
        "license: 'mit",
        "license: [mit",
        "license: mit\nx: {a: [",
        "license: mit\n- other",
        "license: *undefined",
        "license: mit\nx: *undefined",
        "license: mit\n\tx: 1",
        "license: mit\n%x",
        "license: mit\n@x: 1",
        "a: b: c",
    ],
)
def test_a_yaml_error_anywhere_raises(tmp_path, front):
    error = _raises(tmp_path, front, "front matter fse cannot read as YAML")
    assert isinstance(error.__cause__, yaml.YAMLError)


def test_the_line_of_a_yaml_error(tmp_path):
    _raises(
        tmp_path, "license: mit\nx: [", r"\(line 3\)"
    )  # the line of the file, the first being the ---
    _raises(
        tmp_path,
        "license: 2020-13-01",
        r"invalid timestamp value: month must be in 1\.\.12 \(line 2\)",
    )


@pytest.mark.parametrize(
    "front",
    [
        "license: \x00",
        "license: mit\nx: \x00",
        "license: mit\nx: \x07",
        "license: ﻿",
    ],
)
def test_a_nul_or_control_character_is_a_corrupt_card(tmp_path, front):
    """PyYAML raises a ReaderError, without a problem or a mark, already when the loader is
    made."""
    if front.endswith("﻿"):
        pytest.skip("a BOM is a printable character for PyYAML")
    error = _raises(tmp_path, front, "front matter fse cannot read as YAML")
    assert isinstance(error.__cause__, yaml.reader.ReaderError)


@pytest.mark.parametrize(
    "front",
    [
        "license: 2020-13-01",
        "license: 2020-02-30",
        "license: !!int abc",
        "license: !!float x",
        "license: !!bool maybe",
        "license: !!timestamp nope",
        "base_model: [a, !!int x]",
        "language: !!set {a}",
        "license: !!python/name:os.system",
        "license: !!python/object/apply:os.system [x]",
        "license: !!python/object:os.system {}",
        "license: !!unknown x",
    ],
)
def test_a_value_pyyaml_cannot_construct_raises_a_corrupt_card(tmp_path, front):
    """A ValueError of a constructor (an invalid date, a number of an explicit tag) and
    the ConstructorError of a tag no safe loader knows are CorruptArtifactError."""
    error = _raises(tmp_path, front, "cannot read as YAML|fse cannot read")
    assert isinstance(error.__cause__, (yaml.YAMLError, type(None)))


@pytest.mark.parametrize("front", ["license: !!int 1_0_", "license: !!binary '*'"])
def test_a_value_of_an_explicit_tag_is_checked_like_any_other(tmp_path, front):
    value = _read(tmp_path, front)["license"]
    assert not isinstance(value, str)
    with pytest.raises(CorruptArtifactError, match="license that is no string"):
        strings(value, "license", tmp_path, ID)


def test_the_python_tags_of_a_key_fse_reads_run_nothing(tmp_path):
    marker = tmp_path / "marker"
    front = f"license: !!python/object/apply:os.system ['touch {marker}']"
    _raises(tmp_path, front, "fse cannot read")
    _raises(tmp_path, f"license: !!python/object/apply:os.system\n  args: ['touch {marker}']")
    assert not marker.exists()


def test_the_unread_part_of_the_card_is_never_constructed(tmp_path, monkeypatch):
    """Tags that no loader knows, dates that do not exist, merges below the root and a
    Python object in a key fse does not read are only composed, never constructed."""
    marker = tmp_path / "marker"
    constructed = []
    original = yaml.SafeLoader.construct_object

    def spy(self, node, deep=False):
        constructed.append(node.value)
        return original(self, node, deep)

    monkeypatch.setattr(yaml.SafeLoader, "construct_object", spy)
    front = (
        "license: mit\n"
        f"a: !!python/object/apply:os.system ['touch {marker}']\n"
        "b: !!python/name:os.system\n"
        "c: 2020-13-01\n"
        "d: !!int abc\n"
        "e: {x: &x [1], y: *x, <<: {z: 1}}\n"
        "language: [en, de]"
    )
    assert _read(tmp_path, front)["license"] == "mit"
    assert not marker.exists()
    assert constructed == ["mit", "en", "de"]


@pytest.mark.parametrize(
    ("front", "kind"),
    [
        ("license: {a: b}", "mapping"),
        ("license: [a, [b]]", "list"),
        ("license: [a, {b: c}]", "list"),
        ("license:\n- a\n-\n  - b", "list"),
        ("license: !!set {a}", "mapping"),
        ("license: !!omap [a: b]", "list"),
        ("license: !!python/tuple [a]", "list"),
    ],
)
def test_a_value_that_is_no_scalar_or_flat_list_is_refused_unconstructed(
    tmp_path, monkeypatch, front, kind
):
    def refuse(*_args, **_kwargs):
        raise AssertionError("constructed")

    monkeypatch.setattr(yaml.SafeLoader, "construct_object", refuse)
    _raises(tmp_path, front, f"license fse cannot read: a {kind} that is no scalar")


def test_a_nesting_too_deep_to_compose_is_a_corrupt_card(tmp_path):
    for nested in ("[" * 100_000, "{a: " * 100_000, "- " * 50_000 + "x"):
        _raises(tmp_path, f"license: mit\nx: {nested}", "nested too deeply|cannot read as YAML")
    deep = "\n".join("  " * i + "a:" for i in range(5000))
    _raises(tmp_path, f"license: mit\nx:\n{deep}", "nested too deeply|cannot read as YAML")


def test_aliases_are_not_expanded(tmp_path):
    """The billion laughs: each level names the one before it nine times, which would be
    10**9 values, but a node is shared, never copied."""
    levels = ["a: &a0 [x]"]
    levels += [f"a{k}: &a{k} [{', '.join([f'*a{k - 1}'] * 9)}]" for k in range(1, 12)]
    start = time.perf_counter()
    assert _read(tmp_path, "license: mit\nbomb:\n  " + "\n  ".join(levels))["license"] == "mit"
    assert time.perf_counter() - start < 2
    bomb = "\n".join(levels).replace("a: &a0", "&a0")
    _raises(tmp_path, f"license: *a11\n{bomb}")  # an alias before its anchor


def test_a_cycle_is_never_followed(tmp_path):
    assert _read(tmp_path, "license: mit\nloop: &l [*l]")["license"] == "mit"
    assert _read(tmp_path, "license: mit\nloop: &m {a: *m}")["license"] == "mit"
    _raises(tmp_path, "license: &l [*l]", "license fse cannot read: a list that is no scalar")
    _raises(tmp_path, "license: &m {a: *m}", "license fse cannot read: a mapping that is no")


def test_a_large_card_reads_in_bounded_time(tmp_path):
    """A model-index of about 2 MiB (the card of a model with many results)."""
    row = "  - task: {type: sts, name: STS}\n    metrics: [{type: spearman, value: 0.5}]\n"
    front = "license: mit\nmodel-index:\n" + row * 20_000 + "language: [en]"
    assert len(front) > 1_500_000
    start = time.perf_counter()
    assert _read(tmp_path, front) == {"license": "mit", "base_model": None, "language": ["en"]}
    assert time.perf_counter() - start < 20


def test_a_front_matter_that_is_not_utf8_raises_a_unicode_error(tmp_path):
    with pytest.raises(UnicodeDecodeError):
        front_matter(b"---\nlicense: \xff\n---\n", tmp_path / "README.md", KEYS)


def test_the_loader_is_disposed_whatever_happens(tmp_path, monkeypatch):
    disposed = []
    original = yaml.SafeLoader.dispose

    def spy(self):
        disposed.append(True)
        original(self)

    monkeypatch.setattr(yaml.SafeLoader, "dispose", spy)
    _read(tmp_path, "license: mit")
    _raises(tmp_path, "license: [")
    _raises(tmp_path, "license: a\nlicense: b")
    assert disposed == [True] * 3


@pytest.mark.parametrize(
    ("value", "want"),
    [(None, None), ("abc", "abc"), (["a", "b"], ["a", "b"])],
)
def test_strings(tmp_path, value, want):
    assert strings(value, "language", tmp_path, ID) == want


@pytest.mark.parametrize(
    ("value", "key", "match"),
    [
        ([], "language", r"language list fse cannot read: \[\]"),
        ("ABC", "language", "language fse cannot read: 'ABC'"),
        (["a", "B"], "language", "language fse cannot read: 'B'"),
        (True, "language", "language fse cannot read: True"),
        (12, "license", "license that is no string: 12"),
        (["a", None], "license", "license that is no string: None"),
        ([["a"]], "license", "license that is no string"),
    ],
)
def test_strings_of_another_form_raise(tmp_path, value, key, match):
    with pytest.raises(CorruptArtifactError, match=match):
        strings(value, key, tmp_path, ID)
