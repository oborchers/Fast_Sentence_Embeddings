"""The root of a score store: the table of the combinations a head may have, the
strict reading of ``fse.score_heads`` and the strictness about future identifiers."""

import json

import pytest

from fse._container import CorruptArtifactError, canonical_json
from fse._contract import UnsupportedArtifactError
from fse._store._layout import SCORE_STORE_BYTES, create_store, read_store, root_text
from fse._store._scores_info import combination_error, parse_info
from support import scores as sc

FILE = "store.json"


def _roundtrip(info):
    return parse_info(info.metadata(), FILE)


def test_a_root_is_read_back_as_written():
    info = sc.default_info(doc_ids=True)
    assert _roundtrip(info) == info
    assert info.dim == 4
    assert info.threshold_names == ("thresholds.toxic",)
    assert info.metadata()["fse.doc_ids"] == "int64"


def test_the_metadata_names_the_format_and_the_modes():
    metadata = sc.default_info().metadata()
    assert metadata["fse.kind"] == "prediction_scores"
    assert (metadata["fse.format_version"], metadata["fse.transform_version"]) == ("1", "2")
    assert (metadata["fse.doc_ids"], metadata["fse.resume"]) == ("none", "none")


def cls(**changes):
    """A classification head; ``stored`` is the decision unless changed."""
    base = {"task": "classification", "stored": "decision", "flat": None}
    return sc.head(**{**base, **changes})


GOOD = [
    sc.head(),
    sc.head(width=3, flat=None, columns=(0, 3)),
    sc.head(flat=False),
    sc.softmax(),
    sc.softmax(stored="proba"),
    sc.softmax(width=1, labels=(0, 1)),
    sc.softmax(width=2, labels=(0, 1), stored="proba"),
    sc.softmax(width=2, labels=(0, 1)),
    sc.toxic(),
    sc.toxic(stored="proba"),
    cls(link="sigmoid@1", width=1, labels=("n", "p"), columns=(0, 1)),
    cls(link="sigmoid@1", stored="proba", width=2, labels=("n", "p"), columns=(0, 2)),
    cls(link="ovr@1", width=2, labels=("n", "p"), columns=(0, 2)),
    cls(link="none", width=3, labels=(1, 2, 3), columns=(0, 3)),
    cls(
        rule="linear@2", link="softmax@1", stored="proba", width=3, labels=(1, 2, 3), columns=(0, 3)
    ),
]


@pytest.mark.parametrize("good", GOOD)
def test_every_row_of_the_table_is_a_valid_head(good):
    assert combination_error(good) is None
    assert _roundtrip(sc.info(good)).heads == (good,)


def _bad(**changes):
    base = {"labels": (0, 1, 2), "width": 3, "columns": (0, 3), "link": "softmax@1"}
    return cls(**{**base, **changes})


BAD = {
    "regression with a link": sc.head(link="softmax@1"),
    "regression storing proba": sc.head(stored="proba"),
    "regression with labels": sc.head(labels=(0, 1)),
    "regression linear@2": sc.head(rule="linear@2"),
    "regression wide with flat": sc.head(width=2, columns=(0, 2)),
    "regression single without flat": sc.head(flat=None),
    "classification storing values": _bad(stored="values"),
    "classification with flat": _bad(flat=True),
    "classification without labels": _bad(labels=None),
    "duplicate labels": _bad(labels=(0, 0, 1)),
    "mixed labels": _bad(labels=(0, "a", 2)),
    "bool labels": _bad(labels=(True, False, 2)),
    "no link storing proba": _bad(link="none", stored="proba"),
    "multilabel with linear@1": _bad(link="multilabel@1", rule="linear@1"),
    "linear@2 without a link": _bad(link="none", rule="linear@2"),
    "multilabel width": _bad(link="multilabel@1", rule="linear@2", width=2, columns=(0, 2)),
    "one label": _bad(labels=(0,), width=1, columns=(0, 1)),
    "proba width": _bad(stored="proba", width=2, columns=(0, 2)),
    "logit width": _bad(width=2, columns=(0, 2)),
    "sigmoid with two logits": _bad(link="sigmoid@1", labels=(0, 1), width=2, columns=(0, 2)),
    "ovr with one logit": _bad(link="ovr@1", labels=(0, 1), width=1, columns=(0, 1)),
    "a bad name": sc.head("Bad"),
    "a bad model": sc.head(model="xxh3-128:zz"),
    "a bad span": sc.head(columns=(0, 2)),
}


@pytest.mark.parametrize("bad", BAD.values(), ids=BAD.keys())
def test_every_other_known_combination_is_corrupt(bad):
    assert combination_error(bad) is not None
    with pytest.raises(CorruptArtifactError, match=f"head {bad.name!r}"):
        _roundtrip(sc.info(bad))


def _edited(entry_changes=None, **changes):
    metadata = sc.default_info().metadata()
    entries = json.loads(metadata["fse.score_heads"])
    entries[0].update(entry_changes or {})
    metadata["fse.score_heads"] = canonical_json(entries)
    metadata.update(changes)
    return metadata


@pytest.mark.parametrize(
    "changes",
    [
        {"rule": "linear@7"},
        {"link": "softmax@9"},
        {"path": "swapped"},
        {"stored": "logits"},
        {"task": "ranking"},
    ],
)
def test_a_well_formed_unknown_identifier_is_unsupported(changes):
    with pytest.raises(UnsupportedArtifactError, match="unsupported"):
        parse_info(_edited(changes), FILE)


@pytest.mark.parametrize("changes", [{"rule": "Linear 2"}, {"path": "no spaces allowed"}])
def test_a_malformed_identifier_is_corrupt(changes):
    with pytest.raises(CorruptArtifactError):
        parse_info(_edited(changes), FILE)


def test_a_future_format_is_unsupported_before_the_keys_are_checked():
    metadata = _edited(**{"fse.format_version": "7", "fse.extra": "x"})
    with pytest.raises(UnsupportedArtifactError, match="format '7'"):
        parse_info(metadata, FILE)


def test_a_future_resume_rule_is_unsupported_and_a_malformed_one_corrupt():
    with pytest.raises(UnsupportedArtifactError, match=r"fse.resume"):
        parse_info(_edited(**{"fse.resume": "source_rows@1"}), FILE)
    with pytest.raises(CorruptArtifactError):
        parse_info(_edited(**{"fse.resume": "not an id"}), FILE)


def test_the_keys_and_the_json_are_strict():
    metadata = sc.default_info().metadata()
    with pytest.raises(CorruptArtifactError, match="must hold exactly"):
        parse_info({**metadata, "fse.extra": "x"}, FILE)
    for text in ("[]", "{}", '[ {"name": "a"} ]', "[1]"):
        with pytest.raises(CorruptArtifactError):
            parse_info({**metadata, "fse.score_heads": text}, FILE)
    with pytest.raises(CorruptArtifactError, match="fingerprint"):
        parse_info({**metadata, "fse.encoder": "nope"}, FILE)
    with pytest.raises(UnsupportedArtifactError, match=r"fse.doc_ids"):
        parse_info({**metadata, "fse.doc_ids": "maybe"}, FILE)
    with pytest.raises(CorruptArtifactError, match="document-ID"):
        parse_info({**metadata, "fse.doc_ids": 3}, FILE)


def test_columns_must_run_without_a_gap_and_names_must_differ():
    a, b = sc.head("a"), sc.head("b", columns=(2, 3))
    with pytest.raises(CorruptArtifactError, match="without a gap"):
        _roundtrip(sc.info(a, b))
    with pytest.raises(CorruptArtifactError, match="unique"):
        _roundtrip(sc.info(a, sc.head("a", columns=(1, 2))))


def test_two_roots_are_compared_head_by_head(tmp_path):
    info = sc.default_info()
    info.check(sc.default_info(), tmp_path)
    for other, message in (
        (sc.default_info(doc_ids=True), "has_doc_ids"),
        (sc.info(sc.head("other")), "holds the heads"),
        (sc.info(sc.head(), sc.toxic(start=1, stored="proba")), "stored"),
        (sc.info(sc.head(), sc.toxic(start=1, labels=("a", "b", "z"))), "labels"),
    ):
        with pytest.raises(ValueError, match=message):
            info.check(other, tmp_path)


def test_a_different_path_names_the_head_and_both_paths(tmp_path):
    other = sc.info(sc.head(path="unfolded"), sc.toxic(start=1))
    with pytest.raises(ValueError, match=r"'quality'.*'folded'.*'unfolded'"):
        sc.default_info().check(other, tmp_path)


def test_a_score_store_may_exceed_64_kib_up_to_1_mib(tmp_path):
    many = [sc.head(f"h{i:04d}", model=sc.MODEL, columns=(i, i + 1), flat=True) for i in range(900)]
    info = sc.info(*many)
    size = len(root_text(info))
    assert 1 << 16 < size < SCORE_STORE_BYTES
    create_store(tmp_path / "s", info)
    assert read_store(tmp_path / "s").heads == info.heads


def test_a_root_beyond_1_mib_is_corrupt(tmp_path):
    many = [sc.head(f"h{i:05d}", columns=(i, i + 1)) for i in range(5000)]
    info = sc.info(*many)
    assert len(root_text(info)) > SCORE_STORE_BYTES
    create_store(tmp_path / "s", info)
    with pytest.raises(CorruptArtifactError, match="larger than 1048576 bytes"):
        read_store(tmp_path / "s")
