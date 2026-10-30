"""The parameters of unigram@1: UnigramConfig, its checks and its JSON form."""

import dataclasses

import numpy as np
import pytest

from fse._contract import UnsupportedArtifactError
from fse._tokenizer_unigram import METASPACE, UNIGRAM, UnigramConfig, step_of
from support.charsmap import build
from support.unigram import STEPS

SCORES = np.array([-13.5, -13.5, -2.0 - 2.0**-30, -1.0], dtype="<f8").tobytes()
BLOB = build({"\u00a0": " ", "\uff21": "A"})


def _config(**changes) -> UnigramConfig:
    base = {
        "added_tokens": ((0, "[PAD]"), (1, "[UNK]")),
        "unk_id": 1,
        "normalizer": tuple(step_of(s) for s in STEPS),
        "scores": SCORES,
        "charsmap": BLOB,
    }
    return UnigramConfig(**{**base, **changes})


def _vocabulary(tokens):
    data = "".join(tokens).encode()
    offsets = np.cumsum([0, *(len(t.encode()) for t in tokens)], dtype=np.int64)
    return np.frombuffer(data, dtype=np.uint8), offsets


def test_a_config_keeps_its_parameters_and_data():
    config = _config()
    assert config.identifier == UNIGRAM == "unigram@1"
    assert config.steps() == STEPS
    assert config.metaspace == METASPACE
    assert config.scores_array().tolist() == [-13.5, -13.5, -2.0 - 2.0**-30, -1.0]
    assert not config.scores_array().flags.writeable
    assert hash(config) == hash(_config())
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.unk_id = 0  # type: ignore[misc]


def test_the_json_form_round_trips():
    config = _config()
    value = config.to_json()
    assert value == {
        "added_tokens": [[0, "[PAD]"], [1, "[UNK]"]],
        "byte_fallback": False,
        "metaspace": {"prepend_scheme": "always", "replacement": "\u2581", "split": False},
        "normalizer": STEPS,
        "unk_id": 1,
    }
    assert UnigramConfig.from_json(value, SCORES, BLOB) == config
    with pytest.raises(ValueError, match="JSON object of"):
        UnigramConfig.from_json({**value, "x": 1}, SCORES, BLOB)
    with pytest.raises(ValueError, match="added_tokens must be a list"):
        UnigramConfig.from_json({**value, "added_tokens": [(0, "[PAD]")]}, SCORES, BLOB)
    with pytest.raises(ValueError, match="normalizer must be a list"):
        UnigramConfig.from_json({**value, "normalizer": ["precompiled"]}, SCORES, BLOB)
    with pytest.raises(ValueError, match="metaspace must be"):
        UnigramConfig.from_json({**value, "metaspace": []}, SCORES, BLOB)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"added_tokens": ((1, "[UNK]"), (0, "[PAD]"))}, "sorted by id"),
        ({"added_tokens": ((0, "a"), (1, "a"))}, "unique"),
        ({"added_tokens": ((0, ""),)}, "non-empty str"),
        ({"added_tokens": ((True, "a"),)}, "must be an int"),
        ({"added_tokens": [(0, "a")]}, "tuple of"),
        ({"unk_id": -1}, "between 0"),
        ({"unk_id": 4}, "there are 4 pieces"),
        ({"normalizer": [step_of(STEPS[0])]}, "tuple of steps"),
        ({"normalizer": (("type", "precompiled"),)}, "tuple of \\(key, value\\) pairs"),
        ({"normalizer": ((("pattern", "a"), ("type", "replace")),)}, "holds exactly"),
        (
            {"normalizer": (step_of({"type": "replace", "pattern": "", "content": ""}),)},
            "non-empty",
        ),
        (
            {"normalizer": (step_of({"type": "runs", "class": "digit", "min": 1, "content": ""}),)},
            "class",
        ),
        (
            {"normalizer": (step_of({"type": "runs", "class": "space", "min": 0, "content": ""}),)},
            "between 1",
        ),
        ({"normalizer": (step_of({"type": "strip", "left": 1, "right": True}),)}, "bool"),
        ({"normalizer": ()}, "exactly with a precompiled step"),
        ({"charsmap": None}, "exactly with a precompiled step"),
        ({"charsmap": b"\0\0\0\0"}, "positive multiple of 4"),
        ({"scores": b"\0" * 12}, "float64 per piece"),
        ({"scores": np.array([1.0, np.inf], "<f8").tobytes(), "unk_id": 0}, "finite"),
        ({"byte_fallback": 0}, "bool"),
    ],
)
def test_malformed_parameters_raise(changes, message):
    with pytest.raises(ValueError, match=message):
        _config(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"byte_fallback": True}, "byte_fallback"),
        ({"metaspace": (("prepend_scheme", "never"), *METASPACE[1:])}, "supports the Metaspace"),
        ({"normalizer": (step_of({"type": "lowercase"}),)}, "no normalizer step"),
        ({"normalizer": (step_of(STEPS[0]),) * 2}, "at most one precompiled"),
    ],
)
def test_what_this_fse_does_not_support_raises(changes, message):
    with pytest.raises(UnsupportedArtifactError, match=message):
        _config(**changes)


def test_the_vocabulary_is_checked():
    config = _config()
    config.check_vocabulary(*_vocabulary(["[PAD]", "[UNK]", "ab", "a"]))
    with pytest.raises(ValueError, match="4 scores for 3 tokens"):
        config.check_vocabulary(*_vocabulary(["[PAD]", "[UNK]", "a"]))
    with pytest.raises(ValueError, match="whose token is"):
        config.check_vocabulary(*_vocabulary(["[PAD]", "[unk]", "ab", "a"]))
    shifted = _config(added_tokens=((0, "[PAD]"), (5, "[UNK]")))
    with pytest.raises(ValueError, match="has id 5 of 4"):
        shifted.check_vocabulary(*_vocabulary(["[PAD]", "[UNK]", "ab", "a"]))
