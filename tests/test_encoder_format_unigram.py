"""The keys and tensors of encoder 1 for unigram@1: the parameters in fse.tokenizer_config,
the float64 scores in tokenizer.scores and the charsmap verbatim in tokenizer.charsmap."""

import json

import numpy as np
import pytest

from fse._contract import Normalize, UnsupportedArtifactError
from fse._encoder._format import EncoderContent, from_artifact, to_artifact
from fse._tokenizer_spec import TokenizerSpec
from fse._tokenizer_unigram import UnigramConfig, step_of
from fse._wordvectors import WordVectors
from support.charsmap import build

R = "\u2581"
TOKENS = ["[PAD]", "[UNK]", R, "a", "b", "ab"]
SCORES = np.array([-13.5, -13.5, -1.0, -1.0, -1.0, -2.0 - 2.0**-30], dtype="<f8")
BLOB = build({"\u00a0": " ", "\uff21": "A"})
STEPS = (step_of({"type": "precompiled"}), step_of({"type": "strip", "left": True, "right": True}))
CONFIG = UnigramConfig(((0, "[PAD]"), (1, "[UNK]")), 1, STEPS, SCORES.tobytes(), BLOB)
SPEC = TokenizerSpec("unigram@1", CONFIG, "drop", 512, 3584)


def _table():
    wv = WordVectors.from_arrays(np.ones((len(TOKENS), 3), np.float32), words=TOKENS)
    tensors, keys, _ = wv._parts()
    return dict(tensors), {k: v for k, v in keys.items() if k in ("fse.dim", "fse.dtype")}


def _artifact(spec=SPEC):
    content = EncoderContent(tokenizer=spec, steps=(Normalize(),), oov_weight=None, max_ngrams=None)
    return to_artifact(content, *_table())


def test_the_parameters_and_tensors_round_trip():
    tensors, metadata = _artifact()
    assert metadata["fse.tokenizer"] == "unigram@1"
    assert json.loads(metadata["fse.tokenizer_config"]) == CONFIG.to_json()
    assert tensors["tokenizer.scores"].dtype == np.dtype("<f8")
    assert tensors["tokenizer.scores"].tobytes() == SCORES.tobytes()
    assert tensors["tokenizer.charsmap"].tobytes() == BLOB
    content, table, _ = from_artifact(tensors, metadata)
    assert content.tokenizer == SPEC
    assert "tokenizer.scores" not in table
    without = TokenizerSpec("unigram@1", UnigramConfig((), 1, (), SCORES.tobytes()), "keep")
    tensors, metadata = _artifact(without)
    assert sorted(t for t in tensors if t.startswith("tokenizer.")) == ["tokenizer.scores"]
    assert from_artifact(tensors, metadata)[0].tokenizer == without


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda t: t.pop("tokenizer.scores"), "needs the F64 vector tokenizer.scores"),
        (lambda t: t.update({"tokenizer.scores": SCORES.astype(np.float32)}), "F64 vector"),
        (lambda t: t.update({"tokenizer.scores": SCORES.reshape(2, 3)}), "F64 vector"),
        (lambda t: t.update({"tokenizer.scores": SCORES[:5].copy()}), "5 scores for 6 tokens"),
        (lambda t: t.update({"tokenizer.charsmap": np.zeros(8, np.uint32)}), "U8 vector"),
        (lambda t: t.pop("tokenizer.charsmap"), "exactly with a precompiled step"),
        (lambda t: t.update({"tokenizer.charsmap": np.zeros(4, np.uint8)}), "multiple of 4"),
        (lambda t: t.update({"tokenizer.extra": np.zeros(1, np.uint8)}), "does not use"),
        (lambda t: t.update({"tokenizer.scores": np.full(6, np.nan, "<f8")}), "finite"),
    ],
)
def test_broken_tensors_raise(edit, match):
    tensors, metadata = _artifact()
    edit(tensors)
    with pytest.raises(ValueError, match=match):
        from_artifact(tensors, metadata)


def test_the_tensors_belong_to_unigram_1_only():
    tensors, metadata = _artifact()
    config = json.loads(metadata["fse.tokenizer_config"])
    config["byte_fallback"] = True
    with pytest.raises(UnsupportedArtifactError, match="byte_fallback"):
        from_artifact(
            tensors,
            {
                **metadata,
                "fse.tokenizer_config": json.dumps(
                    config, separators=(",", ":"), sort_keys=True, ensure_ascii=False
                ),
            },
        )
    wordpiece = {
        "fse.tokenizer": "wordpiece@1",
        "fse.tokenizer_config": json.dumps(
            {
                "added_tokens": [[0, "[PAD]"], [1, "[UNK]"]],
                "clean_text": True,
                "continuing_subword_prefix": "##",
                "handle_chinese_chars": True,
                "lowercase": True,
                "max_input_chars_per_word": 100,
                "strip_accents": True,
                "unk_id": 1,
            },
            separators=(",", ":"),
        ),
    }
    with pytest.raises(ValueError, match="holds the tensors"):
        from_artifact(tensors, {**metadata, **wordpiece})
