"""Shared by the tests of fse.StaticModel: models in the folder layouts model2vec reads,
texts that cross its two cuts, and the comparison with model2vec's own ``encode``."""

import json
import shutil

import numpy as np

from fse._encoder._model2vec_rules import UNSET
from support import converted as cv
from support import model2vec as mv
from support import sentence_transformers as stm
from support import wordpiece as wp

TOKENS = wp.vocabulary()
UNK = TOKENS.index("[UNK]")
LAYOUTS = ("model2vec", "st-root", "st-nested")


def texts(count: int = 400) -> list[str]:
    """Texts of every length, the empty one, one of unknown words only, and ones that
    cross the token cut (many unknown ids before the 512th token) and the character cut."""
    long = " ".join(TOKENS[5:400]) * 3
    unknown = "[UNK] " * 700 + "hello world"
    return [*wp.corpus(TOKENS, count=count), long, unknown, "", "zzzq", "x" * 5000]


def model_folder(root, layout: str = "model2vec", *, normalize: bool = True, config=None):
    """A model of ``TOKENS`` in one of ``LAYOUTS``. A model2vec one is written by model2vec;
    the others by hand in the layout of sentence-transformers, whose
    config_sentence_transformers.json holds ``config``."""
    if layout == "model2vec":
        return mv.save_model(root, normalize=normalize)
    folder = stm.write_model(root, config=config)
    if layout == "st-root":
        for name in ("model.safetensors", "tokenizer.json"):
            shutil.move(folder / "0_StaticEmbedding" / name, folder / name)
        (folder / "0_StaticEmbedding").rmdir()
        modules = json.loads((folder / "modules.json").read_text())
        modules[0]["path"] = "."
        (folder / "modules.json").write_text(json.dumps(modules))
    return folder


def close(model, texts, want, *, max_length=UNSET, normalize=None):
    """``model.encode`` agrees with model2vec's rows ``want`` within the bound of
    tests/support/converted.py, for the settings given to ``encode``; gives the largest
    ratio of error to bound."""
    cut = model.max_length if max_length is UNSET else max_length
    norm = model.normalize if normalize is None else normalize
    got = model.encode(texts, max_length=max_length, normalize=normalize)
    encoder = model._variants.get(cut, norm)
    assert got.dtype == np.float32
    return cv.assert_close(got, want, encoder, texts, norm)
