"""The cases of the head conformance suites: tables, heads, sentences and the backends."""

import numpy as np

from fse import _native
from fse._contract import Link
from fse._encoder import Encoder
from fse._runtime._head import HeadSpec
from fse._wordvectors import FastTextParams, WordVectors

F32 = np.float32
TINY = F32(np.finfo(F32).smallest_subnormal)
DIM = 37  # two blocks of dot16 and a tail
WORDS = 40
# Every CPU backend: the kernel of each instruction set of this machine and the reference.
BACKENDS = [*(("cpu", isa) for isa in _native.available_isas()), ("numpy", None)]
LINKS = list(Link)


def case_spec(dtype, fasttext, normalize=False, tokenizer="fse_words@1", huge=False):
    """The EncodeSpec of a table of ``WORDS`` words of ``DIM`` values with -0.0, subnormal
    and large values; with ``huge`` the row of w0 is 1e30 everywhere."""
    rng = np.random.default_rng(7)
    rows = rng.standard_normal((WORDS, DIM)).astype(F32)
    rows[1, :5] = -0.0
    rows[2, 3] = TINY * 5
    rows[4] *= F32(1e4)
    if huge:
        rows[0] = F32(1e30)
    kwargs = {}
    if fasttext:
        kwargs = {
            "fasttext": FastTextParams(3, 5, 61),
            "ngrams": rng.standard_normal((61, DIM)).astype(F32),
        }
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(WORDS)], **kwargs)
    if dtype != "float32":
        wv = wv.quantize(dtype)
    extra = {"max_ngrams": 7, "oov_weight": 0.6} if fasttext else {}
    encoder = Encoder.from_word_vectors(wv, tokenizer=tokenizer, normalize=normalize, **extra)
    return encoder._spec


def head_of(spec, k, normalized, link=None, seed=3):
    """A head of k outputs: sigmoid for one, ovr for two, softmax for more, unless given."""
    rng = np.random.default_rng(seed + k)
    weight = (rng.standard_normal((k, spec.output_dim)) * 0.5).astype(F32)
    bias = rng.standard_normal(k).astype(F32)
    if k > 1:
        bias[0] = -0.0
    if link is None:
        link = Link.SIGMOID if k == 1 else Link.OVR if k == 2 else Link.SOFTMAX
    return HeadSpec(weight, bias, normalized, link)


def sentences(n=120, seed=11):
    """Empty, one-token, unknown-only and long sentences among ordinary ones."""
    rng = np.random.default_rng(seed)
    out = ["", "w3", "zzq", "w1", "w2 w2", " ".join(["w5", "w9"] * 150)]
    for i in range(n):
        size = int(rng.integers(1, 30))
        tokens = [f"w{t}" if t < WORDS else f"unk{t}x" for t in rng.integers(1, WORDS + 12, size)]
        out.append("" if i % 23 == 0 else " ".join(tokens))
    return out
