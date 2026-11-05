"""The groups of heads of the conformance suites: compositions of heads of one table."""

import numpy as np

from fse._contract import Link
from fse._runtime import HeadSpec, Member
from support.head_cases import head_of

F32 = np.float32


def heads_of(spec, name):
    """The heads of the composition ``name`` as (head name, HeadSpec) in keyword order:

    F2 two raw heads (k = 1 sigmoid, k = 17 softmax); F3 raw heads of multilabel k = 3,
    ovr k = 2 and softmax k = 4 at a temperature (label rule linear@2); N2 two heads on
    normalized means; M a raw head, a head on normalized means, two regressors (one and
    three targets) and a head without a link, raw and normalized mixed."""
    o = spec.output_dim
    if name == "F2":
        return [("a", head_of(spec, 1, False)), ("b", head_of(spec, 17, False))]
    if name == "F3":
        return [
            ("multi", _linear2(spec, 3, Link.MULTILABEL, False)),
            ("pair", head_of(spec, 2, False)),
            ("soft", _linear2(spec, 4, Link.SOFTMAX, False)),
        ]
    if name == "N2":
        return [("a", head_of(spec, 3, True)), ("b", head_of(spec, 5, True, seed=9))]
    if name == "M":
        none = head_of(spec, 2, True, seed=21)
        return [
            ("raw", head_of(spec, 4, False)),
            ("norm", head_of(spec, 3, True)),
            ("quality", _regressor(1, False, o)),
            ("targets", _regressor(3, True, o)),
            ("plain", HeadSpec(none.weight, none.bias, True, None)),
        ]
    raise ValueError(name)


COMPOSITIONS = ("F2", "F3", "N2", "M")


def members_of(spec, name):
    return tuple(Member(n, h) for n, h in heads_of(spec, name))


def _linear2(spec, k, link, normalized):
    base = head_of(spec, k, normalized, link=link, seed=5)
    thresholds = np.linspace(0.3, 0.7, k).astype(F32) if link is Link.MULTILABEL else None
    return HeadSpec(base.weight, base.bias, normalized, link, thresholds, _rule2())


def _rule2():
    from fse._contract import Head

    return Head.LINEAR2


def _regressor(k, normalized, dim):
    rng = np.random.default_rng(40 + k)
    weight = (rng.standard_normal((k, dim)) * 0.3).astype(F32)
    return HeadSpec(weight, rng.standard_normal(k).astype(F32), normalized, None)


def outputs_for(members, rule="proba"):
    """The runtime output of every head for ``rule``: "logits" (every head), "proba" (heads
    with a link, else logits) or "labels" (heads with a link, else logits)."""
    if rule == "logits":
        return tuple("logits" for _ in members)
    return tuple(rule if m.head.link is not None else "logits" for m in members)


TINY = F32(np.finfo(F32).smallest_subnormal)


def witness_spec(dtype="float32", normalize=False, dim=37):
    """An EncodeSpec whose words are witnesses: w0 is zero, w1 is -0.0 everywhere, w2 has a
    subnormal value everywhere, w3 and on are ordinary."""
    from fse._encoder import Encoder
    from fse._wordvectors import WordVectors

    rng = np.random.default_rng(3)
    rows = rng.standard_normal((20, dim)).astype(F32)
    rows[0] = 0.0
    rows[1] = -0.0
    rows[2] = TINY * 3
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(20)])
    if dtype != "float32":
        wv = wv.quantize(dtype)
    return Encoder.from_word_vectors(wv, normalize=normalize)._spec


def witness_head(spec, k, link=None, normalized=False):
    """k outputs of positive weights (so -0.0 and subnormal rows stay so after the fold) and
    the bias -0.0, equal in every column (a full tie for the label)."""
    weight = np.full((k, spec.output_dim), 0.25, dtype=F32)
    bias = np.full(k, -0.0, dtype=F32)
    if link is None:
        link = Link.SIGMOID if k == 1 else Link.SOFTMAX
    return HeadSpec(weight, bias, normalized, link)


def overflow_spec(normalize=False, big_row=False):
    """A table of dim 4 whose w5 and w6 are [1, 0, 0, 0] and w7 is [0, 1, 0, 0]: a head
    with 3e38 in the first (second) column of A overflows when "w5 w6" is pooled ("w7" plus
    its bias in the logits), a head with small weights does not."""
    from fse._encoder import Encoder
    from fse._wordvectors import WordVectors

    rows = np.full((8, 4), 0.1, dtype=F32)
    if big_row:
        rows[4] = 10.0  # a fold with 3e38 overflows
    rows[5] = rows[6] = (1, 0, 0, 0)
    rows[7] = (0, 1, 0, 0)
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(8)])
    return Encoder.from_word_vectors(wv, normalize=normalize)._spec


def column_head(spec, column, size, k=1, bias=0.0, normalized=False):
    """A head of k outputs whose weight is ``size`` in ``column`` and 0 elsewhere."""
    weight = np.zeros((k, spec.output_dim), dtype=F32)
    weight[:, column] = size
    return HeadSpec(weight, np.full(k, bias, dtype=F32), normalized, None)
