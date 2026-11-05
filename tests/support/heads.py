"""A group of heads and the reference it must equal: each head alone, bit for bit."""

import numpy as np

from fse._runtime import HeadsPipeline, Pipeline, plan_heads
from fse._text import StringSource, make_vocabulary
from support import head as support
from support.head_legacy import legacy_make_head
from support.heads_cases import outputs_for

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def own_of(spec, backend="cpu", isa=None):
    return Pipeline(spec, backend, isa=isa, vocabulary=make_vocabulary(spec.word_vectors))


def group(spec, members, backend="cpu", isa=None, *, vectors=False, own=None):
    """The HeadsPipeline of ``members`` on ``backend``."""
    own = own or own_of(spec, backend, isa)
    plan = plan_heads(spec, members, backend, own=own, with_vectors=vectors, isa=isa, workers=2)
    return HeadsPipeline(plan)


def expected(spec, member, texts, output, *, vectors):
    """The reference of one head on ``texts``: its ``output`` ("logits", "proba" or
    "labels"), on the unfolded rows when the vectors are asked for."""
    head = member.head
    z = support.logits(
        spec,
        head.weight,
        head.bias,
        texts,
        normalized=head.normalized,
        folded=not vectors and not head.normalized,
    )
    return support.outputs(z, output, head.link)


def assert_same(got, want, output):
    if output == "labels":
        assert got.tolist() == want.tolist()
    else:
        np.testing.assert_array_equal(bits(got), bits(want))


def alone(spec, member, backend, isa, vectors, texts, output):
    """The frozen single-head pipeline of 2.0 on the same backend."""
    pipe = legacy_make_head(
        spec, member.head, backend, own=own_of(spec, backend, isa), with_vectors=vectors, isa=isa
    )
    return pipe.infer(StringSource(texts), output=output, workers=1)


def check_group(spec, members, backend, isa, vectors, texts):
    """Every head of the group, for every output, equals that head alone on the same
    backend, instruction set and path, and the reference; the vectors equal the single ones."""
    got = group(spec, members, backend, isa, vectors=vectors)
    for rule in ("logits", "proba", "labels"):
        outputs = outputs_for(members, rule)
        results, vecs = got.infer(StringSource(texts), outputs=outputs, workers=3)
        for member, output, result in zip(members, outputs, results, strict=True):
            one, one_vectors = alone(spec, member, backend, isa, vectors, texts, output)
            assert_same(result, one, output)
            if output != "labels":
                assert_same(result, expected(spec, member, texts, output, vectors=vectors), output)
            if vectors:
                np.testing.assert_array_equal(bits(vecs), bits(one_vectors))
