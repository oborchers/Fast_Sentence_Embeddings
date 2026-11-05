"""Conformance of groups of heads: every head of a group has the bits of the same head alone
on the same backend, instruction set and path (the frozen single-head pipeline of 2.0), and
of the reference composed from fse._reference_head. The meta-tests in
tests/test_conformance_heads_meta.py check that these cases catch the usual mistakes."""

import numpy as np
import pytest

from fse._backends import get_backend
from fse._runtime import HeadsPipeline, Member, Pipeline, plan_heads
from fse._runtime._head_common import fold
from fse._text import StringSource, TokenListSource
from support import head as ref_support
from support.head_cases import BACKENDS, case_spec, head_of, sentences
from support.head_legacy import legacy_make_head
from support.heads import assert_same, bits, check_group, expected, group
from support.heads_cases import (
    COMPOSITIONS,
    members_of,
    outputs_for,
    witness_head,
    witness_spec,
)

F32 = np.float32
TEXTS = sentences()


@pytest.mark.parametrize("vectors", [False, True], ids=["pooled", "vectors"])
@pytest.mark.parametrize("composition", COMPOSITIONS)
@pytest.mark.parametrize("normalize", [False, True], ids=["raw-encoder", "normalizing-encoder"])
@pytest.mark.parametrize("fasttext", [False, True], ids=["words", "fasttext"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
def test_every_head_of_a_group_is_that_head_alone(
    backend, dtype, fasttext, normalize, composition, vectors
):
    name, isa = backend
    spec = case_spec(dtype, fasttext, normalize=normalize)
    check_group(spec, members_of(spec, composition), name, isa, vectors, TEXTS)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
@pytest.mark.parametrize("columns", [(17, 1), (30, 1), (31, 1), (32, 1), (64, 1), (1024, 1)])
def test_the_columns_of_the_shared_table_at_every_block_edge(backend, columns):
    name, isa = backend
    if columns[0] == 1024 and name == "numpy":
        pytest.skip("the reference loops in Python; the cpu backends cover the 1024 columns")
    spec = case_spec("float32", fasttext=False)
    members = tuple(Member(f"h{i}", head_of(spec, k, False, seed=i)) for i, k in enumerate(columns))
    check_group(spec, members, name, isa, False, TEXTS[:40])


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
@pytest.mark.parametrize("k", [1, 3, 4, 5, 7, 8, 9, 17])
def test_multilabel_widths_at_the_edges_of_the_sigmoid_blocks(backend, k):
    from fse._contract import Link
    from support.heads_cases import _linear2

    name, isa = backend
    spec = case_spec("float32", fasttext=False)
    members = (
        Member("a", head_of(spec, 2, False)),
        Member("multi", _linear2(spec, k, Link.MULTILABEL, False)),
    )
    check_group(spec, members, name, isa, False, TEXTS[:40])


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
def test_zeros_subnormals_and_ties_in_every_column(backend):
    name, isa = backend
    spec = witness_spec("float32")
    ks = (33, 1, 31)  # columns across the block edges of the shared table
    members = tuple(Member(f"h{i}", witness_head(spec, k)) for i, k in enumerate(ks))
    texts = ["w1", "w2", "w0", "", "w1 w1 w1", "w2 w2", "w3 w1", "unknown"]
    # The witnesses are in the reference before the group is compared with it.
    for member in members:
        z = ref_support.logits(spec, member.head.weight, member.head.bias, texts)
        assert ((bits(z[0]) & 0x7FFFFFFF) == 0).all(), "w1 gives a zero in every column"
        tiny = bits(z[1]) & 0x7FFFFFFF
        assert ((tiny > 0) & (tiny < 0x00800000)).all(), "w2 gives a subnormal in every column"
    check_group(spec, members, name, isa, False, texts)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_standalone_heads_of_one_table(backend, dtype):
    name, isa = backend
    spec = case_spec(dtype, fasttext=True)
    base = head_of(spec, 4, False)
    pool = fold(get_backend(name, isa=isa), spec.pool_input(), base.weight, 1)
    table = Pipeline(spec, name, isa=isa, pool=pool, steps=())
    from fse._runtime import HeadSpec

    low = HeadSpec(None, base.bias, False, base.link)
    high = HeadSpec(None, (base.bias * 2).astype(F32), False, base.link)
    members = (Member("low", low), Member("high", high))
    plan = plan_heads(spec, members, name, own=table, isa=isa)
    got = HeadsPipeline(plan)
    for output in ("logits", "proba", "labels"):
        results, _ = got.infer(StringSource(TEXTS), outputs=(output, output), workers=2)
        for member, result in zip(members, results, strict=True):
            alone = legacy_make_head(spec, member.head, name, own=table, isa=isa)
            want, _ = alone.infer(StringSource(TEXTS), output=output, workers=1)
            assert_same(result, want, output)


@pytest.mark.parametrize("kind", ["tokens", "stream", "small-tasks"])
def test_token_lists_streams_and_small_tasks(kind):
    from fse._stream import StreamSource

    spec = case_spec("float16", fasttext=True, normalize=False, tokenizer="external")
    members = members_of(spec, "M")
    pipeline = group(spec, members)
    texts = TEXTS
    if kind == "tokens":
        source = TokenListSource([t.split(" ") if t else [] for t in texts])
    elif kind == "stream":
        source = StreamSource(iter(texts), chunk_rows=13)
    else:
        source = TokenListSource([t.split(" ") if t else [] for t in texts])
    outputs = outputs_for(members, "proba")
    results, _ = pipeline.infer(source, outputs=outputs, workers=3, chunk_bytes=1 << 10)
    for member, output, result in zip(members, outputs, results, strict=True):
        assert_same(result, expected(spec, member, texts, output, vectors=False), output)
