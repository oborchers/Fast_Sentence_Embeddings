"""The errors of a group of heads name the heads they belong to, and are those of the
earliest failing row."""

import numpy as np
import pytest

from fse._runtime import Member, plan_heads
from fse._text import StringSource
from support.heads import group, own_of
from support.heads_cases import column_head, overflow_spec

F32 = np.float32
BIG = F32(3e38)


def _run(spec, members, texts, vectors=False, workers=1):
    pipeline = group(spec, members, vectors=vectors)
    return pipeline.infer(StringSource(texts), outputs=("logits",) * len(members), workers=workers)


def test_a_fold_that_overflows_names_the_head():
    spec = overflow_spec(big_row=True)
    members = (
        Member("small", column_head(spec, 0, 1.0)),
        Member("big", column_head(spec, 0, BIG)),
    )
    with pytest.raises(ValueError, match="head 'big' does not fold into the table"):
        plan_heads(spec, members, "cpu", own=own_of(spec))


def test_a_lone_head_names_none():
    spec = overflow_spec(big_row=True)
    with pytest.raises(ValueError, match=r"^the head does not fold into the table"):
        plan_heads(
            spec,
            (Member(None, column_head(spec, 0, BIG)),),
            "cpu",
            own=own_of(spec),
        )


def test_an_overflow_in_the_columns_of_one_head_names_that_head():
    spec = overflow_spec()
    members = (Member("small", column_head(spec, 0, 1.0)), Member("big", column_head(spec, 0, BIG)))
    with pytest.raises(ValueError, match=r"^head 'big': row 2 contains NaN or infinite values"):
        _run(spec, members, ["w1", "w1", "w5 w6", "w1"])


def test_an_overflow_in_the_columns_of_several_heads_names_them_all():
    spec = overflow_spec()
    members = (
        Member("a", column_head(spec, 0, BIG)),
        Member("b", column_head(spec, 1, 1.0)),
        Member("c", column_head(spec, 0, BIG, k=2)),
    )
    with pytest.raises(ValueError, match=r"^heads 'a', 'c': row 1 "):
        _run(spec, members, ["w1", "w5 w6"], workers=2)


def test_the_error_is_that_of_the_earliest_row_whatever_the_head():
    spec = overflow_spec()
    a = column_head(spec, 1, BIG, bias=BIG)  # its logits of "w7" are infinite: row 3
    b = column_head(spec, 0, BIG)  # its pooling of "w5 w6" overflows: row 7
    texts = ["w1"] * 3 + ["w7"] + ["w1"] * 3 + ["w5 w6", "w1"]
    with pytest.raises(ValueError, match=r"^head 'a': .*row 3"):
        _run(spec, (Member("b", b), Member("a", a)), texts)


def test_a_pooling_error_in_the_same_row_comes_before_the_logits():
    spec = overflow_spec()
    a = column_head(spec, 0, BIG, bias=BIG)  # "w5 w6": pooling overflows before any logit
    b = column_head(spec, 1, 1.0)
    with pytest.raises(ValueError, match=r"^head 'a': row 1 contains NaN"):
        _run(spec, (Member("b", b), Member("a", a)), ["w1", "w5 w6"])


def test_infinite_logits_name_the_head():
    spec = overflow_spec()
    a = column_head(spec, 1, 1.0, k=2)
    b = column_head(spec, 1, BIG, bias=BIG)
    with pytest.raises(ValueError, match=r"^head 'b': .*row 1"):
        _run(spec, (Member("a", a), Member("b", b)), ["w1", "w7"])


def test_the_rows_of_the_full_table_name_their_readers_and_the_vectors():
    from fse._encoder import Encoder
    from fse._wordvectors import WordVectors

    rows = np.full((4, 4), 0.1, dtype=F32)
    rows[2] = BIG  # two of these overflow the pooled mean
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(4)])
    spec = Encoder.from_word_vectors(wv, normalize=False)._spec
    members = (
        Member("raw", column_head(spec, 0, 1.0)),
        Member("normalized", column_head(spec, 0, 1.0, normalized=True)),
    )
    with pytest.raises(ValueError, match=r"^heads 'raw', 'normalized' and the vectors: row 1 "):
        _run(spec, members, ["w1", "w2 w2"], vectors=True)


def test_a_lone_head_keeps_its_errors_without_a_name():
    spec = overflow_spec()
    with pytest.raises(ValueError, match=r"^row 1 contains NaN or infinite values"):
        _run(spec, (Member(None, column_head(spec, 0, BIG)),), ["w1", "w5 w6"])


def test_a_normalization_error_names_only_what_depends_on_it():
    from fse._encoder import Encoder
    from fse._wordvectors import WordVectors

    rows = np.full((4, 4), 0.1, dtype=F32)
    rows[2] = F32(3e19)  # the squares overflow in the norm, the rows are finite
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(4)])
    spec = Encoder.from_word_vectors(wv, normalize=False)._spec
    members = (
        Member("raw", column_head(spec, 0, 1.0)),
        Member("norm", column_head(spec, 0, 1.0, normalized=True)),
    )
    with pytest.raises(ValueError, match=r"^head 'norm': ") as caught:
        _run(spec, members, ["w1", "w2"], vectors=True)
    assert "raw" not in str(caught.value).split(":")[0]


@pytest.mark.parametrize("shared", [False, True])
def test_cached_group_diagnosis_neither_refolds_nor_starts_threads(shared, monkeypatch):
    import threading

    import fse
    from fse._backends import CpuBackend

    count = (1 << 14) + 1
    rows = np.full((count, 2), 0.1, dtype=F32)
    rows[5] = rows[6] = (1, 0)
    encoder = fse.Encoder.from_arrays(rows, words=[f"w{i}" for i in range(count)], normalize=False)
    heads = fse.Heads(
        small=fse.Regressor.from_arrays(encoder, [1.0, 0.0]),
        big=fse.Regressor.from_arrays(encoder, [float(BIG), 0.0]),
    )
    # Cache a parallel first fold, then use a different call's execution settings.
    heads.predict(["w1"], workers=2)
    started = []
    real_start = threading.Thread.start

    def start(thread):
        started.append(thread.name)
        return real_start(thread)

    def no_refold(*_args):
        pytest.fail("error attribution must read already folded columns")

    with fse.Pool(3) as pool:
        monkeypatch.setattr(threading.Thread, "start", start)
        monkeypatch.setattr(CpuBackend, "scores", no_refold)
        with pytest.raises(ValueError, match=r"^head 'big': row 1 contains NaN"):
            heads.predict(["w1", "w5 w6"], workers=1, pool=pool if shared else None)
    assert started == []
