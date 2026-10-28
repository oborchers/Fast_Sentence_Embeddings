"""The VectorStore: search and score of the stored rows, bit for bit as the reference."""

import ctypes
import ctypes.util
import platform
import sys

import numpy as np
import pytest

import fse
from fse import _reference
from fse._container import CorruptArtifactError
from fse._store import _search
from support.encoder import unfolded
from support.runtime import texts

ENCODER = unfolded()
F32 = np.float32


def bits(x):
    return np.asarray(x, F32).view(np.uint32)


def _store(tmp_path, dtype, sentences, shard_rows=64):
    path = tmp_path / f"store-{dtype}"
    with fse.ShardSink(path, ENCODER, dtype=dtype, shard_rows=shard_rows) as sink:
        ENCODER.encode(sentences, sink=sink)
    codes, scales = _reference.quantize_rows(ENCODER.encode(sentences), dtype)
    return path, codes, scales


def _want_search(codes, queries, k):
    inverse = _reference.inverse_norms(codes)
    cosines = _reference.cosines(codes, inverse, queries)
    ids = np.stack([_reference.top_k(cosines[:, j], k) for j in range(len(queries))])
    return ids, np.stack([cosines[ids[j], j] for j in range(len(queries))])


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("q", [1, 3, 70])
def test_search_gives_the_reference(tmp_path, monkeypatch, dtype, workers, q):
    monkeypatch.setattr(_search, "_MIN_ROWS", 16)  # many blocks per shard
    monkeypatch.setattr(_search, "_TASK_SCORES", 16 * q)
    monkeypatch.setattr(_search, "_QUERY_BLOCK", 32)  # and several blocks of queries
    path, codes, _ = _store(tmp_path, dtype, texts(300))
    queries = np.random.default_rng(q).standard_normal((q, ENCODER.output_dim)).astype(F32)
    with fse.VectorStore.open(path, encoder=ENCODER) as store:
        got = store.search(queries, k=17, workers=workers, mode="exact")
    ids, scores = _want_search(codes, queries, 17)
    np.testing.assert_array_equal(got.ids, ids)
    np.testing.assert_array_equal(bits(got.scores), bits(scores))


def test_ties_go_to_the_smaller_row_across_blocks_and_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(_search, "_MIN_ROWS", 4)
    monkeypatch.setattr(_search, "_TASK_SCORES", 4)
    sentences = ["w1 w2", "w3", "w1 w2", "w4 w5", "w1 w2"] * 30  # many equal rows
    path, codes, _ = _store(tmp_path, "int8", sentences, shard_rows=7)
    query = np.asarray(ENCODER.encode(["w1 w2"])[0], F32)
    with fse.VectorStore.open(path) as store:
        got = store.search(query, k=40, workers=3, mode="exact")
    ids, scores = _want_search(codes, query[None, :], 40)
    assert got.ids.shape == (40,)
    np.testing.assert_array_equal(got.ids, ids[0])
    np.testing.assert_array_equal(bits(got.scores), bits(scores[0]))
    assert got.ids[:5].tolist() == [0, 2, 4, 5, 7]


def test_the_blas_mode_is_close_but_not_bit_defined(tmp_path):
    path = _store(tmp_path, "float16", texts(400))[0]
    queries = np.random.default_rng(5).standard_normal((40, ENCODER.output_dim)).astype(F32)
    with fse.VectorStore.open(path) as store:
        fast = store.search(queries, k=5, mode="blas")
        exact = store.search(queries, k=5, mode="exact")
    np.testing.assert_allclose(fast.scores, exact.scores, rtol=1e-5, atol=1e-6)
    assert (fast.ids[:, 0] == exact.ids[:, 0]).mean() > 0.95


def test_auto_takes_the_contract_for_a_few_queries_and_blas_above(tmp_path):
    path = _store(tmp_path, "int8", texts(400))[0]
    rng = np.random.default_rng(6)
    few = rng.standard_normal((_search.AUTO_EXACT_MAX, ENCODER.output_dim)).astype(F32)
    many = rng.standard_normal((_search.AUTO_EXACT_MAX + 1, ENCODER.output_dim)).astype(F32)
    with fse.VectorStore.open(path) as store:
        for queries, mode in ((few, "exact"), (many, "blas")):
            auto, want = store.search(queries, k=5), store.search(queries, k=5, mode=mode)
            np.testing.assert_array_equal(auto.ids, want.ids)
            np.testing.assert_array_equal(bits(auto.scores), bits(want.scores))
            np.testing.assert_array_equal(
                bits(store.score(queries)), bits(store.score(queries, mode=mode))
            )
        with pytest.raises(ValueError, match="mode"):
            store.search(few, mode="fast")


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_score_gives_the_reference(tmp_path, monkeypatch, dtype):
    monkeypatch.setattr(_search, "_MIN_ROWS", 16)
    monkeypatch.setattr(_search, "_TASK_SCORES", 16)
    path, codes, scales = _store(tmp_path, dtype, texts(200))
    weights = np.random.default_rng(2).standard_normal((3, ENCODER.output_dim)).astype(F32)
    biases = np.array([0.5, -1, 2], F32)
    with fse.VectorStore.open(path) as store:
        got = store.score(weights, biases, workers=3, mode="exact")
        one = store.score(weights[0], 0.5, mode="exact")
        fast = store.score(weights, biases, mode="blas")
    want = _reference.scores(codes, scales, weights, biases)
    np.testing.assert_array_equal(bits(got), bits(want))
    np.testing.assert_array_equal(bits(one), bits(want[:, 0]))
    np.testing.assert_allclose(fast, want, rtol=1e-5, atol=1e-5)


def test_vectors_decode_the_rows(tmp_path):
    path, codes, scales = _store(tmp_path, "int8", texts(150))
    with fse.VectorStore.open(path) as store:
        assert (len(store), store.dim, store.dtype) == (150, ENCODER.output_dim, "int8")
        assert store.encoder == ENCODER.fingerprint
        assert [s.rows for s in store.shards] == [64, 64, 22]
        np.testing.assert_array_equal(
            store.vectors(60, 130), _reference.dequantize(codes, scales)[60:130]
        )
        assert store.vectors(5, 5).shape == (0, ENCODER.output_dim)
        with pytest.raises(ValueError, match="not within the 150 rows"):
            store.vectors(10, 151)


def test_a_store_is_a_snapshot(tmp_path):
    path, _, _ = _store(tmp_path, "float32", texts(70))
    store = fse.VectorStore.open(path)
    with fse.ShardSink(path, ENCODER) as sink:
        ENCODER.encode(texts(30, seed=9), sink=sink)
    assert len(store) == 70
    assert len(fse.VectorStore.open(path)) == 100
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.search(np.ones(ENCODER.output_dim, F32))


def test_arguments_are_checked(tmp_path):
    path, _, _ = _store(tmp_path, "float32", texts(20))
    with fse.VectorStore.open(path) as store:
        dim = ENCODER.output_dim
        with pytest.raises(ValueError, match="k is 21, but the store holds 20 rows"):
            store.search(np.ones(dim, F32), k=21)
        assert len(store.search(np.ones(dim, F32), k=20).ids) == 20
        with pytest.raises(ValueError, match="query 1 is the zero vector"):
            store.search(np.stack([np.ones(dim, F32), np.zeros(dim, F32)]))
        with pytest.raises(ValueError, match="queries must be a vector of 6 values"):
            store.search(np.ones(dim + 1, F32))
        with pytest.raises(ValueError, match="must be finite"):
            store.search(np.full(dim, np.nan, F32))
        with pytest.raises(TypeError, match="must be numbers"):
            store.score(np.array(["a"] * dim))
        with pytest.raises(ValueError, match="bias must be one number or 2 numbers"):
            store.score(np.ones((2, dim), F32), np.ones(3))
    with pytest.raises(ValueError, match="holds the rows of encoder"):
        fse.VectorStore.open(path, encoder=unfolded(steps="normalize"))
    with pytest.raises(TypeError, match=r"VectorStore.open\(path\)"):
        fse.VectorStore(path)
    with pytest.raises(ValueError, match="backend must be one of"):
        fse.VectorStore.open(path, backend="gpu")


def test_verify_catches_a_changed_byte(tmp_path):
    path, _, _ = _store(tmp_path, "int8", texts(100))
    with fse.VectorStore.open(path) as store:
        store.verify()
        name = store.shards[1].name
    file = path / name / "vectors.npy"
    data = bytearray(file.read_bytes())
    data[-1] ^= 1
    file.chmod(0o644)
    file.write_bytes(bytes(data))
    with fse.VectorStore.open(path) as store, pytest.raises(CorruptArtifactError, match="checksum"):
        store.verify()


_UPWARD = {"arm64": 0x00400000, "aarch64": 0x00400000, "x86_64": 0x0800}  # FE_UPWARD


@pytest.mark.skipif(
    sys.platform == "win32" or platform.machine() not in _UPWARD,
    reason="no fesetround of the C library, or its constants are not known here",
)
def test_inputs_are_converted_as_the_contract_whatever_the_rounding_of_the_thread(tmp_path):
    upward, nearest = _UPWARD[platform.machine()], 0  # FE_TONEAREST is 0 on all of them
    # fesetround is in libc on macOS and in libm with glibc.
    libc = ctypes.CDLL(ctypes.util.find_library("c" if sys.platform == "darwin" else "m"))
    path = _store(tmp_path, "float32", texts(50))[0]
    queries = np.random.default_rng(7).standard_normal((2, ENCODER.output_dim))  # float64
    bias = 1 + 2**-30  # not a float32: rounds to 1 to nearest, up to the next float32
    with fse.VectorStore.open(path) as store:
        want = store.search(queries.astype(F32), k=5, mode="exact")
        want_score = store.score(queries.astype(F32), np.float32(1.0), mode="exact")
        assert libc.fesetround(upward) == 0
        try:
            got = store.search(queries, k=5, mode="exact")
            got_score = store.score(queries, bias, mode="exact")
        finally:
            libc.fesetround(nearest)
    np.testing.assert_array_equal(got.ids, want.ids)
    np.testing.assert_array_equal(bits(got.scores), bits(want.scores))
    np.testing.assert_array_equal(bits(got_score), bits(want_score))


def test_a_store_cannot_be_closed_while_it_is_searched(tmp_path, monkeypatch):
    import threading

    path = _store(tmp_path, "float32", texts(50))[0]
    entered, release = threading.Event(), threading.Event()
    query = np.ones(ENCODER.output_dim, F32)
    with fse.VectorStore.open(path) as store:
        real = store._backend.cosines

        def cosines(*args):
            entered.set()
            release.wait(10)  # the search holds the store here
            return real(*args)

        monkeypatch.setattr(store._backend, "cosines", cosines)
        found = []
        search = threading.Thread(target=lambda: found.append(store.search(query, k=3)))
        search.start()
        assert entered.wait(10)
        with pytest.raises(RuntimeError, match="being searched"):
            store.close()
        release.set()
        search.join(10)
        assert len(found) == 1
    with pytest.raises(RuntimeError, match="closed"):
        store.search(query)
