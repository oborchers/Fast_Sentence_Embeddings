"""Task-side ASCII copying preserves bytes, rows, failures and snapshot ownership."""

import gc
import threading
import weakref

import numpy as np
import pytest

from fse import Encoder, MemorySink, _native
from fse._contract import Normalize
from fse._encoder._sources import collection
from fse._runtime import EncodeSpec, Pipeline, _pipeline
from fse._text import CopiedStrings, StringSource, copied_strings
from support.runtime import expected, small_tasks, word_vectors
from support.waiting import arrived, blocked_in_wait


@pytest.fixture(autouse=True)
def reset_copied_string_hooks():
    yield
    _native._string_copy_between(None)
    _native._string_copy_check(None)
    _native._string_copy_fail_stage(0)
    assert _native._test_join_resources() == 0


def pipeline(*, normalize=False, wv=None):
    wv = word_vectors() if wv is None else wv
    return Pipeline(
        EncodeSpec(wv, np.ones(len(wv), np.float32), (Normalize(),) if normalize else ())
    )


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize(
    "rows",
    [
        ["", "", ""],
        ["w1\0w2", "", "w3"],
        ["w1", "w2 w3", "w4 " * 300],
        [chr(i) for i in range(128)],
        ["w1 " * 100_000, "", "w2", "w3"],
    ],
)
def test_bytes_offsets_task_cuts_and_reference_bits(container, rows, monkeypatch):
    small_tasks(monkeypatch)
    source = copied_strings(container(rows))
    assert isinstance(source, CopiedStrings)
    ordinary = StringSource(rows)
    assert not source._offsets.flags.writeable
    assert source._offsets[0] == 0
    assert np.diff(source._offsets).tolist() == [len(row) for row in rows]
    for size in (1, 512, 65536):
        assert source.scan(workers=3, chunk_bytes=size) == ordinary.scan(
            workers=3, chunk_bytes=size
        )
    with pytest.raises(AssertionError, match="not copied every row"):
        _native._test_string_copy_result(source._copy)
    pipe = pipeline()
    actual = pipe.infer(source, workers=3, chunk_bytes=512)
    assert actual.tobytes() == expected(pipe.spec, rows).tobytes()
    data, offsets = _native._test_string_copy_result(source._copy)
    joined, starts = _native.join_strings(rows, 3)
    assert data == joined
    assert offsets.tobytes() == starts.tobytes()


def test_unwritten_bytes_cannot_take_the_ordinary_path():
    source = copied_strings(["w1"])
    for call in (
        source._array,
        lambda: source.encode(None, None, None),
        lambda: source.encode_with(None, None),
    ):
        with pytest.raises(AssertionError, match="only native tasks"):
            call()
    pipe = pipeline()
    pipe._text_rows = False
    with pytest.raises(AssertionError, match="require native tasks"):
        pipe.infer(source, workers=2)


class ListSubclass(list):
    pass


class Encoded(str):
    def encode(self, *_args):
        return b"w3"


def _outcome(call):
    try:
        result = call()
    except (TypeError, ValueError) as error:
        return type(error), str(error)
    return result.tobytes()


@pytest.mark.parametrize(
    "rows",
    [
        ListSubclass(["w1", "w2"]),
        ["w1", Encoded("w2")],
        ["w1", "é"],
        ["w1", "\ud800"],
        ["w1", 3],
        ["w1", ["w2"]],
        [],
        np.array(["w1", "w2"]),
    ],
)
def test_fallback_calls_join_and_keeps_the_error_or_result(rows, monkeypatch):
    encoder = Encoder.from_word_vectors(word_vectors(), normalize=False)
    ordinary = _outcome(lambda: encoder.encode(rows, workers=1))
    calls = []
    join = _native.join_strings

    def spy(*args):
        calls.append(True)
        return join(*args)

    monkeypatch.setattr(_native, "join_strings", spy)
    assert _outcome(lambda: encoder.encode(rows, workers=3)) == ordinary
    assert calls


@pytest.mark.parametrize(("count", "eligible"), [(32767, True), (32768, False), (32769, False)])
def test_row_boundary_routes_to_copy_or_join(count, eligible, monkeypatch):
    encoder = Encoder.from_word_vectors(word_vectors(), normalize=False)
    calls = []
    join = _native.join_strings

    def spy(*args):
        calls.append(True)
        if eligible:
            raise AssertionError("eligible ASCII rows must not join")
        return join(*args)

    monkeypatch.setattr(_native, "join_strings", spy)
    result = encoder.encode(["w1"] * count, workers=3)
    assert len(result) == count
    assert bool(calls) is not eligible


def test_one_worker_and_sinks_keep_join(monkeypatch):
    encoder = Encoder.from_word_vectors(word_vectors(), normalize=False)
    calls = []
    join = _native.join_strings

    def spy(*args):
        calls.append(True)
        return join(*args)

    monkeypatch.setattr(_native, "join_strings", spy)
    encoder.encode(["w1"], workers=1)
    encoder.encode(["w1"], workers=3, sink=MemorySink(encoder.output_dim))
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("stage", "part"), [(1, "snapshot"), (2, "descriptors"), (3, "offsets"), (4, "bytes")]
)
def test_allocation_failures_release_everything(stage, part):
    _native._string_copy_fail_stage(stage)
    with pytest.raises(MemoryError) as caught:
        copied_strings(["w1", "w2"])
    assert str(caught.value) == f"could not allocate the {part} of the strings"
    del caught
    gc.collect()
    assert _native._test_join_resources() == 0
    _native._string_copy_fail_stage(0)
    source = copied_strings(["w1", "w2"])
    assert source is not None
    pipeline().infer(source, workers=2)
    assert _native._test_string_copy_result(source._copy)[0] == b"w1w2"


@pytest.mark.parametrize("replacement", [[], ["w2"], ["w2", "w3", "w4"]])
def test_size_change_during_snapshot_falls_back(replacement):
    rows = ["w1", "w1"]
    _native._string_copy_between(lambda: rows.__setitem__(slice(None), replacement))
    source = collection(rows, "fse_words@1", in_tasks=True)
    assert type(source) is StringSource
    ordinary = StringSource(rows)
    assert source._data == ordinary._data
    assert source._offsets.tobytes() == ordinary._offsets.tobytes()


def test_snapshot_is_unchanged_by_allocation_finalizers(monkeypatch):
    rows = ["w1", "w2"]
    allocate = np.empty

    def empty(*args, **kwargs):
        rows[:] = ["w3"]
        return allocate(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "empty", empty)
        source = copied_strings(rows)
    pipe = pipeline()
    assert pipe.infer(source, workers=2).tobytes() == expected(pipe.spec, ["w1", "w2"]).tobytes()
    assert rows == ["w3"]


def test_thread_replacements_produce_one_complete_snapshot():
    states = [["w1", "w2"], ["w3", "w4", "w5"], ["w6"]]
    rows = list(states[0])
    condition = threading.Condition()
    state = {"wanted": None, "done": False, "stop": False}

    def replace():
        with condition:
            while not state["stop"]:
                while state["wanted"] is None and not state["stop"]:
                    condition.wait(10)
                if state["stop"]:
                    break
                rows[:] = states[state["wanted"]]
                state.update(wanted=None, done=True)
                condition.notify_all()

    thread = threading.Thread(target=replace, name="replace-strings")
    reached = blocked_in_wait(condition, thread.name)
    thread.start()
    try:
        assert arrived(reached, condition)
        pipe = pipeline()
        wants = [expected(pipe.spec, s).tobytes() for s in states]
        for index in range(3):
            source = copied_strings(rows)
            with condition:
                state.update(wanted=index, done=False)
                condition.notify_all()
                assert condition.wait_for(lambda: state["done"], timeout=10)
            actual = pipe.infer(source, workers=2)
            assert actual.tobytes() == wants[(index - 1) % 3 if index else 0]
            del source
    finally:
        with condition:
            state["stop"] = True
            condition.notify_all()
        thread.join(10)
    assert not thread.is_alive()


def test_retry_before_copy_copies_whole_task_once(monkeypatch):
    source = copied_strings(["w1", "w2", "w3", "w4", "w5"])
    pipe = pipeline()
    attempts = []
    snapshots = []

    def before_check(first, last):
        attempts.append(last - first)
        if len(attempts) == 1:
            raise ValueError("before copy")
        if len(attempts) > 2:
            snapshots.append(_native._test_string_copy_result(source._copy)[0])

    # Drive the same native attempt through several prefixes, as earliest does.
    def prefixes(run, count):
        with pytest.raises(ValueError, match="before copy"):
            run(count)
        for n in (2, 1, 4, count):
            run(n)

    _native._string_copy_check(before_check)
    monkeypatch.setattr(_pipeline, "earliest", prefixes)
    actual = pipe.infer(source, workers=2)
    assert attempts == [5, 2, 1, 4, 5]
    assert snapshots == [b"w1w2w3w4w5"] * 3
    assert actual.tobytes() == expected(pipe.spec, ["w1", "w2", "w3", "w4", "w5"]).tobytes()


@pytest.mark.parametrize("chunk_bytes", [1, 1 << 20])
def test_competing_errors_keep_the_earliest_row(chunk_bytes, monkeypatch):
    small_tasks(monkeypatch)
    from fse._wordvectors import WordVectors

    wv = WordVectors.from_arrays(
        np.array(
            [[1, 1], [np.float32(1e-30), 0], [np.finfo(np.float32).max, np.finfo(np.float32).max]],
            np.float32,
        ),
        words=["good", "zero", "huge"],
    )
    pipe = pipeline(normalize=True, wv=wv)
    rows = ["good", "zero", "huge huge", "good", "zero"]
    ordinary = _outcome(lambda: pipe.infer(StringSource(rows), workers=3, chunk_bytes=chunk_bytes))
    source = copied_strings(rows)
    owner = weakref.ref(source._copy)
    actual = _outcome(lambda source=source: pipe.infer(source, workers=3, chunk_bytes=chunk_bytes))
    assert actual == ordinary
    assert actual[0] is ValueError
    assert "row 1" in actual[1]
    del source
    gc.collect()
    assert owner() is None


def test_success_releases_owner():
    source = copied_strings(["w1", "w2"])
    owner = weakref.ref(source._copy)
    pipeline().infer(source, workers=2)
    del source
    gc.collect()
    assert owner() is None


@pytest.mark.parametrize("word_dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_word_and_output_dtypes_preserve_all_bits(word_dtype, dtype, monkeypatch):
    small_tasks(monkeypatch)
    rows = ["", "w1 w2", "w3\0w4", "unknown", "w5 " * 200] * 30
    pipe = pipeline(normalize=True, wv=word_vectors(word_dtype))
    ordinary = pipe.infer(StringSource(rows), workers=3, dtype=dtype, chunk_bytes=256)
    actual = pipe.infer(copied_strings(rows), workers=3, dtype=dtype, chunk_bytes=256)
    parts = actual if isinstance(actual, tuple) else (actual,)
    want = ordinary if isinstance(ordinary, tuple) else (ordinary,)
    assert [p.tobytes() for p in parts] == [p.tobytes() for p in want]
    if dtype == "float32":
        assert actual.tobytes() == expected(pipe.spec, rows).tobytes()


def test_numpy_backend_keeps_join(monkeypatch):
    encoder = Encoder.from_word_vectors(word_vectors(), normalize=False)

    def forbidden(*_args):
        raise AssertionError("the numpy backend must not prepare task copies")

    monkeypatch.setattr(_native, "copy_strings", forbidden)
    actual = encoder.encode(["w1", "w2"], workers=3, backend="numpy")
    assert len(actual) == 2


@pytest.mark.skipif(
    __import__("sys").version_info >= (3, 12), reason="legacy strings end with 3.11"
)
def test_legacy_string_falls_back_before_reading_its_data(monkeypatch):
    import ctypes

    api = ctypes.pythonapi
    api.PyUnicode_FromUnicode.restype = ctypes.py_object
    api.PyUnicode_FromUnicode.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
    api.PyUnicode_AsUnicode.restype = ctypes.c_void_p
    api.PyUnicode_AsUnicode.argtypes = [ctypes.py_object]
    with pytest.warns(DeprecationWarning, match=r"PyUnicode_FromUnicode\(NULL, size\)"):
        legacy = api.PyUnicode_FromUnicode(None, 2)
    wide = ctypes.create_unicode_buffer("w2", 3)
    ctypes.memmove(api.PyUnicode_AsUnicode(legacy), wide, 2 * ctypes.sizeof(ctypes.c_wchar))
    rows = ["w1", legacy]
    assert copied_strings(rows) is None  # Do not ready it before testing the fallback.
    calls = []
    join = _native.join_strings

    def spy(*args):
        calls.append(True)
        return join(*args)

    monkeypatch.setattr(_native, "join_strings", spy)
    source = collection(rows, "fse_words@1", in_tasks=True)
    assert type(source) is StringSource
    assert calls
    assert (
        pipeline().infer(source, workers=2).tobytes()
        == pipeline().infer(StringSource(["w1", "w2"]), workers=2).tobytes()
    )
