"""Bounded child-process probes for native entry, signals and fork ownership."""

import gc
import json
import os
import signal
import sys
import threading
import time
import warnings

import numpy as np

from fse import _native
from fse._text import StringSource, copied_strings
from support.native_team import faults, pipeline, tasks_for
from support.runtime import expected, texts
from support.waiting import until


def entered():
    assert until(lambda: _native._test_team_control()[1] >= 2, timeout=10)


def interrupt():
    if os.name == "nt":
        import _thread

        _thread.interrupt_main()
    else:
        os.kill(os.getpid(), signal.SIGINT)


def main_signal(model, source, twice=False, copy_count=None):
    errors = []

    def sender():
        try:
            entered()
            if copy_count is not None:
                _native._test_team_control(False)
                assert until(lambda: _native._test_team_copy_control()[1] == copy_count)
            interrupt()
            if twice:
                interrupt()
        except BaseException as error:
            errors.append(repr(error))
        finally:
            _native._test_team_copy_control(-1)
            _native._test_team_control(False)

    worker = threading.Thread(target=sender)
    worker.start()
    received = False
    started = time.monotonic()
    try:
        model.infer(source, workers=2)
    except KeyboardInterrupt:
        received = True
    finally:
        worker.join(timeout=15)
    assert not worker.is_alive()
    assert not errors
    assert received
    return {"interrupt": True, "latency": time.monotonic() - started}


def background(model, source, wanted):
    result, errors = [], []

    def encode():
        try:
            result.append(model.infer(source, workers=2))
        except BaseException as error:
            errors.append(repr(error))

    worker = threading.Thread(target=encode)
    worker.start()
    entered()
    return worker, result, errors, wanted


def background_signal(model, source, wanted):
    worker, result, errors, wanted = background(model, source, wanted)
    received = False
    try:
        interrupt()
        until(lambda: not worker.is_alive(), timeout=1)
    except KeyboardInterrupt:
        received = True
    finally:
        _native._test_team_control(False)
        worker.join(timeout=15)
    assert not worker.is_alive()
    assert not errors
    assert received
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].view(np.uint32), wanted.view(np.uint32))
    return {"background_result": True, "interrupt": True}


def fork_probe(model, source, wanted):
    worker, result, errors, wanted = background(model, source, wanted)
    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            _native._test_team_control(False)
            baseline = _native._test_team_resources()
            found = model.infer(source, workers=2)
            np.testing.assert_array_equal(found.view(np.uint32), wanted.view(np.uint32))
            assert _native._test_team_resources() == baseline
            os.write(write_fd, json.dumps({"baseline": baseline, "child": True}).encode())
        except BaseException as error:
            os.write(write_fd, repr(error).encode())
            os._exit(1)
        os._exit(0)
    os.close(write_fd)
    _native._test_team_control(False)
    worker.join(timeout=15)
    _, status = os.waitpid(pid, 0)
    receipt = os.read(read_fd, 4096)
    os.close(read_fd)
    assert status == 0, receipt
    assert not worker.is_alive()
    assert not errors
    assert len(result) == 1
    if sys.version_info >= (3, 12):
        assert any(issubclass(r.category, DeprecationWarning) for r in records)
    else:
        assert not records
    return json.loads(receipt)


def main():
    model = pipeline()
    sentences = texts(2000)
    copied = len(sys.argv) > 2 and sys.argv[2] == "copied"
    source = copied_strings(sentences) if copied else StringSource(sentences)
    task_count = len(tasks_for(model, source, 2))
    assert task_count > len(tasks_for(model, source, 2, refined=False))
    wanted = expected(model.spec, sentences)
    baseline = (_native._test_team_resources(), _native._test_join_resources())
    _native._test_team_control(True)
    mode = sys.argv[1]
    if mode == "interrupt-error":
        previous = _native._test_corrupt(1)
        _native._test_team_copy_control(0)
        try:
            with faults(corrupt_task=0):
                receipt = main_signal(model, source, copy_count=task_count)
        finally:
            _native._test_corrupt(previous)
    elif mode.startswith("main"):
        receipt = main_signal(model, source, mode == "main-twice")
    elif mode == "background":
        receipt = background_signal(model, source, wanted)
    else:
        receipt = fork_probe(model, source, wanted)
    if copied:
        assert _native._test_string_copy_progress(source._copy) == (len(sentences), task_count)
        assert _native._test_string_copy_result(source._copy)[0] == "".join(sentences).encode()
        receipt["copies_accounted"] = True
    gc.collect()
    assert (_native._test_team_resources(), _native._test_join_resources()) == baseline
    receipt["balanced"] = True
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
