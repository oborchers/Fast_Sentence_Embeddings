"""RAM-limited sizing stays above real call durations without allocating large inputs."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return {
        name: importlib.import_module(name)
        for name in (
            "pod4_calibration",
            "pod4_storage",
            "pod4_memory",
            "pod4_readme",
            "pod4_inputs",
        )
    }


def test_memory_rejection_shrinks_input_and_keeps_every_specification(
    modules, monkeypatch, tmp_path
):
    module = modules["pod4_calibration"]
    options = SimpleNamespace(output=tmp_path, deadline=float("inf"), count=0)
    seen, released = [], []
    monkeypatch.setattr(
        module, "bind_input", lambda options, _source, count: setattr(options, "count", count)
    )
    monkeypatch.setattr(module, "release_current", lambda options: released.append(options.count))

    def observe(options, _repo, _dtype, workers):
        seen.append((options.count, workers))
        if options.count > 2_000_000:
            raise modules["pod4_memory"].MemoryAdmissionError(
                {"limit": 64, "current": 20, "additional_budget": 60}
            )
        return options.count / 750_000 if workers == 32 else options.count / 100_000

    monkeypatch.setattr(module, "calibration", observe)
    chosen = module.calibrate(options, [], [("model", "float32", w) for w in (1, 16, 32)])
    assert 1_500_000 <= chosen <= 2_000_000
    assert {(n, w) for n, w in seen if n == chosen} == {(chosen, 1), (chosen, 16), (chosen, 32)}
    assert any(n > 2_000_000 for n in released)
    decision = json.loads((tmp_path / "calibration-decisions.jsonl").read_text().splitlines()[-1])
    assert decision["memory_constrained"] is True
    assert decision["minimum_actual_fse_call_seconds"] >= 2


def test_impossible_memory_duration_pair_fails_instead_of_shortening_calls(
    modules, monkeypatch, tmp_path
):
    module = modules["pod4_calibration"]
    options = SimpleNamespace(output=tmp_path, deadline=float("inf"), count=0)
    monkeypatch.setattr(
        module, "bind_input", lambda options, _source, count: setattr(options, "count", count)
    )
    monkeypatch.setattr(module, "release_current", lambda _options: None)

    def observe(options, *_spec):
        if options.count > 500_000:
            raise modules["pod4_memory"].MemoryAdmissionError(
                {"limit": 64, "current": 20, "additional_budget": 60}
            )
        return 1

    monkeypatch.setattr(module, "calibration", observe)
    with pytest.raises(MemoryError, match=r"duration and memory|bounded calibration"):
        module.calibrate(options, [], [("model", "float32", 32)])


def test_cleanup_preserves_identity_and_pinned_model_inputs(modules, tmp_path):
    module = modules["pod4_storage"]
    from compare_artifacts import file_digest

    folder = tmp_path / "inputs"
    folder.mkdir()
    path = folder / "passages-2-seed0.txt"
    path.write_text("one\ntwo\n")
    receipt = path.with_suffix(".identity.json")
    receipt.write_text(json.dumps({"checksum": file_digest(path)}))
    data = receipt.read_bytes()
    options = SimpleNamespace(inputs=folder, output=tmp_path, pinned_inputs={path}, input_file=path)
    module.release_current(options)
    assert path.exists()
    options.pinned_inputs.clear()
    module.release_current(options)
    assert not path.exists()
    assert not receipt.exists()
    assert options.input_file is None
    assert (tmp_path / "input-identities" / receipt.name).read_bytes() == data


def test_validation_cache_flush_precedes_advice_and_keeps_output(modules, monkeypatch, tmp_path):
    module = modules["pod4_storage"]
    calls = []
    path = tmp_path / "output.npy"
    path.write_bytes(b"unchanged")
    monkeypatch.setattr(module.os, "fsync", lambda _fd: calls.append("sync"))
    monkeypatch.setattr(
        module.os, "posix_fadvise", lambda *_args: calls.append("advice"), raising=False
    )
    monkeypatch.setattr(module.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    module.discard_validation_cache(path)
    assert calls == ["sync", "advice"]
    assert path.read_bytes() == b"unchanged"


def test_released_input_regenerates_byte_identically_without_recounting(
    modules, monkeypatch, tmp_path
):
    import shutil

    inputs = modules["pod4_inputs"]
    storage = modules["pod4_storage"]
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(modules["pod4_memory"], "budget", lambda: (64 * 1024**3, 0))
    monkeypatch.setattr(inputs, "sampled_lengths", lambda *_args: {"statistics": "exact fixture"})
    folder = tmp_path / "inputs"
    source = ["one two three four five", "six seven eight nine ten"]
    path, record = inputs.write_input(folder, source, 12)
    data = path.read_bytes()
    options = SimpleNamespace(inputs=folder, output=tmp_path, pinned_inputs=set(), input_file=path)
    storage.release_current(options)

    def refuse_recount(*_args):
        raise AssertionError("identical regenerated content must reuse its immutable receipt")

    monkeypatch.setattr(inputs, "sampled_lengths", refuse_recount)
    regenerated, reused = inputs.write_input(
        folder, source, 12, receipt_archive=tmp_path / "input-identities"
    )
    assert regenerated.read_bytes() == data
    assert reused == record


def test_later_model_memory_failure_preserves_earlier_measured_cells(
    modules, monkeypatch, tmp_path
):
    panel = modules["pod4_readme"]
    options = SimpleNamespace(
        output=tmp_path,
        cache=tmp_path,
        models=tmp_path,
        inputs=tmp_path / "inputs",
        deadline=float("inf"),
        pinned_inputs=set(),
        failed_models={},
    )
    (tmp_path / "probe.json").write_text('{"passed":true}')
    specs = [(panel.POTION, "float32", 1, None), (panel.STATIC_ST, "float32", 1, "st")]
    monkeypatch.setattr(panel, "cases", lambda: specs)
    monkeypatch.setattr(panel, "options_for", lambda _args: options)
    monkeypatch.setattr(panel, "source_passages", lambda _cache: [])

    def calibrate(options, _source, specs):
        if specs[0][0] == panel.STATIC_ST:
            raise modules["pod4_memory"].MemoryAdmissionError(
                {"limit": 64, "current": 20, "additional_budget": 60}
            )
        options.input_file = tmp_path / "input.txt"
        return 100

    def bind(options, _source, _count):
        options.input_file = tmp_path / "input.txt"
        options.input_record = {"texts": 100}

    monkeypatch.setattr(panel, "calibrate", calibrate)
    monkeypatch.setattr(panel, "bind_input", bind)
    monkeypatch.setattr(
        panel, "measure", lambda _options, *spec: {"cell": panel.case_id(*spec), "eligible": True}
    )
    monkeypatch.setattr(panel, "check_durations", lambda *_args: True)
    monkeypatch.setattr(panel, "release_current", lambda _options: None)
    monkeypatch.setattr(panel, "release", lambda *_args: None)
    monkeypatch.setattr(panel, "export_jsonl", lambda *_args: None)
    monkeypatch.setattr(panel, "emit", lambda *_args: None)
    with pytest.raises(RuntimeError, match="failed, short or noisy"):
        panel.run(SimpleNamespace(output=tmp_path, cache=tmp_path))
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["cells"][0]["cell"] == panel.case_id(*specs[0])
    assert report["cells"][0]["eligible"] is True
    assert report["failed"] == [panel.case_id(*specs[1])]
    assert report["complete"] is False


def test_calibration_reserves_the_other_correctness_output(modules, monkeypatch):
    memory = modules["pod4_memory"]
    monkeypatch.setattr(memory, "budget", lambda: (1000, 400))
    assert memory.encoding(1, 10, 0, 0)["additional_budget"] == 416
    with pytest.raises(memory.MemoryAdmissionError, match="validation output reserved=True"):
        memory.encoding(1, 10, 0, 0, reserve_output=True)


def test_child_memory_rejection_is_preserved_before_bounded_retry(modules, monkeypatch, tmp_path):
    module = modules["pod4_calibration"]
    options = SimpleNamespace(output=tmp_path, deadline=float("inf"))
    budget = {
        "limit": 64_000_000_000,
        "current": 12_000_000_000,
        "additional_budget": 44_000_000_000,
    }
    payload = json.dumps({"event": "memory_admission_rejected", "budget": budget}) + "\n"
    monkeypatch.setattr(module, "ensure_model", lambda *_args: None)
    monkeypatch.setattr(module, "encode_command", lambda *_args: ["child"])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=78, stdout=payload, stderr=""),
    )
    with pytest.raises(modules["pod4_memory"].MemoryAdmissionError) as caught:
        module.calibration(options, "model", "float32", 1)
    assert caught.value.receipt == budget
    raw = json.loads((tmp_path / "calibration.jsonl").read_text())
    assert raw["returncode"] == 78
    assert raw["stdout"] == payload


def test_feasible_short_margin_is_revalidated_when_growth_exhausts_memory(
    modules, monkeypatch, tmp_path
):
    module = modules["pod4_calibration"]
    options = SimpleNamespace(output=tmp_path, deadline=float("inf"), count=0)
    seen = []
    monkeypatch.setattr(
        module, "bind_input", lambda options, _source, count: setattr(options, "count", count)
    )
    monkeypatch.setattr(module, "release_current", lambda _options: None)

    def observe(options, *_spec):
        seen.append(options.count)
        if options.count > 250_000:
            raise modules["pod4_memory"].MemoryAdmissionError(
                {"limit": 64, "current": 20, "additional_budget": 60}
            )
        return 2.2

    monkeypatch.setattr(module, "calibration", observe)
    assert module.calibrate(options, [], [("model", "float32", 32)]) == 250_000
    assert seen[-1] == 250_000
    assert seen.count(250_000) == 2
    decision = json.loads((tmp_path / "calibration-decisions.jsonl").read_text().splitlines()[-1])
    assert decision["revalidated_fallback"] is True


def test_binding_preserves_receipt_before_a_pinned_measurement_can_timeout(
    modules, monkeypatch, tmp_path
):
    from compare_artifacts import file_digest

    module = modules["pod4_calibration"]
    folder = tmp_path / "inputs"
    folder.mkdir()
    path = folder / "passages-1-seed0.txt"
    path.write_text("one two three four five\n")
    data = json.dumps({"checksum": file_digest(path)}).encode()
    path.with_suffix(".identity.json").write_bytes(data)
    monkeypatch.setattr(module, "write_input", lambda *_args, **_kwargs: (path, json.loads(data)))
    options = SimpleNamespace(inputs=folder, output=tmp_path, pinned_inputs={path})
    module.bind_input(options, [], 1)
    assert path.exists()
    assert (
        tmp_path / "input-identities" / path.with_suffix(".identity.json").name
    ).read_bytes() == data
