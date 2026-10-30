"""The arithmetic of benchmarks/roofline_model.py and the report of roofline_report.py on
hand-computed cases, and the hardware profiles that ship with them (PERFORMANCE.md)."""

import importlib
import math
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"

pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "roofline_model.py").exists(),
    reason="benchmarks/ is not part of this checkout",
)


@pytest.fixture
def rm(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("roofline_model")


def _hw(rm, **kw):
    roof = rm.Bandwidth("spec", 100.0, "spec", "a test")
    return rm.Hardware(
        **{
            "name": "t",
            "p_cores": 8,
            "e_cores": 2,
            "l2_cluster_bytes": 0,
            "bandwidths": (roof,),
            **kw,
        }
    )


def _stages(rm, serial: float, parallel: float):
    return rm.Stages(serial, 0.0, 0.0, 0.0, parallel, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("dtype", "row", "token"), [("float32", 1024, 1032), ("float16", 512, 520), ("int8", 256, 268)]
)
def test_bytes_a_token_reads(rm, dtype, row, token):
    model = rm.Model("m", 256, dtype, 10, False)
    assert (model.row_bytes, model.token_bytes) == (row, token)
    assert model.table_bytes == 10 * (token - 4)  # the ids are in the batch, not the table


def test_bytes_per_token_spreads_the_output_row_over_its_text(rm):
    data = rm.Dataset("d", texts=10, bytes=100, tokens=40, distinct_ids=5)
    model = rm.Model("m", 256, "float32", 10, True)
    assert rm.bytes_per_token(model, data) == 1032 + 4 * 256 / 4


@pytest.mark.parametrize(
    ("workers", "ratio", "c"),
    [(1, 1.0, 1), (8, 0.5, 8), (9, 0.5, 8.5), (10, 0.5, 9), (16, 0.5, 9), (10, 1.0, 10)],
)
def test_cores_fill_the_p_cores_first(rm, workers, ratio, c):
    assert rm.cores(_hw(rm, e_core_ratio=ratio), workers) == c


def test_cores_refuse_no_worker(rm):
    with pytest.raises(ValueError, match="workers must be positive"):
        rm.cores(_hw(rm), 0)


def test_bounds_of_a_hand_case(rm):
    data = rm.Dataset("d", texts=1000, bytes=1, tokens=10**6, distinct_ids=1)
    model = rm.Model("m", 256, "float32", 10, False)  # 1032 + 1.024 bytes per token
    stages = _stages(rm, serial=0.1, parallel=0.9)
    b = rm.bounds(_hw(rm), model, data, stages, 8)
    assert b.core == pytest.approx(8 / 9e-7)
    assert b.amdahl == pytest.approx(1 / (1e-7 + 9e-7 / 8))
    memory = 1033.024 / 100e9
    assert b.memory["spec"] == pytest.approx(1 / memory)
    assert b.combined["spec"] == pytest.approx(1 / (1e-7 + max(9e-7 / 8, memory)))
    assert b.binding["spec"] == "work"
    assert rm.serial_fraction(stages) == pytest.approx(0.1)


def test_bounds_name_what_binds(rm):
    data = rm.Dataset("d", texts=1, bytes=1, tokens=10**6, distinct_ids=1)
    model = rm.Model("m", 4096, "float32", 10, False)  # about 16 KiB a token
    assert rm.bounds(_hw(rm), model, data, _stages(rm, 0.0, 0.8), 8).binding["spec"] == "memory"
    assert rm.bounds(_hw(rm), model, data, _stages(rm, 1.0, 0.8), 8).binding["spec"] == "serial"


def test_bounds_take_the_scan_at_n_workers(rm):
    data = rm.Dataset("d", texts=1, bytes=1, tokens=10**6, distinct_ids=1)
    model = rm.Model("m", 8, "float32", 10, False)
    stages = rm.Stages(0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0)
    b = rm.bounds(_hw(rm), model, data, stages, 8, plan=0.1)
    assert b.amdahl == pytest.approx(1 / (1e-7 + 8e-7 / 8))


def test_share_refuses_a_bound_that_is_no_rate(rm):
    assert rm.share(5.0, 10.0) == 0.5
    for bound in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="positive and finite"):
            rm.share(1.0, bound)


def test_shipped_profiles_name_their_sources(rm):
    profiles = rm.load_profiles()
    assert {"m1-pro-10", "m4-pro-14"} <= set(profiles)
    for hw in profiles.values():
        assert hw.notes
        assert all(r.source for r in hw.bandwidths)
        assert any(r.kind == "spec" for r in hw.bandwidths)
    assert profiles["m1-pro-10"].measured
    assert not profiles["m4-pro-14"].measured


_PROFILE = """
[x]
p_cores = 1
e_cores = 0
l2_cluster_bytes = 0
topology_source = "test"
{extra}
[[x.bandwidth]]
name = "spec"
gb_per_s = 1
kind = "{kind}"
source = "{source}"
"""


@pytest.mark.parametrize(
    ("extra", "kind", "source", "message"),
    [
        ("", "spec", "", "has no source"),
        ("", "guess", "s", "must be spec or measured"),
        ("e_core_ratio = 0.5", "spec", "s", "e_core_ratio .* has no source"),
    ],
)
def test_profiles_refuse_numbers_without_source(rm, tmp_path, extra, kind, source, message):
    path = tmp_path / "hw.toml"
    path.write_text(_PROFILE.format(extra=extra, kind=kind, source=source), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        rm.load_profiles(path)


def test_profiles_refuse_a_missing_key(rm, tmp_path):
    path = tmp_path / "hw.toml"
    path.write_text(
        _PROFILE.format(extra="", kind="spec", source="s").replace("e_cores = 0\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="has no 'e_cores'"):
        rm.load_profiles(path)


def _result(hardware: str) -> dict:
    seconds = {1: 0.1, 8: 0.02}
    return {
        "model": {"name": "m", "dim": 256, "dtype": "float32", "vocab": 100, "normalize": True},
        "dataset": {"name": "d", "texts": 100, "bytes": 1000, "tokens": 10**6, "distinct_ids": 50},
        "stages": {
            "join": 0.002,
            "plan": 0.001,
            "caller": 0.001,
            "encode_rest": 0.001,
            "tokenize": 0.04,
            "pool": 0.04,
            "normalize": 0.01,
            "task_rest": 0.005,
        },
        "plan_at": {"8": 0.001},
        "measured": {
            "seconds": seconds,
            "range": {k: (v, v) for k, v in seconds.items()},
            "task_wall": {1: 0.095, 8: 0.12},
            "task_cpu": {1: 0.095, 8: 0.1},
            "tasks": {1: 2, 8: 16},
        },
        "hardware": hardware,
    }


def test_report_gives_the_shares_on_the_measured_machine(rm):
    report = importlib.import_module("roofline_report").report
    text = "\n".join(report(_result("t"), _hw(rm)))
    assert "workers 8: C(N) = 8.0, measured 50.0 M tokens/s" in text
    assert "summed wall 1.26x and thread CPU 1.05x" in text
    assert "  share " in text
    assert "the distinct rows fit" not in text


def test_report_gives_no_share_on_other_hardware(rm):
    report = importlib.import_module("roofline_report").report
    text = "\n".join(report(_result("elsewhere"), _hw(rm)))
    assert "measured on elsewhere, not on t" in text
    assert "  share " not in text


@pytest.mark.usefixtures("rm")  # puts benchmarks/ on the path
def test_dtype_comparison_gives_time_ratios_to_the_first():
    dtype = importlib.import_module("roofline_dtype")
    kept = [
        {"float32": 2.0, "int8": 1.0},
        {"float32": 3.0, "int8": 2.0},
        {"float32": 1.0, "int8": 1.0},
    ]
    summary = dtype._summary(kept, ["float32", "int8"], 8, 1)
    assert summary["int8"]["ratio"] == 1.5
    assert summary["int8"]["range"] == (1.0, 2.0)
    assert summary["float32"]["seconds"] == 2.0
    line = dtype.lines([summary], ["float32", "int8"], 10**6)[0]
    assert line.startswith("workers 8: 3 rounds (1 discarded); float32 0.5 M tokens/s; int8 1.0 M")
    assert "float32 time / int8 time 1.500 (1.000 to 2.000)" in line
    assert dtype.lines([{"workers": 1, "rounds": 0, "discarded": 3}], ["float32"], 1) == [
        "workers 1: 0 rounds (3 discarded)"
    ]


def test_the_callers_part_overlaps_the_workers(rm):
    data = rm.Dataset("d", texts=1, bytes=1, tokens=10**6, distinct_ids=1)
    model = rm.Model("m", 8, "float32", 10, False)
    stages = rm.Stages(0.1, 0.0, 0.5, 0.0, 0.8, 0.0, 0.0, 0.0)  # caller 0.5 s, work 0.8 s
    b = rm.bounds(_hw(rm), model, data, stages, 8)
    assert b.amdahl == pytest.approx(1 / (1e-7 + 5e-7))  # not 1e-7 + 5e-7 + 1e-7
    assert b.binding["spec"] == "caller"
    assert rm.serial_fraction(stages) == pytest.approx(0.1 / 1.4)


def test_a_negative_rest_counts_as_no_serial_time(rm):
    data = rm.Dataset("d", texts=1, bytes=1, tokens=10**6, distinct_ids=1)
    model = rm.Model("m", 8, "float32", 10, False)
    stages = rm.Stages(0.0, 0.0, -0.1, -0.2, 0.8, 0.0, 0.0, 0.0)
    b = rm.bounds(_hw(rm), model, data, stages, 8)
    assert b.amdahl == pytest.approx(b.core)
