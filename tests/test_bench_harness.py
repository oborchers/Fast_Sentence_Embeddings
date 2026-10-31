"""The shared harness of the paired benchmarks (benchmarks/bench_harness.py): the order of
the sides, the raw data of every run, rounds discarded for their load, and a failing child."""

import importlib
import json
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"

pytestmark = [
    pytest.mark.skipif(
        not (BENCHMARKS / "bench_harness.py").exists(),
        reason="benchmarks/ is not part of this checkout",
    ),
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="the benchmark scripts use resource and os.getloadavg (POSIX only)",
    ),
]


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("bench_harness")


def _loads(monkeypatch, harness, values):
    """``harness.load`` returns ``values`` in turn (the last one for ever)."""
    it = iter(values)
    last = [values[-1]]

    def load():
        last[0] = next(it, last[0])
        return last[0]

    monkeypatch.setattr(harness, "load", load)


def test_the_sides_alternate_and_the_first_round_starts_with_the_first(harness):
    assert harness.order(["a", "b", "c"], 0) == ["a", "b", "c"]
    assert harness.order(["a", "b", "c"], 1) == ["c", "b", "a"]
    assert harness.order(["a", "b"], 2) == ["a", "b"]


def test_rounds_run_in_the_order_of_the_attempt_and_keep_every_run(harness):
    seen = []

    def run_one(name):
        seen.append(name)
        return {"seconds": len(seen)}

    done = harness.rounds(["x", "y"], run_one, 3)
    assert seen == ["x", "y", "y", "x", "x", "y"]
    assert [r.result["seconds"] for r in done.runs("x")] == [1, 4, 5]
    assert done.results("y") == [{"seconds": 2}, {"seconds": 3}, {"seconds": 6}]
    assert done.by_round()[1] == {"x": {"seconds": 4}, "y": {"seconds": 3}}
    assert done.discarded == []
    assert all(r.seconds >= 0 for r in done.runs("x"))


def test_a_fixed_order_does_not_alternate_and_the_warm_hook_runs_before_each_run(harness):
    events = []
    done = harness.rounds(
        ["x", "y"],
        lambda name: events.append(("run", name)),
        2,
        warm=lambda name: events.append(("warm", name)),
        alternate=False,
    )
    assert events == [(k, n) for _ in range(2) for n in "xy" for k in ("warm", "run")]
    assert len(done.kept) == 2


def test_the_load_is_taken_before_and_after_every_run(harness, monkeypatch):
    _loads(monkeypatch, harness, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    done = harness.rounds(["x", "y"], lambda _: {"v": 1}, 2)
    (x0, x1), (y0, y1) = done.runs("x"), done.runs("y")
    assert (x0.before, x0.after, y0.before, y0.after) == (1.0, 1.5, 2.0, 2.5)
    # the second round starts with y
    assert (y1.before, y1.after, x1.before, x1.after) == (3.0, 3.5, 4.0, 4.5)
    assert done.loads == [(1.0, 2.5), (3.0, 4.5)]
    assert x0.record() == {"v": 1, "load_average": 1.0, "load_after": 1.5}


def test_a_round_with_a_high_load_is_discarded_but_its_data_is_kept(harness, monkeypatch):
    # round 0: load 1 and 9 after (discarded); round 1: 1 and 1; round 2: 1 and 1
    _loads(monkeypatch, harness, [1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1])
    calls = []
    done = harness.rounds(["x", "y"], lambda n: calls.append(n) or len(calls), 2, max_load=5.0)
    assert len(done.discarded) == 1
    assert [r.result for r in done.discarded[0].values()] == [1, 2]
    assert len(done.kept) == 2
    assert done.results("x") == [4, 5]  # round 1 started with y, round 2 with x
    assert done.loads == [(1, 1), (1, 1)]


def test_no_round_is_discarded_without_a_limit_or_without_a_load_average(harness, monkeypatch):
    _loads(monkeypatch, harness, [99.0])
    assert len(harness.rounds(["x"], lambda _: 0, 2).kept) == 2
    _loads(monkeypatch, harness, [float("nan")])
    done = harness.rounds(["x"], lambda _: 0, 2, max_load=1.0)
    assert len(done.kept) == 2
    assert done.discarded == []


def test_rounds_stop_after_three_times_the_wanted_ones_or_the_budget(harness, monkeypatch):
    _loads(monkeypatch, harness, [99.0])
    done = harness.rounds(["x"], lambda _: 0, 2, max_load=1.0)
    assert (len(done.kept), len(done.discarded)) == (0, 6)
    _loads(monkeypatch, harness, [0.0])
    assert len(harness.rounds(["x"], lambda _: 0, 2, budget_s=-1.0).kept) == 0


def test_the_ratio_of_each_round_to_the_first_side(harness):
    seconds = {"a": iter([2.0, 4.0, 3.0]), "b": iter([3.0, 2.0, 0.0])}
    done = harness.rounds(["a", "b"], lambda n: {"s": next(seconds[n])}, 3)
    # the rounds of b with the value of a as the base
    assert done.ratios("b", "a", lambda r: r["s"]) == [1.5, 0.5, 0.0]
    assert done.ratios("a", "b", lambda r: r["s"]) == [2 / 3, 2.0]  # a zero base is left out


def test_medians(harness):
    assert harness.median([3, 1, 2]) == 2
    assert harness.median([]) == 0.0
    runs = [{"a": 1, "b": 10.0, "flag": True, "name": "x"}, {"a": 3, "b": 20.0}]
    assert harness.median_numbers(runs) == {"a": 2, "b": 15.0}


def _child(tmp_path, body):
    script = tmp_path / "child.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def test_a_child_prints_its_json_on_the_last_line_or_on_all_of_stdout(harness, tmp_path):
    noisy = _child(tmp_path, "print('warming up')\nprint('{\"x\": [1, 2]}')")
    assert harness.run_child(noisy) == {"x": [1, 2]}
    whole = _child(tmp_path, "import json\nprint(json.dumps({'x': 1}, indent=1))")
    assert harness.run_child(whole, stdout="all") == {"x": 1}
    with pytest.raises(json.JSONDecodeError):
        harness.run_child(whole)  # the last line of indented JSON is "}"


def test_a_failing_child_ends_the_run_with_the_tail_of_its_stderr(harness, tmp_path):
    cmd = _child(tmp_path, "import sys\nsys.stderr.write('x' * 5000 + 'the reason')\nsys.exit(3)")
    with pytest.raises(SystemExit) as error:
        harness.run_child(cmd, label={"kind": "case"})
    message = str(error.value)
    assert message.startswith("{'kind': 'case'} failed:\n")
    assert message.endswith("the reason")
    assert len(message) < 3100


def test_a_silent_child_and_a_timeout_end_the_run(harness, tmp_path):
    with pytest.raises(SystemExit, match="printed nothing"):
        harness.run_child(_child(tmp_path, "pass"), label="quiet")
    with pytest.raises(SystemExit, match=r"timed out after 0\.5 s"):
        harness.run_child(_child(tmp_path, "import time\ntime.sleep(30)"), timeout=0.5)


def test_the_environment_of_a_child(harness, tmp_path):
    cmd = _child(tmp_path, "import json, os\nprint(json.dumps(os.environ.get('HARNESS_X')))")
    assert harness.run_child(cmd, env={"HARNESS_X": "1"}) == "1"


def test_the_timeline_compare_runs_on_the_harness(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    run = importlib.import_module("runtime_timeline_run")
    assert run.run_rounds is importlib.import_module("bench_harness").rounds
