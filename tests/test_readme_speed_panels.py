"""Unfinished or incompatible observations must not become public speed claims."""

import copy
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PANEL = runpy.run_path(str(ROOT / "media/readme/speed_panels.py"))


def row(backend="fse", dtype="float32", workers=1):
    return {
        "complete": True,
        "runtime_attributed": True,
        "aa_upper": 1.05,
        "backend": backend,
        "runtime_fingerprint": f"test-only-{backend}",
        "source": {
            "commit": "0" * 40,
            "path": "benchmarks/baseline/test.jsonl",
            "series_id": "test",
        },
        "context": {
            "model": "test/model",
            "revision": "1" * 40,
            "parent_artifact": "test-weights",
            "input_digest": "test-input",
            "texts": 100,
            "corpus": "test-corpus",
            "machine": "test-machine",
            "date": "test-date",
            "workers": workers,
            "table_dtype": dtype,
            "output_dtype": "float32",
            "output_dim": 256,
            "normalize": True,
            "tokenizer_rule": "test-rule",
        },
        "texts_per_second": [100, 200, 300],
        "call_seconds": [2, 3, 4],
    }


def pair(workers=1):
    reference = row("model2vec", workers=workers)
    reference["texts_per_second"] = [10, 200, 30]
    return {"fse": row(workers=workers), "reference": reference, "output_checked": True}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("complete", False),
        ("runtime_attributed", False),
        ("aa_upper", 1.101),
        ("aa_upper", 0.99),
        ("runtime_fingerprint", ""),
        ("texts_per_second", [100, 200]),
        ("texts_per_second", [100, float("nan"), 200]),
    ],
)
def test_unqualified_measurements_fail(field, value):
    sample = row()
    sample[field] = value
    with pytest.raises(ValueError, match=r"complete|variation|identity|rounds|finite"):
        PANEL["measured"](sample)


@pytest.mark.parametrize(
    "key",
    [
        "model",
        "revision",
        "input_digest",
        "machine",
        "workers",
        "normalize",
        "output_dim",
        "table_dtype",
        "tokenizer_rule",
    ],
)
def test_same_model_bars_reject_mixed_computation(key):
    sample = pair()
    sample["reference"]["context"][key] = "different"
    with pytest.raises(ValueError, match="same comparison context"):
        PANEL["comparison"](sample)


def test_speedup_preserves_round_pairing():
    assert PANEL["comparison"](pair()) == 10


def test_compact_table_requires_all_precisions_at_one_and_all_cores():
    rows = [row(dtype=dtype, workers=w) for dtype in PANEL["DTYPES"] for w in (1, 32)]
    assert "| int8 | 200 | 200 |" in PANEL["table"](rows, (1, 32))
    with pytest.raises(ValueError, match="all precisions"):
        PANEL["table"](rows[:2], (1, 32))
    rows[-1]["context"]["workers"] = 10
    with pytest.raises(ValueError, match="all precisions"):
        PANEL["table"](rows, (1, 32))


def test_comparison_requires_same_series_and_checked_output():
    sample = pair()
    sample["reference"]["source"]["series_id"] = "other"
    with pytest.raises(ValueError, match="paired series"):
        PANEL["comparison"](sample)
    sample = pair()
    sample["output_checked"] = False
    with pytest.raises(ValueError, match="Output"):
        PANEL["comparison"](sample)


def test_empty_data_publishes_nothing_and_invalid_pair_writes_nothing(tmp_path):
    palette = json.loads((ROOT / "media/readme/palette.json").read_text())
    data = {"format": "readme-speed-panels-v1", "table": [], "comparisons": []}
    PANEL["publish"](data, tmp_path, palette)
    assert list(tmp_path.iterdir()) == []
    data["table"] = [row(dtype=dtype, workers=w) for dtype in PANEL["DTYPES"] for w in (1, 32)]
    data["selection_rows"] = decision_rows()
    data["table_caption"] = "test caption"
    bad = copy.deepcopy(pair())
    bad["output_checked"] = False
    data["comparisons"] = [bad, pair(32)]
    data["comparison_caption"] = "test caption"
    with pytest.raises(ValueError, match="Output"):
        PANEL["publish"](data, tmp_path, palette)
    assert list(tmp_path.iterdir()) == []


def test_preview_has_no_rates_or_speedup_claims():
    palette = json.loads((ROOT / "media/readme/palette.json").read_text())
    for theme in ("light", "dark"):
        colors = {key: value["value"] for key, value in palette[theme].items()}
        drawing = PANEL["svg"]([], "Unmeasured layout", colors, preview=True)
        assert "Layout preview. No measurements." in drawing
        assert "times as fast" not in drawing
        assert colors["accent"] in drawing


@pytest.mark.parametrize(
    "name",
    [
        "speed-table.md",
        "library-speedups-light.svg",
        "library-speedups-dark.svg",
        "core-selection-evidence.md",
    ],
)
def test_withdrawn_data_cannot_leave_stale_outputs_silently(tmp_path, name):
    target = tmp_path / name
    target.write_text("previously published output")
    palette = json.loads((ROOT / "media/readme/palette.json").read_text())
    data = {"format": "readme-speed-panels-v1", "table": [], "comparisons": []}
    with pytest.raises(ValueError, match="Unselected panel files"):
        PANEL["publish"](data, tmp_path, palette)
    assert target.read_text() == "previously published output"


def test_table_rejects_a_different_executed_fse_runtime():
    rows = [row(dtype=dtype, workers=w) for dtype in PANEL["DTYPES"] for w in (1, 32)]
    rows[-1]["runtime_fingerprint"] = "other-runtime"
    with pytest.raises(ValueError, match="one executed runtime"):
        PANEL["table"](rows, (1, 32))


def test_library_panels_require_both_machine_core_settings():
    with pytest.raises(ValueError, match="every selected core setting"):
        PANEL["paired_core_settings"]([pair()], {"test/model": (1, 32)})
    PANEL["paired_core_settings"]([pair(), pair(32)], {"test/model": (1, 32)})
    with pytest.raises(ValueError, match="measured panel setting"):
        PANEL["comparison"](pair(10))


def decision_rows(rate32=1300):
    rows = []
    for worker, rate in zip(PANEL["PANEL_WORKERS"], (100, 1000, rate32), strict=True):
        sample = row(workers=worker)
        sample["texts_per_second"] = [rate, rate, rate]
        sample["source"]["series_id"] = f"selection_rows-{worker}"
        rows.append(sample)
    return rows


@pytest.mark.parametrize(
    ("rate32", "chosen"), [(1200, (1, 32)), (1199.999, (1, 16)), (900, (1, 16))]
)
def test_32_core_threshold_is_inclusive_and_unrounded(rate32, chosen):
    assert PANEL["select_cores"](decision_rows(rate32)) == chosen


def test_selection_uses_only_one_16_and_32_without_a_curve():
    points = decision_rows(1100)
    assert [r["context"]["workers"] for r in points] == [1, 16, 32]
    points[0]["texts_per_second"] = [2000] * 3
    assert PANEL["select_cores"](points) == (1, 16)


def test_selection_rows_selection_requires_complete_valid_coherent_points():
    with pytest.raises(ValueError, match="cells at 1, 16 and 32"):
        PANEL["select_cores"](decision_rows()[:-1])
    points = decision_rows()
    points[-1]["aa_upper"] = 1.11
    with pytest.raises(ValueError, match="variation"):
        PANEL["select_cores"](points)
    points = decision_rows()
    points[-1]["context"]["input_digest"] = "another-input"
    with pytest.raises(ValueError, match="same comparison context"):
        PANEL["select_cores"](points)


def test_publish_selects_from_selection_rows_and_never_substitutes_32(tmp_path):
    palette = json.loads((ROOT / "media/readme/palette.json").read_text())
    data = {
        "format": "readme-speed-panels-v1",
        "selection_rows": decision_rows(1100),
        "table": [row(dtype=dtype, workers=w) for dtype in PANEL["DTYPES"] for w in (1, 16, 32)],
        "comparisons": [pair(w) for w in (1, 16, 32)],
        "table_caption": "test fixture",
        "comparison_caption": "test fixture",
    }
    PANEL["publish"](data, tmp_path, palette)
    for name in ("speed-table.md", "library-speedups-light.svg", "library-speedups-dark.svg"):
        text = (tmp_path / name).read_text()
        assert "All 32 cores" not in text
        assert "16 cores" in text
        assert "One core" in text
        assert PANEL["LOW_GAIN_NOTE"] in text
    data["table"] = [r for r in data["table"] if r["context"]["workers"] != 16]
    with pytest.raises(ValueError, match="all precisions"):
        PANEL["publish"](data, tmp_path, palette)


def test_selection_rows_is_required_for_every_library_model(tmp_path):
    palette = json.loads((ROOT / "media/readme/palette.json").read_text())
    data = {
        "format": "readme-speed-panels-v1",
        "selection_rows": [],
        "table": [],
        "comparisons": [pair(), pair(32)],
    }
    with pytest.raises(ValueError, match="every displayed model"):
        PANEL["publish"](data, tmp_path, palette)
    assert list(tmp_path.iterdir()) == []


def test_one_16_32_grid_records_direct_decision_sources(tmp_path):
    palette = json.loads((ROOT / "media/readme/palette.json").read_text())
    points = decision_rows(1100)
    data = {
        "format": "readme-speed-panels-v1",
        "selection_rows": points,
        "table": [row(dtype=dtype, workers=w) for dtype in PANEL["DTYPES"] for w in (1, 16, 32)],
        "comparisons": [pair(w) for w in (1, 16, 32)],
        "table_caption": "test fixture",
        "comparison_caption": "test fixture",
    }
    PANEL["publish"](data, tmp_path, palette)
    text = (tmp_path / "speed-table.md").read_text()
    assert "16 cores" in text
    assert "All 32 cores" not in text
    evidence = (tmp_path / "core-selection-evidence.md").read_text()
    assert "Unrounded 32/16 ratio:" in evidence
    assert "8-worker" not in evidence
    assert "https://github.com/oborchers/Fast_Sentence_Embeddings/blob/" in evidence


def test_completely_filtered_precision_table_fails_before_writing(tmp_path):
    palette = json.loads((ROOT / "media/readme/palette.json").read_text())
    data = {
        "format": "readme-speed-panels-v1",
        "selection_rows": decision_rows(1100),
        "table": [row(dtype=dtype, workers=32) for dtype in PANEL["DTYPES"]],
        "comparisons": [],
        "table_caption": "test fixture",
    }
    with pytest.raises(ValueError, match="Missing selected precision"):
        PANEL["publish"](data, tmp_path, palette)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("include_complete_library", [False, True])
def test_completely_filtered_library_cannot_disappear(tmp_path, include_complete_library):
    palette = json.loads((ROOT / "media/readme/palette.json").read_text())
    rust = pair(32)
    rust["reference"]["backend"] = "rust"
    rust["reference"]["runtime_fingerprint"] = "test-only-rust"
    pairs = [pair(), pair(16), rust] if include_complete_library else [rust]
    data = {
        "format": "readme-speed-panels-v1",
        "selection_rows": decision_rows(1100),
        "table": [],
        "comparisons": pairs,
        "comparison_caption": "test fixture",
    }
    with pytest.raises(ValueError, match="every selected core setting"):
        PANEL["publish"](data, tmp_path, palette)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("calls", [[], [2, 2], [2, 1.999, 3], [2, float("nan"), 3]])
def test_short_individual_calls_cannot_hide_behind_a_median(calls):
    sample = row()
    sample["call_seconds"] = calls
    with pytest.raises(ValueError, match="2 seconds|finite"):
        PANEL["measured"](sample)
