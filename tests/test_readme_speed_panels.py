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
    rows = [row(dtype=dtype, workers=w) for dtype in PANEL["DTYPES"] for w in PANEL["WORKERS"]]
    assert "| int8 | 200 | 200 |" in PANEL["table"](rows)
    with pytest.raises(ValueError, match="all precisions"):
        PANEL["table"](rows[:2])
    rows[-1]["context"]["workers"] = 10
    with pytest.raises(ValueError, match="all precisions"):
        PANEL["table"](rows)


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
    data["table"] = [
        row(dtype=dtype, workers=w) for dtype in PANEL["DTYPES"] for w in PANEL["WORKERS"]
    ]
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
    "name", ["speed-table.md", "library-speedups-light.svg", "library-speedups-dark.svg"]
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
    rows = [row(dtype=dtype, workers=w) for dtype in PANEL["DTYPES"] for w in PANEL["WORKERS"]]
    rows[-1]["runtime_fingerprint"] = "other-runtime"
    with pytest.raises(ValueError, match="one executed runtime"):
        PANEL["table"](rows)


def test_library_panels_require_both_machine_core_settings():
    with pytest.raises(ValueError, match="one core and all 32 cores"):
        PANEL["paired_core_settings"]([pair()])
    PANEL["paired_core_settings"]([pair(), pair(32)])
    with pytest.raises(ValueError, match="one core or all 32 cores"):
        PANEL["comparison"](pair(10))
