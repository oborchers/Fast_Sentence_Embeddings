"""The explicit single-pair 100M showcase, last so failures preserve earlier evidence."""

import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

from pod_artifacts import emit, environment
from pod_folded_summary import runtime
from pod_large_inputs import SHOWCASE_ROWS, prepare_one
from pod_pairs import aa_band, append, operation
from pod_speed import POTION, ensure_model



def run(args):
    folder = args.output / "showcase-100m"
    folder.mkdir()
    source = prepare_one(folder / "pipeline-inputs", SHOWCASE_ROWS)
    (folder / "input-identity.json").write_text(json.dumps(source, indent=2) + "\n")
    emit("showcase-100m-input", source)
    options = SimpleNamespace(models=folder / "models", cache=args.cache)
    ensure_model(POTION, "float32", options)
    command = [
        sys.executable,
        str(Path(__file__).with_name("pod_large_child.py")),
        "--inputs",
        str(folder / "pipeline-inputs"),
        "--models",
        str(options.models),
        "--corpus",
        "repeated-shuffled-100m",
        "--workers",
        "32",
        "--mode",
        "shard",
        "--storage-dtype",
        "int8",
        "--durability",
        "none",
    ]
    context = {"series_id": uuid.uuid4().hex, "cell": "showcase/100m/int8/none/32", "round": 0}
    hardware = environment()
    rows = []
    for label in ("fse-a", "fse-b"):
        row = operation(command, label, 32, args.deadline, folder, context, hardware)
        append(folder / "speed-results.jsonl", row)
        rows.append(row)
        emit("showcase-100m-" + label, row)
    band = aa_band(rows)
    same_runtime = runtime(rows[0]) == runtime(rows[1]) and all(runtime(rows[0]).values())
    stable = (
        rows[0]["output_digest"] == rows[1]["output_digest"]
        and same_runtime
        and rows[0].get("runtime_package") == "fse"
    )
    complete = band["aa_max_ratio"] <= 1.1 and stable
    result = {
        **context,
        **band,
        "event": "series_complete",
        "complete": complete,
        "required_rounds": 1,
        "valid_rounds": int(complete),
        "scope": "maintainer-requested showcase: one timed run plus one A/A repeat, not a three-pair matrix cell",
        "primary_fse": rows[0],
        "aa_repeat": rows[1],
        "source": source,
        "runtime_and_output_match": stable,
    }
    append(folder / "speed-results.jsonl", result)
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    emit("showcase-100m-summary", result)
