"""Extract README inputs from immutable pod journals, replaying strict cell checks.

This prepares an audited-data candidate, never edits README or renders a claim.
Publication still needs an independent source review and DR-154's renderer gates.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from pod_readme import POTION, STATIC_ST
from pod_readme_summary import summarize

CURVE = (1, 2, 4, 8, 16, 32)
TOKENIZER_FIELDS = (
    "fse.tokenizer",
    "fse.tokenizer_config",
    "fse.max_chars",
    "fse.max_tokens",
    "fse.unknown",
)


def committed_text(commit, path):
    relative = PurePosixPath(path)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("publication source must be a full immutable commit")
    if not path.startswith("benchmarks/baseline/") or ".." in relative.parts:
        raise ValueError("publication source must stay under benchmarks/baseline")
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True)


def context(rows, checked):
    first = rows[0]
    provenance = first.get("artifact_provenance") or {}
    parent = provenance.get("float32_parent_fingerprint")
    if (
        not parent
        or provenance.get("provenance_verification") != "embedded source and preparation record"
    ):
        raise ValueError("README cells require verified Hub float32-parent lineage")
    if any(row.get("artifact_provenance") != provenance for row in rows):
        raise ValueError("fse controls changed artifact lineage")
    header = checked["encoder_header"]
    return {
        "model": first["model"],
        "revision": first["revision"],
        "parent_artifact": parent,
        "input_digest": first["input_digest"],
        "texts": first["texts"],
        "corpus": first["corpus"],
        "machine": first["machine"],
        "date": first["time_utc"][:10],
        "workers": first["requested_threads"],
        "table_dtype": first["table_dtype"],
        "output_dtype": first["output_dtype"],
        "output_dim": first["shape"][1],
        "normalize": checked["normalize"],
        "tokenizer_rule": json.dumps(
            {
                **{key: header[key] for key in TOKENIZER_FIELDS},
                "preprocessing": first.get("preprocessing"),
            },
            sort_keys=True,
        ),
    }


def panel_row(summary, label, source, details):
    arm = summary["arms"][label]
    return {
        "complete": True,
        "runtime_attributed": arm["runtime_stable"],
        "aa_upper": summary["verified_aa_ratio"],
        "backend": arm["backend"],
        "runtime_fingerprint": arm["runtime"]["runtime_fingerprint"],
        "source": source,
        "context": details,
        "texts_per_second": arm["ordered_texts_per_second"],
    }


def extract_series(done, raw, checks, commit, journal):
    operations = [r for r in raw if r.get("event") == "operation"]
    rounds = [r for r in raw if r.get("event") == "round_complete"]
    labels = list(dict.fromkeys(r["label"] for r in operations))
    checked = checks[done["cell"]]
    summary = summarize(done, operations, rounds, labels, checked)
    if not summary["eligible"]:
        raise ValueError("strict source summary rejects this series")
    controls = [r for r in operations if r["kind"] == "fse"]
    details = context(controls, checked)
    source = {"commit": commit, "path": journal, "series_id": done["series_id"]}
    primary = panel_row(summary, "fse-a", source, details)
    references = [label for label in labels if label not in ("fse-a", "fse-b")]
    if len(references) > 1:
        raise ValueError("one same-model reference is required per library series")
    return summary, primary, references


def document(raw, checked_rows, commit, journal):
    checks = {row["cell"]: row for row in checked_rows}
    if len(checks) != len(checked_rows):
        raise ValueError("ambiguous repeated output checks for one cell")
    groups = {}
    for row in raw:
        groups.setdefault(row.get("series_id"), []).append(row)
    result = {
        "format": "readme-speed-panels-v1",
        "scaling": [],
        "table": [],
        "comparisons": [],
        "excluded": [],
        "independent_source_review_required": True,
        "table_caption": "",
        "comparison_caption": "",
    }
    for done in (r for r in raw if r.get("event") == "series_complete"):
        try:
            summary, primary, references = extract_series(
                done, groups[done["series_id"]], checks, commit, journal
            )
        except (KeyError, ValueError, ZeroDivisionError) as error:
            result["excluded"].append(
                {"cell": done["cell"], "series_id": done["series_id"], "reason": str(error)}
            )
            continue
        add_cell(result, summary, primary, references)
    return result


def add_cell(result, summary, primary, references):
    details = primary["context"]
    if references:
        reference = panel_row(summary, references[0], primary["source"], details)
        result["comparisons"].append(
            {"fse": primary, "reference": reference, "output_checked": True}
        )
        return
    if details["model"] == POTION and details["workers"] in (1, 16, 32):
        result["table"].append(primary)
    if (
        details["model"] in (POTION, STATIC_ST)
        and details["table_dtype"] == "float32"
        and details["workers"] in CURVE
    ):
        result["scaling"].append(primary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--checks", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table-caption", required=True)
    parser.add_argument("--comparison-caption", required=True)
    args = parser.parse_args()
    raw = [json.loads(line) for line in committed_text(args.commit, args.journal).splitlines()]
    checked = [json.loads(line) for line in committed_text(args.commit, args.checks).splitlines()]
    result = document(raw, checked, args.commit, args.journal)
    if not args.table_caption.strip() or not args.comparison_caption.strip():
        parser.error("verified nonempty public captions are required")
    result.update(table_caption=args.table_caption, comparison_caption=args.comparison_caption)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
