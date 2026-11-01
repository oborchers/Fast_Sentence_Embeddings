"""Independent, stdlib-only checks of planned pod cells and measured call receipts."""

import json

from pod_duration import qualifies


def check_cells(expected, summary):
    cells = summary.get("cells", [])
    identifiers = [row["cell"] for row in cells]
    failures = []
    for identifier in expected:
        matching = [row for row in cells if row["cell"] == identifier]
        if len(matching) != 1 or matching[0].get("eligible") is not True:
            failures.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        failures.append("duplicate cell receipt")
    return failures


def validate(root):
    failures = []
    studies = (
        ("readme-panels", "plan.json", "cases", "summary.json"),
        (
            "folded-classification",
            "classification-plan.json",
            "expected_cells",
            "classification-summary.json",
        ),
    )
    for folder, plan_name, key, summary_name in studies:
        path = root / folder
        try:
            expected = json.loads((path / plan_name).read_text())[key]
            summary = json.loads((path / summary_name).read_text())
            failures.extend(check_cells(expected, summary))
            if folder == "readme-panels" and summary.get("complete") is not True:
                failures.append("incomplete README summary")
            operations = [
                json.loads(line) for line in (path / "speed-results.jsonl").read_text().splitlines()
            ]
            controls = [
                row
                for row in operations
                if row.get("event") == "operation"
                and (row.get("runtime_package") == "fse" or row.get("label") in ("fse-a", "fse-b"))
            ]
            if not controls or any(
                row.get("minimum_call_seconds") != 2 or not qualifies(row) for row in controls
            ):
                failures.append(folder + ": missing or short actual fse calls")
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(folder + ": invalid or missing receipt: " + str(error))
    try:
        if json.loads((root / "readme-panels/probe.json").read_text()).get("passed") is not True:
            failures.append("host probe did not pass")
    except (OSError, ValueError):
        failures.append("host probe receipt missing or invalid")
    return [{"reason": reason} for reason in failures]
