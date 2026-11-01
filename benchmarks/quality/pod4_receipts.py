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


def long_calls(path):
    operations = [json.loads(line) for line in path.read_text().splitlines()]
    controls = [
        row
        for row in operations
        if row.get("event") == "operation"
        and (row.get("runtime_package") == "fse" or row.get("label") in ("fse-a", "fse-b"))
    ]
    return bool(controls) and all(
        row.get("minimum_call_seconds") == 2 and qualifies(row) for row in controls
    )


def study(root, path, folder, plan_name, key, summary_name):
    from pod4_raw_receipts import classification, readme

    expected = json.loads((path / plan_name).read_text())[key]
    summary = json.loads((path / summary_name).read_text())
    failures = check_cells(expected, summary)
    reconstructed = readme(path, expected) if folder == "readme-panels" else classification(path)
    failures.extend(check_cells(expected, reconstructed))
    receipt = (
        root / "readme-complete.json"
        if folder == "readme-panels"
        else path / "classifications-complete.json"
    )
    if json.loads(receipt.read_text()).get("status") != "completed":
        failures.append(folder + ": stage completion receipt failed")
    if folder == "readme-panels" and summary.get("complete") is not True:
        failures.append("incomplete README summary")
    if not long_calls(path / "speed-results.jsonl"):
        failures.append(folder + ": missing or short actual fse calls")
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
            failures.extend(study(root, path, folder, plan_name, key, summary_name))
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(folder + ": invalid or missing receipt: " + str(error))
    try:
        if json.loads((root / "readme-panels/probe.json").read_text()).get("passed") is not True:
            failures.append("host probe did not pass")
    except (OSError, ValueError):
        failures.append("host probe receipt missing or invalid")
    return [{"reason": reason} for reason in failures]
