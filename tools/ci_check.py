"""Require selected CI jobs to succeed; unknown and abandoned outcomes fail closed."""

from __future__ import annotations

import json
import os
import subprocess

JOBS = {"changes", "lint", "test", "test-st", "coverage", "sdist", "wheels"}


def required(needs, event):
    if event not in ("push", "pull_request", "workflow_dispatch", "schedule"):
        raise ValueError("unknown CI event")
    if set(needs) != JOBS:
        raise ValueError("CI summary needs the complete expected job inventory")
    outputs = needs["changes"]["outputs"]
    if any(
        outputs.get(key) not in ("true", "false")
        for key in ("code", "lint", "st", "sdist", "native", "targeted")
    ):
        raise ValueError("CI selection outputs must contain explicit boolean strings")
    targeted = outputs["targeted"] == "true"
    if targeted and event != "workflow_dispatch":
        raise ValueError("targeted macOS selection requires workflow_dispatch")
    code = outputs["code"] == "true"
    return {
        "changes": True,
        "lint": outputs["lint"] == "true" and not targeted,
        "test": code,
        "test-st": code and outputs["st"] == "true",
        "coverage": code and event != "pull_request" and not targeted,
        "sdist": code and outputs["sdist"] == "true",
        "wheels": code and outputs["native"] == "true",
    }


def failures(needs, event, observed):
    selected = required(needs, event)
    problems = []
    for name, job in needs.items():
        result = job["result"]
        if result != "success" and not (result == "skipped" and not selected[name]):
            problems.append(f"{name}: {result}")
    # A matrix aggregate must not hide a failed/cancelled/abandoned individual job.
    # Exclude the running summary and its downstream jobs in ci.yml: release
    # needs check, and github-release needs release. Earlier attempts can retain
    # their skipped records during failed-only reruns; they supply no CI evidence.
    for job in observed:
        name = job["name"]
        if name in ("check", "release", "github-release"):
            continue
        group = name.split(" (", 1)[0]
        conclusion = job["conclusion"]
        allowed_skip = group in selected and not selected[group] and conclusion == "skipped"
        if job["status"] != "completed" or (conclusion != "success" and not allowed_skip):
            problems.append(f"observed {name}: {job['status']}/{conclusion}")
    return problems


def inventory(repository, run, attempts):
    # Failed-only reruns retain successful jobs from earlier attempts. Read each
    # attempt explicitly and retain its newest record for every exact matrix name.
    latest = {}
    for attempt in range(1, attempts + 1):
        pages = json.loads(
            subprocess.check_output(
                [
                    "gh",
                    "api",
                    f"repos/{repository}/actions/runs/{run}/attempts/{attempt}/jobs?per_page=100",
                    "--paginate",
                    "--slurp",
                ]
            )
        )
        for page in pages:
            for job in page["jobs"]:
                latest[job["name"]] = job
    if not latest:
        raise ValueError("CI job inventory is empty")
    return list(latest.values())


def main():
    needs = json.loads(os.environ["CI_NEEDS"])
    if needs.get("changes", {}).get("result") != "success":
        raise SystemExit("CI source resolution or selection failed")
    repository, run = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"]
    observed = inventory(repository, run, int(os.environ["GITHUB_RUN_ATTEMPT"]))
    groups = {job["name"].split(" (", 1)[0] for job in observed}
    expected = {name for name, job in needs.items() if job["result"] != "skipped"}
    if expected - groups:
        raise ValueError("CI job inventory is missing selected jobs")
    problems = failures(needs, os.environ["GITHUB_EVENT_NAME"], observed)
    if problems:
        raise SystemExit("CI did not pass:\n" + "\n".join(problems))
    print("All selected CI jobs succeeded. Intentional omissions supply no test evidence.")


if __name__ == "__main__":
    main()
