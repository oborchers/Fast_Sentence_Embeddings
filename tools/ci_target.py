"""Bind an explicit manual macOS run to an open PR's immutable current head."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


def target(pr, repository, number, sha):
    if not re.fullmatch(r"[1-9][0-9]*", number) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("pr-macos requires a positive PR number and full lowercase head SHA")
    if (
        pr["number"] != int(number)
        or pr["state"] != "open"
        or pr["base"]["repo"]["full_name"] != repository
        or pr["base"]["ref"] != "master"
        or pr["head"]["sha"] != sha
    ):
        raise ValueError(
            "requested PR must be open against master with exactly the requested head SHA"
        )
    source = pr["head"]["repo"]["full_name"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source):
        raise ValueError("PR head repository is missing or malformed")
    return {"source_repository": source, "source_ref": sha, "target_pr": number}


def main():
    repository = os.environ["GITHUB_REPOSITORY"]
    scope = os.environ.get("CI_SCOPE", "")
    number, sha = os.environ.get("CI_PR_NUMBER", ""), os.environ.get("CI_HEAD_SHA", "")
    targeted = scope == "pr-macos"
    if scope not in ("", "full", "pr-macos"):
        raise ValueError("unknown manual CI scope")
    if targeted:
        if os.environ["GITHUB_EVENT_NAME"] != "workflow_dispatch":
            raise ValueError("pr-macos requires an explicit workflow_dispatch")
        # Validate before constructing the API path. Never interpolate input into shell code.
        if not re.fullmatch(r"[1-9][0-9]*", number) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("pr-macos requires a PR number and full lowercase head SHA")
        pr = json.loads(
            subprocess.check_output(["gh", "api", f"repos/{repository}/pulls/{number}"])
        )
        values = target(pr, repository, number, sha)
        print(
            f"Verified PR {number} ({pr['title']}), source {values['source_repository']} at {sha}"
        )
    else:
        if number or sha:
            raise ValueError("PR number and head SHA require scope=pr-macos")
        values = {
            "source_repository": repository,
            "source_ref": os.environ["GITHUB_SHA"],
            "target_pr": "",
        }
    values["targeted"] = str(targeted).lower()
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write("Verified CI source:\n\n")
        stream.write(f"- Controller commit: `{os.environ['GITHUB_SHA']}`\n")
        stream.write(f"- Source repository: `{values['source_repository']}`\n")
        stream.write(f"- Tested commit: `{values['source_ref']}`\n")
        if targeted:
            stream.write(f"- PR {number}: {pr['title']}\n")
            stream.write(
                "- Explicit optional macOS evidence. This does not validate a later PR head.\n"
            )


if __name__ == "__main__":
    main()
