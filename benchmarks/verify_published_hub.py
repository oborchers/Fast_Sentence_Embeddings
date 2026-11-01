"""Verify every approved uploaded repo; never upload, repair, overwrite or delete on Hub.

Run only after fse-main confirms Oliver's approval and all public uploads finish.
Each repo runs the literal approved card in a new child with an empty HF_HOME.
Its owned cache is removed after the child exits; evidence stays outside scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hub_public_readback import FLAGS, check_uploaded_repo
from hub_publication import file_sha256
from hub_readback_evidence import write_atomic


def cache_environment(home):
    env = dict(
        os.environ,
        HF_HOME=str(home),
        HF_HUB_CACHE=str(home / "hub"),
        HUGGINGFACE_HUB_CACHE=str(home / "hub"),
        HF_XET_CACHE=str(home / "xet"),
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_XET_CHUNK_CACHE_SIZE_BYTES="0",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        HF_HUB_VERBOSITY="error",
        HF_DEBUG="0",
        HF_ENDPOINT="https://huggingface.co",
        HF_HUB_OFFLINE="0",
    )
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        env.pop(name, None)
    return env


def uploaded_entries(manifest, receipts, manifest_path):
    if receipts.get("publication_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("upload receipts do not reference this exact approved manifest")
    approved = receipts.get("approved_repositories")
    uploaded = receipts.get("repositories")
    if not isinstance(approved, list) or not isinstance(uploaded, list) or not approved:
        raise ValueError("receipts require the exact approved and uploaded repository lists")
    known = {entry["repo"]: entry for entry in manifest["repositories"]}
    names = [row["repo"] for row in uploaded]
    if (
        len(set(names)) != len(names)
        or len(set(approved)) != len(approved)
        or set(names) != set(approved)
    ):
        raise ValueError("not every approved repo has exactly one completed upload receipt")
    if set(names) - set(known):
        raise ValueError("uploaded repository is not in the approved local manifest")
    for row in uploaded:
        if not re.fullmatch(r"[a-f0-9]{40}", row.get("revision", "")):
            raise ValueError("each upload receipt needs its exact Hub commit revision")
    return [(known[row["repo"]], row["revision"]) for row in uploaded]


def remove_owned_cache(case, owner):
    if (
        case.is_symlink()
        or (case / "hf-home").is_symlink()
        or json.loads((case / "owner.json").read_text()) != owner
    ):
        raise ValueError("scratch ownership changed; refusing cache deletion")
    if {path.name for path in case.iterdir()} != {"owner.json", "hf-home"}:
        raise ValueError("unexpected files in owned cache scratch; refusing deletion")
    allocated = sum(
        path.stat().st_blocks * 512
        for path in case.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    print(
        json.dumps(
            {
                "verified_cache_to_remove": str(case),
                "repo": owner["repo"],
                "allocated_bytes": allocated,
            }
        ),
        flush=True,
    )
    shutil.rmtree(case)
    if case.exists():
        raise OSError("owned readback cache remains after cleanup")
    return allocated


def child_run(args):
    manifest = json.loads(args.manifest.read_text())
    entry = next(value for value in manifest["repositories"] if value["repo"] == args.repo)
    row = check_uploaded_repo(entry, args.revision, args.hf_home)
    if args.result.exists():
        raise FileExistsError(args.result)
    write_atomic(args.result, row)
    return 0 if row["status"] == "passed" else 1


def run_one(entry, revision, args, output):
    if shutil.disk_usage(args.scratch).free < 25 * 2**30:
        raise RuntimeError("less than 25 GiB free before cold Hub verification")
    case = Path(tempfile.mkdtemp(prefix=entry["repo"].replace("/", "--") + "-", dir=args.scratch))
    home = case / "hf-home"
    home.mkdir()
    owner = {"owner": "dr157-hub-public-readback", "repo": entry["repo"], "path": str(case)}
    (case / "owner.json").write_text(json.dumps(owner))
    stem = entry["repo"].replace("/", "--")
    result = output / f"{stem}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--manifest",
        str(args.manifest),
        "--repo",
        entry["repo"],
        "--revision",
        revision,
        "--hf-home",
        str(home),
        "--result",
        str(result),
    ]
    try:
        with (output / f"{stem}.log").open("w") as log:
            completed = subprocess.run(
                command,
                env=cache_environment(home),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        if result.exists():
            row = json.loads(result.read_text())
            row["exit_code"] = completed.returncode
            if completed.returncode != 0:
                row["status"] = "failed"
        else:
            row = {
                "repo": entry["repo"],
                "status": "failed",
                "exit_code": completed.returncode,
                **dict.fromkeys(FLAGS, False),
            }
    except subprocess.TimeoutExpired:
        row = {
            "repo": entry["repo"],
            "status": "failed",
            "error": {"type": "TimeoutExpired"},
            **dict.fromkeys(FLAGS, False),
        }
    except Exception as error:
        row = {
            "repo": entry["repo"],
            "status": "failed",
            "error": {"type": type(error).__name__},
            **dict.fromkeys(FLAGS, False),
        }
    write_atomic(result, row)  # Preserve the complete probe before deleting downloaded bytes.
    try:
        removed = remove_owned_cache(case, owner)
        cache_removed = True
    except Exception as cleanup_error:
        removed, cache_removed = 0, False
        cleanup_failure = type(cleanup_error).__name__
    row["cache_removed"] = cache_removed
    row["cache_allocated_bytes_removed"] = removed
    if not cache_removed:
        row["status"] = "failed"
        row["cleanup_error"] = cleanup_failure
    write_atomic(result, row)
    return row


def table(rows):
    lines = [
        "| Repo | Files/SHA256 | Public API load | Metadata | Probe | Pearson difference |",
        "|---|---|---|---|---|---:|",
    ]
    for row in rows:
        flags = [
            "ok" if row.get(key) is True else "failed" if row.get(key) is False else "not run"
            for key in FLAGS
        ]
        delta = row.get("difference_points")
        value = "not run" if delta is None else f"{delta:+.8f}"
        lines.append(
            f"| {row['repo']} | {flags[0]} | {flags[1]} | {flags[2]} | {flags[3]} | {value} |"
        )
    return lines


def checked_run(entry, revision, args):
    try:
        return run_one(entry, revision, args, args.output)
    except Exception as error:
        return {
            "repo": entry["repo"],
            "status": "failed",
            "uploaded_revision": revision,
            "error": {"type": type(error).__name__},
            **dict.fromkeys(FLAGS, False),
        }


def parent_run(args):
    if args.output.exists():
        raise FileExistsError(
            "readback evidence is never overwritten; choose a fresh result directory"
        )
    if args.scratch.resolve().is_relative_to((Path.home() / ".cache").resolve()):
        raise ValueError("readback scratch must not use the shared cache tree")
    manifest = json.loads(args.manifest.read_text())
    receipts = json.loads(args.upload_receipts.read_text())
    entries = uploaded_entries(manifest, receipts, args.manifest)
    args.output.mkdir(parents=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry, revision in entries:
        row = checked_run(entry, revision, args)
        rows.append(row)
        evidence = {
            "readback_version": 1,
            "publication_manifest_sha256": file_sha256(args.manifest),
            "hub_writes_performed": False,
            "repositories": rows,
            "complete": len(rows) == len(entries),
            "passed": all(value["status"] == "passed" for value in rows),
        }
        write_atomic(args.output / "evidence.json", evidence)
        (args.output / "report.md").write_text(
            "# Fresh public Hub readback\n\n" + "\n".join(table(rows)) + "\n"
        )
        print(json.dumps({"repo": row["repo"], "status": row["status"]}), flush=True)
    return 0 if all(row["status"] == "passed" for row in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upload-receipts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--revision")
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.child:
        return child_run(args)
    if any(value is None for value in (args.upload_receipts, args.output, args.scratch)):
        raise ValueError(
            "public readback requires completed upload receipts, output and owned scratch"
        )
    return parent_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
