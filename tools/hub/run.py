"""Build, check, publish or read back the 30 classic Hub repositories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from payload import PUBLISHED_SHA256, build, check, manifest, render_card
from publish import readback, upload, write_receipt


def rebuild(entries, models, output, discard):
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "rebuild-receipt.json"
    state = {"hub_writes_performed": False, "status": "running", "repositories": []}
    write_receipt(receipt, state)
    for entry in entries:
        required = entry["total_bytes"] + 8 * 2**30
        if shutil.disk_usage(output).free < required:
            raise RuntimeError("insufficient disk space for the next payload and 8 GiB reserve")
        folder, files = build(models, output, entry)
        row = {
            "repo": entry["repo"],
            "files": files,
            "all_bytes_exact": True,
            "output_removed": False,
        }
        state["repositories"].append(row)
        write_receipt(receipt, state)
        if discard:
            # Recheck this exact output, then remove only its verified regular files.
            check(folder, entry)
            if folder.parent != output or folder.is_symlink():
                raise ValueError("cleanup target differs from the newly built payload")
            print(json.dumps({"cleanup": str(folder), "repo": entry["repo"]}), flush=True)
            for name in entry["files"]:
                (folder / name).unlink()
            folder.rmdir()
            if folder.exists():
                raise RuntimeError("verified output cleanup failed")
            row["output_removed"] = True
            write_receipt(receipt, state)
        print(
            json.dumps({"verified": entry["repo"], "completed": len(state["repositories"])}),
            flush=True,
        )
    state["status"] = "complete"
    write_receipt(receipt, state)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cards", "build", "check", "upload", "readback"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", default=PUBLISHED_SHA256)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", help="Limit work to one manifest repository")
    parser.add_argument("--discard-verified-output", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Explicitly enable Hub writes")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.models.resolve()):
        parser.error("--output must be outside the read-only model inputs")
    return args


def main():
    args = arguments()
    entries = manifest(args.manifest, args.manifest_sha256)["repositories"]
    if args.repo:
        entries = [entry for entry in entries if entry["repo"] == args.repo]
        if not entries:
            raise ValueError("--repo is not in the manifest")
    if args.action == "build":
        rebuild(entries, args.models, args.output, args.discard_verified_output)
        return
    if args.action == "cards":
        args.output.mkdir(parents=True, exist_ok=False)
        for entry in entries:
            data = render_card(args.models, entry)
            (args.output / (entry["repo"].split("/")[1] + ".md")).write_bytes(data)
        print(json.dumps({"cards_byte_equal": len(entries), "hub_writes_performed": False}))
        return
    if args.action == "upload":
        args.output.mkdir(parents=True, exist_ok=True)
        upload(
            entries, args.models, args.output, args.output / "publication-receipt.json", args.upload
        )
        return
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        if args.action == "check":
            files = check(args.models / entry["repo"].split("/")[1], entry)
            rows.append({"repo": entry["repo"], "files": files, "all_bytes_exact": True})
        else:
            rows.append(readback(entry, args.output))
        write_receipt(
            args.output / "verification.json", {"repositories": rows, "hub_writes_performed": False}
        )


if __name__ == "__main__":
    main()
