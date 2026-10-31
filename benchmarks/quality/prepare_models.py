"""Convert the pinned static Hub selection into finished quality artifacts.

Run with nice -n 10. Each model conversion yields the shared heavy-job lock.
Existing artifacts are refused rather than silently reused or overwritten.
The manifest records source revisions separately from the fse build commit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_pretrained import COMMITS  # isort: skip
from quality_coordination import task_reservation  # isort: skip


def prepare(folder: Path, coordination: Path, names: list[str]) -> None:
    from fse import Encoder

    folder.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name in names:
        revision = COMMITS[name]
        stem = name.split("/")[-1]
        paths = [folder / f"{stem}-{dtype}.safetensors" for dtype in ("float32", "float16", "int8")]
        if any(path.exists() for path in paths):
            raise FileExistsError(f"artifacts already exist for {name} in {folder}")
        with task_reservation(coordination):
            encoder = Encoder.from_pretrained(name, revision=revision, local_files_only=True)
            encoder.save(paths[0])
            for dtype, path in zip(("float16", "int8"), paths[1:], strict=True):
                encoder.quantize(dtype).save(path)
            record = {
                "source": name,
                "revision": revision,
                "commit": commit,
                "fse": version("fse"),
                "python": sys.version,
                "artifacts": [path.name for path in paths],
                "output_dim": encoder.output_dim,
            }
            with (folder / "sources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        del encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(COMMITS), default=sorted(COMMITS))
    args = parser.parse_args()
    prepare(args.folder, args.coordination, args.models)


if __name__ == "__main__":
    main()
