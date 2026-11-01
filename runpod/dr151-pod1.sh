#!/usr/bin/env bash
# Image python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 60 GB disk. Never launches a pod.
set -uo pipefail
started=$(/usr/bin/date +%s)
export DR151_DEADLINE=$((started + 10800 - 90))
/usr/bin/mkdir -p /dr151-results /dr151-cache
cat > /dr151-finalize.py <<'DR151_FINALIZER'
"""Stdlib-only artifact export, including partial JSONL files after a hard timeout."""

import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path


def emit(name, value):
    compressed = gzip.compress(json.dumps(value, ensure_ascii=True).encode(), mtime=0)
    encoded = base64.b64encode(compressed).decode()
    chunks = [encoded[start : start + 160] for start in range(0, len(encoded), 160)]
    print(
        f"DR190_BEGIN {name} {len(compressed)} {hashlib.sha256(compressed).hexdigest()}", flush=True
    )
    for index, chunk in enumerate(chunks):
        print(f"DR190_DATA {name} {index} {chunk}", flush=True)
    print(f"DR190_END {name} {len(chunks)}", flush=True)


def jsonl_value(path, content):
    rows, errors = [], []
    for number, line in enumerate(content.splitlines()):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            errors.append({"line": number + 1, "content": line})
    return {"path": str(path), "format": "jsonl", "rows": rows, "parse_errors": errors}


def failures(value):
    result = []
    if value.get("parse_errors"):
        result.append({"path": value["path"], "reason": "partial JSONL record"})
    for row in value.get("rows", []):
        if not isinstance(row, dict):
            continue
        if row.get("event") == "series_complete" and not row["complete"]:
            result.append({"cell": row.get("cell"), "reason": "incomplete or noisy cell"})
        if row.get("status") == "failed":
            result.append({"model": row.get("model"), "reason": "failed invocation"})
    return result


def final(root, returncode):
    steps, incomplete = [], []
    for index, path in enumerate(sorted(root.rglob("*"))):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".log", ".txt"):
            continue
        if "pipeline-inputs" in path.parts:
            continue
        name = f"final-file-{index}"
        content = path.read_text(errors="replace")
        value = (
            jsonl_value(path, content)
            if path.suffix == ".jsonl"
            else {"path": str(path), "format": "text", "content": content}
        )
        emit(name, value)
        incomplete.extend(failures(value))
        steps.append({"name": name, "path": str(path)})
    emit(
        "manifest",
        {
            "complete": returncode == 0 and not incomplete,
            "returncode": returncode,
            "status": "complete" if returncode == 0 and not incomplete else "partial or failed",
            "incomplete": incomplete,
            "steps": steps,
        },
    )
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    final(Path(sys.argv[1]), int(sys.argv[2]))
DR151_FINALIZER
cat > /dr151-work.sh <<'DR151_WORK'
set -euxo pipefail
export PATH=/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONUNBUFFERED=1 UV_NO_PROGRESS=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false SETUPTOOLS_SCM_PRETEND_VERSION=2.0.0.dev0
export HF_HOME=/dr151-cache/huggingface FSE_QUALITY_CACHE=/dr151-cache
export HF_HUB_CACHE=/dr151-cache/huggingface/hub
export DEBIAN_FRONTEND=noninteractive
/usr/bin/apt-get update
/usr/bin/apt-get install -y --no-install-recommends build-essential git curl procps util-linux ca-certificates
/usr/bin/git clone --branch dr151-compare --no-checkout https://github.com/oborchers/Fast_Sentence_Embeddings.git /dr151-repo
/usr/bin/git -C /dr151-repo checkout --detach 97b741887dc0bd4730131aa60a7b5242fef8d5ef
test "$(/usr/bin/git -C /dr151-repo rev-parse HEAD)" = 97b741887dc0bd4730131aa60a7b5242fef8d5ef
cd /dr151-repo
export SOURCE_DATE_EPOCH=$(/usr/bin/git show -s --format=%ct HEAD)
/usr/local/bin/python -m pip install --no-cache-dir uv==0.13.0
/usr/local/bin/uv venv --python /usr/local/bin/python /dr151-env
/usr/local/bin/uv pip install --python /dr151-env/bin/python setuptools==84.2.1 setuptools-scm==8.2.0 Cython==3.1.0 numpy==2.5.4
/usr/local/bin/uv pip install --python /dr151-env/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
/usr/local/bin/uv pip install --python /dr151-env/bin/python -r benchmarks/quality/classic-requirements.lock.txt
/usr/local/bin/uv pip install --python /dr151-env/bin/python --no-build-isolation -e .
/usr/local/bin/uv pip check --python /dr151-env/bin/python
/usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_mind.py prepare --models /dr151-results/models --cache /dr151-cache --output /dr151-results --workers 16
/usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_mind.py evaluate --models /dr151-results/models --cache /dr151-cache --output /dr151-results --workers 16
DR151_WORK
remaining=$((DR151_DEADLINE - $(/usr/bin/date +%s)))
/usr/bin/timeout --signal=TERM --kill-after=15 "$remaining" /bin/bash /dr151-work.sh > /dr151-results/bootstrap.log 2>&1
status=$?
/usr/local/bin/python - "$status" <<'DR151_INVOCATION'
import json, sys
from pathlib import Path
Path('/dr151-results/bootstrap-invocation.json').write_text(json.dumps({"command":["/bin/bash","/dr151-work.sh"],"script":Path('/dr151-work.sh').read_text(),"returncode":int(sys.argv[1]),"pinned_commit":"97b741887dc0bd4730131aa60a7b5242fef8d5ef"}))
DR151_INVOCATION
finalize_seconds=$((started + 10800 - 5 - $(/usr/bin/date +%s)))
if [ "$finalize_seconds" -lt 1 ]; then finalize_seconds=1; fi
/usr/bin/timeout --signal=TERM --kill-after=2 "$finalize_seconds" /usr/local/bin/python /dr151-finalize.py /dr151-results "$status"
printf 'ALL_DONE\n'
/usr/bin/sleep infinity
