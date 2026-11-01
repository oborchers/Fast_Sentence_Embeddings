#!/usr/bin/env bash
# Revised Pod 1. python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 200 GB container disk.
set -uo pipefail
started=$(/usr/bin/date +%s)
export DR151_DEADLINE=$((started + 10800 - 90))
/usr/bin/mkdir -p /dr151-results
if ! /usr/bin/mkdir /dr151-cache; then
  printf 'ERROR: /dr151-cache already exists; require a fresh exclusive pod cache.\nALL_DONE\n' >&2
  /usr/bin/sleep infinity
  exit 1
fi
if ! /usr/local/bin/python - <<'DR151_CACHE_OWNER'
import json, uuid
from pathlib import Path
record = {"cache":"/dr151-cache","created_exclusively":True,"owner_token":uuid.uuid4().hex}
for path in ('/dr151-cache/.dr151-owner.json', '/dr151-results/cache-ownership.json'):
    with Path(path).open('x') as stream: json.dump(record, stream)
DR151_CACHE_OWNER
then
  printf 'ERROR: cannot establish exclusive cache ownership.\nALL_DONE\n' >&2
  /usr/bin/sleep infinity
  exit 1
fi
/usr/bin/cat > /dr151-finalize.py <<'DR151_FINALIZER'
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
/usr/bin/cat > /dr151-work.sh <<'DR151_WORK'
set -euo pipefail
run_stage() {
  stage_name=$1
  shift
  printf 'stage_start %s %s\n' "$stage_name" "$(/usr/bin/date -u +%FT%TZ)"
  /usr/bin/df -h /dr151-cache
  if "$@"; then stage_status=0; else stage_status=$?; fi
  printf 'stage_end %s status=%s %s\n' "$stage_name" "$stage_status" "$(/usr/bin/date -u +%FT%TZ)"
  /usr/bin/df -h /dr151-cache
  return "$stage_status"
}
/usr/bin/cat > /dr151-ssh-setup.sh <<'DR151_SSH'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
/usr/bin/apt-get update
/usr/bin/apt-get install -y --no-install-recommends openssh-server
: "${PUBLIC_KEY:?RunPod must inject PUBLIC_KEY for SSH access}"
/usr/bin/mkdir -p /run/sshd /root/.ssh
/usr/bin/chmod 755 /run/sshd
/usr/bin/chmod 700 /root/.ssh
umask 077
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
/usr/bin/chmod 600 /root/.ssh/authorized_keys
/usr/bin/ssh-keygen -l -f /root/.ssh/authorized_keys > /dev/null
/usr/bin/ssh-keygen -A
/usr/sbin/sshd -t
/usr/sbin/sshd
printf 'SSH_READY port=22 authorized_keys_mode=600\n'
DR151_SSH
run_stage ssh /bin/bash /dr151-ssh-setup.sh
printf 'stage_start environment %s\n' "$(/usr/bin/date -u +%FT%TZ)"
/usr/bin/df -h /dr151-cache
set -euxo pipefail
export HF_DATASETS_CACHE=/dr151-cache/huggingface/datasets
export UV_CACHE_DIR=/dr151-cache/uv TMPDIR=/dr151-cache/tmp HF_HUB_DISABLE_XET=1
/usr/bin/mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
/usr/local/bin/python - <<'DR151_SPACE'
import json,shutil
u=shutil.disk_usage('/dr151-cache')
print(json.dumps({'stage':'initial-disk-preflight','total':u.total,'free':u.free,'required_free_gib':140}),flush=True)
if u.free < 140*2**30: raise SystemExit('MIND requires a 200 GB container disk and at least 140 GiB free before setup; no data is sampled or removed to fit.')
DR151_SPACE
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
/usr/bin/git -C /dr151-repo checkout --detach 489c0f33c4d172e79b6143856ab5f0ab1faff2c9
test "$(/usr/bin/git -C /dr151-repo rev-parse HEAD)" = 489c0f33c4d172e79b6143856ab5f0ab1faff2c9
cd /dr151-repo
export SOURCE_DATE_EPOCH=$(/usr/bin/git show -s --format=%ct HEAD)
/usr/local/bin/python -m pip install --no-cache-dir uv==0.13.0
/usr/local/bin/uv venv --python /usr/local/bin/python /dr151-env
/usr/local/bin/uv pip install --python /dr151-env/bin/python setuptools==84.2.1 setuptools-scm==8.2.0 Cython==3.1.0 numpy==2.5.4
/usr/local/bin/uv pip install --python /dr151-env/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
/usr/local/bin/uv pip install --python /dr151-env/bin/python -r benchmarks/quality/classic-requirements.lock.txt
/usr/local/bin/uv pip install --python /dr151-env/bin/python --no-build-isolation -e .
/usr/local/bin/uv pip check --python /dr151-env/bin/python
/dr151-env/bin/python - <<'DR151_CACHE'
import json
from pathlib import Path
import datasets.config as d
import huggingface_hub.constants as h
layout={'hub':str(h.HF_HUB_CACHE),'datasets':str(d.HF_DATASETS_CACHE)}
assert Path(layout['hub']).resolve()==Path('/dr151-cache/huggingface/hub')
assert Path(layout['datasets']).resolve()==Path('/dr151-cache/huggingface/datasets')
Path('/dr151-results/cache-layout.json').write_text(json.dumps(layout,indent=2)+'\n')
print(json.dumps(layout),flush=True)
DR151_CACHE
printf 'stage_end environment %s\n' "$(/usr/bin/date -u +%FT%TZ)"
/usr/bin/df -h /dr151-cache
run_stage mind-prepare /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_mind.py prepare --models /dr151-results/models --cache /dr151-cache --output /dr151-results --workers 16
run_stage mind-evaluate /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_mind.py evaluate --models /dr151-results/models --cache /dr151-cache --output /dr151-results --workers 16
DR151_WORK
remaining=$((DR151_DEADLINE - $(/usr/bin/date +%s)))
/usr/bin/timeout --signal=TERM --kill-after=15 "$remaining" /bin/bash /dr151-work.sh 2>&1 | /usr/bin/tee --output-error=warn /dr151-results/bootstrap.log
status=${PIPESTATUS[0]}
/usr/local/bin/python - "$status" <<'DR151_INVOCATION'
import json, sys
from pathlib import Path
Path('/dr151-results/bootstrap-invocation.json').write_text(json.dumps({"command":["/bin/bash","/dr151-work.sh"],"script":Path('/dr151-work.sh').read_text(),"returncode":int(sys.argv[1]),"pinned_commit":"489c0f33c4d172e79b6143856ab5f0ab1faff2c9"}))
DR151_INVOCATION
finalize_seconds=$((started + 10800 - 5 - $(/usr/bin/date +%s)))
if [ "$finalize_seconds" -lt 1 ]; then finalize_seconds=1; fi
/usr/bin/timeout --signal=TERM --kill-after=2 "$finalize_seconds" /usr/local/bin/python /dr151-finalize.py /dr151-results "$status"
printf 'ALL_DONE\n'
/usr/bin/sleep infinity
