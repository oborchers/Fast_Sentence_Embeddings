#!/usr/bin/env bash
# Additional Pod 3. python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 150 GB disk.
set -uo pipefail
started=$(/usr/bin/date +%s)
export DR151_DEADLINE=$((started + 9000 - 60))
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
/usr/bin/git -C /dr151-repo checkout --detach 3ca670db94ae8924bf6e4c59fc9d4b03b364c633
test "$(/usr/bin/git -C /dr151-repo rev-parse HEAD)" = 3ca670db94ae8924bf6e4c59fc9d4b03b364c633
cd /dr151-repo
export SOURCE_DATE_EPOCH=$(/usr/bin/git show -s --format=%ct HEAD)
/usr/local/bin/python -m pip install --no-cache-dir uv==0.13.0
/usr/local/bin/uv venv --python /usr/local/bin/python /dr151-env
/usr/local/bin/uv pip install --python /dr151-env/bin/python setuptools==84.2.1 setuptools-scm==8.2.0 Cython==3.1.0 numpy==2.5.4
export TQDM_DISABLE=1
/usr/local/bin/uv pip install --python /dr151-env/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
/usr/local/bin/uv pip install --python /dr151-env/bin/python -r benchmarks/quality/pod3-requirements.lock.txt
/usr/local/bin/uv pip install --python /dr151-env/bin/python --no-build-isolation -e .
/usr/local/bin/uv pip check --python /dr151-env/bin/python
printf 'stage_end environment %s\n' "$(/usr/bin/date -u +%FT%TZ)"
/usr/bin/df -h /dr151-cache
set +e
/usr/bin/cat > /dr151-rust-setup.sh <<'DR151_RUST'
set -euxo pipefail
/usr/bin/curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /dr151-rustup.sh
/bin/bash /dr151-rustup.sh -y --profile minimal --default-toolchain 1.98.0
/dr151-env/bin/python -c 'import sys; sys.path.insert(0,"benchmarks"); from compare_rust_build import build_worker; build_worker()'
DR151_RUST
run_stage rust-setup /usr/bin/timeout --signal=TERM --kill-after=10 600 /bin/bash /dr151-rust-setup.sh
rust_status=$?
readme_deadline=$(( $(/usr/bin/date +%s) + 1200 ))
if [ "$readme_deadline" -gt "$((DR151_DEADLINE - 6600))" ]; then readme_deadline=$((DR151_DEADLINE - 6600)); fi
readme_seconds=$((readme_deadline - $(/usr/bin/date +%s)))
if [ "$readme_seconds" -gt 0 ]; then
  run_stage readme-panels /usr/bin/timeout --signal=TERM --kill-after=10 "$readme_seconds" /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_folded.py --stage readme --cache /dr151-cache --output /dr151-results --deadline-epoch "$readme_deadline"
  readme_status=$?
else
  readme_status=124
fi
classification_deadline=$((DR151_DEADLINE - 6600))
classification_seconds=$((classification_deadline - $(/usr/bin/date +%s)))
if [ "$classification_seconds" -gt 0 ]; then
  run_stage folded-classification /usr/bin/timeout --signal=TERM --kill-after=10 "$classification_seconds" /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_folded.py --stage classifications --cache /dr151-cache --output /dr151-results --deadline-epoch "$classification_deadline"
  classification_status=$?
else
  classification_status=124
fi
large_deadline=$((DR151_DEADLINE - 2400))
large_seconds=$((large_deadline - $(/usr/bin/date +%s)))
if [ "$large_seconds" -gt 0 ]; then
  run_stage large-write-paths /usr/bin/timeout --signal=TERM --kill-after=10 "$large_seconds" /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_folded.py --stage large --cache /dr151-cache --output /dr151-results --deadline-epoch "$large_deadline"
  large_status=$?
else
  large_status=124
fi
scaling_deadline=$((DR151_DEADLINE - 2100))
scaling_seconds=$((scaling_deadline - $(/usr/bin/date +%s)))
if [ "$scaling_seconds" -gt 0 ]; then
  run_stage worker-scaling /usr/bin/timeout --signal=TERM --kill-after=10 "$scaling_seconds" /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_folded.py --stage scaling --cache /dr151-cache --output /dr151-results --deadline-epoch "$scaling_deadline"
  scaling_status=$?
else
  scaling_status=124
fi
run_stage showcase-100m /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_folded.py --stage showcase --cache /dr151-cache --output /dr151-results --deadline-epoch "$DR151_DEADLINE"
showcase_status=$?
if [ "$readme_status" -ne 0 ]; then exit "$readme_status"; fi
if [ "$classification_status" -ne 0 ]; then exit "$classification_status"; fi
if [ "$large_status" -ne 0 ]; then exit "$large_status"; fi
if [ "$rust_status" -ne 0 ]; then exit "$rust_status"; fi
if [ "$scaling_status" -ne 0 ]; then exit "$scaling_status"; fi
exit "$showcase_status"
DR151_WORK
remaining=$((DR151_DEADLINE - $(/usr/bin/date +%s)))
/usr/bin/timeout --signal=TERM --kill-after=15 "$remaining" /bin/bash /dr151-work.sh 2>&1 | /usr/bin/tee --output-error=warn /dr151-results/bootstrap.log
status=${PIPESTATUS[0]}
/usr/local/bin/python - "$status" <<'DR151_INVOCATION'
import json, sys
from pathlib import Path
Path('/dr151-results/bootstrap-invocation.json').write_text(json.dumps({"command":["/bin/bash","/dr151-work.sh"],"script":Path('/dr151-work.sh').read_text(),"returncode":int(sys.argv[1]),"pinned_commit":"3ca670db94ae8924bf6e4c59fc9d4b03b364c633"}))
DR151_INVOCATION
finalize_seconds=$((started + 9000 - 5 - $(/usr/bin/date +%s)))
if [ "$finalize_seconds" -lt 1 ]; then finalize_seconds=1; fi
/usr/bin/timeout --signal=TERM --kill-after=2 "$finalize_seconds" /usr/local/bin/python /dr151-finalize.py /dr151-results "$status"
printf 'ALL_DONE\n'
/usr/bin/sleep infinity
