#!/usr/bin/env bash
# Pod 5. python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 200 GB disk. No lifecycle calls.
set -uo pipefail
actual_started=$(/usr/bin/date +%s)
if ! [[ "${DR151_ALLOCATION_EPOCH:-}" =~ ^[0-9]{10}$ ]]; then
  printf 'ERROR missing allocation budget epoch. ALL_DONE\n' >&2
  /usr/bin/sleep infinity
  exit 2
fi
started=$DR151_ALLOCATION_EPOCH
if [ "$started" -gt "$actual_started" ] || [ "$((actual_started - started))" -gt 600 ]; then
  printf 'ERROR invalid or stale allocation budget epoch. ALL_DONE\n' >&2
  /usr/bin/sleep infinity
  exit 2
fi
export DR151_DEADLINE=$((started + 12900 - 120))
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
import hashlib, json, shutil, sys
from pathlib import Path
root = Path('/dr151-results')
status = int(sys.argv[2])
models = Path('/dr151-cache/models')
for pattern in ('*.json', '*.jsonl', '*-fit.npz'):
    for path in models.glob(pattern):
        shutil.copy2(path, root / path.name)
memory = {}
for name in ('memory.current', 'memory.peak', 'memory.max',
             'memory/memory.usage_in_bytes', 'memory/memory.max_usage_in_bytes',
             'memory/memory.limit_in_bytes'):
    path = Path('/sys/fs/cgroup') / name
    if path.is_file():
        memory[name] = path.read_text().strip()
(root / 'container-memory.json').write_text(json.dumps(memory, indent=2) + '\n')
files = []
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'transfer-manifest.json':
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                      'sha256': digest.hexdigest()})
receipt = root / 'complete.json'
complete = status == 0 and receipt.is_file() and json.loads(receipt.read_text())['complete'] is True
(root / 'transfer-manifest.json').write_text(json.dumps(
    {'complete': complete, 'returncode': status, 'files': files}, indent=2) + '\n')
print('FINAL_STATUS ' + ('complete' if complete else 'partial or failed'), flush=True)
DR151_FINALIZER
/usr/bin/cat > /dr151-work.sh <<'DR151_WORK'
set -euo pipefail
run_stage() {
  stage_name=$1
  shift
  printf 'stage_start %s %s\n' "$stage_name" "$(/usr/bin/date -u +%FT%TZ)"
  /usr/bin/df -h /dr151-cache
  /usr/local/bin/python -c 'import shutil; assert shutil.disk_usage("/dr151-cache").free >= 40*2**30, "Pod 5 stage needs 40 GiB free"' || return 1
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
export NLTK_DATA=/dr151-cache/nltk
/usr/bin/mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$NLTK_DATA"
/usr/local/bin/python - <<'DR151_PREFLIGHT'
import json, shutil
from pathlib import Path
free = shutil.disk_usage('/dr151-cache').free
ram = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))) * 1024
limits = []
for name in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
    path = Path(name)
    if path.is_file():
        value = path.read_text().strip()
        if value != 'max':
            limits.append(int(value))
ram = min([ram, *limits])
current = Path('/sys/fs/cgroup/memory.current')
if not current.is_file():
    current = Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
used = int(current.read_text()) if current.is_file() else 0
if ram - used < 56 * 2**30:
    raise SystemExit('Pod 5 needs at least 56 GiB cgroup headroom before setup')
print(json.dumps({'free_bytes': free, 'effective_ram_bytes': ram, 'cgroup_limits': limits}), flush=True)
if free < 150 * 2**30 or ram < 58 * 2**30:
    raise SystemExit('Pod 5 requires 64 GB RAM and 200 GB disk, with 150 GiB initially free')
DR151_PREFLIGHT
export PATH=/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONUNBUFFERED=1 UV_NO_PROGRESS=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false SETUPTOOLS_SCM_PRETEND_VERSION=2.0.0.dev0
export HF_HOME=/dr151-cache/huggingface FSE_QUALITY_CACHE=/dr151-cache
export HF_HUB_CACHE=/dr151-cache/huggingface/hub
export DEBIAN_FRONTEND=noninteractive
/usr/bin/apt-get update
/usr/bin/apt-get install -y --no-install-recommends time build-essential git curl procps util-linux ca-certificates
/usr/bin/git clone --branch dr151-quality --no-checkout https://github.com/oborchers/Fast_Sentence_Embeddings.git /dr151-repo
/usr/bin/git -C /dr151-repo checkout --detach 0a79ef7157d64d5f93a6c1c3ef6ee7cd9af88067
test "$(/usr/bin/git -C /dr151-repo rev-parse HEAD)" = 0a79ef7157d64d5f93a6c1c3ef6ee7cd9af88067
cd /dr151-repo
export SOURCE_DATE_EPOCH=$(/usr/bin/git show -s --format=%ct HEAD)
/usr/local/bin/python -m pip install --no-cache-dir uv==0.13.0
/usr/local/bin/uv venv --python /usr/local/bin/python /dr151-env
/usr/local/bin/uv pip install --python /dr151-env/bin/python setuptools==84.2.1 setuptools-scm==8.2.0 Cython==3.1.0 numpy==2.5.4
/usr/local/bin/uv pip install --python /dr151-env/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
/usr/local/bin/uv pip install --python /dr151-env/bin/python -r benchmarks/quality/classic-requirements.lock.txt
/usr/local/bin/uv pip install --python /dr151-env/bin/python --no-build-isolation -e .
/usr/local/bin/uv pip check --python /dr151-env/bin/python
printf 'stage_end environment %s\n' "$(/usr/bin/date -u +%FT%TZ)"
run_stage classics /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod5.py --workers 2
DR151_WORK
remaining=$((DR151_DEADLINE - $(/usr/bin/date +%s)))
/usr/bin/timeout --signal=TERM --kill-after=15 "$remaining" /bin/bash /dr151-work.sh 2>&1 | /usr/bin/tee --output-error=warn /dr151-results/bootstrap.log
status=${PIPESTATUS[0]}
/usr/local/bin/python - "$status" <<'DR151_INVOCATION'
import json, sys
from pathlib import Path
Path('/dr151-results/bootstrap-invocation.json').write_text(json.dumps({"command":["/bin/bash","/dr151-work.sh"],"script":Path('/dr151-work.sh').read_text(),"returncode":int(sys.argv[1]),"pinned_commit":"0a79ef7157d64d5f93a6c1c3ef6ee7cd9af88067"}))
DR151_INVOCATION
finalize_seconds=$((started + 12900 - 5 - $(/usr/bin/date +%s)))
if [ "$finalize_seconds" -lt 1 ]; then finalize_seconds=1; fi
/usr/bin/timeout --signal=TERM --kill-after=2 "$finalize_seconds" /usr/local/bin/python /dr151-finalize.py /dr151-results "$status"
printf 'ALL_DONE\n'
/usr/bin/sleep infinity
