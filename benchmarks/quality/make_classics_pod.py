"""Generate Pod 5, historical GloVe protocol, with a 215-minute work/export limit."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

from make_folded_pod import script as template
from make_pod_scripts import bootstrap, install_quality
from pod_shell import ssh_first

MEMORY_SOURCE = Path(__file__).with_name("pod5_memory.py").read_text()

FINALIZER = (
    MEMORY_SOURCE
    + """\nimport hashlib, json, shutil, sys
from pathlib import Path
root = Path('/dr151-results')
status = int(sys.argv[2])
models = Path('/dr151-cache/models')
for pattern in ('*.json', '*.jsonl', '*-fit.npz'):
    for path in models.glob(pattern):
        shutil.copy2(path, root / path.name)
try:
    memory = snapshot()
except (OSError, RuntimeError) as error:
    memory = {'error': str(error)}
    status = status or 1
(root / 'container-memory.json').write_text(json.dumps(memory, indent=2) + '\\n')
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
    {'complete': complete, 'returncode': status, 'files': files}, indent=2) + '\\n')
print('FINAL_STATUS ' + ('complete' if complete else 'partial or failed'), flush=True)
"""
)


def script(commit):
    storage = """export HF_DATASETS_CACHE=/dr151-cache/huggingface/datasets
export UV_CACHE_DIR=/dr151-cache/uv TMPDIR=/dr151-cache/tmp HF_HUB_DISABLE_XET=1
export NLTK_DATA=/dr151-cache/nltk
/usr/bin/mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$NLTK_DATA"
/usr/local/bin/python - <<'DR151_PREFLIGHT'
import json, shutil
from pathlib import Path
free = shutil.disk_usage('/dr151-cache').free
if free < 150 * 2**30:
    raise SystemExit('Pod 5 requires 200 GB disk, with 150 GiB initially free')
DR151_PREFLIGHT
"""
    storage = storage.replace(
        "import json, shutil\nfrom pathlib import Path\n",
        MEMORY_SOURCE + "\npreflight()\nimport shutil\n",
    )
    work = ssh_first() + bootstrap(commit).replace(
        "set -euxo pipefail\n", "set -euxo pipefail\n" + storage, 1
    ).replace("--branch dr151-compare", "--branch dr151-quality")
    work = work.replace("build-essential git curl", "time build-essential git curl")
    work += install_quality()
    work += """printf 'stage_end environment %s\\n' "$(/usr/bin/date -u +%FT%TZ)"
run_stage classics /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod5.py --workers 2
"""
    shell = template(commit, limit=12900, export_reserve=120)
    start = shell.index("\n", shell.index("<<'DR151_WORK'")) + 1
    end = shell.index("DR151_WORK\n", start)
    shell = shell[:start] + work + shell[end:]
    start = shell.index("\n", shell.index("<<'DR151_FINALIZER'")) + 1
    end = shell.index("DR151_FINALIZER\n", start)
    shell = shell[:start] + FINALIZER + shell[end:]
    # Every setup stage checks space too, before apt or package downloads.
    shell = shell.replace(
        '  /usr/bin/df -h /dr151-cache\n  if "$@";',
        """  /usr/bin/df -h /dr151-cache
  /usr/local/bin/python -c 'import shutil; assert shutil.disk_usage("/dr151-cache").free >= 40*2**30, "Pod 5 stage needs 40 GiB free"' || return 1
  if "$@";""",
    )
    shell = shell.replace(
        "started=$(/usr/bin/date +%s)",
        """actual_started=$(/usr/bin/date +%s)
if ! [[ "${DR151_ALLOCATION_EPOCH:-}" =~ ^[0-9]{10}$ ]]; then
  printf 'ERROR missing allocation budget epoch. ALL_DONE\\n' >&2
  /usr/bin/sleep infinity
  exit 2
fi
started=$DR151_ALLOCATION_EPOCH
if [ "$started" -gt "$actual_started" ] || [ "$((actual_started - started))" -gt 600 ]; then
  printf 'ERROR invalid or stale allocation budget epoch. ALL_DONE\\n' >&2
  /usr/bin/sleep infinity
  exit 2
fi""",
    )
    return shell.replace(
        "Additional Pod 3. python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 150 GB disk.",
        "Pod 5. python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 200 GB disk. No lifecycle calls.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.commit):
        parser.error("commit must be a full object ID")
    subprocess.run(["git", "cat-file", "-e", args.commit + "^{commit}"], check=True)
    args.output.write_text(script(args.commit))
    args.output.chmod(0o755)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest(), args.output)


if __name__ == "__main__":
    main()
