"""Generate the additional 45-minute folded-classifier pod, never launching it."""

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

from make_pod_scripts import HERE, bootstrap
from pod_shell import ssh_first


def script(commit):
    work = (
        ssh_first()
        + bootstrap(commit)
        + """export TQDM_DISABLE=1
/usr/local/bin/uv pip install --python /dr151-env/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
/usr/local/bin/uv pip install --python /dr151-env/bin/python -r benchmarks/quality/pod-classifier-requirements.lock.txt
/usr/local/bin/uv pip install --python /dr151-env/bin/python --no-build-isolation -e .
/usr/local/bin/uv pip check --python /dr151-env/bin/python
printf 'stage_end environment %s\\n' "$(/usr/bin/date -u +%FT%TZ)"
/usr/bin/df -h /dr151-cache
run_stage folded-classification /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_folded.py --cache /dr151-cache --output /dr151-results --deadline-epoch $DR151_DEADLINE
"""
    )
    return f"""#!/usr/bin/env bash
# Additional Pod 3. python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 20 GB disk.
set -uo pipefail
started=$(/usr/bin/date +%s)
export DR151_DEADLINE=$((started + 2700 - 60))
/usr/bin/mkdir -p /dr151-results
if ! /usr/bin/mkdir /dr151-cache; then
  printf 'ERROR: /dr151-cache already exists; require a fresh exclusive pod cache.\\nALL_DONE\\n' >&2
  /usr/bin/sleep infinity
  exit 1
fi
/usr/local/bin/python - <<'DR151_CACHE_OWNER'
import json, uuid
from pathlib import Path
record = {{"cache":"/dr151-cache","created_exclusively":True,"owner_token":uuid.uuid4().hex}}
for path in ('/dr151-cache/.dr151-owner.json', '/dr151-results/cache-ownership.json'):
    with Path(path).open('x') as stream: json.dump(record, stream)
DR151_CACHE_OWNER
/usr/bin/cat > /dr151-finalize.py <<'DR151_FINALIZER'
{(HERE / "pod_finalize.py").read_text()}DR151_FINALIZER
/usr/bin/cat > /dr151-work.sh <<'DR151_WORK'
{work}DR151_WORK
remaining=$((DR151_DEADLINE - $(/usr/bin/date +%s)))
/usr/bin/timeout --signal=TERM --kill-after=15 "$remaining" /bin/bash /dr151-work.sh 2>&1 | /usr/bin/tee --output-error=warn /dr151-results/bootstrap.log
status=${{PIPESTATUS[0]}}
/usr/local/bin/python - "$status" <<'DR151_INVOCATION'
import json, sys
from pathlib import Path
Path('/dr151-results/bootstrap-invocation.json').write_text(json.dumps({{"command":["/bin/bash","/dr151-work.sh"],"script":Path('/dr151-work.sh').read_text(),"returncode":int(sys.argv[1]),"pinned_commit":"{commit}"}}))
DR151_INVOCATION
finalize_seconds=$((started + 2700 - 5 - $(/usr/bin/date +%s)))
if [ "$finalize_seconds" -lt 1 ]; then finalize_seconds=1; fi
/usr/bin/timeout --signal=TERM --kill-after=2 "$finalize_seconds" /usr/local/bin/python /dr151-finalize.py /dr151-results "$status"
printf 'ALL_DONE\\n'
/usr/bin/sleep infinity
"""


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
