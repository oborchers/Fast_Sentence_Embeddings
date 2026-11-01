"""Generate two standalone pod scripts pinned to a pushed source commit."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/oborchers/Fast_Sentence_Embeddings.git"
BUILD = "setuptools==84.2.1 setuptools-scm==8.2.0 Cython==3.1.0 numpy==2.5.4"


def bootstrap(commit):
    return f"""set -euxo pipefail
export PATH=/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONUNBUFFERED=1 UV_NO_PROGRESS=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false SETUPTOOLS_SCM_PRETEND_VERSION=2.0.0.dev0
export HF_HOME=/dr151-cache/huggingface FSE_QUALITY_CACHE=/dr151-cache
export HF_HUB_CACHE=/dr151-cache/huggingface/hub
export DEBIAN_FRONTEND=noninteractive
/usr/bin/apt-get update
/usr/bin/apt-get install -y --no-install-recommends build-essential git curl procps util-linux ca-certificates
/usr/bin/git clone --branch dr151-compare --no-checkout {REPO} /dr151-repo
/usr/bin/git -C /dr151-repo checkout --detach {commit}
test "$(/usr/bin/git -C /dr151-repo rev-parse HEAD)" = {commit}
cd /dr151-repo
export SOURCE_DATE_EPOCH=$(/usr/bin/git show -s --format=%ct HEAD)
/usr/local/bin/python -m pip install --no-cache-dir uv==0.13.0
/usr/local/bin/uv venv --python /usr/local/bin/python /dr151-env
/usr/local/bin/uv pip install --python /dr151-env/bin/python {BUILD}
"""


def install_quality():
    return """/usr/local/bin/uv pip install --python /dr151-env/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
/usr/local/bin/uv pip install --python /dr151-env/bin/python -r benchmarks/quality/classic-requirements.lock.txt
/usr/local/bin/uv pip install --python /dr151-env/bin/python --no-build-isolation -e .
/usr/local/bin/uv pip check --python /dr151-env/bin/python
"""


def speed_stages():
    base = "/usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_speed.py"
    args = "--models /dr151-results/models --cache /dr151-cache --output /dr151-results --deadline-epoch $DR151_DEADLINE"
    return f"""/usr/local/bin/uv pip install --python /dr151-env/bin/python -c benchmarks/comparison-requirements.lock.txt model2vec==0.10.0 pyarrow==26.0.0 threadpoolctl==3.7.0
/usr/local/bin/uv pip install --python /dr151-env/bin/python --no-build-isolation -e .
set +e
{base} probe {args}
probe_status=$?
set -e
if [ "$probe_status" -ne 0 ]; then exit "$probe_status"; fi
/usr/bin/curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /dr151-rustup.sh
/bin/bash /dr151-rustup.sh -y --profile minimal --default-toolchain 1.98.0
/usr/local/bin/uv pip install --python /dr151-env/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
/usr/local/bin/uv pip install --python /dr151-env/bin/python -r benchmarks/quality/pod-speed-requirements.lock.txt
/usr/local/bin/uv pip check --python /dr151-env/bin/python
/dr151-env/bin/python -c 'import sys; sys.path.insert(0,"benchmarks"); from compare_rust_build import build_worker; build_worker()'
/dr151-env/bin/python -c 'import nltk; nltk.download("punkt_tab", quiet=True, raise_on_error=True)'
for stage in counterparts pipelines float32 int8 classification search float16; do
  {base} "$stage" {args}
done
"""


def script(commit, pod):
    limit = 10800 if pod == 1 else 14400
    work = bootstrap(commit)
    if pod == 1:
        work += install_quality()
        for stage in ("prepare", "evaluate"):
            work += f"/usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_mind.py {stage} --models /dr151-results/models --cache /dr151-cache --output /dr151-results --workers 16\n"
    else:
        work += speed_stages()
    return f"""#!/usr/bin/env bash
# Image python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 60 GB disk. Never launches a pod.
set -uo pipefail
started=$(/usr/bin/date +%s)
export DR151_DEADLINE=$((started + {limit} - 90))
/usr/bin/mkdir -p /dr151-results /dr151-cache
cat > /dr151-finalize.py <<'DR151_FINALIZER'
{(HERE / "pod_finalize.py").read_text()}DR151_FINALIZER
cat > /dr151-work.sh <<'DR151_WORK'
{work}DR151_WORK
remaining=$((DR151_DEADLINE - $(/usr/bin/date +%s)))
/usr/bin/timeout --signal=TERM --kill-after=15 "$remaining" /bin/bash /dr151-work.sh > /dr151-results/bootstrap.log 2>&1
status=$?
/usr/local/bin/python - "$status" <<'DR151_INVOCATION'
import json, sys
from pathlib import Path
Path('/dr151-results/bootstrap-invocation.json').write_text(json.dumps({{"command":["/bin/bash","/dr151-work.sh"],"script":Path('/dr151-work.sh').read_text(),"returncode":int(sys.argv[1]),"pinned_commit":"{commit}"}}))
DR151_INVOCATION
finalize_seconds=$((started + {limit} - 5 - $(/usr/bin/date +%s)))
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
        parser.error("commit must be a full 40-character object ID")
    args.output.mkdir(parents=True, exist_ok=True)
    for pod in (1, 2):
        path = args.output / f"dr151-pod{pod}.sh"
        path.write_text(script(args.commit, pod))
        path.chmod(0o755)
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")


if __name__ == "__main__":
    main()
