"""Revised MIND pod with explicit cache layout, early disk budget and live diagnostics."""

import argparse
import hashlib
import re
import subprocess

from make_folded_pod import script as shell_template
from make_pod_scripts import bootstrap, install_quality
from pod_shell import ssh_first


def script(commit):
    storage = """export HF_DATASETS_CACHE=/dr151-cache/huggingface/datasets
export UV_CACHE_DIR=/dr151-cache/uv TMPDIR=/dr151-cache/tmp HF_HUB_DISABLE_XET=1
/usr/bin/mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
/usr/local/bin/python - <<'DR151_SPACE'
import json,shutil
u=shutil.disk_usage('/dr151-cache')
print(json.dumps({'stage':'initial-disk-preflight','total':u.total,'free':u.free,'required_free_gib':140}),flush=True)
if u.free < 140*2**30: raise SystemExit('MIND requires a 200 GB container disk and at least 140 GiB free before setup; no data is sampled or removed to fit.')
DR151_SPACE
"""
    work = ssh_first() + bootstrap(commit).replace(
        "set -euxo pipefail\n", "set -euxo pipefail\n" + storage, 1
    )
    work += install_quality()
    work += """/dr151-env/bin/python - <<'DR151_CACHE'
import json
from pathlib import Path
import datasets.config as d
import huggingface_hub.constants as h
layout={'hub':str(h.HF_HUB_CACHE),'datasets':str(d.HF_DATASETS_CACHE)}
assert Path(layout['hub']).resolve()==Path('/dr151-cache/huggingface/hub')
assert Path(layout['datasets']).resolve()==Path('/dr151-cache/huggingface/datasets')
Path('/dr151-results/cache-layout.json').write_text(json.dumps(layout,indent=2)+'\\n')
print(json.dumps(layout),flush=True)
DR151_CACHE
printf 'stage_end environment %s\\n' "$(/usr/bin/date -u +%FT%TZ)"
/usr/bin/df -h /dr151-cache
"""
    for stage in ("prepare", "evaluate"):
        work += f"run_stage mind-{stage} /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod_mind.py {stage} --models /dr151-results/models --cache /dr151-cache --output /dr151-results --workers 16\n"
    template = shell_template(commit, limit=10800, export_reserve=90)
    start = template.index("\n", template.index("<<'DR151_WORK'")) + 1
    stop = template.index("DR151_WORK\n", start)
    result = template[:start] + work + template[stop:]
    return result.replace(
        "Additional Pod 3. python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 150 GB disk.",
        "Revised Pod 1. python:3.13, cpu3c, 32 vCPU, 64 GB RAM, 200 GB container disk.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.commit):
        parser.error("commit must be a full object ID")
    subprocess.run(["git", "cat-file", "-e", args.commit + "^{commit}"], check=True)
    from pathlib import Path

    path = Path(args.output)
    path.write_text(script(args.commit))
    path.chmod(0o755)
    print(hashlib.sha256(path.read_bytes()).hexdigest(), path)


if __name__ == "__main__":
    main()
