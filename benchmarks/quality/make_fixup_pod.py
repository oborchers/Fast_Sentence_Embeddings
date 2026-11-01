"""Generate the unlaunched140-minute slim fix-up pod from the SSH-first wrapper."""

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

from make_folded_pod import script as base_script
from pod4_shell import safe_export

LIMIT = 8400
DISK_GB = 100


def script(commit):
    value = base_script(commit, limit=LIMIT, export_reserve=90)
    start = value.index("readme_deadline=")
    end_marker = 'exit "$showcase_status"\n'
    end = value.index(end_marker, start) + len(end_marker)
    stages = """if [ "$rust_status" -ne 0 ]; then exit "$rust_status"; fi
probe_deadline=$(( $(/usr/bin/date +%s) + 900 ))
if [ "$probe_deadline" -gt "$DR151_DEADLINE" ]; then probe_deadline=$DR151_DEADLINE; fi
probe_seconds=$((probe_deadline - $(/usr/bin/date +%s) - 20))
if [ "$probe_seconds" -le 0 ]; then exit 124; fi
run_stage quiet-probe /usr/bin/timeout --signal=TERM --kill-after=10 "$probe_seconds" /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod4.py probe --cache /dr151-cache --output /dr151-results --deadline-epoch "$probe_deadline"
probe_status=$?
if [ "$probe_status" -ne 0 ]; then exit "$probe_status"; fi
/dr151-env/bin/python -c 'import json; assert json.load(open("/dr151-results/readme-panels/probe.json")).get("passed") is True'
receipt_status=$?
if [ "$receipt_status" -ne 0 ]; then exit "$receipt_status"; fi
readme_deadline=$(( $(/usr/bin/date +%s) + 4200 ))
if [ "$readme_deadline" -gt "$((DR151_DEADLINE - 3000))" ]; then readme_deadline=$((DR151_DEADLINE - 3000)); fi
readme_seconds=$((readme_deadline - $(/usr/bin/date +%s)))
if [ "$readme_seconds" -gt 0 ]; then
  run_stage readme-panels /usr/bin/timeout --signal=TERM --kill-after=10 "$readme_seconds" /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod4.py readme --cache /dr151-cache --output /dr151-results --deadline-epoch "$readme_deadline"
  readme_status=$?
else
  readme_status=124
fi
if [ "$readme_status" -eq 75 ]; then exit 75; fi
if [ "$readme_status" -ge 124 ]; then exit "$readme_status"; fi
if [ ! -f /dr151-results/readme-panels/plan.json ] || [ ! -f /dr151-results/readme-panels/speed-results.jsonl ]; then exit 1; fi
run_stage folded-classification /usr/bin/nice -n 10 /dr151-env/bin/python benchmarks/quality/pod4.py classifications --cache /dr151-cache --output /dr151-results --deadline-epoch "$DR151_DEADLINE"
classification_status=$?
if [ "$readme_status" -ne 0 ]; then exit "$readme_status"; fi
exit "$classification_status"
"""
    value = value[:start] + stages + value[end:]
    value = value.replace(
        "run_stage rust-setup /usr/bin/timeout --signal=TERM --kill-after=10 600",
        'rust_seconds=$((DR151_DEADLINE - $(/usr/bin/date +%s) - 20))\nif [ "$rust_seconds" -gt 600 ]; then rust_seconds=600; fi\nif [ "$rust_seconds" -le 0 ]; then exit 124; fi\nrun_stage rust-setup /usr/bin/timeout --signal=TERM --kill-after=10 "$rust_seconds"',
    )
    value = value.replace(
        "/usr/bin/timeout --signal=TERM --kill-after=10",
        "/usr/bin/timeout --foreground --signal=TERM --kill-after=10",
    )
    value = value.replace(
        "# Additional Pod 3.", "# Slim Pod 4 fix-up, never launched by this generator."
    )
    value = value.replace("150 GB disk", f"{DISK_GB} GB disk")
    return safe_export(value)


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
