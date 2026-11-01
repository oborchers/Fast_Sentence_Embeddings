"""Pod4-only bounded supervision and disk-independent failure artifact emission."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILURE = """failure_manifest() {
/usr/local/bin/python - "$1" "$2" <<'DR151_FAILURE'
import base64, gzip, hashlib, json, sys
value = {'complete':False, 'returncode':int(sys.argv[1]), 'reason':sys.argv[2], 'status':'partial or failed'}
blob = gzip.compress(json.dumps(value).encode(), mtime=0)
encoded = base64.b64encode(blob).decode()
chunks = [encoded[i:i+160] for i in range(0,len(encoded),160)]
print('DR190_BEGIN manifest',len(blob),hashlib.sha256(blob).hexdigest(),flush=True)
for i, chunk in enumerate(chunks): print('DR190_DATA manifest',i,chunk,flush=True)
print('DR190_END manifest',len(chunks),flush=True)
DR151_FAILURE
}
"""


def safe_export(value):
    start = value.index("/usr/bin/cat > /dr151-finalize.py")
    end = value.index("DR151_FINALIZER\n", start) + len("DR151_FINALIZER\n")
    finalizer = value[start:end]
    value = value[:start] + value[end:]
    value = value.replace(
        "started=$(/usr/bin/date +%s)\n", "started=$(/usr/bin/date +%s)\n" + FAILURE, 1
    )
    setup = finalizer.replace("/usr/bin/cat >", "if ! /usr/bin/cat >", 1)
    setup += "then\n  failure_manifest 1 finalizer-creation\n  printf 'ALL_DONE\\n'\n  /usr/bin/sleep infinity\n  exit 1\nfi\n"
    setup += """/usr/local/bin/python - <<'DR151_PLAN'
import json, shutil
from pathlib import Path
Path('/dr151-results/pod4-plan.json').write_text(json.dumps({'stages':['probe','readme','classifications']}))
usage = shutil.disk_usage('/dr151-results')
Path('/dr151-results/disk-preflight.json').write_text(json.dumps({'total':usage.total,'free':usage.free}))
if usage.total < 100_000_000_000 or usage.free < 25 * 1024**3:
    raise SystemExit('require 100 GB capacity and at least 25 GiB free')
DR151_PLAN
if [ "$?" -ne 0 ]; then
  failure_manifest 1 disk-preflight
  printf 'ALL_DONE\\n'
  /usr/bin/sleep infinity
  exit 1
fi
"""
    marker = "/usr/bin/mkdir -p /dr151-results\n"
    value = value.replace(marker, marker + setup, 1)
    value = value.replace(
        "\\nALL_DONE\\n' >&2",
        "\\n' >&2\n  failure_manifest 1 cache-ownership\n  printf 'ALL_DONE\\n'",
    )
    start = value.index('/usr/bin/timeout --signal=TERM --kill-after=15 "$remaining"')
    end = value.index("status=${PIPESTATUS[0]}\n", start) + len("status=${PIPESTATUS[0]}\n")
    supervisor = (
        "/usr/local/bin/python - \"$remaining\" <<'DR151_SUPERVISOR'\n"
        + (HERE / "pod_supervise.py").read_text()
        + "DR151_SUPERVISOR\nstatus=$?\n"
    )
    value = value[:start] + supervisor + value[end:]
    value = value.replace(
        ' /dr151-results "$status"\n',
        ' /dr151-results "$status" --no-sentinel\nexport_status=$?\nif [ "$export_status" -ne 0 ]; then\n  failure_manifest "$export_status" export-failed\nfi\n',
    )
    return value
