set -euo pipefail
umask 077
entry=@ENTRY@
revision=@REVISION@
root="${MEASURE_ROOT:-}"
state="$root/$entry-state"
source="$root/measurement-source"
environment="$root/$entry-env"
work="$root/$entry-work"
results="$root/$entry-results"
mkdir -p "$root/$entry-logs" "$state"
exec > >(tee -a "$root/$entry-logs/bootstrap.log") 2>&1
finish() {
  export MEASURE_EXIT_STATUS="$?" MEASURE_ROOT_PATH="$root" MEASURE_ENTRY="$entry" MEASURE_SOURCE="$source"
  set +e
  trap - EXIT
  /usr/local/bin/python - <<'FINISH_RECORD'
@RECORDS@
import json, os, pathlib
root = pathlib.Path(os.environ["MEASURE_ROOT_PATH"])
entry = os.environ["MEASURE_ENTRY"]
status = int(os.environ["MEASURE_EXIT_STATUS"])
if not os.environ.get("MEASURE_CONTROL_PROBE"):
    status = status or 1
names = [entry]
default = pathlib.Path(os.environ["MEASURE_SOURCE"]) / "benchmarks/config/allocation.json"
allocation = pathlib.Path(os.environ.get("MEASURE_ALLOCATION") or default)
if allocation.exists():
    names = [s["name"] for s in json.loads(allocation.read_text())[entry]["steps"]]
folders = [root / (name + "-results") for name in names]
if (root / "allocation-results").exists():
    folders.append(root / "allocation-results")
for folder in folders:  # each folder alone: one failure does not leave the others open
    try:
        seal_unstarted(folder, status)
    except Exception as error:
        print("could not seal", folder, repr(error))
FINISH_RECORD
  if test "${MEASURE_CONTROL_PROBE:-0}" != 1; then sleep infinity; fi
}
trap finish EXIT
first_run=1
if ! (set -o noclobber; printf '%s\n' "$revision" > "$state/started"); then first_run=0; fi
if test "${MEASURE_CONTROL_PROBE:-0}" != 1; then
  if ! test -x /usr/sbin/sshd; then
    /usr/bin/apt-get update
    /usr/bin/apt-get install -y openssh-server git
  fi
  mkdir -p /run/sshd /root/.ssh
  chmod 700 /root/.ssh
  : "${PUBLIC_KEY:?SSH public key required}"
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  /usr/sbin/sshd
fi
if test "$first_run" = 0 || test -e "$source" || test -e "$environment" || test -e "$work" || test -e "$results"; then
  printf 'RESTART_DETECTED: SSH only; no second initialization or measurement.\n'
  exit 77
fi
if test "${MEASURE_CONTROL_PROBE:-0}" = 1; then
  printf '%s\n' 'initialized once' > "$state/initialization"
  exit 0
fi
: "${MEASURE_ALLOCATED_AT:?Set the actual allocation Unix timestamp}"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="$root/measurement-cache/uv"
export HF_HOME="$root/measurement-cache/huggingface"
export HF_DATASETS_CACHE="$root/measurement-cache/datasets"
export RUSTUP_HOME=/root/.rustup CARGO_HOME=/root/.cargo LC_ALL=C.UTF-8
/usr/bin/git clone --filter=blob:none --no-checkout https://github.com/oborchers/Fast_Sentence_Embeddings.git "$source"
/usr/bin/git -C "$source" checkout --detach "$revision"
input="${MEASURE_INPUT:-$source/benchmarks/config/$entry.json}"
allocation="${MEASURE_ALLOCATION:-$source/benchmarks/config/allocation.json}"
mode=--run
rehearsal=()
if test "${MEASURE_REHEARSAL:-0}" = 1; then mode=--rehearse; rehearsal=(--rehearsal); fi
profile=()
if test "$entry" = speed && test "${MEASURE_STUDY:-libraries}" = headline; then profile=(--billing-profile headline_billing); fi
export PYTHONPATH="$source/benchmarks"
lifecycle=(/usr/local/bin/python -m _harness.allocation)
common=(--spec "$allocation" --entry "$entry" --root "$root" --allocated-at "$MEASURE_ALLOCATED_AT" --study-file "$input" --pins-file "$source/benchmarks/config/pins.json" "${profile[@]}")
# The machine check comes before any installation: a rejected machine starts no step.
"${lifecycle[@]}" begin "${common[@]}" "${rehearsal[@]}"
/usr/local/bin/python "$source/benchmarks/$entry.py" --setup "$environment" --input "$input" --allocation "$allocation" --allocated-at "$MEASURE_ALLOCATED_AT"
"${lifecycle[@]}" run "${common[@]}" --param "mode=$mode" --param "input=$input" --param "study=${MEASURE_STUDY:-libraries}" &
printf '%s\n' "$!" > "$root/$entry-logs/allocation.pid"
wait "$!"
