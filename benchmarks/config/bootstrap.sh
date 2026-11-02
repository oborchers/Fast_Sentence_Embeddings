set -euo pipefail
umask 077
entry=@ENTRY@
revision=@REVISION@
root="${MEASURE_ROOT:-}"
state="$root/$entry-state"
results="$root/$entry-results"
source="$root/measurement-source"
environment="$root/$entry-env"
work="$root/$entry-work"
mkdir -p "$root/$entry-logs" "$state"
exec > >(tee -a "$root/$entry-logs/bootstrap.log") 2>&1
finish() {
  export MEASURE_EXIT_STATUS="$?" MEASURE_RESULTS="$results"
  set +e
  trap - EXIT
  /usr/local/bin/python - <<'FINISH_RECORD'
import hashlib,json,os,pathlib,time
root=pathlib.Path(os.environ['MEASURE_RESULTS'])
if not (root/'DONE').exists():
    root.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        temp=root/(name+'.pending')
        with temp.open('w') as stream:
            json.dump(value,stream);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temp,root/name)
    status=int(os.environ['MEASURE_EXIT_STATUS'])
    if not os.environ.get('MEASURE_CONTROL_PROBE'): status=status or 1
    (root/'cells.jsonl').touch(exist_ok=True)
    rows=sum(1 for _ in (root/'cells.jsonl').open())
    write('progress.json',dict(stage='terminal',finished=rows,exit_status=status))
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('DONE','manifest.json')]
    write('manifest.json',dict(files=files))
    write('DONE',dict(exit_status=status,cells=rows,completed_at=time.time(),manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()))
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
/usr/local/bin/python "$source/benchmarks/$entry.py" --setup "$environment" --input "$input" --allocated-at "$MEASURE_ALLOCATED_AT"
mode=--run
if test "${MEASURE_REHEARSAL:-0}" = 1; then mode=--rehearse; fi
extra=()
if test "$entry" = speed; then
  extra=(--reference-python "$environment/reference/bin/python" --study "${MEASURE_STUDY:-libraries}")
fi
"$environment/main/bin/python" "$source/benchmarks/$entry.py" "$mode" --input "$input" --results "$results" --work "$work" --allocated-at "$MEASURE_ALLOCATED_AT" "${extra[@]}" &
printf '%s\n' "$!" > "$root/$entry-logs/measurement.pid"
wait "$!"
