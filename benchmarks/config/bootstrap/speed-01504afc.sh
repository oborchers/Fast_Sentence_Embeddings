set -euo pipefail
umask 077
entry=speed
revision=01504afce057fdf5080c2946aa37e4fdb956ef24

# Refuse an unpriced GPU allocation before directories, SSH setup or network work.
if test "${MEASURE_STUDY:-}" = gpu || test "${MEASURE_BILLING_PROFILE:-}" = gpu_billing || test "${MEASURE_BILLING_PROFILE:-}" = gpu_quality_billing; then
  /usr/local/bin/python - <<'GPU_PRICE_PREFLIGHT'
"""One time and money limit for a whole allocation, counted from ``allocated_at``.

Every step and every cell asks the same budget before it starts. The limit has a reserve for
collection; nothing extends it and nothing retries.
"""

from __future__ import annotations

import json
import math
import os
import time

DEFAULT_RESERVE_SECONDS = 600


POSITIVE_FIELDS = ("month_hours", "planning_minutes", "planning_balance_usd")
NONNEGATIVE_FIELDS = ("compute_usd_per_hour", "disk_usd_per_gb_month", "collection_reserve_seconds")


def _billing_number(key, value):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"billing {key} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (key in POSITIVE_FIELDS and value == 0):
        raise ValueError(f"billing {key} is outside its allowed finite range")


def validate_billing(values, required=False):
    """Check original numeric inputs, including partial profile overrides."""
    if not isinstance(values, dict):
        raise ValueError("billing must be an object")
    fields = (*POSITIVE_FIELDS, *NONNEGATIVE_FIELDS)
    mandatory = set(fields) - {"collection_reserve_seconds"}
    if required and not mandatory.issubset(values):
        raise ValueError("billing is missing a required numeric field")
    for key in fields:
        if key in values:
            _billing_number(key, values[key])


def _billing_profile(contract, profile, study):
    if study is not None and not isinstance(study, str):
        raise ValueError("study must be a string")
    inherited = os.environ.get("MEASURE_BILLING_PROFILE") or None
    if profile is not None and inherited is not None and profile != inherited:
        raise ValueError("explicit and inherited billing profiles disagree")
    profile = profile if profile is not None else inherited or contract.get("billing_profile")
    if profile is None and study is not None and study + "_billing" in contract:
        profile = study + "_billing"
    return profile


def planning_argument(raw):
    """Parse a CLI or inherited JSON number without coercing strings or booleans."""
    value = json.loads(raw)
    _billing_number("planning_minutes", value)
    return value


def add_planning_argument(parser):
    parser.add_argument(
        "--planning-minutes",
        type=planning_argument,
        help="positive finite allocation duration, overriding the selected billing profile",
    )


def resolve_planning_minutes(value):
    inherited = os.environ.get("MEASURE_PLANNING_MINUTES")
    source = "argument"
    if value is not None:
        _billing_number("planning_minutes", value)
    if inherited is not None:
        inherited = planning_argument(inherited)
        if value is not None and value != inherited:
            raise ValueError("step planning minutes differ from the allocation")
        value = inherited
        source = os.environ.get("MEASURE_PLANNING_MINUTES_SOURCE", "argument")
        if source not in ("config", "argument"):
            raise ValueError("unknown planning minutes source")
    return value, source if value is not None else "config"


def planning_override(billing, value):
    value, source = resolve_planning_minutes(value)
    if value is None:
        return billing, source
    result = dict(billing)
    result["planning_balance_usd"] *= value / billing["planning_minutes"]
    result["planning_minutes"] = value
    return result, source


def select_billing(
    contract, profile=None, study=None, *, allow_unpriced=False, planning_minutes=None
):
    """Resolve the same profile for bootstrap, setup, allocation and measurement."""
    profile = _billing_profile(contract, profile, study)
    if study == "gpu" and profile != "gpu_billing":
        raise ValueError("the gpu study requires gpu_billing, not another billing profile")
    validate_billing(contract["billing"])
    override = {}
    if profile is not None:
        if not isinstance(profile, str) or not profile.endswith("_billing"):
            raise ValueError("billing profile must name a declared *_billing object")
        if profile not in contract or not isinstance(contract[profile], dict):
            raise ValueError("unknown billing profile")
        override = contract[profile]
        validate_billing(override)
    billing = {**contract["billing"], **override}
    billing, source = planning_override(billing, planning_minutes)
    billing = _priced_billing(billing, profile, allow_unpriced)
    contract.update(billing=billing, billing_profile=profile, planning_minutes_source=source)
    return contract


def _priced_billing(billing, profile, allow_unpriced):
    gpu_profile = profile in ("gpu_billing", "gpu_quality_billing")
    if gpu_profile:
        billing = gpu_price(billing, allow_unpriced=allow_unpriced)
        if "planning_balance_usd" in billing:
            billing["planning_balance_usd"] *= billing["planning_minutes"] / 60
    validate_billing(billing, required=not (gpu_profile and allow_unpriced))
    return billing


def gpu_price(billing, *, allow_unpriced=False):
    """Require an explicit approved quote, with a one-hour money ceiling."""
    result = {
        k: v
        for k, v in billing.items()
        if k not in ("compute_usd_per_hour", "planning_balance_usd")
    }
    raw = os.environ.get("SPEED_GPU_USD_PER_HOUR")
    if raw is None and allow_unpriced:
        return result
    try:
        price = json.loads(raw) if raw is not None else None
    except (ValueError, TypeError) as error:
        raise ValueError("SPEED_GPU_USD_PER_HOUR requires a positive finite JSON number") from error
    if type(price) not in (int, float) or price <= 0:
        raise ValueError("SPEED_GPU_USD_PER_HOUR requires a positive finite JSON number")
    _billing_number("compute_usd_per_hour", price)
    result.update(compute_usd_per_hour=price, planning_balance_usd=price)
    return result


class WallBudgetExpiredError(TimeoutError):
    """Finish active work safely, then refuse all further operations and waits."""


def check(deadline):
    """Raise when a monotonic deadline has passed; None means no deadline."""
    if deadline is not None and time.monotonic() >= deadline:
        raise WallBudgetExpiredError("total wall-time budget exhausted; no automatic extension")


class Budget:
    """The seconds an allocation may use: the smaller of the planned minutes and the planned
    balance at the hourly cost of compute plus disk."""

    def __init__(
        self,
        billing,
        disk_gb,
        allocated_at,
        clock=time.time,
        *,
        profile=None,
        planning_minutes_source="config",
    ):
        validate_billing(billing, required=True)
        self.billing, self.profile = dict(billing), profile
        self.planning_minutes_source = planning_minutes_source
        self.allocated_at, self.clock = allocated_at, clock
        self.reserve_seconds = billing.get("collection_reserve_seconds", DEFAULT_RESERVE_SECONDS)
        self.hourly = billing["compute_usd_per_hour"] + (
            disk_gb * billing["disk_usd_per_gb_month"] / billing["month_hours"]
        )
        if not math.isfinite(self.hourly) or self.hourly <= 0:
            raise ValueError("positive finite total hourly cost required")
        self.ceiling_seconds = min(
            billing["planning_minutes"] * 60, billing["planning_balance_usd"] / self.hourly * 3600
        )
        if not math.isfinite(self.ceiling_seconds) or self.ceiling_seconds <= 0:
            raise ValueError("positive finite budget ceiling required")

    @classmethod
    def from_contract(cls, contract, allocated_at, clock=time.time):
        return cls(
            contract["billing"],
            contract["machine"]["disk_gb"],
            allocated_at,
            clock,
            profile=contract.get("billing_profile"),
            planning_minutes_source=contract.get("planning_minutes_source", "config"),
        )

    def elapsed(self):
        return max(0.0, self.clock() - self.allocated_at)

    def remaining(self):
        """Seconds left for work, after the collection reserve."""
        return self.ceiling_seconds - self.reserve_seconds - self.elapsed()

    def admit(self, estimate):
        """True when work of ``estimate`` seconds still fits before the reserve."""
        return self.elapsed() + estimate + self.reserve_seconds <= self.ceiling_seconds

gpu_price({})
GPU_PRICE_PREFLIGHT
fi
root="${MEASURE_ROOT:-}"
state="$root/$entry-state"
source="$root/measurement-source"
environment="$root/$entry-env"
work="$root/$entry-work"
results="$root/$entry-results"
mkdir -p "$root/$entry-logs" "$state"
exec > >(tee -a "$root/$entry-logs/bootstrap.log") 2>&1
finish() {
  export MEASURE_EXIT_STATUS="$?" MEASURE_ROOT_PATH="${root:-/}" MEASURE_ENTRY="$entry" MEASURE_SOURCE="$source"
  set +e
  trap - EXIT
  /usr/local/bin/python - <<'FINISH_RECORD'
"""Durable result records: atomic writes, inventory, manifest and DONE.

Standard library only and free of imports from the rest of the package: the bootstrap script
inlines this file to seal the results of a run that never started.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

SEALED = ("manifest.json", "DONE")


def document(path):
    return json.loads(path.read_text())


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as stream:  # a leftover of a dead writer is replaced
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def append(path, value):
    line = json.dumps(value, allow_nan=False, separators=(",", ":"))
    with path.open("a") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def inventory(root):
    """Every file of a result folder except its own manifest and DONE, with size and hash."""
    return [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p not in (root / SEALED[0], root / SEALED[1])
    ]


def exit_code(rows, planned, root):
    """Zero only when every planned cell has a valid row and no fatal record exists."""
    return int(
        len(rows) != planned
        or any(r["status"] != "valid" for r in rows)
        or (root / "fatal.jsonl").exists()
    )


def seal(root, exit_status, cells, label=None, extra=None):
    """Write manifest.json over the folder, then DONE over the manifest. DONE is last: a
    folder without it is not finished and collection refuses it."""
    label = label or {}
    for leftover in root.rglob("*.pending"):  # a dead writer's temporary file is no member
        leftover.unlink()
    atomic(root / SEALED[0], {**label, "files": inventory(root)})
    atomic(
        root / SEALED[1],
        {
            **label,
            "exit_status": exit_status,
            "completed_at": time.time(),
            "cells": cells,
            **(extra or {}),
            "manifest_sha256": sha256(root / SEALED[0]),
        },
    )


def seal_unstarted(root, exit_status, reason=None):
    """Seal a folder of a run that ended before it could seal itself (a crash, a refusal, a
    restart): an empty cell list is enough, the status and the reason carry the failure. A
    folder that is already sealed stays as it is."""
    if (root / "DONE").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    if reason:
        append(root / "fatal.jsonl", {"error": reason})
    (root / "cells.jsonl").touch(exist_ok=True)
    rows = sum(1 for _ in (root / "cells.jsonl").open())
    atomic(
        root / "progress.json", {"stage": "terminal", "finished": rows, "exit_status": exit_status}
    )
    seal(root, exit_status, rows)


def read_rows(root, name="cells.jsonl"):
    path = Path(root) / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

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
# Resolve once before setup; export the same profile to every allocation step.
export PYTHONPATH="$source/benchmarks"
if test "$entry" = speed; then
  MEASURE_BILLING_PROFILE=$(/usr/local/bin/python - "$input" "${MEASURE_STUDY:-libraries}" <<'BILLING_PROFILE'
import json, sys
from _harness.budget import select_billing
print(select_billing(json.load(open(sys.argv[1])), study=sys.argv[2])["billing_profile"] or "")
BILLING_PROFILE
)
  export MEASURE_BILLING_PROFILE
  profile=(--study "${MEASURE_STUDY:-libraries}")
  if test -n "$MEASURE_BILLING_PROFILE"; then profile+=(--billing-profile "$MEASURE_BILLING_PROFILE"); fi
fi
lifecycle=(/usr/local/bin/python -m _harness.allocation)
common=(--spec "$allocation" --entry "$entry" --root "${root:-/}" --allocated-at "$MEASURE_ALLOCATED_AT" --study-file "$input" --pins-file "$source/benchmarks/config/pins.json" "${profile[@]}")
# The machine check comes before any installation: a rejected machine starts no step.
"${lifecycle[@]}" begin "${common[@]}" "${rehearsal[@]}"
setup_study=()
if test "$entry" = speed; then setup_study=(--study "${MEASURE_STUDY:-libraries}"); fi
/usr/local/bin/python "$source/benchmarks/$entry.py" --setup "$environment" --input "$input" --allocation "$allocation" --allocated-at "$MEASURE_ALLOCATED_AT" "${setup_study[@]}"
"${lifecycle[@]}" run "${common[@]}" --param "mode=$mode" --param "input=$input" --param "study=${MEASURE_STUDY:-libraries}" --param "billing_profile=${MEASURE_BILLING_PROFILE:-}" &
printf '%s\n' "$!" > "$root/$entry-logs/allocation.pid"
wait "$!"

