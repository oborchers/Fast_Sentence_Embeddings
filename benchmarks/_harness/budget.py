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


def select_billing(contract, profile=None, study=None, *, allow_unpriced=False):
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
    billing = _priced_billing(billing, profile, allow_unpriced)
    contract.update(billing=billing, billing_profile=profile)
    return contract


def _priced_billing(billing, profile, allow_unpriced):
    gpu_profile = profile in ("gpu_billing", "gpu_quality_billing")
    if profile == "gpu_quality_billing" and billing["planning_minutes"] > 480:
        raise ValueError("GPU quality planning ceiling is eight hours; no automatic extension")
    if gpu_profile:
        billing = gpu_price(billing, allow_unpriced=allow_unpriced)
        if profile == "gpu_quality_billing" and "planning_balance_usd" in billing:
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

    def __init__(self, billing, disk_gb, allocated_at, clock=time.time, *, profile=None):
        validate_billing(billing, required=True)
        self.billing, self.profile = dict(billing), profile
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
        )

    def elapsed(self):
        return max(0.0, self.clock() - self.allocated_at)

    def remaining(self):
        """Seconds left for work, after the collection reserve."""
        return self.ceiling_seconds - self.reserve_seconds - self.elapsed()

    def admit(self, estimate):
        """True when work of ``estimate`` seconds still fits before the reserve."""
        return self.elapsed() + estimate + self.reserve_seconds <= self.ceiling_seconds
