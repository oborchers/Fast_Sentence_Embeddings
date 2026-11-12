"""The data of an allocation (``allocation@1``): machine check, environments and steps.

An allocation is one rented machine from ``allocated_at`` on. It declares what the machine
must be, which environments to install (base, candidate and reference per revision) and the
ordered steps to run, each with its own result folder. Standard library only.
"""

from __future__ import annotations

import json
import math
import re
import string

from _harness import machine

FORMAT = "allocation@1"
ROLES = ("base", "candidate", "reference")
SAFE = re.compile(r"[a-z][a-z0-9-]*")
REVISION = re.compile(r"[0-9a-f]{40}")
SECTIONS = ("machine_check", "rehearsal_machine_check", "environments", "steps", "toolchains")


def _safe_name(kind, name):
    if not isinstance(name, str) or not SAFE.fullmatch(name):
        raise ValueError(f"{kind} name must be lowercase letters, digits and hyphens: {name!r}")
    return name


def _environment(item):
    unknown = set(item) - {"name", "role", "lock", "revision"}
    if unknown or item.get("role") not in ROLES or not isinstance(item.get("lock"), str):
        raise ValueError(f"environment needs name, role {ROLES} and lock only: {item}")
    revision = item.get("revision")
    if revision not in (None, "pins") and not REVISION.fullmatch(str(revision)):
        raise ValueError("environment revision is null, 'pins' or a full commit")
    return {"revision": None, **item, "name": _safe_name("environment", item.get("name"))}


def _positive_seconds(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _step(item, environments, earlier):
    if not isinstance(item, dict):
        raise ValueError("step must be an object")
    unknown = set(item) - {
        "name",
        "environment",
        "command",
        "estimate_seconds",
        "requires",
        "requires_by_seconds",
    }
    estimate = item.get("estimate_seconds")
    command = item.get("command")
    if unknown or item.get("environment") not in environments:
        raise ValueError(f"step needs name, command and a declared environment: {item}")
    if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
        raise ValueError("step command must be a nonempty list of strings")
    if not _positive_seconds(estimate):
        raise ValueError("positive finite estimate_seconds required for every step")
    requires = item.get("requires", [])
    if not isinstance(requires, list) or any(not isinstance(n, str) for n in requires):
        raise ValueError("requires must be a list of earlier step names")
    if len(set(requires)) != len(requires) or any(n not in earlier for n in requires):
        raise ValueError("requires must name unique earlier steps")
    if "requires_by_seconds" in item:
        deadline = item["requires_by_seconds"]
        if not requires or not _positive_seconds(deadline):
            raise ValueError("requires_by_seconds needs prerequisites and a positive finite number")
    return {**item, "name": _safe_name("step", item.get("name"))}


def validate(spec):
    unknown = set(spec) - set(SECTIONS) - {"format"}
    if unknown or spec.get("format") != FORMAT:
        raise ValueError(f"allocation must be {FORMAT} with sections {SECTIONS}")
    environments = [_environment(item) for item in spec["environments"]]
    names = [e["name"] for e in environments]
    if not names or len(set(names)) != len(names):
        raise ValueError("nonempty unique environment names required")
    if not isinstance(spec["steps"], list):
        raise ValueError("steps must be an ordered list")
    steps = []
    earlier = set()
    for item in spec["steps"]:
        step = _step(item, set(names), earlier)
        steps.append(step)
        earlier.add(step["name"])
    step_names = [s["name"] for s in steps]
    if not steps or len(set(step_names)) != len(step_names):
        raise ValueError("nonempty unique step names required")
    machine.validate(spec["machine_check"])
    machine.validate(spec["rehearsal_machine_check"])
    return {**spec, "environments": environments, "steps": steps}


def load(path, entry):
    """The validated allocation of one entry (``speed`` or ``quality``) from a data file."""
    spec = json.loads(path.read_text())[entry]
    return validate(spec)


def render(step, values):
    """The command of a step with ``{name}`` fields filled. A field without a value is an
    error: a step never starts with a half-formed command line."""
    formatter = string.Formatter()
    result = []
    for argument in step["command"]:
        for _, field, _, _ in formatter.parse(argument):
            if field is not None and field not in values:
                raise ValueError(f"step {step['name']} uses the unknown field {{{field}}}")
        result.append(argument.format_map(values))
    return result
