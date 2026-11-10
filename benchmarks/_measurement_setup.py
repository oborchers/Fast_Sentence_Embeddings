"""Private helpers for the central measurement entries."""

from __future__ import annotations

import re
import subprocess
import sys
from functools import partial
from pathlib import Path

from _harness import records as records_module
from _harness.budget import Budget
from _harness.records import atomic, sha256
from _harness.spec import load as load_allocation
from _measurement_config import CONFIG, PINS


def setup_command(*arguments, allocated_at, contract):
    if not Budget.from_contract(contract, allocated_at).admit(60):
        raise RuntimeError("setup exhausted allocation budget")
    subprocess.run([str(x) for x in arguments], check=True)


def _install_environment(execute, uv, root, environment):
    destination = root / environment["name"]
    execute(uv, "venv", "--python", sys.executable, destination)
    execute(
        uv,
        "pip",
        "install",
        "--python",
        destination / "bin/python",
        "--no-deps",
        "--require-hashes",
        "-r",
        CONFIG / environment["lock"],
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "--index-strategy=unsafe-best-match",
    )
    execute(uv, "pip", "check", "--python", destination / "bin/python")


def _install_fse(execute, uv, root, environment, revision):
    """One clone of the product per environment that carries a revision: the candidate in
    ``fse-source``, any other role (base) in ``fse-source-<name>``."""
    candidate = environment["role"] == "candidate"
    source = root / ("fse-source" if candidate else "fse-source-" + environment["name"])
    execute(
        "git",
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        "https://github.com/oborchers/Fast_Sentence_Embeddings.git",
        source,
    )
    execute("git", "-C", source, "checkout", "--detach", revision)
    python = root / environment["name"] / "bin/python"
    flags = ("--no-deps", "--no-build-isolation")
    execute(uv, "pip", "install", "--python", python, *flags, source)
    execute(uv, "pip", "check", "--python", python)


def _install_rust(execute, root):
    installer = root / "tools" / "rustup-init"
    execute(
        "curl",
        "--fail",
        "--location",
        "--output",
        installer,
        "https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init",
    )
    installer.chmod(0o700)
    execute(
        installer, "-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", "1.98.0"
    )
    return sha256(installer)


def setup(root, contract, allocated_at, entry, allocation=None):
    """Install the frozen CPU environments of the allocation without starting measurement:
    every environment (base, candidate, reference) with its own lock, and the product at the
    revision each one declares."""
    definition = load_allocation(allocation or CONFIG / "allocation.json", entry)
    outside_checkout(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "tools").mkdir()
    execute = partial(setup_command, allocated_at=allocated_at, contract=contract)
    execute("apt-get", "update")
    packages = ("build-essential", "pkg-config", "libssl-dev", "curl", "git", "ca-certificates")
    execute("apt-get", "install", "-y", *packages)
    execute(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--prefix",
        root / "tools",
        "uv==" + PINS["versions"]["uv"],
    )
    uv = root / "tools/bin/uv"
    environments = definition["environments"]
    for environment in environments:
        _install_environment(execute, uv, root, environment)
    for environment in environments:
        if environment["revision"]:
            pinned = environment["revision"] == "pins"
            revision = contract["fse_revision"] if pinned else environment["revision"]
            _install_fse(execute, uv, root, environment, revision)
    locks = {e["lock"]: sha256(CONFIG / e["lock"]) for e in environments}
    budget = Budget.from_contract(contract, allocated_at)
    record = {
        "budget_ceiling_seconds": budget.ceiling_seconds,
        "collection_reserve_seconds": budget.reserve_seconds,
        "allocated_at": allocated_at,
        "environments": [{k: e[k] for k in ("name", "role", "revision")} for e in environments],
        "fse_revision": contract["fse_revision"],
        "billing": contract["billing"],
        "billing_profile": contract.get("billing_profile"),
        "python": sys.version,
        "base_image": contract["machine"]["image"],
    }
    if "rust" in definition["toolchains"]:
        record["rustup_installer_sha256"] = _install_rust(execute, root)
        locks["rust.lock"] = sha256(CONFIG / "rust.lock")
        record["system_compiler"] = subprocess.check_output(
            ["cc", "--version"], text=True
        ).splitlines()[0]
    atomic(root / "setup.json", {"locks": locks, **record})


def bootstrap(entry, revision):
    if (
        entry not in ("speed", "quality")
        or not revision
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError("bootstrap requires an entry and full immutable source commit")
    # The record writer of the harness is inlined: the script seals the results of a run that
    # never started on a bare interpreter, before any environment or clone exists.
    records = Path(records_module.__file__).read_text()
    return (
        (CONFIG / "bootstrap.sh")
        .read_text()
        .replace("@RECORDS@", records)
        .replace("@ENTRY@", entry)
        .replace("@REVISION@", revision)
    )


def utility(args, entry):
    import json

    from _harness.allocation import collect_all
    from _harness.collect import collect
    from _measurement_config import contract as load_contract

    if args.collect:
        if not args.archive:
            raise ValueError("--collect requires --archive")
        outside_checkout(args.archive)
        if (args.collect / "allocation-results").is_dir():
            print(json.dumps(collect_all(args.collect, args.archive)))  # a whole allocation
        else:
            print(json.dumps(collect(args.collect, args.archive)))
        return True
    if args.bootstrap:
        print(bootstrap(entry, args.source_revision))
        return True
    if args.describe:
        print(json.dumps(load_contract(args.input), indent=2))
        return True
    return False


def outside_checkout(path):
    checkout = Path(__file__).resolve().parents[1]
    if path.resolve().is_relative_to(checkout):
        raise ValueError("model caches and reconstruction outputs must be outside the checkout")
