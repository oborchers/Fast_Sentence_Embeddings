"""Private helpers for the central measurement entries."""

from __future__ import annotations

import subprocess
import sys
from functools import partial

from _measurement import admit, atomic, sha256
from _measurement_config import CONFIG, PINS


def setup_command(*arguments, allocated_at, contract):
    if not admit(contract, allocated_at, 60):
        raise RuntimeError("setup exhausted allocation budget")
    subprocess.run([str(x) for x in arguments], check=True)


def setup(root, contract, allocated_at, quality=False):
    """Install the two frozen CPU environments without starting measurement."""
    root.mkdir(parents=True, exist_ok=False)
    here = CONFIG
    tools, source = root / "tools", root / "fse-source"
    tools.mkdir()
    execute = partial(setup_command, allocated_at=allocated_at, contract=contract)
    execute("apt-get", "update")
    packages = ("build-essential", "pkg-config", "libssl-dev", "curl", "git", "ca-certificates")
    execute("apt-get", "install", "-y", *packages)
    execute(
        sys.executable, "-m", "pip", "install", "--prefix", tools, "uv==" + PINS["versions"]["uv"]
    )
    uv = tools / "bin/uv"
    environments = (
        (("main", "quality.lock.txt"),)
        if quality
        else (("main", "speed-main.lock.txt"), ("reference", "speed-reference.lock.txt"))
    )
    for name, lock in environments:
        destination = root / name
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
            here / lock,
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "--index-strategy=unsafe-best-match",
        )
        execute(uv, "pip", "check", "--python", destination / "bin/python")
    execute(
        "git",
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        "https://github.com/oborchers/Fast_Sentence_Embeddings.git",
        source,
    )
    execute("git", "-C", source, "checkout", "--detach", contract["fse_revision"])
    flags = ("--no-deps", "--no-build-isolation")
    execute(uv, "pip", "install", "--python", root / "main/bin/python", *flags, source)
    execute(uv, "pip", "check", "--python", root / "main/bin/python")
    if quality:
        atomic(
            root / "setup.json",
            {
                "locks": {"quality.lock.txt": sha256(CONFIG / "quality.lock.txt")},
                "fse_revision": contract["fse_revision"],
                "python": sys.version,
                "base_image": contract["machine"]["image"],
            },
        )
        return
    installer = tools / "rustup-init"
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
    atomic(
        root / "setup.json",
        dict(
            rustup_installer_sha256=sha256(installer),
            locks={
                name: sha256(here / name)
                for name in ("speed-main.lock.txt", "speed-reference.lock.txt", "rust.lock")
            },
            fse_revision=contract["fse_revision"],
            python=sys.version,
            base_image=contract["machine"]["image"],
            system_compiler=subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
        ),
    )


def bootstrap(entry, revision):
    import re

    if (
        entry not in ("speed", "quality")
        or not revision
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError("bootstrap requires an entry and full immutable source commit")
    return (
        (CONFIG / "bootstrap.sh")
        .read_text()
        .replace("@ENTRY@", entry)
        .replace("@REVISION@", revision)
    )


def utility(args, entry):
    import json

    from _measurement_collect import collect
    from _measurement_config import contract as load_contract

    if args.collect:
        if not args.archive:
            raise ValueError("--collect requires --archive")
        print(json.dumps(collect(args.collect, args.archive)))
        return True
    if args.bootstrap:
        print(bootstrap(entry, args.source_revision))
        return True
    if args.describe:
        print(json.dumps(load_contract(args.input), indent=2))
        return True
    return False
