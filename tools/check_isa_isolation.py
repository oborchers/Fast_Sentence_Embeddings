"""Fail if objdump finds ZMM or mask registers outside AVX-512 function symbols.

Inspect unstripped production objects before wheel packing removes local symbols.
The matching production wheel is also executed under SDE's Haswell profile.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

SYMBOL = re.compile(r"^\s*[0-9a-fA-F]+ <([^>]+)>:$", re.MULTILINE)
INSTRUCTION = re.compile(r"^\s*[0-9a-fA-F]+:\s+(.+)$")
REGISTER = re.compile(r"\b(?:zmm\d+|k[0-7])\b", re.IGNORECASE)


def inspect(text: str) -> tuple[int, list[str]]:
    symbol = None
    advanced = 0
    violations = []
    for line in text.splitlines():
        header = SYMBOL.match(line)
        if header:
            symbol = header[1]
            continue
        instruction = INSTRUCTION.match(line)
        if instruction and REGISTER.search(instruction[1]):
            advanced += 1
            if symbol is None or "avx512" not in symbol:
                violations.append(f"{symbol}: {line.strip()}")
    return advanced, violations


def objdump() -> str:
    for name in ("llvm-objdump", "objdump"):
        found = shutil.which(name)
        if found:
            return found
    windows = Path("C:/Program Files/LLVM/bin/llvm-objdump.exe")
    if windows.is_file():
        return str(windows)
    raise RuntimeError("objdump or llvm-objdump is required to verify instruction isolation")


def check_paths(objects: list[Path], allow_stripped: bool = False) -> None:
    if not objects:
        raise RuntimeError("no native binary supplied for instruction isolation")
    tool = objdump()
    for path in objects:
        text = subprocess.check_output([tool, "-d", "--no-show-raw-insn", str(path)], text=True)
        advanced, violations = inspect(text)
        named = any("avx512" in match[1] for match in SYMBOL.finditer(text))
        # PE wheels normally omit local COFF symbols and their PDB. Object-level
        # MSVC proof and Haswell execution remain mandatory in the SDE job.
        if allow_stripped and sys.platform == "win32" and not named and advanced:
            print(
                f"{path}: UNVERIFIED per-symbol isolation: shipped PE has no static AVX-512 symbols"
            )
            return
        if not advanced or violations:
            raise RuntimeError(
                f"{path}: {advanced} advanced instructions, violations: {violations}"
            )
        print(f"{path}: {advanced} ZMM/mask instructions, all in AVX-512 symbols")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects", nargs="*")
    parser.add_argument("--installed-wheel", action="store_true")
    parser.add_argument("--allow-stripped-windows", action="store_true")
    args = parser.parse_args()
    objects = [Path(path) for path in args.objects]
    if args.installed_wheel:
        from fse import _native

        if "avx512" not in _native._compiled_isas():
            print("Installed wheel has no AVX-512 variant; register isolation is not applicable")
            return
        objects = [Path(_native.__file__)]
    check_paths(objects, args.allow_stripped_windows)


if __name__ == "__main__":
    main()
