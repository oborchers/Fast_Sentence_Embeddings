"""Fail if objdump finds ZMM or mask registers outside AVX-512 function symbols.

Inspect unstripped production objects before wheel packing removes local symbols.
The matching production wheel is also executed under SDE's Haswell profile.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

SYMBOL = re.compile(r"^\s*[0-9a-fA-F]+ <([^>]+)>:$")
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


def main() -> None:
    objects = [Path(path) for path in sys.argv[1:]] if len(sys.argv) > 1 else [
        *Path("build").rglob("_native.o"), *Path("build").rglob("_native.obj")
    ]
    if not objects:
        raise RuntimeError("no unstripped production native object was retained")
    tool = objdump()
    for path in objects:
        text = subprocess.check_output([tool, "-d", "--no-show-raw-insn", str(path)], text=True)
        advanced, violations = inspect(text)
        if not advanced or violations:
            raise RuntimeError(
                f"{path}: {advanced} advanced instructions, violations: {violations}"
            )
        print(f"{path}: {advanced} ZMM/mask instructions, all in AVX-512 symbols")


if __name__ == "__main__":
    main()
