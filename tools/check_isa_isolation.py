"""Fail if objdump finds ZMM or mask registers outside AVX-512 function symbols.

Inspect unstripped production objects before wheel packing removes local symbols.
The matching production wheel is also executed under SDE's Haswell profile.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SYMBOL = re.compile(r"^\s*[0-9a-fA-F]+ <([^>]+)>:$", re.MULTILINE)
INSTRUCTION = re.compile(r"^\s*([0-9a-fA-F]+):\s+(.+)$")
REGISTER = re.compile(r"\b(?:zmm\d+|k[0-7])\b", re.IGNORECASE)


def inspect(text: str, symbols: dict[int, str] | None = None) -> tuple[int, list[str]]:
    symbol = None
    advanced = 0
    violations = []
    for line in text.splitlines():
        header = SYMBOL.match(line)
        if header:
            symbol = header[1]
            continue
        instruction = INSTRUCTION.match(line)
        if instruction and REGISTER.search(instruction[2]):
            advanced += 1
            if symbols is not None:
                symbol = symbols.get(int(instruction[1], 16))
            if symbol is None or "avx512" not in symbol:
                violations.append(f"{symbol}: {line.strip()}")
    return advanced, violations


def find_tool(*names: str) -> str:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
        windows = Path(f"C:/Program Files/LLVM/bin/{name}.exe")
        if windows.is_file():
            return str(windows)
    raise RuntimeError(f"required LLVM/binutils tool not found: {names}")


def resolve_addresses(path: Path, text: str, pdb: Path) -> dict[int, str]:
    if not pdb.is_file():
        raise RuntimeError(f"production PDB missing: {pdb}")
    addresses = sorted(
        {
            int(match[1], 16)
            for line in text.splitlines()
            if (match := INSTRUCTION.match(line)) and REGISTER.search(match[2])
        }
    )
    result = subprocess.check_output(
        [find_tool("llvm-symbolizer"), "--output-style=JSON", "--no-inlines", f"--obj={path}"],
        input="".join(f"0x{address:x}\n" for address in addresses),
        text=True,
    )
    # LLVM reads the matching PDB from the PE debug directory. No newer --pdb option
    # is needed, so the CI proof also works with the runner's existing LLVM version.
    records = [json.loads(line) for line in result.splitlines() if line.strip()]
    return resolved_symbols(records, addresses)


def resolved_symbols(records: list[dict], addresses: list[int]) -> dict[int, str]:
    symbols = {}
    for record in records:
        address = int(record["Address"], 0)
        frames = record["Symbol"]
        name = frames[0]["FunctionName"] if len(frames) == 1 else ""
        if address in symbols or not name or name == "??":
            raise RuntimeError(f"unresolved or ambiguous production address: {address:x}")
        symbols[address] = name
    if set(symbols) != set(addresses):
        raise RuntimeError("symbolizer omitted or added production instruction addresses")
    return symbols


def check_paths(objects: list[Path], allow_stripped: bool = False, pdb: Path | None = None) -> None:
    if not objects:
        raise RuntimeError("no native binary supplied for instruction isolation")
    tool = find_tool("llvm-objdump", "objdump")
    for path in objects:
        text = subprocess.check_output([tool, "-d", "--no-show-raw-insn", str(path)], text=True)
        symbols = resolve_addresses(path, text, pdb) if pdb is not None else None
        advanced, violations = inspect(text, symbols)
        named = any("avx512" in match[1] for match in SYMBOL.finditer(text))
        # PE wheels normally omit local COFF symbols and their PDB. Object-level
        # MSVC proof and Haswell execution remain mandatory in the SDE job.
        if allow_stripped and pdb is None and sys.platform == "win32" and not named and advanced:
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
    parser.add_argument("--pdb", type=Path)
    parser.add_argument("--allow-stripped-windows", action="store_true")
    args = parser.parse_args()
    objects = [Path(path) for path in args.objects]
    if args.installed_wheel:
        _native = importlib.import_module("fse._native")

        if "avx512" not in _native._compiled_isas():
            print("Installed wheel has no AVX-512 variant; register isolation is not applicable")
            return
        objects = [Path(_native.__file__)]
    check_paths(objects, args.allow_stripped_windows, args.pdb)


if __name__ == "__main__":
    main()
