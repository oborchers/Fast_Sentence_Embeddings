"""The production disassembly check fails closed on leaked advanced instructions."""

import runpy
from pathlib import Path

import pytest

INSPECT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/check_isa_isolation.py"))[
    "inspect"
]


@pytest.mark.parametrize("symbol", ["fse_loop_f32_avx512", "fse_loop_f32_avx512.constprop.0"])
def test_accepts_named_avx512_functions(symbol):
    text = f"0000 <{symbol}>:\n 0: vfmadd213ps %zmm0,%zmm1,%zmm2\n 4: kmovw %eax,%k1\n"
    assert INSPECT(text) == (2, [])


@pytest.mark.parametrize("symbol", ["fse_loop_f32_avx2", "__pyx_module_exec_fse__native", None])
@pytest.mark.parametrize("instruction", ["vmovups zmm0, [rax]", "kmovw k1, eax"])
def test_rejects_vector_or_mask_registers_outside_avx512(symbol, instruction):
    header = f"0000 <{symbol}>:\n" if symbol else ""
    advanced, violations = INSPECT(header + f" 0: {instruction}\n")
    assert advanced == 1
    assert len(violations) == 1


def test_tracks_symbol_boundaries_and_ignores_avx2_and_comments():
    text = """0000 <fse_loop_f32_avx512>:
 0: vfmadd213ps %zmm0,%zmm1,%zmm2
0010 <fse_loop_f32_avx2>:
 10: vmovups %ymm0,(%rax)
 14: kmovw %eax,%k2
"""
    advanced, violations = INSPECT(text)
    assert advanced == 2
    assert len(violations) == 1
    assert "fse_loop_f32_avx2" in violations[0]


def test_post_link_addresses_are_checked_against_resolved_functions():
    text = "0000 <.text>:\n 1000: vmovups zmm0, [rax]\n 1010: kmovw k1, eax\n"
    assert INSPECT(text, {0x1000: "fse_loop_f32_avx512", 0x1010: "fse_loop_f32_avx512"}) == (2, [])
    count, violations = INSPECT(text, {0x1000: "fse_loop_f32_avx512", 0x1010: "fse_run"})
    assert count == 2
    assert len(violations) == 1
    assert "fse_run" in violations[0]


@pytest.mark.parametrize("name", ["", "??"])
def test_unresolved_debug_symbols_fail_closed(name):
    resolve = INSPECT.__globals__["resolved_symbols"]
    with pytest.raises(RuntimeError, match="unresolved"):
        resolve([{"Address": "0x1000", "Symbol": [{"FunctionName": name}]}], [0x1000])


def test_debug_symbols_must_cover_exactly_the_disassembled_addresses():
    resolve = INSPECT.__globals__["resolved_symbols"]
    record = {"Address": "0x1000", "Symbol": [{"FunctionName": "fse_loop_f32_avx512"}]}
    assert resolve([record], [0x1000]) == {0x1000: "fse_loop_f32_avx512"}
    with pytest.raises(RuntimeError, match="omitted or added"):
        resolve([record], [0x1000, 0x1010])
    with pytest.raises(RuntimeError, match="ambiguous"):
        resolve([record, record], [0x1000])
