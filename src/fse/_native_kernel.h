/* The weighted mean kernel of fse 2.0 (docs/architecture/numerical-contract.md, "Numerical contract (`transform_version` 2)").

   Author: Oliver Borchers. Licensed under GNU General Public License v3.0.

   Every contribution does acc[i] = fma(f, x[i], acc[i]) in token order, with the factor
   f = weight (times the scale for int8 rows) and the stored row x read exactly; a
   sentence finishes as (1/n) * acc with 1/n rounded first; an empty sentence is the
   zero row. Lanes of a vector are independent elements, so vectorizing over the
   dimension keeps every element's sequence of fused operations: every instruction set
   gives the bits of fse._reference. Nothing else may contract a multiply and an add,
   so this file is built with -ffp-contract=off (or /fp:precise) and refuses fast math. */

#ifndef FSE_NATIVE_KERNEL_H
#define FSE_NATIVE_KERNEL_H

#include "_native_kernel_base.h"
#include "_native_kernel_mean.h"
#include "_native_kernel_steps.h"
#include "_native_kernel_store.h"
#include "_native_kernel_search.h"
#include "_native_kernel_head.h"

/* For the tests: set flush-to-zero and denormals-are-zero (FZ on arm64) and rounding
   toward +infinity, each if asked, and return the previous environment. */
static uint64_t fse_test_fpenv_set(int flush, int round_up) {
    fse_fpenv env = {0};
#if defined(FSE_X86)
    unsigned int csr = _mm_getcsr();
    env.saved = csr;
    csr &= ~(0x8000u | 0x0040u | 0x6000u);
    if (flush) csr |= 0x8000u | 0x0040u;
    if (round_up) csr |= 0x4000u;
    _mm_setcsr(csr);
#elif defined(FSE_NEON)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    env.saved = fpcr;
    uint64_t want = fpcr & ~((1ull << 24) | (3ull << 22));
    if (flush) want |= 1ull << 24;
    if (round_up) want |= 1ull << 22;
    __asm__ __volatile__("msr fpcr, %0" : : "r"(want));
#else
    (void)flush;
    (void)round_up;
#endif
    return env.saved;
}
/* The environment of the kernel for a whole Python call, numpy conversions included. */
static uint64_t fse_fpenv_push(void) { return fse_fpenv_enter().saved; }
static void fse_fpenv_pop(uint64_t saved) {
    fse_fpenv env;
    env.saved = saved;
    fse_fpenv_leave(env);
}

static void fse_test_fpenv_restore(uint64_t saved) {
    fse_fpenv env;
    env.saved = saved;
    fse_fpenv_leave(env);
}
static uint64_t fse_test_fpenv_get(void) {
#if defined(FSE_X86)
    return _mm_getcsr();
#elif defined(FSE_NEON)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr;
#else
    return 0;
#endif
}

/* --- Instruction sets ---------------------------------------------------------------- */

static int fse_has_avx2(void) {
#if defined(FSE_X86)
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 1);
    int fma = (info[2] >> 12) & 1, osxsave = (info[2] >> 27) & 1, f16c = (info[2] >> 29) & 1;
    if (!(fma && osxsave && f16c)) return 0;
    if ((_xgetbv(0) & 6) != 6) return 0; /* the OS saves the ymm state */
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
#else
    __builtin_cpu_init();
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    int f16c = __get_cpuid(1, &eax, &ebx, &ecx, &edx) && ((ecx >> 29) & 1u);
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && f16c;
#endif
#else
    return 0;
#endif
}

static int fse_has_isa(int isa) {
    if (isa == FSE_ISA_PORTABLE) return 1;
    if (isa == FSE_ISA_AVX2) return fse_has_avx2();
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) return 1;
#endif
    return 0;
}

typedef int64_t (*fse_loop_t)(const fse_params *, const fse_batch *, float *, int64_t *, int);

static fse_loop_t fse_loop_for(int isa, int dtype) {
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2)
        return dtype == FSE_F32 ? fse_loop_f32_avx2 : dtype == FSE_F16 ? fse_loop_f16_avx2 : fse_loop_i8_avx2;
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON)
        return dtype == FSE_F32 ? fse_loop_f32_neon : dtype == FSE_F16 ? fse_loop_f16_neon : fse_loop_i8_neon;
#endif
    return dtype == FSE_F32 ? fse_loop_f32_portable : dtype == FSE_F16 ? fse_loop_f16_portable : fse_loop_i8_portable;
}

static int64_t fse_run(int isa, const fse_params *p, const fse_batch *b, float *out,
                       int64_t *counts, int finish) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_loop_for(isa, p->dtype)(p, b, out, counts, finish);
    fse_fpenv_leave(env);
    return bad;
}

#endif /* FSE_NATIVE_KERNEL_H */
