/* Raw-register detection, shared by production readers and synthetic test snapshots. */
#ifndef FSE_NATIVE_KERNEL_ISA_H
#define FSE_NATIVE_KERNEL_ISA_H
#include "_native_kernel_base.h"

typedef struct {
    uint32_t max_leaf, leaf1_ecx, leaf7_ebx, leaf7_ecx, vendor, family;
    uint64_t xcr0;
} fse_cpu_features;

static int fse_avx2_eligible(fse_cpu_features r) {
    const uint32_t required = (1u << 12) | (1u << 27) | (1u << 28) | (1u << 29);
    return r.max_leaf >= 7 && (r.leaf1_ecx & required) == required &&
           (r.xcr0 & 6u) == 6u && (r.leaf7_ebx & (1u << 5)) != 0;
}
static int fse_avx512_eligible(fse_cpu_features r, int compiled) {
    const uint32_t required = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
    return compiled && fse_avx2_eligible(r) && (r.xcr0 & 0xE6u) == 0xE6u &&
           (r.leaf7_ebx & required) == required;
}
/* Empty until a full optimization decision approves a CPU family. */
static int fse_avx512_preferred(fse_cpu_features r) { (void)r; return 0; }

static fse_cpu_features fse_cpu_snapshot;
static int fse_cpu_initialized;
#if defined(FSE_TEST_TILES)
static int fse_cpu_read_count;
#endif

static fse_cpu_features fse_read_cpu(void) {
#if defined(FSE_TEST_TILES)
    ++fse_cpu_read_count;
#endif
    fse_cpu_features r = {0};
#if defined(FSE_X86)
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    r.max_leaf = (uint32_t)info[0];
    r.vendor = (uint32_t)info[1];
    if (r.max_leaf < 1) return r;
    __cpuidex(info, 1, 0);
    uint32_t signature = (uint32_t)info[0];
    r.leaf1_ecx = (uint32_t)info[2];
#else
    uint32_t a, b, c, d;
    __cpuid_count(0, 0, a, b, c, d);
    r.max_leaf = a;
    r.vendor = b;
    if (r.max_leaf < 1) return r;
    __cpuid_count(1, 0, a, b, c, d);
    uint32_t signature = a;
    r.leaf1_ecx = c;
#endif
    r.family = (signature >> 8) & 15u;
    if (r.family == 15u) r.family += (signature >> 20) & 255u;
    if (r.max_leaf < 7) return r;
#if defined(_MSC_VER)
    __cpuidex(info, 7, 0);
    r.leaf7_ebx = (uint32_t)info[1];
    r.leaf7_ecx = (uint32_t)info[2];
#else
    __cpuid_count(7, 0, a, b, c, d);
    r.leaf7_ebx = b;
    r.leaf7_ecx = c;
#endif
    if (r.leaf1_ecx & (1u << 27)) {
#if defined(_MSC_VER)
        r.xcr0 = _xgetbv(0);
#else
        uint32_t lo, hi;
        __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        r.xcr0 = ((uint64_t)hi << 32) | lo;
#endif
    }
#endif
    return r;
}
/* Called once under the GIL at extension import, before kernels become reachable. */
static void fse_initialize_cpu(void) {
    if (!fse_cpu_initialized) {
        fse_cpu_snapshot = fse_read_cpu();
        fse_cpu_initialized = 1;
    }
}
static int fse_test_cpu_reads(void) {
#if defined(FSE_TEST_TILES)
    return fse_cpu_read_count;
#else
    return -1;
#endif
}
static int fse_isa_compiled(int isa) {
    if (isa == FSE_ISA_PORTABLE) return 1;
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) return 1;
#endif
#if defined(FSE_AVX512)
    if (isa == FSE_ISA_AVX512) return 1;
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) return 1;
#endif
    return 0;
}
static int fse_has_avx2(void) {
    return fse_isa_compiled(FSE_ISA_AVX2) && fse_avx2_eligible(fse_cpu_snapshot);
}
static int fse_has_isa(int isa) {
    if (!fse_isa_compiled(isa)) return 0;
    if (isa == FSE_ISA_AVX2) return fse_has_avx2();
    if (isa == FSE_ISA_AVX512)
        return fse_avx512_eligible(fse_cpu_snapshot, fse_isa_compiled(isa));
    return 1;
}
static int fse_default_isa(void) {
    fse_cpu_features r = fse_cpu_snapshot;
    if (fse_avx512_preferred(r) && fse_has_isa(FSE_ISA_AVX512)) return FSE_ISA_AVX512;
    if (fse_has_avx2()) return FSE_ISA_AVX2;
    return fse_isa_compiled(FSE_ISA_NEON) ? FSE_ISA_NEON : FSE_ISA_PORTABLE;
}
/* Production has no synthetic detection hook. */
static int fse_test_isa_detection(fse_cpu_features r, int compiled) {
#if defined(FSE_TEST_TILES)
    return fse_avx2_eligible(r) | (fse_avx512_eligible(r, compiled) << 1) |
           (fse_avx512_preferred(r) << 2);
#else
    (void)r; (void)compiled; return -1;
#endif
}
#endif
