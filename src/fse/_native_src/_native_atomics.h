/* Signed, aligned 64-bit atomics shared by the task team and its test harness. */
#ifndef FSE_NATIVE_ATOMICS_H
#define FSE_NATIVE_ATOMICS_H
#include <stdint.h>
#ifdef _MSC_VER
#include <windows.h>
typedef __declspec(align(8)) struct { int64_t value; } fse_atomic;
static int64_t fse_atomic_load(fse_atomic *p) {
    return InterlockedCompareExchange64((volatile LONG64 *)&p->value, 0, 0);
}
static int64_t fse_atomic_add(fse_atomic *p, int64_t n) {
    return InterlockedExchangeAdd64((volatile LONG64 *)&p->value, n);
}
static void fse_atomic_store(fse_atomic *p, int64_t n) {
    InterlockedExchange64((volatile LONG64 *)&p->value, n);
}
static void fse_atomic_min(fse_atomic *p, int64_t n) {
    int64_t old = fse_atomic_load(p), seen;
    while (n < old) {
        seen = InterlockedCompareExchange64((volatile LONG64 *)&p->value, n, old);
        if (seen == old) break;
        old = seen;
    }
}
#else
typedef struct __attribute__((aligned(8))) { int64_t value; } fse_atomic;
static int64_t fse_atomic_load(fse_atomic *p) {
    return __atomic_load_n(&p->value, __ATOMIC_SEQ_CST);
}
static int64_t fse_atomic_add(fse_atomic *p, int64_t n) {
    return __atomic_fetch_add(&p->value, n, __ATOMIC_SEQ_CST);
}
static void fse_atomic_store(fse_atomic *p, int64_t n) {
    __atomic_store_n(&p->value, n, __ATOMIC_SEQ_CST);
}
static void fse_atomic_min(fse_atomic *p, int64_t n) {
    int64_t old = fse_atomic_load(p);
    while (n < old && !__atomic_compare_exchange_n(
               &p->value, &old, n, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
}
#endif
#endif
