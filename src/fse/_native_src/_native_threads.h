/* Native threads for a parallel loop the calling thread waits for (_native_lists.pxi and _native_strings.pxi).
 *
 * fse_parallel(n, fn, arg) runs fn(arg, 0) to fn(arg, n - 1), part 0 in the calling
 * thread and the others in threads of their own, and returns when all have ended. A
 * thread that cannot be started runs its part in the calling thread afterwards, so every
 * part runs exactly once. The parts must not touch the Python C API: the caller keeps
 * holding the GIL while it waits, so no Python code runs meanwhile.
 */
#ifndef FSE_NATIVE_THREADS_H
#define FSE_NATIVE_THREADS_H

#include <stdlib.h>
#include "_native_atomics.h"

typedef void (*fse_part)(void *arg, int part);

/* For the tests: bit i set makes the start of the thread of part i fail. */
static unsigned fse_test_failed_starts = 0;
/* Test-only failure before bookkeeping allocation; no part runs on failure. */
static int fse_test_fail_bookkeeping = 0;

/* Actual live native allocations for join parts and parallel bookkeeping. Only
 * the calling thread allocates/frees these while holding the GIL. Never reset
 * this counter between calls: a missed free must remain visible to the tests. */
static fse_atomic fse_test_join_live = {0};
static fse_atomic fse_test_team_live = {0};

static void *fse_join_calloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (p) fse_atomic_add(&fse_test_join_live, 1);
    return p;
}

static void fse_join_free(void *p) {
    if (p) {
        free(p);
        fse_atomic_add(&fse_test_join_live, -1);
    }
}

typedef struct {
    fse_part fn;
    void *arg;
    int part;
} fse_task;

#ifdef _WIN32
#include <windows.h>
#include <process.h>
typedef HANDLE fse_thread;

static unsigned __stdcall fse_thread_main(void *p) {
    fse_task *t = (fse_task *)p;
    t->fn(t->arg, t->part);
    return 0;
}

static int fse_start(fse_thread *thread, fse_task *task) {
    *thread = (HANDLE)_beginthreadex(NULL, 0, fse_thread_main, task, 0, NULL);
    return *thread ? 0 : -1;
}

static void fse_join(fse_thread thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}
#else
#include <pthread.h>
typedef pthread_t fse_thread;

static void *fse_thread_main(void *p) {
    fse_task *t = (fse_task *)p;
    t->fn(t->arg, t->part);
    return NULL;
}

static int fse_start(fse_thread *thread, fse_task *task) {
    return pthread_create(thread, NULL, fse_thread_main, task) ? -1 : 0;
}

static void fse_join(fse_thread thread) { pthread_join(thread, NULL); }
#endif

/* Caller-owned storage. Failed starts reduce the team; slot zero drains the queue.
 * Separate started/joined handles let each call check its own resource balance. */
typedef struct {
    fse_thread thread, joined_thread;
    fse_task task;
    int started, joined;
} fse_team_slot;

static void fse_team(int n, fse_part fn, void *arg, fse_team_slot *slots,
                     unsigned failed_starts) {
    int i;
    for (i = 1; i < n; i++) {
        slots[i].task.fn = fn;
        slots[i].task.arg = arg;
        slots[i].task.part = i;
        slots[i].started = !(i < 32 && (failed_starts >> i) & 1u) &&
                           fse_start(&slots[i].thread, &slots[i].task) == 0;
    }
    fn(arg, 0);
    for (i = 1; i < n; i++) {
        if (slots[i].started) {
            fse_join(slots[i].thread);
            slots[i].joined_thread = slots[i].thread;
            slots[i].joined = 1;
        }
    }
}

static int fse_team_balanced(int n, fse_team_slot *slots) {
    int i;
    for (i = 1; i < n; i++) {
        if (slots[i].started != slots[i].joined) return 0;
        if (!slots[i].started) continue;
#ifdef _WIN32
        if (slots[i].thread != slots[i].joined_thread) return 0;
#else
        if (!pthread_equal(slots[i].thread, slots[i].joined_thread)) return 0;
#endif
    }
    return 1;
}

/* 0, or -1 if the bookkeeping could not be allocated (then nothing ran). */
static int fse_parallel(int n, fse_part fn, void *arg) {
    fse_thread *threads;
    fse_task *tasks;
    char *started;
    int i;
    if (fse_test_fail_bookkeeping) return -1;
    if (n <= 1) {
        if (n == 1) fn(arg, 0);
        return 0;
    }
    threads = (fse_thread *)fse_join_calloc((size_t)n, sizeof(fse_thread));
    tasks = (fse_task *)fse_join_calloc((size_t)n, sizeof(fse_task));
    started = (char *)fse_join_calloc((size_t)n, 1);
    if (!threads || !tasks || !started) {
        fse_join_free(threads);
        fse_join_free(tasks);
        fse_join_free(started);
        return -1;
    }
    for (i = 1; i < n; i++) {
        tasks[i].fn = fn;
        tasks[i].arg = arg;
        tasks[i].part = i;
        started[i] = !(i < 32 && (fse_test_failed_starts >> i) & 1u) &&
                     fse_start(&threads[i], &tasks[i]) == 0;
    }
    fn(arg, 0);
    for (i = 1; i < n; i++) {
        if (started[i])
            fse_join(threads[i]);
        else
            fn(arg, i);
    }
    fse_join_free(threads);
    fse_join_free(tasks);
    fse_join_free(started);
    return 0;
}

#endif
