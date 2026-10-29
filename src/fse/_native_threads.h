/* Native threads for a parallel loop the calling thread waits for (_native_lists.pxi).
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

typedef void (*fse_part)(void *arg, int part);

/* For the tests: bit i set makes the start of the thread of part i fail. */
static unsigned fse_test_failed_starts = 0;

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

/* 0, or -1 if the bookkeeping could not be allocated (then nothing ran). */
static int fse_parallel(int n, fse_part fn, void *arg) {
    fse_thread *threads;
    fse_task *tasks;
    char *started;
    int i;
    if (n <= 1) {
        if (n == 1) fn(arg, 0);
        return 0;
    }
    threads = (fse_thread *)calloc((size_t)n, sizeof(fse_thread));
    tasks = (fse_task *)calloc((size_t)n, sizeof(fse_task));
    started = (char *)calloc((size_t)n, 1);
    if (!threads || !tasks || !started) {
        free(threads);
        free(tasks);
        free(started);
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
    free(threads);
    free(tasks);
    free(started);
    return 0;
}

#endif
