/* Compile the production thread runner and atomic protocol, without Python. */
#include <assert.h>
#include <string.h>
#include "_native_threads.h"

#define ITEMS 257
#define WORKERS 8
#define CALLS 4

typedef struct {
    fse_atomic next, failed_at;
    fse_atomic visits[ITEMS];
    fse_team_slot slots[WORKERS];
    unsigned failed_starts;
    int failure;
} call_state;

typedef struct { call_state calls[CALLS]; } concurrent_state;

static void worker(void *arg, int slot) {
    call_state *call = (call_state *)arg;
    int64_t i;
    void *owned = fse_join_calloc(1, 16);
    (void)slot;
    assert(owned);
    for (;;) {
        i = fse_atomic_add(&call->next, 1);
        if (i >= ITEMS || i > fse_atomic_load(&call->failed_at)) break;
        assert(fse_atomic_add(&call->visits[i], 1) == 0);
        if (i == call->failure) fse_atomic_min(&call->failed_at, i);
    }
    fse_join_free(owned);
}

static void run_call(void *arg, int index) {
    concurrent_state *state = (concurrent_state *)arg;
    call_state *call = &state->calls[index];
    int i, prefix;
    fse_atomic_store(&call->next, 0);
    fse_atomic_store(&call->failed_at, ITEMS);
    fse_team(WORKERS, worker, call, call->slots, call->failed_starts);
    assert(fse_team_balanced(WORKERS, call->slots));
    prefix = call->failure < 0 ? ITEMS : call->failure;
    assert(fse_atomic_load(&call->failed_at) == prefix);
    for (i = 0; i < prefix; i++) assert(fse_atomic_load(&call->visits[i]) == 1);
    for (i = prefix; i < ITEMS; i++) assert(fse_atomic_load(&call->visits[i]) <= 1);
}

int main(void) {
    concurrent_state state;
    const unsigned masks[] = {0, 2, 10, 254};
    const int failures[] = {-1, 0, 128, 256};
    int round, i;
    for (round = 0; round < 32; round++) {
        memset(&state, 0, sizeof(state));
        for (i = 0; i < CALLS; i++) {
            state.calls[i].failed_starts = masks[(round + i) % 4];
            state.calls[i].failure = failures[(round / 4 + i) % 4];
        }
        assert(fse_parallel(CALLS, run_call, &state) == 0);
        assert(fse_atomic_load(&fse_test_join_live) == 0);
        assert(fse_atomic_load(&fse_test_team_live) == 0);
    }
    return 0;
}
