/* The byte trie of a wordpiece@1 vocabulary for the greedy match (docs/architecture/tokenizers.md, "wordpiece@1"): a static double array built once at load, never persisted.

   A transition from node s by byte c goes to t = base[s] + c when 0 <= t < size and
   check[t] == s; value[t] is the id of the token that ends at t, or FSE_TRIE_NONE. Node 0
   is the root (check[0] = -2, so no transition leads to it); a free unit has check -1.
   The build first inserts the tokens into a temporary trie of linked children, then places
   its nodes breadth first, each at the first base where all its children are free
   (first-fit over the free units of the last blocks, as darts-clone does). All indices are int64_t and checked against the capacity
   before every read and write; the units are int32_t, so the double array stops at
   FSE_TRIE_MAX_UNITS (or a smaller limit, for the tests). */
#ifndef FSE_NATIVE_WORDPIECE_TRIE_H
#define FSE_NATIVE_WORDPIECE_TRIE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FSE_TRIE_NONE 0xFFFFFFFFu
#define FSE_TRIE_MAX_UNITS 0x7FFFFFFFll
#define FSE_TRIE_OK 0
#define FSE_TRIE_NO_MEMORY 1
#define FSE_TRIE_TOO_LARGE 2

typedef struct {
    int32_t *base;
    int32_t *check;
    uint32_t *value;
    int64_t size;  /* units allocated */
    /* The allocations of the build: calls so far, blocks held, and the call that fails
       (-1 for none; a private switch of the tests). */
    int64_t calls, live, fail_at;
} fse_trie;

/* realloc (malloc for p NULL) of the build, counted; NULL for the call fail_at. */
static void *fse_trie_realloc(fse_trie *t, void *p, size_t n) {
    if (t->calls++ == t->fail_at) return NULL;
    void *q = realloc(p, n);
    if (q && !p) t->live++;
    return q;
}

static void fse_trie_release(fse_trie *t, void *p) {
    if (!p) return;
    t->live--;
    free(p);
}

/* The temporary trie of the build: children as a linked list. */
typedef struct {
    int32_t child, sibling;
    uint32_t value;
    uint8_t label;
} fse_trie_node;

typedef struct {
    fse_trie_node *nodes;
    int64_t count, capacity;
} fse_trie_nodes;

static void fse_trie_free(fse_trie *t) {
    fse_trie_release(t, t->base);
    fse_trie_release(t, t->check);
    fse_trie_release(t, t->value);
    t->base = t->check = NULL;
    t->value = NULL;
    t->size = 0;
}

/* A new node of the temporary trie, or -1 when it cannot grow. */
static int64_t fse_trie_new_node(fse_trie *t, fse_trie_nodes *a, uint8_t label, int64_t limit) {
    if (a->count == a->capacity) {
        int64_t want = a->capacity ? 2 * a->capacity : 1024;
        if (want > limit) want = limit;
        if (a->count >= want || (uint64_t)want > SIZE_MAX / sizeof(fse_trie_node)) return -1;
        fse_trie_node *grown = (fse_trie_node *)fse_trie_realloc(t, a->nodes, (size_t)want * sizeof(fse_trie_node));
        if (!grown) return -1;
        a->nodes = grown;
        a->capacity = want;
    }
    fse_trie_node *node = &a->nodes[a->count];
    node->child = node->sibling = -1;
    node->value = FSE_TRIE_NONE;
    node->label = label;
    return a->count++;
}

/* Insert key[0:n] with the id value; returns 0, or an FSE_TRIE_ error. */
static int fse_trie_insert(fse_trie *t, fse_trie_nodes *a, const uint8_t *key, int64_t n, uint32_t value,
                           int64_t limit) {
    int64_t s = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t c = a->nodes[s].child;
        while (c >= 0 && a->nodes[c].label != key[i]) c = a->nodes[c].sibling;
        if (c < 0) {
            c = fse_trie_new_node(t, a, key[i], limit);
            if (c < 0) return a->count >= limit ? FSE_TRIE_TOO_LARGE : FSE_TRIE_NO_MEMORY;
            a->nodes[c].sibling = a->nodes[s].child;
            a->nodes[s].child = (int32_t)c;
        }
        s = c;
    }
    if (a->nodes[s].value == FSE_TRIE_NONE) a->nodes[s].value = value;  /* the first id wins */
    return FSE_TRIE_OK;
}

/* Grow the allocation of the double array (and of the free list) to at least want units:
   free units have check -1, base 0 and no value. */
static int fse_trie_reserve(fse_trie *t, int32_t **next, int32_t **prev, int64_t want,
                            int64_t limit) {
    if (want <= t->size) return FSE_TRIE_OK;
    if (want > limit) return FSE_TRIE_TOO_LARGE;
    int64_t size = t->size ? t->size : 1024;
    while (size < want) size = size > limit / 2 ? limit : 2 * size;
    if ((uint64_t)size > SIZE_MAX / sizeof(int32_t)) return FSE_TRIE_NO_MEMORY;
    size_t bytes = (size_t)size * sizeof(int32_t);
    int32_t *base = (int32_t *)fse_trie_realloc(t, t->base, bytes);
    if (base) t->base = base;
    int32_t *check = (int32_t *)fse_trie_realloc(t, t->check, bytes);
    if (check) t->check = check;
    uint32_t *value = (uint32_t *)fse_trie_realloc(t, t->value, bytes);
    if (value) t->value = value;
    int32_t *n = (int32_t *)fse_trie_realloc(t, *next, bytes);
    if (n) *next = n;
    int32_t *p = (int32_t *)fse_trie_realloc(t, *prev, bytes);
    if (p) *prev = p;
    if (!base || !check || !value || !n || !p) return FSE_TRIE_NO_MEMORY;
    for (int64_t k = t->size; k < size; k++) {
        t->base[k] = 0;
        t->check[k] = -1;
        t->value[k] = FSE_TRIE_NONE;
    }
    t->size = size;
    return FSE_TRIE_OK;
}

/* The units of the build in blocks of 256, as darts-clone keeps them: the free units of the
   last FSE_TRIE_OPEN_BLOCKS blocks in a circular list; older blocks are closed, so a search
   for a base never looks at more than their units. */
#define FSE_TRIE_OPEN_BLOCKS 16
typedef struct {
    int32_t *next, *prev;
    int64_t head;    /* a free unit of the open blocks, or -1 */
    int64_t used;    /* the units in blocks */
    int64_t closed;  /* the units below are in closed blocks */
} fse_trie_units;

static void fse_trie_unlink(fse_trie_units *f, int64_t u) {
    if (f->next[u] == u) {
        f->head = -1;
        return;
    }
    f->next[f->prev[u]] = f->next[u];
    f->prev[f->next[u]] = f->prev[u];
    if (f->head == u) f->head = f->next[u];
}

/* Unit u (free, in a block) now belongs to node owner. */
static void fse_trie_occupy(fse_trie *t, fse_trie_units *f, int64_t u, int64_t owner) {
    t->check[u] = (int32_t)owner;
    if (u >= f->closed) fse_trie_unlink(f, u);
}

/* One more block of free units at the end; closes the oldest open block beyond the limit. */
static int fse_trie_add_block(fse_trie *t, fse_trie_units *f, int64_t limit) {
    int64_t begin = f->used, end = begin + 256;
    int err = fse_trie_reserve(t, &f->next, &f->prev, end + 256, limit);
    if (err) return err;
    for (int64_t u = begin; u < end; u++) {
        if (f->head < 0) {
            f->head = f->next[u] = f->prev[u] = (int32_t)u;
            continue;
        }
        int64_t last = f->prev[f->head];
        f->next[last] = (int32_t)u;
        f->prev[u] = (int32_t)last;
        f->next[u] = (int32_t)f->head;
        f->prev[f->head] = (int32_t)u;
    }
    f->used = end;
    while (f->used - f->closed > FSE_TRIE_OPEN_BLOCKS * 256) {
        for (int64_t u = f->closed; u < f->closed + 256; u++)
            if (t->check[u] == -1) fse_trie_unlink(f, u);
        f->closed += 256;
    }
    return FSE_TRIE_OK;
}

/* Whether all children of temporary node s fit at base b (a unit beyond the blocks is free;
   the allocation holds 256 units more than the blocks). */
static int fse_trie_fits(const fse_trie *t, const fse_trie_nodes *a, int64_t s, int64_t b) {
    for (int32_t c = a->nodes[s].child; c >= 0; c = a->nodes[c].sibling)
        if (t->check[b + a->nodes[c].label] != -1) return 0;
    return 1;
}

/* Place the children of temporary node s, which sits at unit at: the first free unit f of
   the open blocks for which every child unit f - low + label is free, else a new block.
   Appends the placed children (temporary node, unit) to the queue. */
static int fse_trie_place(fse_trie *t, fse_trie_units *f, const fse_trie_nodes *a, int64_t s,
                          int64_t at, int64_t *queue, int64_t *tail, int64_t limit) {
    int32_t first = a->nodes[s].child;
    if (first < 0) return FSE_TRIE_OK;
    int64_t low = 255, high = 0, b = -1;
    for (int32_t c = first; c >= 0; c = a->nodes[c].sibling) {
        if (a->nodes[c].label < low) low = a->nodes[c].label;
        if (a->nodes[c].label > high) high = a->nodes[c].label;
    }
    if (f->head >= 0) {
        int64_t u = f->head;
        do {
            if (u - low >= 1 && fse_trie_fits(t, a, s, u - low)) {
                b = u - low;
                break;
            }
            u = f->next[u];
        } while (u != f->head);
    }
    if (b < 0) b = f->used;  /* the next block is free */
    while (b + high >= f->used) {
        int err = fse_trie_add_block(t, f, limit);
        if (err) return err;
    }
    t->base[at] = (int32_t)b;
    for (int32_t c = first; c >= 0; c = a->nodes[c].sibling) {
        int64_t u = b + a->nodes[c].label;
        fse_trie_occupy(t, f, u, at);
        t->value[u] = a->nodes[c].value;
        queue[2 * *tail] = c;
        queue[2 * *tail + 1] = u;
        (*tail)++;
    }
    return FSE_TRIE_OK;
}

/* Build the trie of the count keys bytes[offsets[k]:offsets[k + 1]], key k with the id k,
   in at most limit units. Returns 0, or an FSE_TRIE_ error and frees all it allocated. */
static int fse_trie_build(fse_trie *t, const uint8_t *bytes, const int64_t *offsets,
                          int64_t count, int64_t limit) {
    fse_trie_nodes a = {NULL, 0, 0};
    fse_trie_units f = {NULL, NULL, -1, 0, 0};
    int64_t *queue = NULL, head = 0, tail = 0;
    int err = fse_trie_new_node(t, &a, 0, limit) < 0 ? FSE_TRIE_NO_MEMORY : FSE_TRIE_OK;
    for (int64_t k = 0; k < count && !err; k++)
        err = fse_trie_insert(t, &a, bytes + offsets[k], offsets[k + 1] - offsets[k], (uint32_t)k, limit);
    if (!err && (uint64_t)a.count > SIZE_MAX / (2 * sizeof(int64_t))) err = FSE_TRIE_NO_MEMORY;
    if (!err) {
        queue = (int64_t *)fse_trie_realloc(t, NULL, (size_t)a.count * 2 * sizeof(int64_t));
        if (!queue) err = FSE_TRIE_NO_MEMORY;
    }
    if (!err) err = fse_trie_add_block(t, &f, limit);
    if (!err) {
        fse_trie_occupy(t, &f, 0, -2);  /* the root: no transition leads to it */
        t->value[0] = a.nodes[0].value;
        queue[0] = 0;
        queue[1] = 0;
        tail = 1;
    }
    while (!err && head < tail) {
        int64_t s = queue[2 * head], at = queue[2 * head + 1];
        head++;
        err = fse_trie_place(t, &f, &a, s, at, queue, &tail, limit);
    }
    fse_trie_release(t, a.nodes);
    fse_trie_release(t, queue);
    fse_trie_release(t, f.next);
    fse_trie_release(t, f.prev);
    if (err) {
        fse_trie_free(t);
        return err;
    }
    /* Down to the units in blocks; a failed shrink keeps the larger arrays. */
    size_t kept = (size_t)f.used * sizeof(int32_t);
    int32_t *base = (int32_t *)fse_trie_realloc(t, t->base, kept);
    if (base) t->base = base;
    int32_t *check = (int32_t *)fse_trie_realloc(t, t->check, kept);
    if (check) t->check = check;
    uint32_t *value = (uint32_t *)fse_trie_realloc(t, t->value, kept);
    if (value) t->value = value;
    t->size = f.used;
    return FSE_TRIE_OK;
}

/* The node of key[0:n] from node s, or -1. */
static inline int64_t fse_trie_walk(const fse_trie *t, int64_t s, const uint8_t *key, int64_t n) {
    for (int64_t i = 0; i < n && s >= 0; i++) {
        int64_t u = (int64_t)t->base[s] + key[i];
        s = (u < t->size && t->check[u] == s) ? u : -1;
    }
    return s;
}

/* The greedy step: the id of the longest token w[k:j] read from node s (the root, or the
   node of the continuation prefix) with j > k on a code-point boundary of w[0:n] (j == n,
   or w[j] is no continuation byte), and *end = j; FSE_TRIE_NONE when there is none. */
static inline uint32_t fse_trie_longest(const fse_trie *t, int64_t s, const uint8_t *w,
                                        int64_t k, int64_t n, int64_t *end) {
    uint32_t best = FSE_TRIE_NONE;
    for (int64_t i = k; i < n; i++) {
        int64_t u = (int64_t)t->base[s] + w[i];
        if (u >= t->size || t->check[u] != s) break;
        s = u;
        if (t->value[s] != FSE_TRIE_NONE && (i + 1 == n || (w[i + 1] & 0xC0) != 0x80)) {
            best = t->value[s];
            *end = i + 1;
        }
    }
    return best;
}

#endif /* FSE_NATIVE_WORDPIECE_TRIE_H */
