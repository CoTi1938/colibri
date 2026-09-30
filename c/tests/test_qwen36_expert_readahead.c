/* Expert readahead: moe_xf_run announces the experts it is about to load
 * (expert_announce -> expert_misses -> expert_advise). What must hold:
 *
 *   1. expert_misses returns the distinct experts of the requested rows that
 *      are not resident, in order of first appearance: a resident expert, a
 *      repeat, a negative id and an id past n_experts are skipped, and an
 *      index entry whose slot now holds another expert counts as a miss, as it
 *      does for the loader (slot_indexed);
 *   2. only the rows [from, to) are read, and seen[] carries over, so a
 *      second call does not announce an expert again;
 *   3. expert_announce clips `to` at S and returns the first row it did not
 *      announce, doing nothing when from >= to; the advise itself is a hint
 *      that finds no tensor in an empty checkpoint and must not fail. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                           \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

enum { LAYERS = 2, EXPERTS = 16, K = 4 };
static Model g_m;
static LCache g_cache[LAYERS];
static Slot g_slots[2];
static int g_by_expert[EXPERTS];
static int g_active[LAYERS] = { 0, 1 };

/* Layer 0 holds experts 3 and 7; the index also still points expert 4 at the
 * slot that now holds 7, which must read as a miss. */
static void model_setup(void) {
    memset(&g_m, 0, sizeof g_m);
    g_m.c.n_layers = LAYERS; g_m.c.n_experts = EXPERTS; g_m.c.topk = K;
    g_m.cache = g_cache; g_m.active_of = g_active;
    for (int e = 0; e < EXPERTS; e++) g_by_expert[e] = -1;
    g_slots[0].eid = 3; g_slots[1].eid = 7;
    g_by_expert[3] = 0; g_by_expert[7] = 1; g_by_expert[4] = 1;
    g_cache[0].slots = g_slots; g_cache[0].slot_by_expert = g_by_expert;
    g_cache[0].n = 2; g_cache[0].cap = 2;
}

static const int g_idx[] = {
     3,  5,  7,  5,      /* row 0: 3 and 7 resident, 5 twice */
     9,  3, -1, 12,      /* row 1: a negative id */
     5, 14, 99,  4,      /* row 2: 99 past n_experts, 4 a stale index entry */
     0,  9, 15,  1,      /* row 3 */
};
enum { ROWS = (int)(sizeof g_idx / sizeof g_idx[0]) / K };

static int same(const int *got, int n, const int *want, int m) {
    if (n != m) return 0;
    for (int i = 0; i < n; i++) if (got[i] != want[i]) return 0;
    return 1;
}

static void test_misses(void) {
    /* Room for every id in the rows (99 the largest), not only for n_experts:
     * with expert_misses inlined, gcc cannot tell that the bound on n_experts
     * keeps 99 out, and warns that seen[99] is past the end. */
    unsigned char seen[128];
    int miss[128];
    memset(seen, 0, sizeof seen);
    int n = expert_misses(&g_m, 0, g_idx, K, 0, 2, seen, miss);
    static const int want01[] = { 5, 9, 12 };
    CHECK(same(miss, n, want01, 3));
    n = expert_misses(&g_m, 0, g_idx, K, 2, 4, seen, miss);   /* 5 and 9 already seen */
    static const int want23[] = { 14, 4, 0, 15, 1 };
    CHECK(same(miss, n, want23, 5));
    memset(seen, 0, sizeof seen);
    n = expert_misses(&g_m, 0, g_idx, K, 1, 2, seen, miss);   /* row 1 alone */
    static const int want1[] = { 9, 12 };
    CHECK(same(miss, n, want1, 2));
    memset(seen, 0, sizeof seen);
    n = expert_misses(&g_m, 1, g_idx, K, 0, 1, seen, miss);   /* layer 1 holds nothing */
    static const int want_l1[] = { 3, 5, 7 };
    CHECK(same(miss, n, want_l1, 3));
    /* ids at and past n_experts, first of their kind: never announced */
    static const int past[K] = { EXPERTS, EXPERTS + 1, 99, 2 };
    memset(seen, 0, sizeof seen);
    n = expert_misses(&g_m, 0, past, K, 0, 1, seen, miss);
    static const int want_past[] = { 2 };
    CHECK(same(miss, n, want_past, 1));
}

static void test_announce(void) {
    CHECK(expert_announce(&g_m, 0, g_idx, ROWS, K, 0, 1) == 1);
    CHECK(expert_announce(&g_m, 0, g_idx, ROWS, K, 1, 1 + 8) == ROWS);   /* clipped at S */
    CHECK(expert_announce(&g_m, 0, g_idx, ROWS, K, ROWS, ROWS + 8) == ROWS);
    CHECK(expert_announce(&g_m, 0, g_idx, 1, K, 0, 9) == 1);             /* a decode step */
}

int main(void) {
    model_setup();
    test_misses();
    test_announce();
    printf("test_qwen36_expert_readahead: OK\n");
    return 0;
}
