/* ensure_kv grows the KV cache all or nothing, and serve_one grows it before
 * ACCEPT. What must hold:
 *
 *   1. from no cache and from a 16-position cache holding 10 recorded rows,
 *      tainted or not, every allocation of the growth fails once in turn.
 *      Each such call returns 0 and leaves the K/V buffers, their contents,
 *      the attention scores, the row stride (max_t == kv_cap), the kv_len and
 *      the prefix record, taint included, exactly as they were, with nothing
 *      left allocated; the only exception is the record's own allocation,
 *      after the cache has grown, whose failure drops reuse as before;
 *   2. a growth that succeeds keeps the rows (per head, at the new stride) and
 *      the record, as before;
 *   3. after a refusal the cache still fits a shorter request without any
 *      allocation, the record still matches a prompt that extends it, and a
 *      snapshot of the held prefix still passes the check pin_restore makes
 *      (kv_prefix_holds) -- none of which a tainted record allows;
 *   4. serve_one, when the cache cannot grow, answers the request with
 *      `ERROR <id> REQUEST_ALLOCATION_FAILED ...` and no ACCEPT before it;
 *   5. serve_loop, with memory that stays short from the growth onward, sends
 *      that refusal, then skips the grid it sends after every request (its
 *      buffers fail too) instead of writing through NULL, reads the next
 *      frame -- here the end of input -- and leaks nothing; once memory is
 *      back the grid is sent again.
 *
 * The injection needs shadow allocators, left out on Windows as in
 * tests/test_798_guards.c; there only the successful growth runs. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#ifndef _WIN32
/* Declared under their own names before the macros below hijack every token
 * that follows; the system headers are already included above. Out of line:
 * inlined into the encoder, the NULL an armed seam returns reached an
 * unchecked snprintf in bpe_piece, and gcc -O3 warned about it
 * (-Wformat-truncation). That allocation gets its check with #1800. */
__attribute__((noinline)) static void *test_malloc_seam(size_t n);
__attribute__((noinline)) static void *test_calloc_seam(size_t count, size_t size);
__attribute__((noinline)) static void *test_realloc_seam(void *p, size_t n);
__attribute__((noinline)) static void test_free_seam(void *p);
#define malloc test_malloc_seam
#define calloc test_calloc_seam
#define realloc test_realloc_seam
#define free test_free_seam
#endif

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#ifndef _WIN32
#undef malloc
#undef calloc
#undef realloc
#undef free

/* One ordinal over every allocation of a call, one over calloc alone, and the
 * count of blocks allocated since arming and not yet freed. g_short: from the
 * calloc numbered g_short_from on, every allocation fails. */
static long g_alloc_n = 0, g_fail_at = -1, g_calloc_n = 0, g_fail_calloc_at = -1, g_live = 0;
static long g_short_from = -1;
static int g_short = 0;
static void *test_malloc_seam(size_t n) {
    if (++g_alloc_n == g_fail_at || g_short) return NULL;
    void *p = malloc(n);
    if (p) g_live++;
    return p;
}
static void *test_calloc_seam(size_t count, size_t size) {
    if (++g_calloc_n == g_short_from) g_short = 1;
    if (++g_alloc_n == g_fail_at || g_calloc_n == g_fail_calloc_at || g_short) return NULL;
    void *p = calloc(count, size);
    if (p) g_live++;
    return p;
}
static void *test_realloc_seam(void *p, size_t n) {
    if (++g_alloc_n == g_fail_at || g_short) return NULL;
    void *q = realloc(p, n);
    if (q && !p) g_live++;
    return q;
}
static void test_free_seam(void *p) {
    if (p) g_live--;
    free(p);
}
static void arm(long ordinal) {
    g_alloc_n = g_calloc_n = g_live = 0;
    g_fail_at = ordinal; g_fail_calloc_at = g_short_from = -1; g_short = 0;
}
static void arm_calloc(long k) { arm(-1); g_fail_calloc_at = k; }
static void arm_short_from_calloc(long k) { arm(-1); g_short_from = k; }
static void disarm(void) { arm(-1); }
#endif

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                           \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

/* Four layers, two of them attention, two KV heads of eight; three experts
 * per layer, none resident, so the grid serve_loop sends is all zeros. */
enum { LAYERS = 4, HEADS = 2, HD = 8, EXPERTS = 3, SMALL = 16, BIG = 64, HELD = 10 };
static uint8_t g_attn[LAYERS] = { 0, 1, 0, 1 };
static LCache g_cache[LAYERS];
static Model g_m;

static void model_reset(void) {
    if (g_m.K) {
        for (int i = 0; i < LAYERS; i++) { free(g_m.K[i]); free(g_m.V[i]); }
        free(g_m.K); free(g_m.V);
    }
    free(g_m.attn_sc);
    kv_prefix_free(&g_m.kvp);
    memset(&g_m, 0, sizeof g_m);
    g_m.c.n_layers = LAYERS; g_m.c.is_attn = g_attn;
    g_m.c.kv_heads = HEADS; g_m.c.k_head_dim = HD;
    g_m.c.n_experts = EXPERTS; g_m.cache = g_cache;
}

static float val(int kv, int layer, int h, int pos, int d) {
    return (float)(kv * 100000 + layer * 10000 + h * 1000 + pos * 10 + d);
}

/* A SMALL cache whose first HELD positions are written and recorded. */
static void model_holding(void) {
    model_reset();
    g_m.max_t = SMALL;
    CHECK(ensure_kv(&g_m) == 1 && g_m.kv_cap == SMALL);
    for (int i = 0; i < LAYERS; i++) if (g_attn[i])
        for (int h = 0; h < HEADS; h++) for (int p = 0; p < HELD; p++) for (int d = 0; d < HD; d++) {
            g_m.K[i][((size_t)h * SMALL + p) * HD + d] = val(0, i, h, p, d);
            g_m.V[i][((size_t)h * SMALL + p) * HD + d] = val(1, i, h, p, d);
        }
    int ids[HELD];
    for (int p = 0; p < HELD; p++) ids[p] = 100 + p;
    kv_prefix_record(&g_m.kvp, ids, 0, HELD);
    g_m.kv_len = HELD;
}

static int rows_held(int stride) {
    for (int i = 0; i < LAYERS; i++) if (g_attn[i])
        for (int h = 0; h < HEADS; h++) for (int p = 0; p < HELD; p++) for (int d = 0; d < HD; d++)
            if (g_m.K[i][((size_t)h * stride + p) * HD + d] != val(0, i, h, p, d) ||
                g_m.V[i][((size_t)h * stride + p) * HD + d] != val(1, i, h, p, d)) return 0;
    return 1;
}

static int record_held(void) {
    if (!g_m.kvp.fed || g_m.kvp.len != HELD || g_m.kv_len != HELD) return 0;
    for (int p = 0; p < HELD; p++) if (g_m.kvp.fed[p] != 100 + p) return 0;
    return 1;
}

static void test_growth_keeps_rows_and_record(void) {
    model_holding();
    g_m.max_t = BIG;
    CHECK(ensure_kv(&g_m) == 1);
    CHECK(g_m.kv_cap == BIG && g_m.max_t == BIG && g_m.kvp.cap == BIG);
    CHECK(rows_held(BIG) && record_held());
}

#ifndef _WIN32
static void test_first_allocation_refused_at_every_ordinal(void) {
    int record_only = 0;
    for (long k = 1;; k++) {
        model_reset();
        g_m.max_t = SMALL;
        arm(k);
        int r = ensure_kv(&g_m);
        long seen = g_alloc_n, live = g_live;
        disarm();
        if (seen < k) { CHECK(r == 1 && g_m.K && g_m.attn_sc && g_m.kvp.cap == SMALL); break; }
        if (r == 1) {             /* the record's allocation, after the cache: reuse off, as before */
            CHECK(g_m.K && g_m.kv_cap == SMALL && !g_m.kvp.fed && g_m.kvp.len == 0);
            record_only++;
            continue;
        }
        CHECK(!g_m.K && !g_m.V && !g_m.attn_sc && g_m.kv_cap == 0 && live == 0);
    }
    CHECK(record_only == 1);
}

static void test_growth_refused_at_every_ordinal(int tainted) {
    int record_only = 0, refused = 0;
    for (long k = 1;; k++) {
        model_holding();
        g_m.kvp.tainted = tainted;
        float **K = g_m.K, **V = g_m.V, *sc = g_m.attn_sc; int *fed = g_m.kvp.fed;
        g_m.max_t = BIG;
        arm(k);
        int r = ensure_kv(&g_m);
        long seen = g_alloc_n, live = g_live;
        disarm();
        if (seen < k) {
            CHECK(r == 1 && g_m.kv_cap == BIG && rows_held(BIG) && record_held());
            CHECK(g_m.kvp.tainted == tainted);
            break;
        }
        if (r == 1) {
            CHECK(g_m.kv_cap == BIG && rows_held(BIG) && !g_m.kvp.fed && g_m.kv_len == 0);
            record_only++;
            continue;
        }
        refused++;
        CHECK(g_m.K == K && g_m.V == V && g_m.attn_sc == sc && g_m.kvp.fed == fed);
        CHECK(g_m.kv_cap == SMALL && g_m.max_t == SMALL && g_m.kvp.cap == SMALL);
        CHECK(rows_held(SMALL) && record_held() && g_m.kvp.tainted == tainted && live == 0);
    }
    CHECK(record_only == 1 && refused >= 5);      /* K and V arrays, scores, two layers' rows */
    /* After a refusal the cache still serves what fits, with no allocation;
     * the record still matches a prompt that extends it, and a snapshot of
     * the held prefix would still be restored -- unless the record is tainted. */
    model_holding();
    g_m.kvp.tainted = tainted;
    g_m.max_t = BIG;
    arm(1);
    CHECK(ensure_kv(&g_m) == 0);
    disarm();
    g_m.max_t = HELD + 2;
    arm(1);
    CHECK(ensure_kv(&g_m) == 1 && g_alloc_n == 0 && g_m.max_t == SMALL);
    disarm();
    int longer[HELD + 2];
    for (int p = 0; p < HELD + 2; p++) longer[p] = 100 + p;
    CHECK(kv_prefix_reuse(&g_m.kvp, longer, HELD + 2) == (tainted ? 0 : HELD));
    CHECK(kv_prefix_holds(&g_m.kvp, longer, HELD) == !tainted);
}

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(text, 1, strlen(text), f) == strlen(text) && fclose(f) == 0);
}

/* Runs fn with fd 1 on a pipe, and fd 0 on another holding `input` when that
 * is not NULL; returns what fn wrote, NUL-terminated. */
static void run_captured(void (*fn)(void), const char *input, char *out, size_t cap) {
    int ofd[2], ifd[2], saved_in = -1;
    CHECK(pipe(ofd) == 0);
    if (input) {
        CHECK(pipe(ifd) == 0);
        CHECK(write(ifd[1], input, strlen(input)) == (ssize_t)strlen(input));
        close(ifd[1]);
        saved_in = dup(0);
        dup2(ifd[0], 0); close(ifd[0]);
    }
    fflush(stdout);
    int saved_out = dup(1);
    dup2(ofd[1], 1); close(ofd[1]);
    fn();
    fflush(stdout);
    dup2(saved_out, 1); close(saved_out);
    if (input) { dup2(saved_in, 0); close(saved_in); clearerr(stdin); }
    size_t n = 0;
    ssize_t got;
    while (n < cap - 1 && (got = read(ofd[0], out + n, cap - 1 - n)) > 0) n += (size_t)got;
    out[n] = 0;
    close(ofd[0]);
}

static void load_test_tokenizer(void) {
    if (g_tok) return;
    mkdir("tests/tmp_kv_growth", 0700);
    write_file("tests/tmp_kv_growth/tokenizer.json",
        "{\"model\":{\"vocab\":{\"i\":0,\"n\":1,\"\\u0120\":2,\"in\":3,\"\\u0120in\":4},"
        "\"merges\":[[\"i\",\"n\"],[\"\\u0120\",\"in\"]]},\"added_tokens\":[]}");
    load_tokenizer("tests/tmp_kv_growth/tokenizer.json");
    CHECK(g_tok != NULL);
}

static ServeReq g_req;
static void call_serve_one(void) { serve_one(&g_m, &g_req); }
static void call_serve_loop(void) { serve_loop(&g_m); }
static void call_emap_emit(void) { emap_emit(&g_m); }

/* The engine answers before it would ACCEPT: one ERROR line, nothing else,
 * and the cache it holds untouched. */
static void test_serve_one_refuses_before_accept(void) {
    load_test_tokenizer();
    model_holding();
    float **K = g_m.K; int *fed = g_m.kvp.fed;
    static char payload[] = "in in in";                 /* 3 tokens: 3 + 40 > SMALL */
    memset(&g_req, 0, sizeof g_req);
    snprintf(g_req.id, sizeof g_req.id, "req1");
    g_req.payload = payload; g_req.plen = (int)strlen(payload); g_req.max_tok = 40;
    char out[512];
    arm_calloc(1);                                      /* the K array: the growth's first allocation */
    run_captured(call_serve_one, NULL, out, sizeof out);
    disarm();
    CHECK(strcmp(out, "ERROR req1 REQUEST_ALLOCATION_FAILED growing the KV cache to 43 tokens\n") == 0);
    CHECK(g_m.K == K && g_m.kvp.fed == fed && g_m.kv_cap == SMALL && g_m.max_t == SMALL);
    CHECK(rows_held(SMALL) && record_held());
}

/* The whole loop, with memory short from the growth on: the boot grid takes
 * the first calloc, the growth's K array the second. */
static void test_serve_loop_survives_the_refusal(void) {
    load_test_tokenizer();
    model_holding();
    float **K = g_m.K; int *fed = g_m.kvp.fed;
    static const char boot[] = "\x01\x01READY\x01\x01\nSTAT 0 0.00 0.0 ";
    const char *grid = "EMAP 4 3 000000000000000000000000\n";
    char out[1024];
    arm_short_from_calloc(2);
    run_captured(call_serve_loop, "SUBMIT req1 0 8 40 0 1\nin in in\n", out, sizeof out);
    long live = g_live;
    int short_reached = g_short;
    disarm();
    CHECK(short_reached && live == 0);
    CHECK(strncmp(out, boot, sizeof boot - 1) == 0);
    const char *tail = strstr(out, "\nEMAP ");
    CHECK(tail != NULL);
    char want[256];
    snprintf(want, sizeof want, "\n%s%s", grid,
             "ERROR req1 REQUEST_ALLOCATION_FAILED growing the KV cache to 43 tokens\n");
    CHECK(strcmp(tail, want) == 0);
    CHECK(g_m.K == K && g_m.kvp.fed == fed && g_m.kv_cap == SMALL && g_m.max_t == SMALL);
    CHECK(rows_held(SMALL) && record_held());
    run_captured(call_emap_emit, NULL, out, sizeof out);
    CHECK(strcmp(out, grid) == 0);
}
#endif

int main(void) {
    test_growth_keeps_rows_and_record();
#ifndef _WIN32
    test_first_allocation_refused_at_every_ordinal();
    test_growth_refused_at_every_ordinal(0);
    test_growth_refused_at_every_ordinal(1);
    test_serve_one_refuses_before_accept();
    test_serve_loop_survives_the_refusal();
#else
    printf("allocation-failure injection skipped on Windows (no shadow allocators)\n");
#endif
    printf("test_qwen36_kv_growth: OK\n");
    return 0;
}
