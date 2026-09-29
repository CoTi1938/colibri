/* Allocation failures inside the text encoder are reported, not fatal. push_id,
 * bpe_piece and encode_text return 0 on any failed malloc/realloc, release what
 * they allocated, and leave the caller's id buffer valid to free; the serving
 * loop then answers that one request with an ERROR frame and stays up, and the
 * CLI and the edge adapter report and return. Before this, the two growth
 * reallocs exit(1)'d the whole engine (#1588) and every other allocation in the
 * encoder was unchecked, so a failure was a NULL write.
 *
 * Injects real failures through shadow allocators (the technique of
 * tests/test_798_guards.c): at each growth step directly, then at every
 * allocation ordinal of four encodings chosen so that together they reach every
 * allocation site of the encoder: the expected ids show that the merges ran, and
 * the test checks that both buffers grew. The shadows also count the blocks a
 * call still holds, so a failure path that leaks is caught. As in
 * test_798_guards.c, the shadows are left out on Windows; there the four
 * encodings run once, with nothing failing. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#ifndef _WIN32
/* Declared under their own names before the macros below hijack every malloc /
 * realloc / free token that follows. The system headers are already included
 * above, so the engine's re-includes do not redeclare the originals under the
 * seam names. */
static void *test_malloc_seam(size_t n);
static void *test_realloc_seam(void *p, size_t n);
static void test_free_seam(void *p);
#define malloc test_malloc_seam
#define realloc test_realloc_seam
#define free test_free_seam
#endif

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#ifndef _WIN32
#undef malloc
#undef realloc
#undef free

/* One counter for malloc and realloc, so an ordinal means "the k-th allocation
 * of the call", and one for realloc alone, to aim at a growth step. g_live
 * counts the blocks allocated since arming and not yet freed; g_last_grown is
 * the last block a realloc handed back. */
static long g_alloc_n = 0, g_fail_at = -1, g_realloc_n = 0, g_fail_realloc_at = -1, g_live = 0;
static void *g_last_grown = NULL;
static void *test_malloc_seam(size_t n) {
    if (++g_alloc_n == g_fail_at) return NULL;
    void *p = malloc(n);
    if (p) g_live++;
    return p;
}
static void *test_realloc_seam(void *p, size_t n) {
    ++g_alloc_n; ++g_realloc_n;
    if (g_alloc_n == g_fail_at || g_realloc_n == g_fail_realloc_at) return NULL;
    void *q = realloc(p, n);
    if (q && !p) g_live++;
    if (q) g_last_grown = q;
    return q;
}
static void test_free_seam(void *p) {
    if (p) g_live--;
    free(p);
}
static void arm(long ordinal) {
    g_alloc_n = g_realloc_n = g_live = 0; g_last_grown = NULL;
    g_fail_at = ordinal; g_fail_realloc_at = -1;
}
static void arm_realloc(long k) { arm(-1); g_fail_realloc_at = k; }
static void disarm(void) { arm(-1); }
#endif

static int g_nfails = 0;
static void check(int cond, const char *what) {
    if (!cond) { printf("FAIL: %s\n", what); g_nfails++; }
}

#ifndef _WIN32
/* push_id: cap 2, two pushes fill it, the third needs the realloc. */
static void test_push_id_growth(void) {
    int *ids = malloc(2 * sizeof(int)); int n = 0, cap = 2;
    check(push_id(&ids, &n, &cap, 10) && push_id(&ids, &n, &cap, 20), "push_id: fills the buffer");
    arm(1);
    int r = push_id(&ids, &n, &cap, 30);
    disarm();
    check(r == 0, "push_id: returns 0 on a failed realloc");
    check(n == 2 && cap == 2 && ids[0] == 10 && ids[1] == 20,
          "push_id: the caller's buffer is untouched after the failure");
    check(push_id(&ids, &n, &cap, 30) && n == 3 && cap == 4 && ids[2] == 30,
          "push_id: the same push succeeds once memory is back");
    free(ids);
}

/* A capacity that cannot double is refused before any realloc is attempted. */
static void test_push_id_overflow_guard(void) {
    int *ids = malloc(sizeof(int)); int n = INT_MAX / 2 + 1, cap = n;
    arm(1);
    int r = push_id(&ids, &n, &cap, 1);
    long allocations = g_alloc_n;
    disarm();
    check(r == 0 && allocations == 0 && n == cap && cap == INT_MAX / 2 + 1,
          "push_id: refuses a capacity that would overflow, without allocating");
    free(ids);
}
#endif

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    check(f != NULL, "fixture: open the tiny tokenizer.json");
    if (!f) return;
    check(fwrite(text, 1, strlen(text), f) == strlen(text) && fclose(f) == 0,
          "fixture: write the tiny tokenizer.json");
}

/* "in" and " in" merge (i+n, then Ġ+in); <|x|> is an added token. */
static const char *tok_json =
    "{\"model\":{\"vocab\":{\"i\":0,\"n\":1,\"\\u0120\":2,\"in\":3,\"\\u0120in\":4},"
    "\"merges\":[[\"i\",\"n\"],[\"\\u0120\",\"in\"]]},"
    "\"added_tokens\":[{\"id\":5,\"content\":\"<|x|>\",\"special\":true}]}";

/* bpe_piece starts its symbol array at 16 and encode_text its id buffer at 1024:
 * the 18-byte piece grows the first, the two 1025-id texts grow the second, once
 * at an added token and once inside bpe_piece. */
#define PIECE18 "ininininininininin"
#define RUN 1024
typedef struct { const char *what; const char *text; const int *want; int nwant; } Case;
static int want_a[] = {3, 4, 5, 3, 4}, want_b[9], want_c[RUN + 1], want_d[RUN + 1];
static char text_c[RUN + 6], text_d[RUN + 2];
static const Case cases[] = {
    {"merges and an added token", "in in<|x|>in in", want_a, 5},
    {"an 18-byte piece", PIECE18, want_b, 9},
    {"1024 ids, then an added token", text_c, want_c, RUN + 1},
    {"1025 ids from one piece", text_d, want_d, RUN + 1},
};
#define NCASES ((int)(sizeof cases / sizeof cases[0]))

static void build_cases(void) {
    for (int k = 0; k < 9; k++) want_b[k] = 3;
    memset(text_c, 'n', RUN); memcpy(text_c + RUN, "<|x|>", 6);
    memset(text_d, 'n', RUN + 1); text_d[RUN + 1] = 0;
    for (int k = 0; k <= RUN; k++) want_c[k] = want_d[k] = 1;   /* "n" is id 1 */
    want_c[RUN] = 5;
}

static void check_ids(const Case *c, int r, const int *ids, int n) {
    if (r == 1 && ids != NULL && n == c->nwant && memcmp(ids, c->want, (size_t)n * sizeof(int)) == 0) return;
    printf("FAIL: %s: expected %d ids, the call returned %d with %d\n", c->what, c->nwant, r, n);
    g_nfails++;
}

#ifndef _WIN32
/* Each allocation ordinal of the encoding fails once, in turn. Every such call
 * must return 0, hand out no ids and hold no blocks; the first call the seam
 * never reaches is a complete run, which must produce the expected ids and hold
 * only the id buffer. Reports its reallocs and whether the id buffer it handed
 * out came from one, i.e. had to grow. */
static void sweep(const Case *c, long *reallocs, int *ids_grew) {
    for (long ordinal = 1;; ordinal++) {
        int *ids = (int *)&ordinal; int n = -1;   /* poisoned: encode_text must reset both */
        arm(ordinal);
        int r = encode_text(c->text, &ids, &n);
        long seen = g_alloc_n, live = g_live, grown = g_realloc_n; void *last = g_last_grown;
        disarm();
        if (seen < ordinal) {                      /* the seam was never reached */
            check_ids(c, r, ids, n);
            if (live != 1) {
                printf("FAIL: %s: a complete run holds %ld blocks, not just the id buffer\n", c->what, live);
                g_nfails++;
            }
            *reallocs = grown; *ids_grew = r == 1 && (void *)ids == last;
            if (r == 1) free(ids);
            printf("%s: %ld allocation ordinals refused one at a time\n", c->what, seen);
            return;
        }
        if (r != 0 || ids != NULL || n != 0 || live != 0) {
            printf("FAIL: %s: allocation %ld failed but the call returned %d (ids=%p n=%d, %ld blocks held)\n",
                   c->what, ordinal, r, (void *)ids, n, live);
            g_nfails++;
            if (r == 1) free(ids);
        }
    }
}

/* bpe_piece on the 18-byte piece: the first realloc is the symbol array's growth
 * at the 17th byte; with the id buffer one slot from full, the second is the id
 * buffer's, at the second id. Either failure returns 0, keeps what the caller
 * had, including the one id pushed before it, and holds no blocks. */
static void test_bpe_piece_growth(void) {
    int len = (int)strlen(PIECE18);
    int *ids = malloc(2 * sizeof(int)); int n = 1, cap = 2; ids[0] = 7;
    arm_realloc(1);
    int r = bpe_piece(PIECE18, len, &ids, &n, &cap);
    long live = g_live;
    disarm();
    check(r == 0 && n == 1 && cap == 2 && ids[0] == 7 && live == 0,
          "bpe_piece: a failed symbol-array growth returns 0, keeps the caller's ids, holds nothing");
    arm_realloc(2);
    r = bpe_piece(PIECE18, len, &ids, &n, &cap);
    live = g_live;
    disarm();
    check(r == 0 && n == 2 && cap == 2 && ids[0] == 7 && ids[1] == 3 && live == 0,
          "bpe_piece: a failed id-buffer growth returns 0, keeps the id pushed before it, holds nothing");
    r = bpe_piece(PIECE18, len, &ids, &n, &cap);
    int all = r == 1 && n == 11 && ids[0] == 7;
    for (int k = 1; all && k < n; k++) all = ids[k] == 3;
    check(all, "bpe_piece: the same piece encodes once memory is back");
    free(ids);
}

/* The serving loop: with the very first allocation of the encoding refused,
 * serve_one must answer that request with an ERROR frame and return, so the
 * engine keeps serving. It never reaches the model, which stays untouched. */
static void test_serve_one_answers_and_keeps_running(void) {
    int fds[2];
    if (pipe(fds) != 0) { check(0, "serve_one: pipe"); return; }
    fflush(stdout);
    int saved = dup(1);
    dup2(fds[1], 1); close(fds[1]);
    static Model m;
    static char payload[] = "in in";
    ServeReq q = {0};
    snprintf(q.id, sizeof q.id, "req1");
    q.payload = payload; q.plen = (int)strlen(payload); q.max_tok = 4;
    arm(1);                                        /* the id buffer itself */
    serve_one(&m, &q);
    disarm();
    fflush(stdout);
    dup2(saved, 1); close(saved);
    char out[256] = {0};
    ssize_t got = read(fds[0], out, sizeof out - 1);
    close(fds[0]);
    check(got > 0 && strncmp(out, "ERROR req1 REQUEST_ALLOCATION_FAILED", 36) == 0,
          "serve_one: answers the request with an ERROR frame and returns");
}
#endif

int main(void) {
#ifndef _WIN32
    test_push_id_growth();
    test_push_id_overflow_guard();
#endif

    const char *dir = "tests/tmp_encode_oom";
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0700);
#endif
    char path[256];
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    write_file(path, tok_json);
    load_tokenizer(path);
    check(g_tok != NULL, "fixture: the tiny tokenizer loaded");
    build_cases();

#ifndef _WIN32
    long reallocs[NCASES]; int ids_grew[NCASES];
    for (int k = 0; k < NCASES; k++) sweep(&cases[k], &reallocs[k], &ids_grew[k]);
    check(reallocs[1] > 0, "the sweep reached the growth of bpe_piece's symbol array");
    check(ids_grew[2] && ids_grew[3], "the sweep reached the growth of the id buffer, at an added token and inside bpe_piece");
    test_bpe_piece_growth();
    test_serve_one_answers_and_keeps_running();
#else
    for (int k = 0; k < NCASES; k++) {
        int *ids = NULL, n = 0;
        int r = encode_text(cases[k].text, &ids, &n);
        check_ids(&cases[k], r, ids, n);
        if (r == 1) free(ids);
    }
    printf("allocation-failure injection skipped on Windows (no shadow allocators)\n");
#endif

    if (g_nfails) { printf("%d check(s) failed\n", g_nfails); return 1; }
    printf("OK\n");
    return 0;
}
