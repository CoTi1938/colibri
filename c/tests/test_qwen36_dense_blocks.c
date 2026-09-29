/* load_tq quantizes a dense matrix stored as plain BF16/F16/F32 a block of rows
 * at a time (load_tq_blocked) instead of staging its whole f32 copy first.
 * What must hold:
 *
 *   1. the int8 rows/scales and the int4 blocks/scales are byte-identical to
 *      quantizing the whole f32 matrix (the previous path: load_t_n, then
 *      qw_quantize) for BF16, F16 and F32, with an uneven last block, a row
 *      wider than the block (one row per block), a width that takes no int4
 *      copy, and a `language_model.`-prefixed tensor;
 *   2. the buffers kept are qw_quantize's: int8 without an int4 tag, int4 alone
 *      with one, both with COLI_DENSE_KEEP_I8=1 (a child process, POSIX only:
 *      the setting is read once per process);
 *   3. a tensor it does not stream -- an MLX affine U32 weight, a size the
 *      config disagrees with -- is left to the whole-matrix path: it returns 0
 *      and touches nothing;
 *   4. no allocation is as large as the matrix's raw or f32 copy, while the
 *      whole-matrix path makes one (POSIX: a shadow malloc records the largest
 *      request);
 *   5. through load_tq, the call model_init_range makes: plain data takes the
 *      blocked path, with the int4 copy only for the tags COLI_DENSE_INT4
 *      names; COLI_KEEP_F32=1, a load that does not quantize and a real MLX
 *      affine triple take the whole-matrix path, the first two keeping the
 *      f32 matrix, and every buffer matches the previous path.
 *
 * The block is shrunk to three rows of 128 floats so the fixture stays tiny. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif

#ifndef _WIN32
/* Declared under its own name before the macro below hijacks every malloc
 * token that follows; the system headers are already included above. */
static void *test_malloc_seam(size_t n);
#define malloc test_malloc_seam
#endif

#define QW_LOAD_BLOCK_BYTES ((size_t)1536)
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#ifndef _WIN32
#undef malloc
static size_t g_largest = 0;
static void *test_malloc_seam(size_t n) {
    if (n > g_largest) g_largest = n;
    return malloc(n);
}
#endif

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                           \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

/* ---- fixture ------------------------------------------------------------- */

typedef struct { const char *name, *dtype; int O, I; void *bytes; size_t nbytes; } Tensor;

static uint32_t g_rng = 12345u;
static uint32_t rnd(void) { g_rng = g_rng * 1664525u + 1013904223u; return g_rng >> 8; }

/* Row 2 is all zeros (scale 1 on both quantizers); the rest spans signs and
 * several orders of magnitude, F16 subnormals included. */
static void *make_values(const char *dtype, int O, int I, size_t *nbytes) {
    size_t n = (size_t)O * I, esz = strcmp(dtype, "F32") ? 2 : 4;
    unsigned char *p = calloc(n, esz);
    CHECK(p != NULL);
    for (size_t k = 0; k < n; k++) {
        if (k / (size_t)I == 2) continue;
        uint32_t r = rnd();
        if (esz == 4) {
            float v = ((float)(r % 20001) - 10000.f) / 2500.f * (float)(1u << (r % 5));
            memcpy(p + k * 4, &v, 4);
        } else {
            uint16_t h;
            if (!strcmp(dtype, "F16")) {
                h = (uint16_t)(r & 0xFFFFu);
                if (((h >> 10) & 0x1F) == 0x1F) h &= 0xBFFF;        /* no inf/NaN */
            } else {
                uint16_t e = (uint16_t)(117 + r % 20);             /* 2^-10 .. 2^9 */
                h = (uint16_t)(((r >> 20) & 1u) << 15 | e << 7 | ((r >> 8) & 0x7F));
            }
            memcpy(p + k * 2, &h, 2);
        }
    }
    *nbytes = n * esz;
    return p;
}

static void write_fixture(const char *path, const Tensor *t, int n) {
    char header[4096];
    size_t hlen = 0, off = 0;
    hlen += (size_t)snprintf(header + hlen, sizeof header - hlen, "{");
    for (int i = 0; i < n; i++) {
        hlen += (size_t)snprintf(header + hlen, sizeof header - hlen,
            "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
            i ? "," : "", t[i].name, t[i].dtype, t[i].O, t[i].I, off, off + t[i].nbytes);
        off += t[i].nbytes;
    }
    hlen += (size_t)snprintf(header + hlen, sizeof header - hlen, "}");
    CHECK(hlen < sizeof header);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    uint64_t h64 = (uint64_t)hlen;
    CHECK(fwrite(&h64, 8, 1, f) == 1);
    CHECK(fwrite(header, 1, hlen, f) == hlen);
    for (int i = 0; i < n; i++)
        CHECK(fwrite(t[i].bytes, 1, t[i].nbytes, f) == t[i].nbytes);
    CHECK(fclose(f) == 0);
}

static Tensor g_t[] = {
    { "model.b.weight", "BF16", 10, 128, NULL, 0 },
    { "model.h.weight", "F16", 10, 128, NULL, 0 },
    { "model.f.weight", "F32", 10, 128, NULL, 0 },
    { "model.odd.weight", "F16", 7, 96, NULL, 0 },       /* 96 % 64 != 0: no int4 copy */
    { "model.wide.weight", "F16", 3, 512, NULL, 0 },     /* one row per block */
    { "language_model.model.pref.weight", "F16", 4, 64, NULL, 0 },
};
#define NT ((int)(sizeof g_t / sizeof g_t[0]))
static const uint32_t g_affine_word[2 * 4] = { 1, 2, 3, 4, 5, 6, 7, 8 };
/* A real MLX affine Q8 triple, 4 x 64 in groups of 32. */
static uint32_t g_affq_words[4 * 16];
static uint16_t g_affq_scales[4 * 2], g_affq_biases[4 * 2];
static uint16_t bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return (uint16_t)(bits >> 16);
}

static void build_fixture(const char *dir) {
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0700);
#endif
    Tensor all[NT + 4];
    for (int i = 0; i < NT; i++) {
        g_t[i].bytes = make_values(g_t[i].dtype, g_t[i].O, g_t[i].I, &g_t[i].nbytes);
        all[i] = g_t[i];
    }
    all[NT] = (Tensor){ "model.aff.weight", "U32", 2, 4, (void *)g_affine_word, sizeof g_affine_word };
    for (int k = 0; k < 4 * 64; k++)
        g_affq_words[k / 4] |= (uint32_t)((k * 37 + 11) & 0xFF) << (8 * (k % 4));
    for (int g = 0; g < 8; g++) {
        g_affq_scales[g] = bf16_bits(0.03125f * (float)(1 + g));
        g_affq_biases[g] = bf16_bits(-0.5f * (float)g);
    }
    all[NT + 1] = (Tensor){ "model.affq.weight", "U32", 4, 16, g_affq_words, sizeof g_affq_words };
    all[NT + 2] = (Tensor){ "model.affq.scales", "BF16", 4, 2, g_affq_scales, sizeof g_affq_scales };
    all[NT + 3] = (Tensor){ "model.affq.biases", "BF16", 4, 2, g_affq_biases, sizeof g_affq_biases };
    char path[512];
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    write_fixture(path, all, NT + 4);
}

/* ---- checks -------------------------------------------------------------- */

static Model g_m;

static int same(const void *a, const void *b, size_t n) {
    if (!a || !b) return a == b;
    return memcmp(a, b, n) == 0;
}

/* The previous path, kept as the reference: the whole f32 matrix, then
 * qw_quantize. */
static void reference(const char *name, int O, int I, const char *tag, QW *ref) {
    float *w = load_t_n(&g_m, name, (int64_t)I * O);
    memset(ref, 0, sizeof *ref);
    qw_quantize(w, I, O, tag, ref);
    free(w);
    ref->w = NULL;
}

/* via_load_tq: through load_tq, which picks the path, rather than straight
 * into load_tq_blocked. */
static void compare(const char *name, int O, int I, const char *tag, int want_q, int want_q4,
                    int via_load_tq) {
    QW ref, got;
    reference(name, O, I, tag, &ref);
    memset(&got, 0, sizeof got);
    if (via_load_tq) load_tq(&g_m, name, I, O, 1, tag, &got);
    else CHECK(load_tq_blocked(&g_m, name, I, O, tag, &got) == 1);
    size_t n = (size_t)O * I;
    CHECK(got.w == NULL && got.I == I && got.O == O && got.ng == ref.ng);
    CHECK((got.q != NULL) == want_q && (ref.q != NULL) == want_q);
    CHECK((got.q4 != NULL) == want_q4 && (ref.q4 != NULL) == want_q4);
    CHECK(same(got.q, ref.q, n) && same(got.sc, ref.sc, (size_t)O * sizeof(float)));
    CHECK(same(got.q4, ref.q4, n / 2));
    CHECK(same(got.sg, ref.sg, (size_t)O * (I / 64) * sizeof(float)));
    qw_free(&ref); qw_free(&got);
}

static const char *plain(const Tensor *t) {
    return strncmp(t->name, "language_model.", 15) ? t->name : t->name + 15;
}

/* COLI_DENSE_BITS=4 is set for the whole process: a tag takes the int4 copy
 * (dropping int8 unless kept), NULL does not. */
static void test_identical(int keep_i8) {
    for (int i = 0; i < NT; i++) {
        const Tensor *t = &g_t[i];
        int int4 = t->I % 64 == 0;
        compare(plain(t), t->O, t->I, NULL, 1, 0, 0);
        compare(plain(t), t->O, t->I, "t", !int4 || keep_i8, int4, 0);
    }
}

static void test_left_to_whole_matrix_path(void) {
    QW poison, out;
    memset(&poison, 0x5A, sizeof poison);
    out = poison;
    /* Asked with its packed shape, so only the dtype tells it from F32 data. */
    CHECK(load_tq_blocked(&g_m, "model.aff.weight", 4, 2, NULL, &out) == 0);
    CHECK(memcmp(&out, &poison, sizeof out) == 0);
    CHECK(load_tq_blocked(&g_m, "model.aff.weight", 32, 2, NULL, &out) == 0);
    CHECK(memcmp(&out, &poison, sizeof out) == 0);
    CHECK(load_tq_blocked(&g_m, "model.b.weight", 128, 9, NULL, &out) == 0);   /* 9 rows asked, 10 stored */
    CHECK(memcmp(&out, &poison, sizeof out) == 0);
    CHECK(load_tq_blocked(&g_m, "model.none.weight", 128, 10, NULL, &out) == 0);
    CHECK(memcmp(&out, &poison, sizeof out) == 0);
}

/* Through load_tq. The int4 selector is read on every call; COLI_KEEP_F32
 * too. */
static void test_load_tq(void) {
    const Tensor *t = &g_t[1];                                   /* F16 [10,128] */
    size_t n = (size_t)t->O * t->I;
    compare(t->name, t->O, t->I, "t", 0, 1, 1);
    compare(t->name, t->O, t->I, NULL, 1, 0, 1);
    setenv("COLI_DENSE_INT4", "lmhead", 1);
    compare(t->name, t->O, t->I, "lmhead", 0, 1, 1);
    compare(t->name, t->O, t->I, "router", 1, 0, 1);
    unsetenv("COLI_DENSE_INT4");
    compare("model.affq.weight", 4, 64, "t", 0, 1, 1);           /* expanded, then quantized */
    compare("model.affq.weight", 4, 64, NULL, 1, 0, 1);

    float *w = load_t_n(&g_m, t->name, (int64_t)n);
    QW ref, got;
    reference(t->name, t->O, t->I, "t", &ref);
    setenv("COLI_KEEP_F32", "1", 1);
    memset(&got, 0, sizeof got);
    load_tq(&g_m, t->name, t->I, t->O, 1, "t", &got);
    unsetenv("COLI_KEEP_F32");
    CHECK(got.w != NULL && memcmp(got.w, w, n * sizeof(float)) == 0);
    CHECK(got.q == NULL && same(got.q4, ref.q4, n / 2));
    CHECK(same(got.sg, ref.sg, (size_t)t->O * (t->I / 64) * sizeof(float)));
    qw_free(&got);
    memset(&got, 0, sizeof got);
    load_tq(&g_m, t->name, t->I, t->O, 0, "t", &got);           /* not quantized */
    CHECK(got.w != NULL && memcmp(got.w, w, n * sizeof(float)) == 0);
    CHECK(got.q == NULL && got.q4 == NULL);
    qw_free(&got); qw_free(&ref); free(w);
}

#ifndef _WIN32
/* The raw BF16 copy is 2560 bytes and the f32 one 5120; the blocked load's
 * largest allocation is its 1536-byte block. The whole-matrix path allocates
 * the f32 copy, which the same check must catch. */
static void test_no_whole_matrix_staging(void) {
    QW out;
    memset(&out, 0, sizeof out);
    g_largest = 0;
    load_tq(&g_m, "model.b.weight", 128, 10, 1, NULL, &out);
    CHECK(out.q != NULL && out.w == NULL);
    CHECK(g_largest < (size_t)10 * 128 * 2);
    qw_free(&out);
    g_largest = 0;
    float *w = load_t_n(&g_m, "model.b.weight", 10 * 128);
    CHECK(g_largest >= (size_t)10 * 128 * sizeof(float));
    free(w);
}

static void test_keep_i8_child(const char *dir) {
    fflush(stdout); fflush(stderr);
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        setenv("COLI_DENSE_KEEP_I8", "1", 1);
        st_init(&g_m.S, dir);
        test_identical(1);
        _exit(0);
    }
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
#endif

int main(void) {
    setenv("COLI_DENSE_BITS", "4", 1);
    setenv("COLI_DENSE_I8", "1", 1);
    unsetenv("COLI_DENSE_INT4");
    unsetenv("COLI_DENSE_KEEP_I8");
    unsetenv("COLI_KEEP_F32");
#ifdef COLI_CUDA
    unsetenv("COLI_CUDA");
#endif
    const char *dir = "tests/tmp_dense_blocks";
    build_fixture(dir);
#ifndef _WIN32
    test_keep_i8_child(dir);         /* before this process reads COLI_DENSE_KEEP_I8 */
#endif
    st_init(&g_m.S, dir);
    test_identical(0);
    test_left_to_whole_matrix_path();
    test_load_tq();
#ifndef _WIN32
    test_no_whole_matrix_staging();
#else
    printf("allocation-size and COLI_DENSE_KEEP_I8 checks skipped on Windows\n");
#endif
    printf("test_qwen36_dense_blocks: OK\n");
    return 0;
}
