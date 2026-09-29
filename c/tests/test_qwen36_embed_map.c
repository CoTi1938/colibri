/* The token embedding stays mapped in its stored dtype (q36_embed_map) and
 * embed_row widens one row on demand, instead of loading the whole table as
 * f32. What must hold, through q36_load_embedding, the call model_init_range
 * makes:
 *
 *   1. a BF16, F16 or F32 table is mapped, not loaded, and every row is
 *      bit-identical to the f32 table the previous path loaded (load_t_n),
 *      with rows longer than the conversion buffer and a tensor starting at an
 *      odd file offset (rows at odd addresses);
 *   2. a `language_model.`-prefixed table maps the same way;
 *   3. COLI_EMBED_MMAP=0 and a mapping that fails (POSIX: mmap refused) load
 *      the f32 table instead, and embed_row reads it; a real MLX affine
 *      triple is expanded by load_t_n to scale * q + bias;
 *   4. a table whose size disagrees with the config, or a U32 table with the
 *      table's element count, is not mapped: it is left to load_t_n, which
 *      refuses the first (tests/test_qwen36_dense_affine.c);
 *   5. without a mapping, embed_row reads the f32 table, which is what a Model
 *      built by hand in the other tests provides. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef _WIN32
#include <sys/mman.h>
/* Declared under its own name before the macro below renames every later use,
 * the one in compat_map_readonly included; sys/mman.h is already in. */
static int g_mmap_fail = 0, g_mmap_refused = 0;
static void *test_mmap_seam(void *addr, size_t len, int prot, int flags, int fd, off_t off);
#define mmap test_mmap_seam
#endif

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#ifndef _WIN32
#undef mmap
static void *test_mmap_seam(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (g_mmap_fail) { g_mmap_refused++; errno = ENOMEM; return MAP_FAILED; }
    return mmap(addr, len, prot, flags, fd, off);
}
#endif

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                           \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

#define V 37
#define D 600          /* more than embed_row's 512-half buffer */

static uint32_t g_rng = 777u;
static uint32_t rnd(void) { g_rng = g_rng * 1664525u + 1013904223u; return g_rng >> 8; }

static void *values(const char *dtype, size_t *nbytes) {
    size_t n = (size_t)V * D, esz = strcmp(dtype, "F32") ? 2 : 4;
    unsigned char *p = malloc(n * esz);
    CHECK(p != NULL);
    for (size_t k = 0; k < n; k++) {
        uint32_t r = rnd();
        if (esz == 4) {
            float v = ((float)(r % 20001) - 10000.f) / 3000.f;
            memcpy(p + k * 4, &v, 4);
        } else {
            uint16_t h = (uint16_t)(r & 0xFFFFu);
            if (!strcmp(dtype, "F16") && ((h >> 10) & 0x1F) == 0x1F) h &= 0xBFFF;     /* no inf/NaN */
            if (!strcmp(dtype, "BF16") && ((h >> 7) & 0xFF) == 0xFF) h &= 0xBFFF;
            memcpy(p + k * 2, &h, 2);
        }
    }
    *nbytes = n * esz;
    return p;
}

typedef struct { const char *name, *dtype, *shape; const void *bytes; size_t nbytes; } Tensor;

/* One pad byte ahead of the tensors, and the header padded with spaces so
 * that the first one's absolute offset (8 + header + 1) is odd. */
static void write_tensors(const char *dir, const Tensor *t, int n) {
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0700);
#endif
    char header[2048];
    size_t off = 1;
    int hlen = snprintf(header, sizeof header,
        "{\"model.pad\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}");
    for (int i = 0; i < n; i++) {
        hlen += snprintf(header + hlen, sizeof header - (size_t)hlen,
            ",\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%zu,%zu]}",
            t[i].name, t[i].dtype, t[i].shape, off, off + t[i].nbytes);
        off += t[i].nbytes;
    }
    hlen += snprintf(header + hlen, sizeof header - (size_t)hlen, "}");
    CHECK(hlen > 0 && hlen + 1 < (int)sizeof header);
    if (hlen % 2) header[hlen++] = ' ';
    char path[512];
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    uint64_t h64 = (uint64_t)hlen;
    unsigned char pad = 0;
    CHECK(fwrite(&h64, 8, 1, f) == 1 && fwrite(header, 1, (size_t)hlen, f) == (size_t)hlen);
    CHECK(fwrite(&pad, 1, 1, f) == 1);
    for (int i = 0; i < n; i++) CHECK(fwrite(t[i].bytes, 1, t[i].nbytes, f) == t[i].nbytes);
    CHECK(fclose(f) == 0);
}

static void write_table(const char *dir, const char *name, const char *dtype,
                        const char *shape, const void *bytes, size_t nbytes) {
    Tensor t = { name, dtype, shape, bytes, nbytes };
    write_tensors(dir, &t, 1);
}

static void open_model(Model *m, const char *dir, int vocab, int hidden) {
    memset(m, 0, sizeof *m);
    m->c.vocab = vocab; m->c.hidden = hidden;
    st_init(&m->S, dir);
}

/* embed_row against the f32 table load_t_n loads from the same checkpoint. */
static void check_rows(const Model *m, Model *ref) {
    float *table = load_t_n(ref, "model.embed_tokens.weight", (int64_t)V * D);
    float out[D];
    for (int tok = 0; tok < V; tok++) {
        memset(out, 0xA5, sizeof out);
        embed_row(m, tok, out);
        CHECK(memcmp(out, table + (size_t)tok * D, sizeof out) == 0);
    }
    free(table);
}

static void test_rows_identical(const char *dtype, const char *dir, const char *name) {
    size_t nbytes;
    void *bytes = values(dtype, &nbytes);
    char shape[32];
    snprintf(shape, sizeof shape, "[%d,%d]", V, D);
    write_table(dir, name, dtype, shape, bytes, nbytes);
    free(bytes);
    static Model m;
    open_model(&m, dir, V, D);
    st_tensor *t = st_find(&m.S, name);
    CHECK(t != NULL && t->off % 2 == 1);
    q36_load_embedding(&m);
    CHECK(m.embed == NULL && m.embed_map.data != NULL);
    check_rows(&m, &m);
}

static void test_f32_fallbacks(void) {
    const char *dir = "tests/tmp_embed_map_plain";
    size_t nbytes;
    void *bytes = values("F16", &nbytes);
    write_table(dir, "model.embed_tokens.weight", "F16", "[37,600]", bytes, nbytes);
    free(bytes);
    static Model m;
    open_model(&m, dir, V, D);
    setenv("COLI_EMBED_MMAP", "0", 1);
    q36_load_embedding(&m);
    unsetenv("COLI_EMBED_MMAP");
    CHECK(m.embed != NULL && m.embed_map.data == NULL);
    check_rows(&m, &m);
    free(m.embed);
#ifndef _WIN32
    static Model f;
    open_model(&f, dir, V, D);
    g_mmap_fail = 1;
    q36_load_embedding(&f);
    g_mmap_fail = 0;
    CHECK(g_mmap_refused == 1 && f.embed != NULL && f.embed_map.data == NULL);
    check_rows(&f, &f);
    free(f.embed);
#endif
}

/* Not mapped, and so left to load_t_n: a size that disagrees with the config,
 * and packed U32 words with the table's element count (only the dtype tells
 * those from data). */
static void test_not_mapped(void) {
    static Model m;
    open_model(&m, "tests/tmp_embed_map_plain", V - 1, D);
    CHECK(q36_embed_map(&m) == 0 && m.embed_map.data == NULL);
    m.c.vocab = V;
    CHECK(q36_embed_map(&m) == 1);
    static uint32_t words[V * D];
    write_table("tests/tmp_embed_map_u32", "model.embed_tokens.weight", "U32", "[37,600]", words, sizeof words);
    static Model a;
    open_model(&a, "tests/tmp_embed_map_u32", V, D);
    CHECK(q36_embed_map(&a) == 0 && a.embed_map.data == NULL);
}

/* An MLX affine Q8 table, 3 x 64 in groups of 32, with bf16-exact scales and
 * biases: load_t_n expands it to scale * q + bias. */
static uint16_t bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return (uint16_t)(bits >> 16);
}
static void test_affine_table_expands(void) {
    enum { AV = 3, AD = 64, GS = 32 };
    static uint32_t words[AV * AD / 4];
    static uint16_t scales[AV * AD / GS], biases[AV * AD / GS];
    for (int r = 0; r < AV; r++)
        for (int j = 0; j < AD; j++)
            words[(r * AD + j) / 4] |= (uint32_t)((r * 71 + j * 5) & 0xFF) << (8 * (j % 4));
    for (int g = 0; g < AV * AD / GS; g++) {
        scales[g] = bf16_bits(0.5f * (float)(1 + g));
        biases[g] = bf16_bits(0.25f * (float)g - 1.0f);
    }
    const Tensor t[] = {
        { "model.embed_tokens.weight", "U32", "[3,16]", words, sizeof words },
        { "model.embed_tokens.scales", "BF16", "[3,2]", scales, sizeof scales },
        { "model.embed_tokens.biases", "BF16", "[3,2]", biases, sizeof biases },
    };
    write_tensors("tests/tmp_embed_map_affine", t, 3);
    static Model m;
    open_model(&m, "tests/tmp_embed_map_affine", AV, AD);
    q36_load_embedding(&m);
    CHECK(m.embed != NULL && m.embed_map.data == NULL);
    float out[AD];
    for (int r = 0; r < AV; r++) {
        embed_row(&m, r, out);
        for (int j = 0; j < AD; j++) {
            int g = (r * AD + j) / GS;
            float want = 0.5f * (float)(1 + g) * (float)((r * 71 + j * 5) & 0xFF) + (0.25f * (float)g - 1.0f);
            CHECK(out[j] == want);
        }
    }
    free(m.embed);
}

static void test_unmapped_reads_f32(void) {
    static Model m;
    memset(&m, 0, sizeof m);
    m.c.vocab = 2; m.c.hidden = D;
    static float table[2 * D];
    for (int i = 0; i < 2 * D; i++) table[i] = (float)i * 0.5f;
    m.embed = table;
    float out[D];
    embed_row(&m, 1, out);
    CHECK(memcmp(out, table + D, sizeof out) == 0);
}

int main(void) {
    unsetenv("COLI_EMBED_MMAP");
    test_rows_identical("BF16", "tests/tmp_embed_map_bf16", "model.embed_tokens.weight");
    test_rows_identical("F16", "tests/tmp_embed_map_f16", "model.embed_tokens.weight");
    test_rows_identical("F32", "tests/tmp_embed_map_f32", "model.embed_tokens.weight");
    test_rows_identical("F16", "tests/tmp_embed_map_prefixed", "language_model.model.embed_tokens.weight");
    test_f32_fallbacks();
    test_not_mapped();
    test_affine_table_expands();
    test_unmapped_reads_f32();
#ifdef _WIN32
    printf("mapping-failure injection skipped on Windows (no mmap seam)\n");
#endif
    printf("test_qwen36_embed_map: OK\n");
    return 0;
}
