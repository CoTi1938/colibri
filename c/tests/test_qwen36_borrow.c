/* Production-gather lifetime/progress regression. Synthetic expert records only:
 * no checkpoint, forward pass, tokenizer or GPU. The spy replaces computation,
 * never gathering, victim selection, publication or borrowing. In numeric cases
 * it delegates to the real helper and compares with independently owned inputs.
 * I/O and condition-wait wrappers only impose deterministic thread schedules.
 */
#define _GNU_SOURCE
#include <stdatomic.h>
#include <errno.h>
#include <pthread.h>
#include "../st.h"
#include "../expert_ffn.h"

static void checked_compute(float *, const float *, int, int, int, int,
                            const int *, const float *, const XfExpert *const *, int, void *);
static void controlled_read(shards *, const char *, void *, int);
static int controlled_wait(pthread_cond_t *, pthread_mutex_t *);
#define xf_moe_run checked_compute
#define st_read_raw controlled_read
#define pthread_cond_wait controlled_wait
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#undef pthread_cond_wait
#undef st_read_raw
#undef xf_moe_run

/* Two input groups also exercise independent gate/up/down scale indexing. */
enum { H = 128, F = 64, NE = 32, KTEST = 8, MAX_PAIRS = 16 };
static atomic_int failures, read_calls;
#define CHECK(c, ...) do { if (!(c)) { atomic_fetch_add(&failures, 1); \
    fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); } } while (0)
static pthread_mutex_t test_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t test_cv = PTHREAD_COND_INITIALIZER;
static struct {
    int compute_entered, compute_continue, read_entered, read_continue;
    int wait_entered, wait_returned, wait_continue, finished;
} events;
static int pause_compute, pause_read_eid = -1, pause_wait_return;
static int real_compute, expected_s, expected_k, compute_calls;
static shards fixture;
static FILE *fixture_file;
static XfExpert stable[NE];

static struct timespec deadline(int seconds) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_sec += seconds; return ts;
}
static void await_locked(int *flag, const char *what) {
    struct timespec until = deadline(20);
    while (!*flag) {
        int rc = pthread_cond_timedwait(&test_cv, &test_mx, &until);
        if (rc) { fprintf(stderr, "FAIL: waiting for %s: %s\n", what, strerror(rc)); exit(2); }
    }
}
static void await(int *flag, const char *what) {
    pthread_mutex_lock(&test_mx); await_locked(flag, what); pthread_mutex_unlock(&test_mx);
}
static void signal_event(int *flag) {
    pthread_mutex_lock(&test_mx); *flag = 1; pthread_cond_broadcast(&test_cv); pthread_mutex_unlock(&test_mx);
}
static void *watchdog(void *arg) {
    (void)arg;
    pthread_mutex_lock(&test_mx);
    struct timespec until = deadline(90);
    while (!events.finished) {
        if (pthread_cond_timedwait(&test_cv, &test_mx, &until) == ETIMEDOUT) {
            fputs("FAIL: cache test deadlocked\n", stderr); exit(2);
        }
    }
    pthread_mutex_unlock(&test_mx); return NULL;
}
static void reset_schedule(void) {
    /* Only called after all actors from the preceding case have joined. */
    pthread_mutex_lock(&test_mx);
    events.compute_entered = events.compute_continue = 0;
    events.read_entered = events.read_continue = 0;
    events.wait_entered = events.wait_returned = events.wait_continue = 0;
    pthread_mutex_unlock(&test_mx);
    pause_compute = pause_wait_return = 0; pause_read_eid = -1;
    compute_calls = 0; real_compute = 0;
}
static int weight_value(int eid, int matrix, int i) {
    return (eid * 3 + matrix * 5 + i * 7) % 16 - 8;
}
static float scale_value(int eid, int matrix, int i) {
    return (1 + eid + matrix * NE + i % 3) / 1024.f;
}
static void make_fixture(void) {
    fixture_file = tmpfile(); if (!fixture_file) { perror("tmpfile"); exit(2); }
    fixture.n = fixture.cap = 2 * NE;
    fixture.t = calloc((size_t)fixture.n, sizeof(st_tensor));
    size_t wb = (size_t)3 * F * H / 2, ns = (size_t)3 * F * H / 64;
    uint8_t *pairs = malloc(wb); float *sc = malloc(ns * sizeof(float));
    for (int e = 0; e < NE; e++) {
        for (int mat = 0; mat < 3; mat++) {
            for (int i = 0; i < F * H; i += 2)
                pairs[(size_t)mat * F * H / 2 + i / 2] = (uint8_t)(
                    (weight_value(e, mat, i) & 15) | ((weight_value(e, mat, i + 1) & 15) << 4));
            for (int i = 0; i < F * H / 64; i++)
                sc[mat * F * H / 64 + i] = scale_value(e, mat, i);
        }
        for (int part = 0; part < 2; part++) {
            st_tensor *t = &fixture.t[2 * e + part]; char name[128];
            snprintf(name, sizeof(name), "model.layers.0.mlp.experts.%d.%s", e, part ? "qs" : "merged_weight");
            t->name = strdup(name); t->fd = fileno(fixture_file); t->off = ftell(fixture_file);
            t->dtype = part ? 2 : 3; t->numel = part ? (int64_t)ns : (int64_t)wb;
            t->nbytes = part ? (int64_t)(ns * sizeof(float)) : (int64_t)wb;
            t->rank = 1; t->shape[0] = t->numel;
            if (fwrite(part ? (void *)sc : (void *)pairs, 1, (size_t)t->nbytes, fixture_file) != (size_t)t->nbytes) exit(2);
        }
        uint8_t *pw = malloc(wb); float *ss = malloc(ns * sizeof(float));
        memcpy(ss, sc, ns * sizeof(float));
        for (int mat = 0; mat < 3; mat++) xf_repack_pairs_signed(
            pw + (size_t)mat * F * H / 2, pairs + (size_t)mat * F * H / 2,
            mat == 2 ? H : F, mat == 2 ? F : H);
        stable[e] = (XfExpert){pw, pw + F * H / 2, pw + F * H,
                              ss, ss + F * H / 64, ss + 2 * F * H / 64};
    }
    fflush(fixture_file); free(pairs); free(sc);
}
static void init_model(Model *m, int cap) {
    memset(m, 0, sizeof(*m)); m->S = fixture;
    m->c.n_layers = 1; m->c.n_experts = NE; m->c.topk = KTEST;
    m->c.hidden = H; m->c.inter = F; m->c.expert_gs = 64;
    m->active_of = calloc(1, sizeof(int)); m->is_pinned = calloc(NE, 1); m->is_queued = calloc(NE, 1);
    m->cache = calloc(1, sizeof(LCache)); LCache *lc = m->cache;
    lc->cap = cap; lc->slots = calloc((size_t)cap, sizeof(Slot));
    lc->slot_by_expert = malloc(NE * sizeof(int));
    for (int e = 0; e < NE; e++) lc->slot_by_expert[e] = -1;
}
static void free_model(Model *m) {
    for (int i = 0; i < m->cache[0].n; i++) {
        Slot *s = &m->cache[0].slots[i]; free(s->g); free(s->pw); free(s->gs);
        free(s->g4); free(s->u4); free(s->d4);
    }
    free(m->cache[0].slots); free(m->cache[0].slot_by_expert); free(m->cache);
    free(m->active_of); free(m->is_pinned); free(m->is_queued);
    if (m->ehit) { free(m->ehit[0]); free(m->ehit); }
}
static void check_inputs(int S, int K, const int *idx, const XfExpert *const *ex) {
    for (int p = 0; p < S * K; p++) {
        if (idx[p] < 0) { CHECK(ex[p] == NULL, "hole has an expert"); continue; }
        CHECK(ex[p] != NULL, "missing expert %d", idx[p]); if (!ex[p]) continue;
        const uint8_t *w[3] = {ex[p]->g4, ex[p]->u4, ex[p]->d4};
        const float *sc[3] = {ex[p]->gs, ex[p]->us, ex[p]->ds};
        int bad_w = 0, bad_s = 0;
        for (int mat = 0; mat < 3; mat++) {
            int I = mat == 2 ? F : H, O = mat == 2 ? H : F;
            for (int r = 0; r < O; r++) for (int i = 0; i < I; i++)
                bad_w += xf_get(w[mat] + (size_t)r * I / 2, i) != weight_value(idx[p], mat, r * I + i);
            for (int i = 0; i < I * O / 64; i++) bad_s += sc[mat][i] != scale_value(idx[p], mat, i);
        }
        CHECK(!bad_w && !bad_s, "expert %d has overwritten data: %d weights, %d scales", idx[p], bad_w, bad_s);
    }
}
static void checked_compute(float *out, const float *x, int S, int K, int h, int f,
                            const int *idx, const float *val, const XfExpert *const *ex, int mode, void *scratch) {
    CHECK(S == expected_s && K == expected_k, "changed partition: S=%d K=%d, wanted %d/%d", S, K, expected_s, expected_k);
    compute_calls++; check_inputs(S, K, idx, ex);
    if (pause_compute) {
        pthread_mutex_lock(&test_mx); events.compute_entered = 1; pthread_cond_broadcast(&test_cv);
        await_locked(&events.compute_continue, "compute continuation"); pthread_mutex_unlock(&test_mx);
        check_inputs(S, K, idx, ex); /* Protection must survive the whole pause. */
    }
    if (real_compute) xf_moe_run(out, x, S, K, h, f, idx, val, ex, mode, scratch);
    else memset(out, 0, (size_t)S * h * sizeof(float));
}
static void controlled_read(shards *S, const char *name, void *out, int drop) {
    int eid = -1; sscanf(name, "model.layers.0.mlp.experts.%d.merged_weight", &eid);
    if (eid == pause_read_eid) {
        pthread_mutex_lock(&test_mx); events.read_entered = 1; pthread_cond_broadcast(&test_cv);
        await_locked(&events.read_continue, "read continuation"); pthread_mutex_unlock(&test_mx);
    }
    atomic_fetch_add(&read_calls, 1); st_read_raw(S, name, out, drop);
}
static int controlled_wait(pthread_cond_t *cv, pthread_mutex_t *mx) {
    signal_event(&events.wait_entered);
    int rc = pthread_cond_wait(cv, mx);
    if (pause_wait_return) {
        /* Preserve pthread_cond_wait's return-with-mutex-held contract, but let
         * prefetch try repeatedly after publication and before demand rescans. */
        pthread_mutex_unlock(mx);
        pthread_mutex_lock(&test_mx); events.wait_returned = 1; pthread_cond_broadcast(&test_cv);
        await_locked(&events.wait_continue, "waiter rescan"); pthread_mutex_unlock(&test_mx);
        pthread_mutex_lock(mx);
    }
    return rc;
}

typedef struct { Model *m; int S; int idx[MAX_PAIRS]; float val[MAX_PAIRS], x[2 * H], out[2 * H]; } Run;
static void init_run(Run *r, Model *m, int S, int repeated) {
    memset(r, 0, sizeof(*r)); r->m = m; r->S = S;
    for (int s = 0; s < S; s++) for (int k = 0; k < KTEST; k++) {
        int p = s * KTEST + k; r->idx[p] = 1 + k + (repeated ? 0 : s * KTEST);
        r->val[p] = (k + 1) / 64.f;
    }
    for (int i = 0; i < S * H; i++) r->x[i] = (i % 17 - 8) / 256.f;
    expected_s = m->cache[0].cap >= S * KTEST ? S : 1;
    expected_k = m->cache[0].cap >= KTEST ? KTEST : 1;
}
static void *run_gather(void *arg) {
    Run *r = arg; moe_xf_run(r->m, 0, r->x, r->S, r->out, r->idx, r->val); return NULL;
}
static void expected_output(const Run *r, float *out) {
    /* Existing computational partitions, independent stable expert storage. */
    void *scratch = malloc(xf_moe_scratch_bytes(expected_s, expected_k, H, F));
    memset(out, 0, (size_t)r->S * H * sizeof(float));
    for (int s0 = 0; s0 < r->S; s0 += expected_s) for (int k0 = 0; k0 < KTEST; k0 += expected_k) {
        int idx[MAX_PAIRS]; float val[MAX_PAIRS], tmp[H]; const XfExpert *ex[MAX_PAIRS];
        for (int s = 0; s < expected_s; s++) for (int k = 0; k < expected_k; k++) {
            int src = (s0 + s) * KTEST + k0 + k, dst = s * expected_k + k;
            idx[dst] = r->idx[src]; val[dst] = r->val[src]; ex[dst] = idx[dst] < 0 ? NULL : &stable[idx[dst]];
        }
        if (expected_k == KTEST) xf_moe_run(out + s0 * H, r->x + s0 * H, expected_s, KTEST, H, F, idx, val, ex, 0, scratch);
        else {
            xf_moe_run(tmp, r->x + s0 * H, 1, 1, H, F, idx, val, ex, 0, scratch);
            for (int d = 0; d < H; d++) out[s0 * H + d] += tmp[d];
        }
    }
    free(scratch);
}
static int cached(Model *m, int eid) {
    pthread_mutex_lock(&g_pilot_mx);
    int found = slot_indexed(m, 0, eid) != NULL;
    pthread_mutex_unlock(&g_pilot_mx); return found;
}
static void force_reuse(Model *m) {
    /* Two disjoint cap-sized working sets must EACH fit simultaneously. Merely
     * fetching many experts could keep recycling the spare slots and miss a
     * leaked borrow. Clear advisory retention, not read ownership, for this test. */
    int cap = m->cache[0].cap;
    pthread_mutex_lock(&g_pilot_mx);
    memset(m->is_pinned, 0, NE);
    for (int i = 0; i < m->cache[0].n; i++) m->cache[0].slots[i].pinned = 0;
    pthread_mutex_unlock(&g_pilot_mx);
    for (int pass = 0; pass < 2; pass++) {
        int first = pass ? NE - cap : 0;
        for (int e = first; e < first + cap; e++) {
            Slot *s; expert_get(m, 0, e, &s);
            CHECK(s->eid == e, "ordinary lookup returned wrong expert");
        }
        for (int e = first; e < first + cap; e++)
            CHECK(cached(m, e), "unreleased storage prevented a complete replacement working set (expert %d)", e);
    }
}
static void case_partition(int S, int cap, int repeated, int numeric) {
    reset_schedule(); real_compute = numeric;
    Model m; init_model(&m, cap); m.is_pinned[0] = 1;
    Slot *hot; expert_get(&m, 0, 0, &hot);
    Run r; init_run(&r, &m, S, repeated); float expected[2 * H];
    if (numeric) expected_output(&r, expected);
    run_gather(&r);
    CHECK(compute_calls == (S / expected_s) * (KTEST / expected_k), "wrong number of computational partitions");
    if (numeric) CHECK(memcmp(r.out, expected, (size_t)S * H * sizeof(float)) == 0, "changed computation S=%d cap=%d", S, cap);
    force_reuse(&m); free_model(&m);
    printf("partition S=%d K=8 cap=%d repeated=%d real-helper=%d checked\n", S, cap, repeated, numeric);
}
static void attempt_prefetch(Model *m, int eid) {
    pthread_mutex_lock(&g_pilot_mx); m->is_queued[eid] = 1; pthread_mutex_unlock(&g_pilot_mx);
    pilot_realload(m, 0, eid);
    pthread_mutex_lock(&g_pilot_mx); CHECK(!m->is_queued[eid], "prefetch left a stale queued flag"); pthread_mutex_unlock(&g_pilot_mx);
}
static void *prefetch_read(void *arg) { attempt_prefetch(arg, 20); return NULL; }
static void *legacy_demand(void *arg) {
    Slot *slot; expert_get(arg, 0, 20, &slot);
    CHECK(slot->eid == 20, "unborrowed waiter returned the wrong expert"); return NULL;
}
static void case_compute_lifetime(int cap) {
    reset_schedule(); pause_compute = 1;
    Model m; init_model(&m, cap); m.is_pinned[0] = 1;
    /* Fill all slots: in cap=16, spare victims exist but expert 0 stays advisory-pinned. */
    for (int e = 0; e < cap; e++) { Slot *s; expert_get(&m, 0, e, &s); }
    Run r; init_run(&r, &m, cap == 16 ? 2 : 1, 1);
    pthread_t thread; if (pthread_create(&thread, NULL, run_gather, &r)) exit(2);
    await(&events.compute_entered, "compute entry");
    int before = atomic_load(&read_calls);
    /* Expert 17 has the SAME packed weights as the oldest borrowed expert 1,
     * but different scales. A weight-only lifetime check would miss this. */
    attempt_prefetch(&m, 17);
    CHECK(atomic_load(&read_calls) == before + (cap == 16), "prefetch did not respect borrowed/unpinned capacity");
    if (cap == 16) {
        CHECK(cached(&m, 0), "prefetch evicted advisory-pinned expert");
        CHECK(cached(&m, 17), "prefetch could not use unborrowed, unpinned storage");
    }
    pthread_t legacy;
    if (cap == 8) {
        if (pthread_create(&legacy, NULL, legacy_demand, &m)) exit(2);
        await(&events.wait_entered, "ordinary demand blocked by all eight borrows");
    }
    signal_event(&events.compute_continue); pthread_join(thread, NULL);
    if (cap == 8) {
        pthread_join(legacy, NULL);
        CHECK(cached(&m, 20), "release did not wake the blocked ordinary lookup");
    }
    attempt_prefetch(&m, 21);
    CHECK(cached(&m, 21), "released storage cannot be reused by prefetch");
    force_reuse(&m); free_model(&m);
    printf("compute-interval protection and reuse cap=%d checked\n", cap);
}
static void case_waiter_rescan(int pinned_victim) {
    reset_schedule(); pause_read_eid = 20; pause_wait_return = 1;
    Model m; init_model(&m, 8); m.is_pinned[20] = (uint8_t)pinned_victim;
    for (int e = 1; e <= 7; e++) { Slot *s; expert_get(&m, 0, e, &s); }
    pthread_t reader, demand;
    if (pthread_create(&reader, NULL, prefetch_read, &m)) exit(2);
    await(&events.read_entered, "admitted prefetch read");
    Run r; init_run(&r, &m, 1, 0);
    if (pthread_create(&demand, NULL, run_gather, &r)) exit(2);
    await(&events.wait_entered, "demand waiting after seven borrows");
    signal_event(&events.read_continue);
    pthread_join(reader, NULL); /* Publication must finish despite the demand claim. */
    await(&events.wait_returned, "demand before victim rescan");
    int before = atomic_load(&read_calls);
    for (int i = 0; i < 32; i++) attempt_prefetch(&m, 21);
    CHECK(atomic_load(&read_calls) == before, "prefetch overtook waiting demand after capacity became available");
    CHECK(cached(&m, 20), "prefetch stole the published rescan victim");
    signal_event(&events.wait_continue); pthread_join(demand, NULL);
    pause_read_eid = -1; pause_wait_return = 0;
    CHECK(compute_calls == 1, "waiter changed the computational partition");
    force_reuse(&m); free_model(&m);
    printf("seven borrows + paused read, publication, repeated prefetch and waiter rescan (pinned=%d) checked\n", pinned_victim);
}
static void case_gather_priority(int cap) {
    /* cap=16: first partition, with never-allocated capacity. cap=8: second
     * partition, with reusable slots from the first. Both need a fresh claim. */
    reset_schedule(); pause_read_eid = cap == 16 ? 1 : 9;
    Model m; init_model(&m, cap);
    Run r; init_run(&r, &m, 2, 0);
    pthread_t demand; if (pthread_create(&demand, NULL, run_gather, &r)) exit(2);
    await(&events.read_entered, "demand read while prefetch has eligible capacity");
    int before = atomic_load(&read_calls);
    for (int i = 0; i < 32; i++) attempt_prefetch(&m, 20);
    CHECK(atomic_load(&read_calls) == before && !cached(&m, 20), "prefetch admitted a miss ahead of an active gather");
    signal_event(&events.read_continue); pthread_join(demand, NULL);
    pause_read_eid = -1;
    CHECK(compute_calls == 2 / expected_s, "gather priority changed computational partitions");
    attempt_prefetch(&m, 20);
    CHECK(cached(&m, 20), "gather claim survived the partition");
    force_reuse(&m); free_model(&m);
    printf("per-batch priority with eligible capacity cap=%d checked\n", cap);
}
static void case_advisory_pins(void) {
    reset_schedule(); Model m; init_model(&m, 2);
    m.is_pinned[0] = m.is_pinned[1] = 1;
    for (int e = 0; e < 2; e++) { Slot *s; expert_get(&m, 0, e, &s); }
    int before = atomic_load(&read_calls);
    attempt_prefetch(&m, 20);
    CHECK(atomic_load(&read_calls) == before && !cached(&m, 20), "prefetch reclaimed an advisory pin");
    Slot *s; expert_get(&m, 0, 2, &s);
    CHECK(s->eid == 2 && !cached(&m, 0) && cached(&m, 1), "demand could not reclaim the oldest advisory pin");
    force_reuse(&m); free_model(&m);
    puts("legacy access adds no release obligation; demand/prefetch pin policies checked");
}
static void case_routing_holes(int empty) {
    reset_schedule(); real_compute = 1;
    Model m; init_model(&m, 8);
    Run r; init_run(&r, &m, 2, 0);
    for (int k = 0; k < KTEST; k++) if (empty || !(k & 1)) {
        r.idx[KTEST + k] = -1; r.val[KTEST + k] = 0;
    }
    float expected[2 * H]; expected_output(&r, expected);
    run_gather(&r);
    CHECK(compute_calls == 2 && !memcmp(r.out, expected, sizeof(expected)), "routing holes changed computation or partitions");
    attempt_prefetch(&m, 20); CHECK(cached(&m, 20), "hole/empty partition leaked its gather claim");
    force_reuse(&m); free_model(&m);
    printf("borrow cleanup across reused scratch with %s routing checked\n", empty ? "empty" : "partial");
}
int main(int argc, char **argv) {
    setenv("QWEN_EXPERT_KERNEL", "1", 1);
#ifdef _OPENMP
    omp_set_num_threads(2);
#endif
    pthread_t guard; if (pthread_create(&guard, NULL, watchdog, NULL)) return 2;
    make_fixture();
    case_partition(1, 8, 0, 0); /* Original self-eviction reproducer, through production gather. */
    if (!(argc > 1 && !strcmp(argv[1], "--gather-only"))) {
        int caps[] = {1, 2, 7, 8, 15, 16};
        for (size_t i = 0; i < sizeof(caps) / sizeof(caps[0]); i++) case_partition(2, caps[i], 0, 1);
        case_partition(2, 16, 1, 1);
        case_compute_lifetime(8); case_compute_lifetime(16);
        case_waiter_rescan(0); case_waiter_rescan(1);
        case_gather_priority(16); case_gather_priority(8); case_advisory_pins();
        case_routing_holes(0); case_routing_holes(1);
    }
    for (int e = 0; e < NE; e++) { free((void *)stable[e].g4); free((void *)stable[e].gs); }
    for (int i = 0; i < fixture.n; i++) free(fixture.t[i].name);
    free(fixture.t); fclose(fixture_file);
    signal_event(&events.finished); pthread_join(guard, NULL);
    if (atomic_load(&failures)) { fprintf(stderr, "%d failure(s)\n", atomic_load(&failures)); return 1; }
    puts("qwen36 production borrowing: ok"); return 0;
}
