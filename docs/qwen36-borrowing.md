# Qwen3.6 packed-expert borrowing: contract and validation

This is a **CPU cache-lifetime correctness repair**, not Metal support or a performance claim. It protects explicitly borrowed expert weights **and scales** while the packed-expert runner uses them. It does not give legacy, unborrowed `expert_get()` callers a new lifetime guarantee or release obligation.

## Finding and upstream refresh

On 2026-09-14, `git fetch origin dev` resolved to:

```text
1efb7c82a0c473984cc0258bfc9e12c0cae94914
```

The production-gather regression was compiled against an unmodified export of that revision and failed as expected:

```text
FAIL: expert 1 has overwritten data: 24576 weights, 384 scales
1 failure(s)
```

Reproducer: cap=8, K=8, an unrelated advisory-pinned expert already resident, followed by a gather of eight distinct experts. The original ordinary victim scan can overwrite a previously gathered expert instead of reclaiming the unrelated pin. No background prefetch is needed.

This is a dated finding at the identified revision, not a claim about future upstream HEAD. Refresh it again before submission if upstream moves.

### Post-CI refresh on current integration tip

A further fetch at **2026-09-14 20:06:55 UTC** resolved `origin/dev` to **`9b4eb1afbddecc47f9b12055a8f4bb9b9993ff93`**. The Qwen engine, expert helper, tier and relevant common headers were byte-identical to the original tested base; the shared Makefile had gained unrelated targets.

In a disposable export of that new tip, the unmodified production-gather reproducer again failed with the expected exit 1. The original five-file implementation patch applied cleanly, compiled with `-Werror`, and passed the complete model-free borrowing regression on the M1 with two-thread OpenMP. No model inference ran.

[Supplemental refresh log and exact commands](https://github.com/CoTi1938/colibri/blob/6ee21113ff66d32f34ba6e09753bb22d0ef0b99b/.github/validation/evidence/upstream-refresh-9b4eb1a.log) ([SHA-256](https://github.com/CoTi1938/colibri/blob/6ee21113ff66d32f34ba6e09753bb22d0ef0b99b/.github/validation/evidence/upstream-refresh-9b4eb1a.log.sha256)). This establishes the refreshed finding and local patch compatibility. It does **not** relabel the earlier Linux/Windows runs as validation of the entire newer upstream tree, and the published implementation history has not been rewritten.

## Frozen contract

- `Slot.pinned` remains an advisory retention preference.
- `Slot.borrows` protects all weight and scale storage in the slot from replacement.
- A successful explicit acquisition increments ownership **inside the same mutex critical section** as lookup or publication.
- Existing `expert_get()` remains unborrowed. The new `expert_borrow()` is used by `moe_xf_run()`.
- All victim selection respects outstanding borrows, regardless of which access API caused the miss.
- Computation, quantization, routing order and the existing batch partitions are unchanged.
- Each successful acquisition is released once after its computational partition, including repeated acquisitions of the same slot.

The additional pointer array has one entry per routed pair in the **existing bounded partition**, not per pair in the entire prompt. The partition rules remain:

```text
S*K <= cap: whole prompt chunk
K <= cap:   one token
otherwise:  one token–expert pair
```

### Per-partition demand priority

```text
register layer claim under cache mutex
    → gather / acquire (may wait for admitted I/O)
    → clear claim
    → compute while borrows remain held
    → release borrows and wake waiters
```

A claim is a priority gate, **not an atomic reservation of all batch slots**. New prefetch misses for that layer are declined while a claim is active. Cache hits and publication of previously admitted reads are still allowed. Declining a prefetch consumes that attempt and clears its queued flag; there is no immediate retry loop.

After gathering, prefetch may use eligible spare storage during computation. It still may not overwrite borrowed or loading storage, or evict advisory pins.

Progress is scoped to one foreground gatherer per model/cache, eventual completion of admitted I/O, and scheduling sufficient to acquire the cache mutex. Since each partition needs no more than `cap` acquisitions, its own borrows cannot fill every slot while it still needs another distinct expert. Existing reads may temporarily occupy the remainder; after publication they can be reclaimed, including advisory-pinned results. New prefetch admissions cannot repeatedly overtake the registered gather.

## Eviction and publication audit

All following transitions use the existing `g_pilot_mx`:

| Path | Eligibility / behavior |
|---|---|
| Demand: unused capacity | Allocate a new slot; mark it loading before unlocking for I/O |
| Demand: ordinary victim | `expert_victim(lc, 0)`: not loading, not borrowed, not pinned; oldest first |
| Demand: pinned fallback | `expert_victim(lc, 1)`: not loading and not borrowed; advisory pins may be reclaimed |
| Demand: waiter rescan | After condition-wait, repeat **both** victim scans under the mutex; no weaker rescan predicate |
| Prefetch: miss admission | Requires no active layer demand-gather claim, even when capacity has never been allocated |
| Prefetch: replacement | `expert_victim(lc, 0)` only; no pinned fallback |
| Demand publication | Publish after the real loader returns; acquire a requested borrow before unlocking; notify waiters |
| Prefetch publication | Publish even if demand has since claimed priority; clear the queue flag and notify waiters |
| Borrow release | Decrement per acquisition, clear the caller's entry, and notify waiters after releasing the partition |

The two production paths that select and overwrite cache slots are `expert_get_impl()` and `pilot_realload()`. Both now use the same victim predicate. `load_expert_merged()` still writes weights and scales only after its caller has selected a slot and marked it loading. Pin-learning operations change retention metadata, not the protected weight/scale contents.

Destruction requires quiescent callers as before. This patch does not make destruction concurrent with borrowed readers safe.

### Caller audit

- CLI generation/serving runs the foreground sequentially.
- The Segment adapter holds `engine->run_lock` around layer execution.
- The Edge adapter's embedding/head operations do not run this expert gatherer.
- CUDA warmstart is parallel but is a separate, full-residency path; `expert_get()` keeps its existing unborrowed behavior. No new CUDA release obligation is introduced.
- Legacy unborrowed CPU paths are **not** newly certified safe against concurrent replacement. This repair's lifetime claim is specifically for explicit borrows in the packed runner.

## Production-runner tests

`c/tests/test_qwen36_borrow.c` includes the production engine and drives `moe_xf_run()`, the actual lookup/loader, and the actual prefetch admission path. It constructs an immutable in-memory tensor index over a temporary file of synthetic expert payloads. It does not load a checkpoint or run a model forward pass.

Fixture geometry: H=128, F=64, E=32, K=8. Gate/up/down have separately identifiable weights and FP32 group scales. The file payload is 442,368 bytes. Safetensors header parsing is not under test.

### Partition and arithmetic checks

| S | K | cap | Expected computation |
|---:|---:|---|---|
| 1 | 8 | 8 | Original pinned self-eviction case |
| 2 | 8 | 1, 2, 7 | One routed pair per invocation |
| 2 | 8 | 8, 15 | One token per invocation |
| 2 | 8 | 16 | Whole chunk |
| 2 | 8 | 16, repeated experts | Whole chunk with multiple acquisitions of shared storage |

The compute wrapper asserts the actual partition sizes and call count. Numeric cases delegate to the **real `xf_moe_run()` helper** and compare bitwise against independently owned stable experts, using the same existing computational partitions. These are same-build synthetic comparisons, not whole-model parity claims. Partial and empty routing in a later partition exercise cleanup when the borrow array is reused.

### Lifetime and progress checks

- The compute spy checks every gate/up/down weight and scale, pauses while another actor attempts replacement, and checks the inputs again before returning.
- Expert 17 has the same packed weights as expert 1 but different scales: the contention case would catch scale corruption even if a weight-only check passed.
- At cap=8 all selected storage is borrowed: prefetch must decline and ordinary demand must wait. Release must wake that waiter.
- At cap=16 with repeated experts, unborrowed **and unpinned** victims remain available to prefetch while computation is paused. The unrelated pin must survive that prefetch.
- After computation, prefetch can reuse released storage.
- Release balance is checked by removing advisory retention and filling **two disjoint cap-sized working sets**, asserting each is fully resident. Merely cycling many requests through spare slots would not detect a partial borrow leak.
- Seven acquired borrows plus an eighth slot occupied by a paused, previously admitted read: demand waits; the read publishes; demand completes the same partition. Both unpinned and pinned publication victims are tested.
- A condition-wait wrapper forces repeated prefetch attempts after publication but before the demand rescan. It returns with the cache mutex held, preserving the wait API's contract. No gathering or ownership operation is replaced.
- Repeated prefetch attempts also occur while demand is paused in its first read with unused capacity, and in the second partition with reusable capacity. Both require an active per-partition claim.
- Prefetch with only advisory-pinned victims declines; ordinary unborrowed demand may reclaim the oldest pin.

Thread coordination uses condition-variable barriers rather than sleep-based assertions. Timed waits/watchdogs fail a deadlocked test; they do not implement runtime timeout recovery.

All normal partition paths release their acquisitions, including repeated references and routing holes. The current loader/allocation failures remain fail-stop; this change introduces no recoverable I/O/OOM protocol. New descriptor allocations occur before any gather claim. Internal ownership imbalance or a condition-wait error aborts rather than allowing silent counter underflow or an unchecked wait failure.

## Validation performed

The tested implementation is commit **`e2a0868ef57630883f1269e765ee4a69a366409d`**, based on `1efb7c82a0c473984cc0258bfc9e12c0cae94914`. The subsequent documentation-only completion records additional evidence without changing the engine, build rules or regression test. The recorded run IDs and source hashes identify the implementation actually tested; they must not be presented as runs of a later documentation commit.

### Local macOS validation

Platform: **Apple M1, 8 GiB RAM, macOS 26.6.2 (25G83)**. Compiler: **Apple clang 21.0.0 (clang-2100.3.34.2)**, macOS SDK **27.0**. OpenMP: **Homebrew libomp 23.1.1**, installed for this validation.

| Check | Result |
|---|---|
| Refreshed upstream production-gather reproducer | Expected failure, exit 1 |
| Patched regression, `-O3`, OpenMP disabled | Pass |
| Patched regression, `-O3`, OpenMP enabled, 2 threads | Pass |
| Repetition | 10/10 additional complete runs in each configuration |
| ASan + UBSan, `-O1`, OpenMP disabled | Pass; leak detection disabled |
| ASan + UBSan, `-O1`, OpenMP enabled, 2 threads | Pass; leak detection disabled |
| TSan, `-O1`, OpenMP disabled; real pthread actors | Pass |
| Qwen CPU executable build | Pass; executable not run |
| Qwen Segment/Edge object build | Pass; adapters not run with a model |
| Engine, adapter object and new regression rebuilt with `-Werror` | Pass at `-O3` with OpenMP; new regression rerun successfully |
| Existing Qwen cache-index, context-allocation, dense/shared, JSON-escape and tokenizer tests | Pass |
| Existing fake-CUDA tier invariants, warmstart and decode-offer tests | Pass; no CUDA device used |
| Existing standalone `test_expert_ffn` | Pass, including shipping expert dimensions; synthetic inputs only |
| `git diff --check` | Pass |

Normal `-O3` builds above emitted no compiler warnings. ASan/UBSan builds emitted an existing `sprintf` deprecation warning in the unchanged JSON-escaping code (`qwen36.c:379`). The same warning was reproduced compiling the test against unmodified upstream. It is documented rather than suppressed or folded into this ownership patch.

### Negative controls

Temporary copies of the patched source were deliberately broken; the working source was not modified:

| Mutation | Detection |
|---|---|
| Demand's pinned fallback ignores borrows | Exit 2: an expected blocked-demand event never occurs |
| Prefetch victim scan ignores borrows | Exit 1: lifetime/scale checks fail |
| Prefetch ignores the gather-priority claim | Exit 1: forbidden admissions/read counts detected |
| Release only once per distinct slot after repeated acquisitions | Exit 1: complete replacement working sets no longer fit |

The minimal pinned reproducer alone is not a sufficient test of the fallback predicate: its unrelated pin is oldest, so that particular weakened fallback may still choose the correct victim. The full contention suite detects the mutation.

### Linux and native Windows validation

Both platforms passed the isolated [GitHub Actions run 34888121651](https://github.com/CoTi1938/colibri/actions/runs/34888121651), using CI commit `bd41870edf052576b0e7be5aeb1a9086aa4c4fd5`. The earlier [run 34886600927](https://github.com/CoTi1938/colibri/actions/runs/34886600927) also passed; the hardware details below belong specifically to the later run, not retroactively to the earlier VM.

| Property | Linux guest | Native Windows guest, not WSL |
|---|---|---|
| OS | Ubuntu 24.04.5 LTS | Windows Server 2022 Datacenter, 10.0.20348 |
| Kernel/build | 6.17.0-1022-azure | Build 20348 |
| Exposed CPU model | AMD EPYC 7763 64-Core Processor | Intel Xeon Platinum 8573C |
| Allocated logical CPUs | 4 (2 exposed cores × 2 threads) | 4 (2 exposed cores × 2 threads) |
| Reported guest RAM | 16,766,414,848 bytes | 17,174,360,064 bytes |
| Runner image | ubuntu24 / 20260907.300.1 | win22 / 20260907.297.1 |
| GCC | Ubuntu GCC 13.3.0 | MSYS2 UCRT64 GCC 16.1.0 |
| Sanitizer compiler | Ubuntu Clang 18.1.3 | Not run |
| Compiled ISA | x86-64-v3, AVX2/FMA | x86-64-v3, AVX2/FMA |

These are virtual-machine guest observations, not a claim to all physical cores named in the CPU brand string or a measurement of the underlying SSD. Each `environment.json` preserves OS/kernel, CPU/topology, RAM, VM identity, guest filesystem, runner image, toolchain, compiler macros and run/source identity. Physical disk type, live CPU frequency and peak process RSS were not measured.

On **both platforms**:

- The five reviewed file hashes matched before and after validation.
- Three complete serial and three complete two-thread OpenMP regressions passed. Serial used `-fno-openmp` to disable code generation; the platform Makefile still linked the OpenMP runtime.
- Engine and adapter compilation with `-Werror` passed; engine binaries were not executed.
- All nine explicitly whitelisted related model-free unit/component tests passed, including the fake-CUDA tests.
- The production-gather negative control failed on the identified upstream base with the expected exit 1.
- All four defect mutations were detected: pinned-fallback mutation exited 2; the other three exited 1.

Linux additionally passed **ASan/UBSan and TSan**, without OpenMP in those sanitizer builds. Leak detection was disabled. TSan alone used per-process ASLR disabling through `setarch -R`; no global kernel settings were changed. No Windows sanitizer result is claimed.

A non-blocking Actions annotation reports that the pinned artifact-upload action declares Node 20 and was forced by the runner to execute on Node 24. This is recorded tooling provenance, not a failed C test.

The validation workflow, collector and scripts are separate from the implementation commit. Exact platform commands and setup are preserved at the tested CI revision:

- [Workflow and native Windows setup](https://github.com/CoTi1938/colibri/blob/bd41870edf052576b0e7be5aeb1a9086aa4c4fd5/.github/workflows/qwen36-cache-validation.yml).
- [Whitelisted builds, repetitions, upstream control and mutations](https://github.com/CoTi1938/colibri/blob/bd41870edf052576b0e7be5aeb1a9086aa4c4fd5/.github/validation/qwen36-cache.sh).
- [Linux sanitizer commands](https://github.com/CoTi1938/colibri/blob/bd41870edf052576b0e7be5aeb1a9086aa4c4fd5/.github/validation/qwen36-sanitizers.sh).
- [Environment collector](https://github.com/CoTi1938/colibri/blob/bd41870edf052576b0e7be5aeb1a9086aa4c4fd5/.github/validation/environment.py).

### Requirement-to-evidence matrix

The following are passes on the recorded macOS, Linux and Windows configurations, not a proof over every possible execution.

| Requirement | Production-runner evidence | Result on all three platforms |
|---|---|---|
| Borrowed weights and scales remain unchanged throughout computation | `case_compute_lifetime()`: check, pause, attempt replacement, recheck; equal weights but different scales for experts 1/17 | Pass |
| Demand normal/fallback/rescan never overwrites borrowed storage | Pinned self-eviction case; `case_waiter_rescan(0/1)`; ordinary demand blocked during compute | Pass |
| Admitted I/O can publish; later prefetch cannot starve a claimed gather | Seven borrows plus a paused read, publication-before-rescan barriers, 32 repeated prefetch attempts | Pass |
| Claims apply to each unchanged computational partition | `case_gather_priority(16/8)`, including unused capacity and the second partition | Pass |
| Repeated-reference acquisition/release remains balanced | Repeated experts across S=2/cap=16; both disjoint cap-sized replacement working sets fit after release | Pass |
| Partial/empty routing does not leak previous borrows or claims | `case_routing_holes(0/1)` plus subsequent prefetch and complete reuse | Pass |
| Retention preferences and legacy access remain distinct from borrowing | Pinned-only prefetch rejection, demand pin reclamation, legacy access requiring no release | Pass; no broader unborrowed lifetime guarantee |
| Partitions and computation remain unchanged | Actual invocation sizes/counts and same-build bitwise comparisons against the real helper on independently owned inputs | Pass |
| Tests detect the intended failures, not just successful execution | Upstream negative control and all four deliberately defective variants | Expected failures detected |

## Durable evidence bundle

Raw local and cross-platform records are archived in the fork, separately from the implementation patch, so they do not depend solely on Actions' 30-day artifact retention:

- [Immutable evidence archive](https://github.com/CoTi1938/colibri/blob/e465cb78a2b7f7c3a5d154d29a4d477dc2458c21/.github/validation/evidence/qwen36-borrowing-2026-09-14.tar.gz).
- [SHA-256 sidecar](https://github.com/CoTi1938/colibri/blob/e465cb78a2b7f7c3a5d154d29a4d477dc2458c21/.github/validation/evidence/qwen36-borrowing-2026-09-14.tar.gz.sha256).

Archive SHA-256:

```text
313183535c0ec7e1d77b62b2d07ebc3edec2f47a0256227504097c844b1cb0dd
```

The archive includes 51 original evidence files, raw compiler/test and negative-control logs, environment/run JSON, source hashes, and a `MANIFEST.json` with per-file SHA-256. Historical logs are preserved unchanged, including descriptions of the pre-commit workspace; their hashes identify the tested source snapshot. The later documentation completion does not change that snapshot's runtime/build/test files.

To retrieve and verify it without running tests or inference:

```sh
curl -fL 'https://raw.githubusercontent.com/CoTi1938/colibri/e465cb78a2b7f7c3a5d154d29a4d477dc2458c21/.github/validation/evidence/qwen36-borrowing-2026-09-14.tar.gz' -o evidence.tar.gz
printf '%s  %s\n' '313183535c0ec7e1d77b62b2d07ebc3edec2f47a0256227504097c844b1cb0dd' 'evidence.tar.gz' | shasum -a 256 -c -
```

## Reproduction commands

The following are the **local macOS** commands, run from the repository root in Bash. `OMPC`/`OMPL` override the macOS Makefile path; they are not a portable way to disable OpenMP on every platform. Use the pinned CI scripts above for Linux/native Windows. All executions below are model-free tests. `qwen36` itself is only compiled.

```sh
# Serial and OpenMP, force rebuild because flags differ.
make -B -C c tests/test_qwen36_borrow OMPC= OMPL=
./c/tests/test_qwen36_borrow
make -B -C c tests/test_qwen36_borrow
OMP_NUM_THREADS=2 ./c/tests/test_qwen36_borrow

# Build production entry points without running them; reject compiler warnings.
make -B -C c qwen36 build/segment/qwen36.o tests/test_qwen36_borrow EXTRA_CFLAGS=-Werror
OMP_NUM_THREADS=2 ./c/tests/test_qwen36_borrow

# Existing related, model-free tests (CUDA tests use the fake backend).
make -C c tests/test_qwen36_cache_index tests/test_qwen36_ctx \
  tests/test_qwen36_dense_batch tests/test_qwen36_tier_invariants \
  tests/test_qwen36_tier_int8_engine tests/test_qwen36_tier_int8_decode \
  tests/test_expert_ffn
for t in test_qwen36_cache_index test_qwen36_ctx test_qwen36_dense_batch \
  test_qwen36_tier_invariants test_qwen36_tier_int8_engine \
  test_qwen36_tier_int8_decode test_expert_ffn; do
  OMP_NUM_THREADS=2 ./c/tests/$t || exit 1
done
make -C c tests/test_qwen36_json_escape tests/test_qwen36_tok_merges
(cd c && ./tests/test_qwen36_json_escape && ./tests/test_qwen36_tok_merges)
```

Sanitizers on the tested macOS toolchain:

```sh
out=$(mktemp -d /tmp/qwen36-borrow-sanitizers.XXXXXX)
flags='-std=gnu11 -O1 -g -Wall -Wextra -Wno-unused-function -Wno-unused-parameter -Wno-misleading-indentation -fno-omit-frame-pointer'
clang $flags -fsanitize=address,undefined c/tests/test_qwen36_borrow.c \
  -o "$out/asan" -lm -pthread
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 "$out/asan"

omp=$(brew --prefix libomp)
clang $flags -Xclang -fopenmp -I"$omp/include" -L"$omp/lib" -lomp \
  -fsanitize=address,undefined c/tests/test_qwen36_borrow.c \
  -o "$out/asan-openmp" -lm -pthread
OMP_NUM_THREADS=2 ASAN_OPTIONS=detect_leaks=0 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 "$out/asan-openmp"

clang $flags -fsanitize=thread c/tests/test_qwen36_borrow.c \
  -o "$out/tsan" -lm -pthread
TSAN_OPTIONS=halt_on_error=1 "$out/tsan"
```

To repeat the negative upstream observation without modifying the checkout:

```sh
up=$(mktemp -d /tmp/qwen36-upstream-repro.XXXXXX)
base=1efb7c82a0c473984cc0258bfc9e12c0cae94914
git archive "$base" c | tar -xf - -C "$up"
cp c/tests/test_qwen36_borrow.c "$up/c/tests/"
clang -std=gnu11 -O3 -Wall -Wextra -Wno-unused-function \
  -Wno-unused-parameter -Wno-misleading-indentation \
  "$up/c/tests/test_qwen36_borrow.c" -o "$up/repro" -lm -pthread
"$up/repro" --gather-only   # expected exit 1 on the identified upstream revision
```

## Explicit exclusions

- No checkpoint inference, real-activation replay, teacher-forced model comparison, quality benchmark, or full-model correctness claim.
- No claim that the full model is practical on 8 GiB RAM; startup memory is unchanged.
- No Metal implementation or real-GPU validation in this patch; fake-CUDA checks are not hardware-CUDA validation.
- No numerical policy, quantization, batch-partition or SIMD redesign; no speedup claim. Incremental locking/allocation overhead has not been benchmarked; CI job durations are not performance measurements.
- Results cover only the recorded macOS, Ubuntu and native Windows Server environments, not every OS release, compiler or hardware configuration. Windows sanitizer coverage is not claimed.
- No full `make check`/model-oracle campaign claimed. Only the explicitly listed builds and model-free tests were run.
- No TSan claim for libomp, leak-checking claim, arbitrary concurrent foreground-gatherer guarantee, cancellation/timeout recovery, or concurrent-destruction guarantee.
- Legacy unborrowed accesses retain their previous caller-specific lifetime requirements.

## Required reassessment

**This repair and its validation must be reassessed, and the implementation, tests and documentation updated if necessary, when model inference becomes feasible or additional validation methods become available.** The current results establish the scoped borrowing/progress contract under the recorded checks; they do not close broader validation permanently.

Reassessment should:

1. Identify the new method, exact source revision, platform/toolchain and inputs; obtain authorization before any model inference or real-activation replay.
2. Repeat the existing model-free regressions, then apply the new checks. Once feasible, model-based checks may include real-activation replay, teacher-forced comparisons and cache-pressure scenarios against a separately verified reference. They are not reported as performed here.
3. Investigate counterexamples or regressions, revise code/tests where warranted, and update the acceptance claims and exclusions to match the new evidence. A previous passing run is not grounds to dismiss a new failure.
4. Preserve the earlier evidence and publish dated follow-up results rather than rewriting historical observations. Changes affecting the frozen contract require explicit review, not an unannounced scope expansion.

Link or attach the evidence bundle when submitting the future PR. This documentation completion opens no PR and performs no inference. The report is reproducible evidence for a bounded contract, not a guarantee that every possible execution is error-free.
