# Metal PR2308 — Apple M1 qualification, 14 November 2026

A: 9e7bd192289ec5f9686f33c8f74468425f96587b (original PR2308 head).
B: 1b17e9c1761e98f1839997cf5016702bf02fb83e (rebase/qwen36-metal-graph).
The comparison measures the complete integration, not any individual merge fix.
Hardware: MacBookAir10,1, Apple M1, 7 GPU cores, 8 GiB RAM; macOS 26.6.2 (25G83).
Compiler: Apple clang 21.0.0 (clang-2100.3.34.2), SDK 27.0, Homebrew libomp.
Exact commands and compiler output are in qualification-logs.json and environment.json.

Checkpoint: Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64, revision c619aa594ad1e70af82168fb6b4878427896e21c.
GGUF: unsloth/Qwen3.6-35B-A3B-GGUF, revision a483e9e6cbd595906af30beda3187c2663a1118c, Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.
GGUF SHA256: ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61. Per-file checkpoint and sidecar hashes are in provenance.json.

## Build and functional scope

Both runtime and frozen harness builds completed. The candidate kernel suite passed with zero failures.
Candidate graph suite: 32 tests passed, no skips, on the actual M1; environment registry: 810 variables.
The kernel log ends with PASSED: 0 failure(s); the graph count refers to test cases.
The candidate frozen-harness build emitted two unused-variable warnings (N and greedy) in the included qwen36.c; the build succeeded and the full compiler output is included.
Real-model results, including negative attempts and corrected CPU-reference configuration, are in the functional logs.
Checkpoint CPU exact initially used f16 KV while Metal exact used f32; aligning only Q36_KV=f32 passed the unchanged 2e-4 screen.
With CPU KV aligned, relative maximum logit differences were 2.904913e-6 (checkpoint) and 1.627458e-6 (GGUF); A/B full logits were bitwise identical within both exact and fast modes on each format.
Stateful comparison: 10/10 passed per format, including actual prefix reuse, pin 0/1 and reset/replay; outputs and layer dumps matched fresh executions.
GGUF dense mapping: 1910087680 bytes; successful lock: 1914208256 bytes, matching an independent page-rounded tensor-range calculation (not a count of unique physical pages).
Lock/unlock balance was checked after decode and for a one-token request with no decode; real dense/expert GPU product counters were nonzero.
GGUF fast versus exact has a relative maximum logit gap 0.0735429 on this small sample. Both modes separately match A/B; this is not a new integration divergence or a quality score.
Q36_KV controls the CPU path; setting it does not force fast Metal to float KV. No claim is made that KV alone explains the fast/exact gap.
The 16-token/4-step numerical screen is not a whole-model quality guarantee. Stateful tests compare 3 turns plus reset/replay, pin 0/1, to fresh processes.

## Measurement protocol

Within each format, A/B use the same effective weights/sidecars, input text and frozen C harness. Expert cache requested 536870912 bytes.
Checkpoint and GGUF are distinct quantizations; their outputs are not required to be identical to each other.
Actual cache: checkpoint 303 slots of 1769472 bytes (536150016 total); GGUF 263 slots of 2039808 bytes (536469504 total).
At this fixed budget the GGUF log selects the eight-expert prompt schedule; cache geometry is recorded rather than retuned between versions.
Default fast kernels, without MTP or approximate expert accelerators. Forced next tokens keep the measured work matched.
Per text, separate cohorts: 64+32 steps, 512+32 steps, 64+32 steps in one process; and one 512-token prompt + 32 steps then exit.
The latter is cold engine/expert-cache state; the OS filesystem cache was not purged.
Four processes per variant, ABBAABBA, nominal thermal state and 60 s rest before each. One excluded A warmup per block.
The original plan prose said a warmup per version; the frozen historical driver actually warms A only. This deviation is disclosed and identical for both formats.
First decode predicts the second new token; the first comes from prefill. Steady decode covers forward steps 2–32.
The historical harness emits 32 selected tokens but its final forward predicts an unused 33rd. These are forward-step rates, not end-to-end chat throughput.
Ready time excludes earlier tokenizer loading/tokenization. No concurrent builds or model transfers ran during speed measurements; diagnostic timer instrumentation was disabled.
Ready includes runtime model and Metal setup, including newLibraryWithSource and pipeline creation; the Metal driver's compilation cache was not isolated.
Ready is printed with 0.01 s resolution; report small absolute loading differences alongside percentages and ranges.
The second-token column sums timed prefill and first forward per run, then takes median/range. It excludes untimed bookkeeping between calls and is not end-to-end serving latency.
RSS/footprint are whole-process peaks: the three sequence rows share the peak over the entire sequence.
Swap deltas are system-wide writes during a process, not total pre-existing swap or process-private I/O.
Logical expert misses are not physical SSD reads. Loader seconds sum overlapping worker activity and cannot be added to wall time.

All values below are median [min,max]; sample counts are explicit. GGUF code-sequence retains 3 A and 4 B after excluding one swap-overlapping run; no replacement run was collected for this cohort.
The GGUF ready-time check started with one A+B pair, followed by a full 4+4 repeat. Both are retained; the original clean cohort remains primary. Missing data is never reported as zero.

## CHECKPOINT

Review decision: Checkpoint qualification and collection complete. First-decode performance concern remains open; no unconditional performance promotion.

| case | n A/B | prefill A → B (tokens/s) | first decode A → B (ms) | second token from ready A → B (s) | steady decode A → B (steps/s) |
|---|---|---|---|---|---|
| chat-cold512, 512 cold | 4/4 | 102.94 [102.54, 104.91] → 101.94 [101.91, 102.81] | 377.7 [353.5, 396.6] → 387.1 [383.4, 471.6] | 5.345 [5.277, 5.358] → 5.410 [5.406, 5.452] | 7.228 [7.146, 7.245] → 7.181 [7.128, 7.199] |
| chat-sequence, 64 first | 4/4 | 20.39 [19.71, 21.00] → 20.78 [20.77, 21.00] | 327.8 [300.7, 344.1] → 315.7 [314.9, 335.5] | 3.454 [3.373, 3.592] → 3.395 [3.363, 3.417] | 7.599 [7.509, 7.679] → 7.634 [7.599, 7.729] |
| chat-sequence, 512 after64 | 4/4 | 107.59 [104.32, 109.07] → 108.68 [107.35, 109.29] | 345.5 [321.3, 357.9] → 332.3 [316.8, 343.3] | 5.094 [5.037, 5.266] → 5.042 [5.010, 5.107] | 7.447 [7.385, 7.464] → 7.399 [6.821, 7.431] |
| chat-sequence, 64 after512 | 4/4 | 22.54 [22.33, 23.21] → 22.58 [22.47, 22.69] | 264.5 [251.6, 280.4] → 256.9 [243.0, 334.6] | 3.103 [3.012, 3.146] → 3.091 [3.063, 3.183] | 7.705 [7.657, 7.826] → 7.736 [7.693, 7.769] |
| code-cold512-repeat1, 512 cold | 4/4 | 100.06 [97.16, 101.49] → 98.10 [97.47, 101.05] | 412.0 [380.7, 464.3] → 418.9 [360.2, 508.3] | 5.540 [5.454, 5.684] → 5.608 [5.487, 5.761] | 7.339 [7.208, 7.503] → 7.410 [7.342, 7.465] |
| code-sequence, 64 first | 4/4 | 22.27 [21.80, 22.62] → 23.27 [23.17, 23.42] | 339.2 [302.7, 381.2] → 305.0 [294.2, 315.3] | 3.218 [3.133, 3.305] → 3.057 [3.027, 3.073] | 7.793 [7.692, 7.886] → 7.872 [7.793, 7.901] |
| code-sequence, 512 after64 | 4/4 | 104.72 [101.35, 107.98] → 106.65 [105.25, 108.90] | 334.9 [296.2, 421.4] → 365.9 [329.4, 385.0] | 5.199 [5.089, 5.473] → 5.158 [5.054, 5.245] | 7.470 [7.392, 7.651] → 7.619 [7.546, 7.725] |
| code-sequence, 64 after512 | 4/4 | 24.83 [24.50, 25.43] → 25.16 [25.11, 25.38] | 257.1 [235.6, 284.1] → 298.6 [265.8, 330.0] | 2.822 [2.797, 2.878] → 2.829 [2.814, 2.879] | 7.890 [7.771, 7.941] → 7.970 [7.909, 8.084] |
| prose-cold512, 512 cold | 4/4 | 121.60 [119.64, 123.61] → 121.36 [119.87, 122.50] | 362.2 [317.6, 370.7] → 367.8 [330.0, 379.6] | 4.578 [4.460, 4.641] → 4.588 [4.554, 4.603] | 9.043 [9.011, 9.117] → 9.125 [8.915, 9.191] |
| prose-sequence, 64 first | 4/4 | 25.52 [25.25, 26.11] → 25.37 [24.80, 25.65] | 275.2 [265.7, 287.7] → 262.4 [257.5, 272.2] | 2.784 [2.724, 2.812] → 2.785 [2.754, 2.853] | 8.873 [8.834, 8.926] → 8.928 [8.829, 8.976] |
| prose-sequence, 512 after64 | 4/4 | 132.56 [131.37, 132.95] → 130.41 [127.62, 132.06] | 266.1 [256.0, 287.7] → 271.8 [263.8, 296.0] | 4.125 [4.114, 4.185] → 4.197 [4.144, 4.308] | 9.145 [9.087, 9.351] → 9.119 [9.053, 9.215] |
| prose-sequence, 64 after512 | 4/4 | 27.95 [27.46, 28.43] → 27.67 [27.61, 27.83] | 244.8 [228.3, 248.2] → 252.9 [240.6, 265.2] | 2.534 [2.479, 2.579] → 2.565 [2.542, 2.583] | 9.015 [8.929, 9.123] → 8.962 [8.940, 8.985] |

| process cohort | n A/B | ready A → B (s) | RSS A → B (MiB) | footprint A → B (MiB) |
|---|---|---|---|---|
| chat-cold512 | 4/4 | 0.190 [0.190, 0.200] → 0.185 [0.180, 0.190] | 1472.71 [1454.30, 1489.19] → 1491.93 [1481.67, 1507.27] | 982.83 [982.46, 983.41] → 985.36 [983.85, 986.30] |
| chat-sequence | 4/4 | 0.190 [0.180, 0.190] → 0.190 [0.180, 0.190] | 1563.42 [1486.97, 1568.69] → 1561.67 [1549.59, 1595.39] | 986.41 [985.93, 986.87] → 986.84 [986.10, 986.99] |
| code-cold512-repeat1 | 4/4 | 0.190 [0.180, 0.190] → 0.190 [0.190, 0.200] | 1453.95 [1418.47, 1473.34] → 1449.23 [1398.83, 1454.98] | 982.12 [981.88, 983.09] → 984.93 [984.69, 985.65] |
| code-sequence | 4/4 | 0.190 [0.180, 0.190] → 0.190 [0.180, 0.200] | 1535.12 [1422.42, 1564.58] → 1551.13 [1492.33, 1566.14] | 986.07 [985.68, 986.41] → 985.65 [984.77, 986.63] |
| prose-cold512 | 4/4 | 0.190 [0.180, 0.190] → 0.190 [0.190, 0.190] | 1631.86 [1612.28, 1658.36] → 1581.32 [1566.45, 1604.95] | 984.64 [983.93, 985.51] → 985.05 [984.38, 985.27] |
| prose-sequence | 4/4 | 0.185 [0.170, 0.200] → 0.190 [0.180, 0.200] | 1539.34 [1529.78, 1570.56] → 1539.54 [1515.77, 1550.25] | 988.47 [988.16, 988.59] → 988.46 [986.91, 988.62] |

All individual samples, repeats, exclusions and per-run power/thermal snapshots: checkpoint-aggregate.json, checkpoint-samples.csv and checkpoint-measurement-logs.json.

Conditions: AC, 80%, all before/after thermal 0 and system swapout delta 0.

Protocol/exclusion note: Historical frozen driver uses one excluded A warmup per block, not one per version as plan prose stated. Keep identical protocol on GGUF.

Protocol/exclusion note: One code-cold512 process overlapped hook tests; original retained and full cohort repeated cleanly.

## GGUF

Review decision: GGUF functional qualification and measurement collection complete. No A/B numerical mismatch in the checked scope. Retain the repeated cold-code startup concern; not an unconditional performance promotion. Checkpoint first-decode concern also remains open.

| case | n A/B | prefill A → B (tokens/s) | first decode A → B (ms) | second token from ready A → B (s) | steady decode A → B (steps/s) |
|---|---|---|---|---|---|
| chat-cold512, 512 cold | 4/4 | 61.44 [59.85, 61.51] → 61.51 [61.22, 61.92] | 390.6 [388.4, 404.6] → 396.8 [390.0, 400.1] | 8.725 [8.712, 8.960] → 8.717 [8.668, 8.763] | 5.399 [5.211, 5.481] → 5.397 [5.252, 5.421] |
| chat-sequence, 64 first | 4/4 | 14.82 [14.31, 14.86] → 14.85 [14.72, 14.92] | 386.4 [378.0, 400.3] → 388.2 [384.1, 394.2] | 4.703 [4.692, 4.872] → 4.696 [4.680, 4.741] | 5.495 [5.348, 5.529] → 5.460 [5.383, 5.525] |
| chat-sequence, 512 after64 | 4/4 | 62.74 [57.91, 63.12] → 62.78 [62.44, 62.86] | 324.3 [317.8, 332.6] → 319.9 [315.2, 326.1] | 8.485 [8.430, 9.174] → 8.474 [8.464, 8.526] | 5.479 [5.317, 5.519] → 5.444 [5.386, 5.581] |
| chat-sequence, 64 after512 | 4/4 | 15.50 [14.67, 15.55] → 15.46 [15.42, 15.63] | 326.2 [311.1, 338.5] → 307.1 [302.3, 314.7] | 4.453 [4.430, 4.699] → 4.447 [4.396, 4.464] | 5.623 [5.478, 5.671] → 5.655 [5.496, 5.660] |
| code-cold512, 512 cold | 4/4 | 60.55 [60.26, 60.69] → 60.52 [56.13, 60.85] | 393.4 [392.2, 394.5] → 387.8 [384.5, 391.9] | 8.849 [8.831, 8.888] → 8.849 [8.798, 9.514] | 5.598 [5.580, 5.609] → 5.627 [5.597, 5.689] |
| code-sequence, 64 first | 3/4 | 16.30 [16.26, 16.47] → 16.33 [15.98, 16.43] | 389.3 [385.9, 392.9] → 391.9 [384.2, 401.5] | 4.316 [4.279, 4.323] → 4.312 [4.296, 4.389] | 5.665 [5.632, 5.711] → 5.635 [5.487, 5.658] |
| code-sequence, 512 after64 | 3/4 | 62.13 [61.97, 62.14] → 61.85 [61.51, 61.92] | 340.5 [337.4, 341.9] → 338.8 [335.8, 349.7] | 8.581 [8.577, 8.603] → 8.617 [8.605, 8.674] | 5.699 [5.678, 5.762] → 5.691 [5.548, 5.734] |
| code-sequence, 64 after512 | 3/4 | 17.14 [17.04, 17.22] → 17.02 [16.91, 17.08] | 291.9 [288.0, 298.9] → 297.4 [288.4, 307.6] | 4.021 [4.009, 4.056] → 4.058 [4.035, 4.093] | 5.853 [5.735, 5.877] → 5.713 [5.633, 5.867] |
| prose-cold512, 512 cold | 4/4 | 67.75 [66.51, 68.17] → 67.51 [66.87, 67.98] | 375.8 [369.6, 380.3] → 378.1 [374.3, 380.1] | 7.932 [7.884, 8.076] → 7.963 [7.905, 8.035] | 6.623 [6.571, 6.640] → 6.625 [6.506, 6.657] |
| prose-sequence, 64 first | 4/4 | 18.37 [18.16, 18.44] → 18.31 [18.21, 18.37] | 376.0 [373.3, 385.3] → 375.1 [368.8, 379.9] | 3.865 [3.844, 3.900] → 3.874 [3.856, 3.884] | 6.442 [6.295, 6.465] → 6.438 [6.417, 6.488] |
| prose-sequence, 512 after64 | 4/4 | 69.04 [67.49, 69.39] → 69.16 [68.47, 69.24] | 302.7 [299.2, 306.6] → 300.0 [298.3, 302.3] | 7.718 [7.678, 7.893] → 7.704 [7.696, 7.776] | 6.761 [6.653, 6.763] → 6.767 [6.732, 6.775] |
| prose-sequence, 64 after512 | 4/4 | 19.54 [19.32, 19.73] → 19.51 [19.37, 19.64] | 283.6 [276.9, 292.4] → 286.8 [281.4, 302.5] | 3.559 [3.521, 3.605] → 3.569 [3.540, 3.601] | 6.766 [6.680, 6.860] → 6.662 [6.632, 6.811] |

| process cohort | n A/B | ready A → B (s) | RSS A → B (MiB) | footprint A → B (MiB) |
|---|---|---|---|---|
| chat-cold512 | 4/4 | 0.160 [0.140, 0.170] → 0.140 [0.140, 0.220] | 2553.64 [2513.86, 2594.81] → 2563.05 [2513.61, 2602.80] | 1091.62 [1091.53, 1091.66] → 1091.51 [1091.48, 1091.53] |
| chat-sequence | 4/4 | 0.220 [0.180, 0.250] → 0.155 [0.140, 0.180] | 2533.35 [2517.98, 2617.12] → 2509.99 [2503.27, 2527.31] | 1097.17 [1097.05, 1097.20] → 1097.04 [1097.03, 1097.12] |
| code-cold512 | 4/4 | 0.145 [0.140, 0.210] → 0.195 [0.140, 0.220] | 2512.39 [2512.22, 2530.41] → 2513.26 [2512.97, 2581.59] | 1091.66 [1091.53, 1091.72] → 1091.56 [1091.23, 1091.70] |
| code-sequence | 3/4 | 0.170 [0.140, 0.200] → 0.185 [0.160, 0.190] | 2521.08 [2521.08, 2533.12] → 2514.70 [2494.92, 2520.83] | 1097.20 [1097.14, 1097.23] → 1097.00 [1091.76, 1097.01] |
| prose-cold512 | 4/4 | 0.190 [0.140, 0.290] → 0.145 [0.140, 0.230] | 2494.44 [2386.41, 2639.34] → 2513.30 [2375.59, 2599.39] | 1091.68 [1091.61, 1091.72] → 1091.55 [1089.98, 1091.69] |
| prose-sequence | 4/4 | 0.165 [0.140, 0.200] → 0.155 [0.130, 0.230] | 2523.27 [2520.31, 2526.89] → 2524.32 [2512.80, 2527.44] | 1097.23 [1097.11, 1097.33] → 1097.12 [1097.09, 1097.20] |

All individual samples, repeats, exclusions and per-run power/thermal snapshots: gguf-aggregate.json, gguf-samples.csv and gguf-measurement-logs.json.

Conditions: All nonexcluded timed processes have AC power, nominal before/after thermal state and zero system swapout delta. Original code-sequence A run 4 has +8608 system swapout pages; source process unknown, excluded. No other within-run swap increase observed.

Protocol/exclusion note: One excluded A warmup per block, matching the frozen driver; initial plan prose said one per version.

Protocol/exclusion note: GGUF code-sequence retains 3 A and 4 B after excluding A run 4; it was not repeated. This departs from the planned 4+4 count. All original 8 raw processes remain.

Protocol/exclusion note: A single A+B ready diagnostic was followed by one full 4+4 repeat. Both are supplementary; the original clean cold-code cohort remains primary.

Performance observation: GGUF cold-code ready time: original medians A 145 ms [140–210] versus B 195 ms [140–220] (+50 ms, +34.48%); full repeat A 155 ms [140–200] versus B 175 ms [160–220] (+20 ms, +12.90%). The intervening single A+B diagnostic was 140 ms for both. Preserve all three sets; the single pair does not overturn the repeated median direction.

Performance observation: Ready is logged at 10 ms resolution and excludes prior tokenizer/input preparation. The frozen harness timer brackets model_init through bench_model plus serve_eos_ids; it does not isolate I/O, allocation or Metal setup. No new swap/thermal event occurred in either cold-code cohort. The absolute gap is small and ranges overlap, but both complete cohorts exceed the predeclared 10% relative screen: retain a startup performance concern, with cause unassigned, rather than an unconditional performance promotion. No further automatic experiments or speculative runtime changes. Ready includes runtime Metal library/pipeline setup (q36_gpu_init -> q36_compile/newLibraryWithSource and newComputePipelineStateWithFunction); the driver compilation cache was not isolated.

Performance observation: GGUF prefill and decode medians show no regression above 5% in the retained primary comparisons or the cold-code repeat. Reduced code-sequence ready medians are 170 ms versus 185 ms (+8.82%, 3 A/4 B); that cohort was not repeated after the documented exclusion.

## Checkpoint first-decode concern

Code 512 after 64: first cohort A 334.85 ms → B 365.90 ms (+9.27%); repeat A 363.30 ms → B 394.15 ms (+8.49%).
The final 64: first cohort A 257.10 ms → B 298.55 ms (+16.12%); repeat A 257.70 ms → B 278.95 ms (+8.25%).
Both cohorts retain the same direction, with overlapping ranges and varying magnitude. Pooled medians are descriptive, not a replacement for either cohort.
The first 64 code prompt changes from a 10.10% first-decode improvement in the original cohort to a 7.11% slowdown in the repeat; the apparent original gain is not stable.
A/B logits fingerprints, token hashes and logical miss counts match. No new swap or thermal event was observed.
The core Metal graph/device/cache/sidecar source blobs are identical; the broader host runtime differs. This does not establish equal execution cost.
Worker loader sums do not consistently account for the wall-time increase. Cause remains unassigned; this report is not an unconditional no-regression endorsement.
Original and repeat logs and the bounded diagnosis are retained. No speculative runtime patch or extra retries were used to erase the finding.
Original code-cold512 had one run overlapping the hook unit suite; it is explicitly excluded. The entire cohort was repeated cleanly and the repeat is primary.

## Historical comparison

historical-reference.json records the October measurements at fixed source revisions. They are context, not a controlled extension of today’s A/B.
The historical model byte hashes and complete execution environment were not recovered; matching filenames/sizes do not prove identical weights.
The historical record uses bench.c; this campaign uses the frozen qwen36_bench.c. The harnesses differ, although both force 32 steps and separate the first decode step.
Historical 512 after 64 is not a cold512 result. No historical value is substituted for an unmeasured current result.

Historical medians → current B medians, context only (not a controlled integration delta):

| format / text / prompt | prefill (tokens/s) | first decode (ms) | steady (steps/s) |
|---|---|---|---|
| checkpoint / prose / 64 first | 26.95 → 25.37 | 262.0 → 262.4 | 9.158 → 8.928 |
| checkpoint / prose / 512 after64 | 136.48 → 130.41 | 254.0 → 271.8 | 9.462 → 9.119 |
| checkpoint / code / 64 first | 23.70 → 23.27 | 303.0 → 305.0 | 8.251 → 7.872 |
| checkpoint / code / 512 after64 | 114.55 → 106.65 | 291.0 → 365.9 | 8.034 → 7.619 |
| checkpoint / chat / 64 first | 21.73 → 20.78 | 312.0 → 315.7 | 8.005 → 7.634 |
| checkpoint / chat / 512 after64 | 117.59 → 108.68 | 290.0 → 332.3 | 8.044 → 7.399 |
| gguf / prose / 64 first | 18.36 → 18.31 | 366.0 → 375.1 | 6.620 → 6.438 |
| gguf / prose / 512 after64 | 70.12 → 69.16 | 303.0 → 300.0 | 7.105 → 6.767 |
| gguf / code / 64 first | 16.64 → 16.33 | 383.0 → 391.9 | 5.869 → 5.635 |
| gguf / code / 512 after64 | 63.35 → 61.85 | 330.0 → 338.8 | 5.985 → 5.691 |
| gguf / chat / 64 first | 15.57 → 14.85 | 377.0 → 388.2 | 5.724 → 5.460 |
| gguf / chat / 512 after64 | 65.09 → 62.78 | 313.0 → 319.9 | 5.741 → 5.444 |

## Reproduction and evidence

provenance.json identifies model revisions/SHA256, sidecars, harness hashes, commands and review decisions.
The measurement logs contain exact per-block setup commands. ${CAMPAIGN} denotes the local campaign root; ${HOME} replaces a personal path.
Build the same frozen qwen36_bench.c against each pinned checkout and use the saved input files and QWEN_EXPERT_CACHE_BYTES=536870912.
qualification-logs.json includes binaries/A and B build.txt and version.txt: exact frozen-harness link commands, source hashes and compiler output.
Public files are flattened. Use each manifest.json entry's source field to restore harness/bench, harness/qwen36_bench.c, harness/promessi_sposi.txt and harness/testi/{codice,chat}.txt before running the saved commands with Python 3.12.
Use the driver directly with checkouts at the pinned commits and an explicit model path; campaign orchestration scripts require local review/state files that are not part of this public package. For example:

```sh
python3.12 harness/bench build /path/to/checkout-A binaries/A
python3.12 harness/bench build /path/to/checkout-B binaries/B
python3.12 harness/bench run results/prose-cold512 A=binaries/A/qwen36-bench B=binaries/B/qwen36-bench --model /path/to/model --text prose --lengths 512 --gen 32 --runs 4 --rest 60 --env-only --env QWEN_EXPERT_CACHE_BYTES=536870912
```

Use new output directories for code/chat and for --lengths 64,512,64. The model path is required; the redacted default paths are not usable local paths.
For exact CPU checks, use COLI_METAL=0 Q36_KV=f32 plus COLI_DENSE_IDOT=0 QWEN_EXPERT_ACT=f32 for checkpoint, or GQ_EXACT=1 for GGUF.
Import GGUF with coli import --mode link. Copy mode changes expert layout and is outside these measurements.
Binary logit/layer dump hashes are published; the full original binary proofs are preserved in the verified private archive. Text logs and scalar samples are included here.
Real-MTP performance, other hardware, LFM3 and embedded-GGUF-MTP/vision execution are not qualified by these real-model runs. Synthetic coverage is reported separately.
