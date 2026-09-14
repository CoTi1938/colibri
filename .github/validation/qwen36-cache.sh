#!/usr/bin/env bash
# Explicitly whitelisted CPU/component tests. Never invoke an inference engine,
# make check/test, a model oracle, a converter or a checkpoint downloader.
set -euo pipefail
platform=${1:?linux or windows}
case "$platform" in
  linux) exe=''; py=python3 ;;
  windows) exe='.exe'; py=python ;;
  *) echo "Unsupported platform: $platform" >&2; exit 2 ;;
esac
repo=$PWD
out="$repo/evidence/$platform"
mkdir -p "$out"
exec > >(tee "$out/validation.log") 2>&1
set -x
base=1efb7c82a0c473984cc0258bfc9e12c0cae94914
printf 'NO MODEL INFERENCE; NO REAL GPU EXECUTION\n'
git rev-parse HEAD
git log -2 --format='%H %s'
uname -a
gcc --version
make --version
"$py" --version
if [[ "$platform" == linux ]]; then lscpu; free -h; fi
sha256sum -c .github/validation/qwen36-sources.sha256
cp .github/validation/qwen36-sources.sha256 "$out/reviewed-sources.sha256"

# x86-64-v3 exercises the AVX2/FMA helper on the standard hosted x64 runners.
# Serial disables OpenMP code generation; the Makefile still links its runtime.
args=(CC=gcc ARCH=x86-64-v3 CUDA=0 HIP=0 CUDA_DLL=0 HIP_DLL=0 METAL=0 VK=0)
for mode in serial openmp; do
  extra=-Werror
  if [[ "$mode" == serial ]]; then extra+=' -fno-openmp'; fi
  make -B -C c "tests/test_qwen36_borrow$exe" "${args[@]}" "EXTRA_CFLAGS=$extra"
  for repetition in 1 2 3; do
    printf '\n--- %s repetition %s ---\n' "$mode" "$repetition"
    (cd c && "./tests/test_qwen36_borrow$exe")
  done
done

# These two production entry points are compiled, NEVER executed.
make -B -C c "qwen36$exe" build/segment/qwen36.o "${args[@]}" EXTRA_CFLAGS=-Werror
related=(test_qwen36_cache_index test_qwen36_ctx test_qwen36_dense_batch
         test_qwen36_tier_invariants test_qwen36_tier_int8_engine
         test_qwen36_tier_int8_decode test_expert_ffn test_qwen36_json_escape
         test_qwen36_tok_merges)
targets=()
for test in "${related[@]}"; do targets+=("tests/$test$exe"); done
make -C c "${targets[@]}" "${args[@]}" EXTRA_CFLAGS=-Werror
for test in "${related[@]}"; do
  printf '\n--- %s (synthetic/fake backend only) ---\n' "$test"
  (cd c && "./tests/$test$exe")
done

# Baseline engine remains untouched; only the new test and its build recipe are
# supplied. That recipe is the reviewed Makefile's four-line test-target addition.
work=$(mktemp -d)
git archive --format=tar "$base" c | tar -xf - -C "$work"
cp c/tests/test_qwen36_borrow.c "$work/c/tests/"
cp c/Makefile "$work/c/Makefile"
make -B -C "$work/c" "tests/test_qwen36_borrow$exe" "${args[@]}" EXTRA_CFLAGS=-Werror
set +e
"$work/c/tests/test_qwen36_borrow$exe" --gather-only > "$out/upstream-negative.log" 2>&1
status=$?
set -e
printf 'Upstream negative exit=%s (expected 1)\n' "$status"
test "$status" -eq 1
grep -F 'FAIL: expert 1 has overwritten data:' "$out/upstream-negative.log"

for mutation in fallback_ignores_borrows prefetch_ignores_borrows \
                prefetch_ignores_demand_claim release_only_once_per_distinct_slot; do
  "$py" .github/validation/qwen36-mutate.py "$mutation" c/qwen36.c "$work/c/qwen36.c"
  make -B -C "$work/c" "tests/test_qwen36_borrow$exe" "${args[@]}" EXTRA_CFLAGS=-Werror
  set +e
  "$work/c/tests/test_qwen36_borrow$exe" > "$out/mutation-$mutation.log" 2>&1
  status=$?
  set -e
  printf 'Mutation %s: exit=%s (must be nonzero)\n' "$mutation" "$status"
  test "$status" -ne 0
  grep -E 'FAIL:|failure\(s\)' "$out/mutation-$mutation.log"
done
sha256sum -c .github/validation/qwen36-sources.sha256
git diff --exit-code
printf 'PASS: %s model-free cache validation\n' "$platform"
