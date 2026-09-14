#!/usr/bin/env bash
# Linux-only, standalone synthetic cache regression. No inference.
set -euo pipefail
out="$PWD/evidence/linux"
mkdir -p "$out"
exec > >(tee "$out/sanitizers.log") 2>&1
set -x
clang --version
flags=(-std=gnu11 -O1 -g -Wall -Wextra -Wno-unused-function
       -Wno-unused-parameter -Wno-misleading-indentation
       -fno-omit-frame-pointer -march=x86-64-v3)
clang "${flags[@]}" -fsanitize=address,undefined c/tests/test_qwen36_borrow.c \
  -o "$out/borrow-asan" -lm -pthread
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  "$out/borrow-asan"
clang "${flags[@]}" -fsanitize=thread c/tests/test_qwen36_borrow.c \
  -o "$out/borrow-tsan" -lm -pthread
# Only the sanitizer process gets ASLR disabled, avoiding TSan shadow-map
# collisions on newer Linux kernels. Do not change global kernel settings.
# A refusal is an explicit CI failure, not a silently skipped sanitizer run.
setarch "$(uname -m)" -R env TSAN_OPTIONS=halt_on_error=1 "$out/borrow-tsan"
printf 'PASS: Linux ASan/UBSan and TSan model-free checks\n'
