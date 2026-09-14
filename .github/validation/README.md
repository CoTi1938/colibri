# Validation-only branch

This branch exists solely to validate the reviewed CPU borrowing repair on Linux x86-64 and native Windows UCRT64. Do not open it as the implementation PR.

The parent correctness commit contains the five reviewed files. This separate CI commit removes the inherited workflows on **this branch only**, preventing model-oracle, deployment or release jobs from being triggered here. Its workflow runs only on this fork and validation branch (or explicit dispatch).

Allowed execution:

- The standalone synthetic cache regression, serial and two-thread OpenMP, three repetitions each.
- Production engine and adapter **compilation only**.
- Explicitly listed model-free unit/component tests; CUDA-labelled tests use the fake backend.
- The production-gather negative test against pinned base `1efb7c82a0c473984cc0258bfc9e12c0cae94914`.
- Four deliberate defect mutations in a disposable source tree.
- Linux Clang ASan/UBSan and TSan on synthetic data. TSan alone uses per-process ASLR disabling; no global kernel settings change.

No model inference, checkpoint download, real-activation replay, real GPU work, `make check`, broad test suite, or model oracle is invoked. There are no pull-request triggers.

`qwen36-sources.sha256` verifies the reviewed five-file snapshot before and after validation. Raw command/provenance logs and individual negative-control logs are uploaded as Actions artifacts, including on failure.

Both platforms compile for x86-64-v3. Serial disables OpenMP code generation with `-fno-openmp`; the platform Makefile still links the OpenMP runtime. Linux sanitizer builds do not enable OpenMP. These are correctness checks, not performance measurements or a model-level validation claim.
