# Handoff from B1+B2 to B3

This is the short version. The full evidence is in `REPORT_cloud_b1_b2_demux.md`.

## State

- **PR:** https://github.com/Kovax007/lc0/pull/1, branch `cloud/b1-b2-demux-backend-api`, into `lc0ex-master` (e309c6af).
  Open, not merged, not watched.
- **Inputs:** upstream `master` dab9898f, LeelaChessZero/lc0#2479 f7d5378a, LeelaChessZero/lc0#2482 9618d825.
  All three were merged with `--no-ff`. #2482 was not merged upstream as of 2026-10-14.
- **Commits:**
  - B1 merge `1039e22`
  - #2479 merge `464022f`
  - #2482 merge `7de114c` (conflicts resolved here)
  - `63e5740` demux child-Q
  - `495f631` backendcompare child-Q
  - REPORT and this handoff
- **VM results:** all green. `meson test` passes 10/10. The new `DemuxChildQTest` passes. Search results are identical
  pre-merge vs merged and single vs demux. `backendcompare` errors are all 0. The `-Dlc0ex-runtime=true` build links.
- **Not run anywhere yet:** `lc0ex-cuda` on a GPU, single or under demux.

## What the code does now

- `demux` is a Backend-API backend. Children are created with `BackendManager::CreateFromParams`, so `lc0ex-cuda` is a
  valid child.
- **Syntax:**
  `--backend=demux --backend-opts="[batch_step=N,]SHARED...,g0(backend=lc0ex-cuda,gpu=0[,demux_threads=T]),g1(...)"`.
  - Each child gets the parent's scalars (minus `batch_step`) with its own subdict merged over them.
  - `backend` and `demux_threads` are dropped before the child sees its options.
- **Routing:** groups of `batch_step` (default 1) go to the children in turn. The starting child rotates with every
  computation.
  - `batch_step` ≥ top rung keeps each gather whole on one GPU.
  - `batch_step=1` splits each gather across the GPUs.
- **Child-Q:** the caller's `EvalResultPtr` (`cq` / `cs` spans included) is handed to the child that evaluates the
  sample, so there is no offset code.
  - `has_childq` is AND over the children.
  - With mixed children it is false, demux warns on stderr, and the spans are never forwarded.
  - All of this is in `network_demux.cc`: 16 lines in `AggregateDemuxAttributes`, the constructor and `AddInput`.
- `wrapper.cc` keeps the fork's `FillChildQ` unchanged, on top of #2482's per-`ComputeBlocking` `NetworkComputation`.
- `backendcompare` prints |cq| and |cs| error per legal move (mean, p50, p90, p99, max) when both sides have child-Q.

## Gotchas B3 will hit

- **abseil.** #2482's `optionsdict.cc` (`absl::StrCat` with `std::string_view`) fails against Ubuntu 24.04's system
  `libabsl-dev`. Use the wrap: remove the package, or pass `--force-fallback-for=absl_flat_hash_map`. Not patched, and
  worth reporting on #2482.
- **Behaviour changes that #2482 itself brings:**
  - per-child `threads=` is now `demux_threads=`, and an old `threads=` now errors in the child;
  - demux with no children now throws;
  - `maximum_batch_size` is derived from the children.
- **Next upstream merge.** If #2482 lands upstream changed or squashed, expect a conflict in `network_demux.cc`.
  Re-apply the 16 child-Q lines from `63e5740`.
- **Tests:** a gtest `main` that generates moves must call `InitializeMagicBitboards()`. The `random` backend needs
  `--weights=""` or autodiscover (no file), and `childq=true` gives it the synthetic head.
- **Deterministic node-limited search** needs `--smart-pruning-factor=0`, because the stopper has a 200 ms wall-clock
  tolerance. Also use `--threads=1` and a fixed `--minibatch-size`.
- **CUDA headers on a CPU box:** `apt-get install --no-install-recommends nvidia-cuda-dev` (12.0) gives `cuda.h` and
  `libcuda.so`, which is enough to build `-Dlc0ex-runtime=true -Dplain_cuda=false`.

## Open items (candidates for B3)

1. **GPU validation.** Run the commands in the REPORT section "Commands for a maintainer on a multi-GPU box":
   - preload;
   - `backendcompare` single vs demux, with `batch_step=64` as the decisive check and `batch_step=1` as informative;
   - `backendbench` demux vs N processes vs `multi`.
2. **`multiplexing` and `roundrobin`** are still old-API and cannot take `lc0ex-cuda` children.
3. **`multi`** takes `has_childq` from its first device without aggregating. Left untouched on purpose.
4. **Search on `lc0ex-master` does not read child-Q yet.** Only the NN cache requests and stores it.
5. **Decide whether demux should replace `multi`** once GPU numbers exist.
