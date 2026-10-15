# B3 report: the two live engine branches on the new base

The same file is on both branches. Machine: a CPU VM (4 cores, 15 GB, Ubuntu 24.04, gcc 13.3, meson 1.3.2) with no GPU.

## Summary

| new branch | head | made from | conflicts | VM result |
|---|---|---|---|---|
| `cloud/b3-fleet-datagen-on-new-base` | merge `95bfa25`, meson fix `7f3e3da`, this REPORT | `fleet-cv4ex` ce44d88 + new base | 8 files, 23 hunks | builds; `meson test` 10/10; datagen and search identical to `fleet-cv4ex` |
| `cloud/b3-serving-on-new-base` | merge `169af41`, this REPORT | `lc0ex-speedups-20260908` f905a99 + new base | 4 files, 26 hunks | builds; `meson test` 10/10; dag-preview search identical to f905a99 |

- **Both branches are merges** (`git merge --no-ff`): no rebase, no history rewritten. The live branches, `lc0ex-master` and PR #1's
  branch are untouched.
- **No conflict needed a design decision.** The one place where one could have been needed, the child-Q NN-cache layout,
  is the same patch on both sides: fleet's 23b2a4d and the base's e309c6a. So the PRs are not drafts.
- **Serving branch.** It is exactly the new base plus f905a99 (policy blend): `git diff 113221d
  cloud/b3-serving-on-new-base` is 3 files, +263 −2, and those are f905a99's lines. Everything else on
  `lc0ex-speedups-20260908` was already in the new base. That includes the padding fix: 96663b0 and fleet's fef478e are the
  same code.
- **Fleet branch.** It is `fleet-cv4ex` with the new base's upstream, demux and tool changes. Fleet's datagen code (memcache
  split, `childq_classic`, selfplay, training data, batch merging) is unchanged.
  - One extra commit, 7f3e3da, is a one-line meson fix. Without it the tests do not link with `-Dgtest=true`, and that was
    already broken on `fleet-cv4ex` itself. The `lc0` executable compiles the same sources with or without it.
- **Datagen equality.** The brief's deterministic settings are not deterministic on `fleet-cv4ex`: old vs old differs.
  - The cause is the fleet recipe itself: forced exploration, `--temp-endgame` and `--resign-playthrough` all draw from an
    unseeded global `Random`.
  - I ran the brief's fallback, search equality on 20 positions with the fleet's search options, and two stronger
    checks.
  - **Every comparison is identical, old vs merged:** games, PGNs and every training-data file byte for byte. So there was no
    difference to bisect.
- **Serving equality.** dag-preview with the TCEC options, `go nodes 2000`, 6 FENs: same best move and same final `info`
  line, old vs merged. The PolicyBlend1..3 options are listed, accepted and range-checked the same way.
- **Demux.** On both merged branches, the B1+B2 child-Q gtest passes 4/4, and `backendcompare` of demux over two
  `random,childq=true` children against one gives 0 on every row over 1,000 positions.
- **Upstream changes the engines now carry** are listed per engine in
  "[Upstream commits that change behaviour](#upstream-commits-that-change-behaviour)". The ones a user of the engine can
  see:
  - fleet: none in the datagen path besides `--opening-seed` (new) and the version string;
  - serving:
    - the command-line flag `--history-fill-new` is back to `--history-fill` (the UCI name `HistoryFill` is unchanged);
    - `Ponder` handling is reverted (#2409);
    - dag-preview's `n_in_flight` fix (#2388);
    - the child-Q interface in `lc0ex-cuda`.

## Heads used

| what | ref | commit |
|---|---|---|
| new base | `cloud/b1-b2-demux-backend-api`, PR #1's head | **113221d** = fd6ba8e4 + `HANDOFF_B3.md` (one doc-only commit) |
| `lc0ex-master` | not used: PR #1 is not merged | e309c6af (as of 2026-10-14 23:00 UTC) |
| fleet live branch | `fleet-cv4ex` | ce44d88f |
| serving live branch | `lc0ex-speedups-20260908` | f905a99d |
| merge base, fleet / new base | | d8ce4825 (upstream #2402) |
| merge base, serving / new base | | e2115ea0 (upstream #2330) |

- **Why 113221d and not fd6ba8e4.** 113221d is the head of PR #1, so merging PR #1 later adds nothing to these branches. The
  brief's counts also match it: fleet lacks **42** commits of 113221d (41 of fd6ba8e4), serving lacks **86** (85). The
  code is identical to fd6ba8e4.
- **Commit counts.** `fleet-cv4ex` has 35 own commits and `lc0ex-speedups-20260908` has 33, as in the brief.

## Fleet branch: conflicts and resolutions

```bash
git worktree add -b cloud/b3-fleet-datagen-on-new-base ../b3-fleet origin/fleet-cv4ex
git merge --no-ff --no-commit 113221d
# CONFLICT (content): src/neural/backends/cuda/network_cuda.cc, src/neural/backends/network_demux.cc,
#   src/neural/memcache.cc, src/neural/wrapper.cc, src/selfplay/game.cc, src/selfplay/tournament.h
# CONFLICT (add/add): src/tools/backendcompare.cc, src/tools/backendcompare.h
```

| file | hunks | resolution | why |
|---|---|---|---|
| `src/neural/backends/network_demux.cc` | 1 | **base's** (`--theirs`) | Fleet's only change here was 23b2a4d's old-API child-Q forwarding (`GetChildQVal` / `GetChildSigmaVal` through `GetParent`). The base replaces the whole file with #2482's Backend-API demux plus PR #1's child-Q AND-aggregation: the same resolution as PR #1. |
| `src/neural/wrapper.cc` | 2 | hunk 1 **fleet's**, hunk 2 **base's** | Hunk 1: fleet's `attrs_.maximum_batch_size = network_->GetMaximumBatchSize()` (29b91af, with `Network::GetMaximumBatchSize()` in `network.h`, which auto-merged) against the base's hard-coded 1024. Hunk 2: #2482's local `computation` in `ComputeBlocking()` with fleet's `FillChildQ` line, the same as PR #1. |
| `src/neural/memcache.cc` | 2 | **fleet's** (both hunks) | The result is byte-identical to fleet's file. Fleet has the invisible split of a full batch (29b91af), `--nncache-rule50-key-from` and the stats. The base's only change to this file is e309c6a's child-Q cache entry, which fleet already has as 23b2a4d (the same patch, different context line). **The child-Q cache layout is the same on both sides, so no design decision was needed.** |
| `src/selfplay/game.cc` | 7 | **fleet's** (all hunks) | The result is byte-identical to fleet's file. The base's only change to this file is upstream #2429 (350e79c: resign adjudication after the training frame is appended). Fleet already carries it: its comment cites "upstream PR #2429", and the block appears once, after `training_data_.Add`. The 7 hunks are #2429's moved block meeting fleet's `tree_`→`tree` / `search_`→`search` renames, `--replay-pgn`, and the V8 `root_data`. |
| `src/selfplay/tournament.h` | 1 | **both** | Two unrelated members: fleet's `games_to_replay_` (`--replay-pgn`) and upstream #2423's `opening_random_` (`--opening-seed`). `tournament.cc` auto-merged both features (checked by reading). |
| `src/neural/backends/cuda/network_cuda.cc` | 2 | **upstream's** | Fleet's guard (15b6bf5) kept graph capture and `GetMiniBatchSize()` below `max_batch_size_`. Upstream #2449 / #2416 do the same in their own form: `opt_batch_size_` is clamped to `[min_batch, max_batch]`, and the capture loop starts at 1 and so ends at `GetMiniBatchSize()` ≤ `max_batch_size_`. This is the plain `cuda` backend; the fleet runs `lc0ex-cuda`. |
| `src/tools/backendcompare.cc` | 7 | **base's** | Fleet's file is the base's original (2607c3d) without `--dump-test`. The base's adds `--dump-test` and the child-Q rows (495f631): a superset. |
| `src/tools/backendcompare.h` | 1 | **base's** | Header comment only. |

**Files both sides changed that merged without conflict:**

- **Result equals fleet's file**, because the base's change is identical to fleet's:
  - `proto/lc0ex*.proto`, `proto/net.proto`, `scripts/compile_proto.py`, `src/main.cc`;
  - `src/neural/backend.{h,cc}`, `backends/backend_multi.cc`;
  - all of `backends/lc0ex-cuda/` (identical, as the brief says);
  - `network_check.cc`, `network_mux.cc`, `network_random.cc`, `network_record.cc`;
  - `network.h`, `register.h`, `src/chess/pgn.h` (fleet already had #2425's en-passant fix).
- **Result equals the base's file:** `src/neural/backends/onnx/network_onnx.cc` (child-Q, the same patch on both sides).
- **Mixed, checked by reading the diff against fleet:**
  - `meson.build` and `meson_options.txt`: fleet's `childq_classic` and `batchmerge.cc`, plus upstream's HIP / CoreX
    options and per-backend static libraries;
  - `src/selfplay/tournament.cc`: fleet's `--replay-pgn` plus #2423;
  - `src/trainingdata/reader.cc`: #2452's `<=`→`<` loop fix on fleet's reader.

**Extra commit 7f3e3da (meson, not a merge resolution):**

- **The problem.** Fleet's `trainingdata.cc`, a common file, has the `childq_classic::Node` overload of `Add()` under
  `LC0_CHILDQ_CLASSIC`, but `childq_classic/node.cc` was in the executable's `files` only. `lc0_lib`, which every gtest
  links, failed with `undefined reference to lczero::childq_classic::Edge::GetP() const` (and 2 more).
  - This also fails on `fleet-cv4ex` ce44d88 with `-Dgtest=true` (log below). The fleet builds with `-Dgtest=false`, so
    nobody saw it.
- **The fix.** `childq_classic/node.cc` moves to `common_files`, as `classic/node.cc` already is. `lc0` compiles the same
  sources as before (`files += common_files`).

## Serving branch: conflicts and resolutions

```bash
git worktree add -b cloud/b3-serving-on-new-base ../b3-serving f905a99
git merge --no-ff --no-commit 113221d
# CONFLICT (content): meson.build
# CONFLICT (add/add): src/neural/backends/lc0ex-cuda/network_fingerprint.cc, network_lc0ex_cuda.cc, src/tools/backendcompare.cc
```

**What the two runtimes are.**

- The base's runtime came in as 2607c3d ("lc0ex runtime on upstream master", the fleet's deployed source) plus e309c6a
  (child-Q).
- Serving's pre-blend runtime, at 96663b0, is 2607c3d's runtime with three differences only:
  - two statements wrapped differently;
  - one line less in `net.proto`;
  - no `--dump-test` in `backendcompare`.
- So "older" here means: serving lacks e309c6a. Serving's only newer code is f905a99.

| file | hunks | resolution | why |
|---|---|---|---|
| `src/neural/backends/lc0ex-cuda/network_lc0ex_cuda.cc` | 17 | **base's runtime + f905a99 re-applied** | Started from the base's file. f905a99's change was replayed with `git merge-file` (base 96663b0's file, theirs f905a99's file); 5 of its 7 hunks applied cleanly. Two spots touched code that e309c6a also changed, and I kept both sides (see below). **Check:** the set of lines added and removed between the base's file and the result equals the set f905a99 adds and removes. |
| `src/neural/backends/lc0ex-cuda/network_fingerprint.cc` | 1 | **base's** | Serving's file plus e309c6a's child-Q lines. A net without child-Q outputs gets the same fingerprint as before, so existing programs still load. |
| `src/tools/backendcompare.cc` | 7 | **base's** | Serving's file equals fleet's: the base's minus `--dump-test` and the child-Q rows. |
| `meson.build` | 1 | **base's** (`onnx_pb = gen.process(...)`, for the per-backend static libraries) | Every line serving added to `meson.build` / `meson_options.txt` is in the result (checked line by line). |

The two hand-joined spots in `network_lc0ex_cuda.cc`:

- **`UploadWeights`, ONNX initializers without a buffer.** The base's child-Q guard (throws when a child-Q initializer has no
  buffer) comes first, then f905a99's `continue` for `/policy/blend/` tensors. The prefixes don't overlap.
- **The constructor.** The base's `attributes_.has_childq = programs_.front().output_childq_mean != nullptr;`, then f905a99's
  `UploadWeights(weights, *executable_, backend_options, ReadPolicyBlendWeights(backend_options, options))`.

**The padding fix.**

- 96663b0 ("pad between batch sizes with the last position") and fleet's fef478e ("padding fix between rung batches")
  start from the same blob (64a0f74). They make the same change: `padded_batch`, `entries_[std::min(sample,
  actual_batch - 1)]`, and copies of `spec.batch_size` rows. The only difference is that fef478e also re-wraps two statements
  (clang-format).
- **One copy survives: the base's** (from fef478e through 2607c3d), in `Lc0exCudaBackendComputation::ComputeBlocking()`:
  the comment "A batch between rungs runs the next rung's program…", `padded_batch`, and both `CopyFromHostAsync` calls with
  `spec.batch_size`.

**Kept from f905a99, all of it:**

- `kPolicyBlend1Id`..`kPolicyBlend3Id` in `shared_params.{h,cc}` (UCI `PolicyBlend1..3`, flags `--policy-blend-1..3`,
  range −10..10, default −10 = not set), which auto-merged;
- `HalfBitsFromFloat`, `RequireBlendTensor`, `PolicyBlendUci`, `ReadPolicyBlendWeights` (backend options `policy_blend_<i>`
  win over the UCI options), `ApplyPolicyBlend`;
- the `/policy/blend/` initializer skip;
- `NEED_RESTART` when a PolicyBlend option changes.

**Result:** `git diff 113221d cloud/b3-serving-on-new-base` is exactly f905a99's change: `network_lc0ex_cuda.cc` +242 −2,
`shared_params.cc` +18, `shared_params.h` +3.

## Upstream commits that change behaviour

Every non-merge commit of the new base that a branch did not have, by its effect on that branch's engine. "Only other
backends / tools" means the commit changes code that the engine's command never runs.

### Fleet: datagen engine (`selfplay`, `childq-classic`, `lc0ex-cuda`, NN cache, batch merging, training data)

These are the 42 commits `origin/fleet-cv4ex..113221d`: 39 non-merge commits, plus PR #1's three merge commits (1039e22,
464022f, 7de114c), which bring in the same changes.

**What a fleet user can see:**

- **1805363 (#2446):** the version is now `v0.34.0-dev` (was `v0.33.0-dev`), in the UCI `id name` and the banner.
- **f5befb7 (#2423):** new option `--opening-seed`.
  - With `--openings-mode=shuffled`, which the fleet uses, the openings are now shuffled by a private generator. It is
    seeded from `std::random_device` unless `--opening-seed` is given, where before the shuffle used the global `Random`.
  - Still a random shuffle per process. Statistically no change for the fleet, and the new option is not set.
- **2663cf3 (#2439):** the log file's timestamps are formatted differently. Output only.

**No effect on legal play:**

- **58f3268 (#2451):** a position with two or more checking pawns or knights is classified as a double check. Only illegal
  FENs can have that.

**Already on `fleet-cv4ex`, so no change:**

- 350e79c (#2429): resign after the training frame;
- 2cd063a (#2425): PGN en passant;
- 2607c3d and e309c6a: the same code as fleet's 15b6bf5 / fef478e / 23b2a4d.

**Only other backends or tools:**

- plain `cuda` / `cudnn` backends: ef33106 (HIP compatibility), d0fcd2b, 3c99ccb, 7e270a6 (#2449, minibatch heuristic),
  1322b36 (#2416, graph capture option), c614eae (bf16 backend), 926e99f, f1b5fb7, cc51106, a3f24cf (also warning fixes in
  `multigame.cc`, `override` only);
- onnx backends: 7928782, 043e723;
- build-only options, default off: da092e3 (CoreX), b9a3f04 (python bindings);
- rescorer and the training-data reader, not used by selfplay: a3fa69e, 08a23a2;
- demux: f7d5378 (#2479), the #2482 commits (0e0bae8, e78a1f3, 42d63e9, ddafeed, e23179a, 9618d82) and 63e5740;
- `backendcompare`: 495f631.

**No behaviour:** 1227b4c (an unused dag field), af8ae6a, f13ae3a, dab9898, fd6ba8e, 113221d.

**Tests:** the fleet engine was compared with `fleet-cv4ex` in Test 2 below. There was no difference, so nothing to bisect.

### Serving: match engine (`dag-preview`, UCI, `lc0ex-cuda`, `multi`)

These are the 86 commits `origin/lc0ex-speedups-20260908..113221d`: the 42 above plus 44 older upstream commits
(`origin/lc0ex-speedups-20260908..d8ce482`).

**Changes the engine's behaviour or interface:**

- **f40854a (#2400):** the command-line flag `--history-fill-new` is renamed back to **`--history-fill`**.
  - The UCI option `HistoryFill` and its default `fen_only` are unchanged.
  - **A launcher that passes `--history-fill-new` must change.**
- **d039730 (#2409), "revert ponder behavior":**
  - the `Ponder` option is "Not currently used" again;
  - a `go ponder` without `Ponder=true` no longer throws;
  - `ponderhit` while not pondering is ignored instead of throwing;
  - `position` no longer sets up the search when `Ponder` is off; `go` does.
  - Not testable deterministically here; read from the diff.
- **37ff141 (#2388):** fixes an `n_in_flight` underflow in dag-preview by restructuring
  `SearchWorker::PickNodesToExtendTask`.
  - With 1 thread, the result is identical (Test 3).
  - With several threads, which a match runs, results were never reproducible, so this needs the GPU search runs below.
- **ddd5c13 (#2407):** dag-preview's `CollisionsManager` is replaced by `absl::Cleanup`. A refactor.
- **909a408 (#2376):** with `UCI_Chess960=false`, a castling move is printed as `e1g1` / `e1c1` only when the king is on
  the e-file and the rook on a / h. Otherwise it is printed king-to-rook. Standard chess is unchanged.
- **66a94d5 (#2367):** the NN cache no longer calls the backend's `ComputeBlocking()` for an empty computation.
- **e309c6a (the base's child-Q interface, not upstream):**
  - `lc0ex-cuda` reports `has_childq` when its program has a child-Q output. The NN cache then stores cq / cs for every
    entry, which takes more memory per entry.
  - The net and the program must agree on having the head: this is in the fingerprint.
  - **A program without the head behaves as before.**
- **1805363:** the version is `v0.34.0-dev`.
- **2663cf3:** log timestamps.
- **58f3268:** multi-check, illegal FENs only.

**Checked to be identical:**

- **3f37760 (#2340), fp16 conversions:**
  - `FP16toFP32` was compared for all 65,536 inputs; `lc0ex-cuda` uses it for FP16 child-Q rows.
  - `FP32toFP16` was compared for all 2³² inputs; the ONNX converter uses it for f16 initializers.
  - Both are bit-identical old vs new, for the generic path and for the F16C path that `-march=native` selects.
- **Weights→ONNX conversion at load** (`lc0ex-cuda` converts a `.pb.gz` without an ONNX model):
  - Commits: 8417bc4, acb4d47, c3c3367, 701bd83, and 310de9c (FLOAT32 weights in `weights_adapter.cc`).
  - With `lc0ex-cuda`'s default converter options (opset 17, `alt_mish` and `alt_layernorm` off, f16), the initializers are
    computed the same way. The arithmetic for LINEAR16 / FLOAT16 / BFLOAT16 is unchanged.
  - The graph-side changes do not reach `lc0ex-cuda`, which uploads initializers by name into a precompiled program:
    - native `Mish` op only at opset ≥ 18;
    - `ffn_activation` for the PE-dense embedding;
    - a new `Selu` path.
  - **The GPU `backendcompare` below is the real check.**
- **6de5d44 (#2331), NVTX tracing:** adds `LCTRACE_FUNCTION_SCOPE` macros, which are empty unless built with
  `-Dtrace_library=nvtx` (default `off`).

**Only selfplay, training data or rescorer** (not the match engine): 2cd063a, 350e79c, f5befb7, 08a23a2, a3fa69e,
2e13e3a (#2348, rescorer reads V7).

**Tools:**

- `backendbench`: 67fa9e1 (threads, statistics), 9e715d2 (CV), 473514b (first batch time), the new
  `--header-only-once`;
- `backendcompare`: `--dump-test` and the child-Q rows.

**Only other backends:**

- cuda / cudnn: bb616f9, 2537586 (graph support), a1b1197, 6777ad2 (CUTLASS fused MHA), 9aba44a, 76b8925, 66a94d5's cuda
  parts, 5ed0f11, 3538b61, plus the 42-list's cuda commits;
- onnx (and coreml, migraphx, trt, dml): 74a710c, d83051f, dbfed8f, 9f35ac0, 3bf83d5, e482a65, 711b7c0, 7f572ae, 4425d4c,
  701bd83, b38ec9e, 1a8b563, 5d8b306;
- sycl / xla: parts of bb616f9, b600668;
- demux: b600668 (#2339), plus the 42-list's demux commits;
- HIP, CoreX, python: 0dc3df6 and the 42-list's.

**Build / CI only:** 26575ac, be0079a, afaa5ab, 15290b7, d8ce482, and the appveyor parts of others.

**Options, compared by diffing `--help` of old and merged, for every mode:**

- **fleet:**
  - `selfplay` gains `--opening-seed`, and `backendcompare` gains `--dump-test`;
  - `childq-classic`, `classic`, `dag-preview` and `backendbench` are unchanged;
  - **no existing option changed its name, default or range.**
- **serving:**
  - `--history-fill-new` → `--history-fill`;
  - `selfplay` gains `--opening-seed`, `backendcompare` gains `--dump-test`, and `backendbench` gains
    `--[no-]header-only-once`;
  - the `Ponder` help text changed (#2409);
  - no default changed.

## Commands and results

`B` = `-Dblas=true -Dgtest=true`. `X` = `-Dlc0ex-runtime=true -Dplain_cuda=false -Dblas=true -Dgtest=true`, with
`nvidia-cuda-dev` 12.0 for `cuda.h` / `libcuda.so`, as in B1+B2's Test 6. `libabsl-dev` is not installed, so meson uses
`subprojects/abseil-cpp.wrap` (20240722.0). All builds are meson `release` (LTO, `-march=native`).

| tree | `B` build | `meson test` (`B`) | `X` build | `meson test` (`X`) | backends in `X` |
|---|---|---|---|---|---|
| fleet old (ce44d88) | `lc0` links; **`liblc0_lib.so` fails** (childq_classic, pre-existing) | 8/8 OK with 7f3e3da's one-line meson change applied locally (not committed there) | `lc0` links; lib fails the same way | – | `blas eigen trivial lc0ex-cuda random check roundrobin recordreplay multiplexing demux multi` |
| **fleet merged** (7f3e3da) | OK | **10/10 OK** | OK | **10/10 OK** | same |
| serving old (f905a99) | OK | 7/7 OK | OK | 7/7 OK | same |
| **serving merged** (169af41) | OK | **10/10 OK** | OK | **10/10 OK** | same |

```
fleet old, B:  [311/336] Linking target liblc0_lib.so
               FAILED: liblc0_lib.so
               undefined reference to `lczero::childq_classic::Edge::GetMove(bool) const'
               undefined reference to `lczero::childq_classic::Edge::GetP() const'
               undefined reference to `lczero::childq_classic::Node::Edges() const'
merged (both), B and X:
 1/10 FP16 OK   2/10 HashCat OK   3/10 PositionTest OK   4/10 OptionsParserTest OK   5/10 SyzygyTest OK
 6/10 EncodePositionForNN OK   7/10 DemuxTest OK   8/10 DemuxChildQTest OK   9/10 EngineTest OK   10/10 ChessBoard OK
 Ok: 10  Fail: 0
```

- **Where the two extra tests come from.** The merged trees have 10 tests where serving old has 7 and fleet old 8: the new
  base adds `FP16` (#2340, which fleet already had), `DemuxTest` (#2482) and `DemuxChildQTest` (B1+B2).
- **The fleet's own meson options configure without errors on both fleet trees.** Those options are `-Dbuildtype=release
  -Db_lto=true -Dcc_cuda=89 -Dcudnn=false -Dcutlass=true -Dgtest=false -Dlc0ex-runtime=true -Dnative_cuda=true -Dnvcc=true`.
  There is no nvcc on the VM, so CUDA and CUTLASS were not compiled.

### Test 2: datagen equality (fleet)

The fleet's recipe-39 command, run through `run_datagen.sh` (script at the end):
- the backend is `--backend=random --backend-opts=childq=true`;
- the brief's changes: `--noise-epsilon=0 --temperature=0 --openings-mode=sequential` with `openings20.pgn` (20 fixed
  6-ply openings), `--parallelism=1 --games=20`;
- every other flag as given.

Output per game: the training file (`$XDG_CACHE_HOME/lc0/data-*/game_NNNNNN.gz`) and the `gameready` line (moves and
result), converted to PGN with python-chess.

**2a. The brief's settings, old vs old: not deterministic.**

```
./run_datagen.sh fleet_old_lc0 dg_old1  ->  rc=0 time=122.3s games=20 tdfiles=20
./run_datagen.sh fleet_old_lc0 dg_old2  ->  rc=0 time=153.0s games=19 tdfiles=19
games differ from game 0, ply 7 (a2a4 vs h3g1); 20 of 20 training files differ
```

The global `Random` (`utils/random.cc`) is seeded from `std::random_device`, there is no seed option, and the recipe draws
from it with `--temperature=0` and `--noise-epsilon=0` too:

- **Forced exploration** (`--forced-exploration-visits=220`) picks its low-policy child with `Random::GetDiscrete`
  (`childq_classic/search.cc:255`). It does this in UCI mode as well.
- **`--temp-endgame=0.60`** is the temperature after `--temp-cutoff-move=40`, whatever `--temperature` is. The games of
  2a diverge again around move 40 when forced exploration is off.
- **`--resign-playthrough=20`** turns resign off in a random 20 % of games (`tournament.cc:498`).
- The player-1 colour is random too (`first_game_black_`), but that only changes the `player1` label.

**2b. The brief's fallback: search equality on 20 fixed positions with the fleet's search options.**

- Setup: `lc0 childq-classic --threads=1 --backend=random --backend-opts=childq=true` plus every search flag of the recipe
  (list in `search_flags` below), `go nodes 800` (the recipe's `--visits`). The positions are startpos and 19 random-playout
  FENs.
- Compared: the `bestmove` line and the final `info` line, with `time` and `nps` removed.

```
all fleet search options:         old vs old        6/20 identical   (forced exploration -> Random)
  + --forced-exploration-visits=0: old vs old       20/20 identical
                                   old vs merged    20/20 identical
```

**2c. Real binaries, recipe with the three random draws off: identical.**

- Setup: run 2a's command plus `--forced-exploration-visits=0 --temp-endgame=0 --resign-playthrough=0`.
- Old, old and merged were run at the same time.

```
dgd_old1   rc=0 time=102.5s games=19 tdfiles=19
dgd_old2   rc=0 time=102.5s games=19 tdfiles=19
dgd_merged rc=0 time=102.4s games=19 tdfiles=19
old vs old:    games IDENTICAL, training files differing: 0 of 19
old vs merged: games IDENTICAL, training files differing: 0 of 19
PGN sha256 (old1, old2, merged): 0cb0b73dfe6854e0… for all three
training-file set sha256 (old1, merged): b26cb008e3cac716… for both
```

- **The full recipe, with a test-only seeded `Random`: identical.**
- **The test-only patch.** It seeds the global generator from `LC0_TEST_RANDOM_SEED` (diff at the end), and was applied
  identically to ce44d88 and 7f3e3da in separate worktrees. **It is not committed anywhere.**
- **What it buys.** 2a's exact command is reproducible, including forced exploration, the endgame temperature and resign
  playthrough.

```
LC0_TEST_RANDOM_SEED=20261015
  old  run 1, run 2:    rc=0 games=20 tdfiles=20 (117 s each)   -> games IDENTICAL, 0 of 20 training files differ
  merged run 1, run 2:  rc=0 games=20 tdfiles=20 (115 s each)   -> games IDENTICAL
  old vs merged:        games IDENTICAL, 0 of 20 training files differ
  PGN sha256 7792e76d452ce729… (old 1, old 2, merged 1, merged 2); training-file set sha256 dbb262007514a069… (old, merged)
LC0_TEST_RANDOM_SEED=7
  old vs merged:        rc=0 games=19 tdfiles=19 each; games IDENTICAL, 0 of 19 training files differ
  PGN sha256 592984062f0caacc… (both); training-file set sha256 b084b23d2710da5b… (both)
seeded search equality, all fleet search options, 20 FENs, go nodes 800:
  old vs merged 20/20 identical; old vs old 20/20 identical
```

- **The files are V8.** The training files start with version `8`, `input_format` 1 (386,260 bytes uncompressed for game
  0).
- **The 19-game runs.** The game that is missing ends `UNDECIDED`, and `PlayOneGame` reports only decided games. This
  happens on `fleet-cv4ex` too, and in each deterministic pair it is the same game in old and merged.

**2d. Every recipe option exists and parses on the merged binary** (the brief's check).

- Setup: the recipe verbatim, with randomness on, `--parallelism=24`, `--openings-mode=shuffled`, `--backend=random
  --backend-opts=childq=true`, `--games=4` and `openings20.pgn`.

```
./run_fleet_verbatim.sh fleet_merged_lc0 verbatim_merged  ->  rc=0 games=4 tdfiles=4 errors=0   (16 s)
```

**Result.** No difference between `fleet-cv4ex` and the merged branch in any comparison, so there was nothing to trace or
bisect.

### Test 3: serving equality (`dag-preview`)

- Setup: `lc0 dag-preview --backend=random --cpuct=2.15 --cpuct-base=45569 --cpuct-factor=3.973 --policy-softmax-temp=1.36
  --fpu-value=0.58 --smart-pruning-factor=0 --threads=1 --minibatch-size=32`, `go nodes 2000`, a fresh process per search.
- **About `childq=true`.** f905a99's binary cannot take `--backend-opts=childq=true`: it answers `error Unknown boolean option:
  random.childq`, because the child-Q interface is not on `lc0ex-speedups-20260908`. So old runs plain `random`. Merged runs
  both `random,childq=true` (the brief's) and `random`. `childq=true` only adds the synthetic cq / cs and does not change
  q / d / m / policy.

```
old vs merged_childq:        6/6 identical (bestmove + info)
old vs merged:               6/6 identical
old vs old_again:            6/6 identical
old vs merged_childq_again:  6/6 identical
positive control, merged with seed=1:  0/6 identical
```

| FEN | bestmove | final info (time / nps removed) |
|---|---|---|
| startpos | g1h3 ponder d7d5 | depth 4 seldepth 8 nodes 2009 score cp 15 pv g1h3 d7d5 b2b4 b8d7 f2f3 d7c5 e2e3 e8d7 |
| kiwipete | d2e3 ponder f6g8 | depth 4 seldepth 7 nodes 2024 score cp 17 pv d2e3 f6g8 e1d2 g8h6 e5g6 a6c8 |
| perft4 `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1` | c4c5 ponder b6c5 | depth 4 seldepth 8 nodes 2014 score cp 4 pv c4c5 b6c5 f3d4 a3d3 f1e1 f6d5 |
| endgame `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1` | b4c4 ponder d6d5 | depth 5 seldepth 9 nodes 2030 score cp 9 pv b4c4 d6d5 c4c1 h4g5 c1c6 d5d4 |
| QGD `r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 0 8` | c3b1 ponder d8d6 | depth 4 seldepth 8 nodes 2022 score cp 3 pv c3b1 d8d6 b1c3 c6d4 f3e5 d6b6 |
| mate in 1 `6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1` | d1d8 | depth 2 seldepth 6 nodes 2058 score mate 1 pv d1d8 |

**Policy-blend UCI options:**

```
uci (old and merged, identical):
option name PolicyBlend1 type string default -10.000000
option name PolicyBlend2 type string default -10.000000
option name PolicyBlend3 type string default -10.000000
--policy-blend-1=0.1 --policy-blend-2=0.2 --policy-blend-3=-10, then setoption PolicyBlend1 0.3 / PolicyBlend2 -0.25 /
PolicyBlend3 2.5, go nodes 2000 (threads 1, smart pruning 0), twice per binary:
  old:    bestmove c2c4 ponder b7b6 (x2), no error
  merged: bestmove c2c4 ponder b7b6 (x2), no error
--policy-blend-1=11:  both  "Flag '--policy-blend-1' must be between -10 and 10."
```

- **What this does and does not show.** The blend itself (`ApplyPolicyBlend`) acts only in `lc0ex-cuda` with a blend weights
  file, so it is covered by the GPU commands below.
- **The `random` dumps.** The old binary, with 2607c3d's `backendcompare.cc` swapped in for `--dump-test` (see the GPU
  section), and the merged binary write byte-identical `random` dumps over 1,000 positions (135,328 bytes).

### Test 4: demux on both merged branches

```
network_demux_childq_test (b3-fleet and b3-serving, identical):
[ OK ] DemuxChildQ.RandomChildrenMatchSingleBackend
[ OK ] DemuxChildQ.EverySampleComesWhollyFromOneChild
[ OK ] DemuxChildQ.MixedChildrenReportNoChildQ
[ OK ] DemuxChildQ.CreatesBackendApiChildren
[  PASSED  ] 4 tests.
```

```bash
lc0 backendcompare --backend=demux --backend-opts="childq=true,a(backend=random),b(backend=random)" \
    --ref-backend=random --ref-backend-opts="childq=true" --fens=fens_1000.txt --max-positions=1000 \
    --batch-size=32 --dump-test=dump.bin        # and --batch-size=7; on both merged branches
```

```
# positions compared: 1000  (skipped 0)
metric                            mean         p50         p90         p99         max
policy KL(ref||test)          0.000000    0.000000    0.000000    0.000000    0.000000
policy L1                     0.000000    0.000000    0.000000    0.000000    0.000000
|q_test - q_ref|              0.000000    0.000000    0.000000    0.000000    0.000000
|d_test - d_ref|              0.000000    0.000000    0.000000    0.000000    0.000000
|m_test - m_ref|              0.000000    0.000000    0.000000    0.000000    0.000000
|cq_test - cq_ref| (move)     0.000000    0.000000    0.000000    0.000000    0.000000
|cs_test - cs_ref| (move)     0.000000    0.000000    0.000000    0.000000    0.000000
top-1 agreement          : 100.0000 %
(the same table for b3-fleet and b3-serving, batch sizes 32 and 7)
--dump-test sha256: 42da8f14…6052bb for all four demux runs and for single random: the same as in B1+B2
positive control (ref seed=1): |cq| p99 1.8042, max 1.9968; |cs| p99 0.2255; top-1 4 %  (both branches)
```

## Time per step

| step | time |
|---|---|
| reading B1+B2's REPORT, fetching, unshallowing, trial counts | ~5 min |
| fleet merge: 8 conflicts, reading both sides of each (merge 23:00, commit 23:10:07) | ~10 min |
| serving merge: 4 conflicts, `merge-file` re-application of f905a99 and its check (commit 23:11:19) | ~8 min (overlapping) |
| fleet merged `B` build (setup 4 s, compile 132 s), diagnosis of the lib link failure | ~4 min |
| fleet old `B` build (122 s), meson fix 7f3e3da, rebuild 36 s, `meson test` 3 s | ~3 min |
| serving merged `B` (126 s) / serving old `B` (119 s), `meson test` 3 s each | ~4 min |
| four `X` builds in sequence (162 / 148 / 140 / 259 s; the last ran alongside other builds) | ~12 min (background) |
| datagen 2a, two runs (122 s, 153 s, under build load) and diagnosis of the non-determinism | ~10 min |
| search equality 2b (5 s per 20 searches) | ~3 min |
| seeded test builds (237 s / 104 s), seeded datagen runs (4 × ~2 min, two at a time), seeded search | ~12 min |
| 2c (3 runs in parallel, 102 s), 2d (16 s), option diffs | ~5 min |
| serving equality, control, policy-blend checks (each search set < 1 s) | ~5 min |
| demux gtest and `backendcompare` (< 1 s per run) | ~2 min |
| fp16 exhaustive checks (2³² conversions ~20 s per variant), reading the upstream commits | ~15 min |
| REPORT, PRs | ~25 min |

## What I could not test on the VM

- **Anything that runs `lc0ex-cuda`** (no GPU, no `.lc0ex` program, no network file):
  - the fleet's real datagen;
  - the padding path;
  - the policy blend applied to a blend weights file;
  - child-Q decoding from a real program;
  - the fingerprint check on real programs;
  - the weights→ONNX conversion of a real `.pb.gz`.
- **Whether `lc0ex-cuda` is bitwise deterministic across runs and batch sizes.** That decides whether the GPU comparisons
  below must give exact equality.
- **Multi-threaded search equality** (TCEC runs more than one thread). It is not reproducible on either binary. The
  dag-preview change #2388 matters most here.
- **Pondering (#2409):** only by reading the diff.
- **The fleet's real build:** nvcc, CUTLASS, `-Dcc_cuda=89`. Only `meson setup` with the fleet's options was run, and it
  configures on both fleet trees.
- **The plain `cuda` backend changes,** and the ONNX backends (onnxruntime is not installed).
- **The 24-hour soak.**

## Commands for the maintainer (GPU)

```bash
git clone https://github.com/Kovax007/lc0 lc0-b3 && cd lc0-b3
git fetch origin cloud/b3-fleet-datagen-on-new-base cloud/b3-serving-on-new-base fleet-cv4ex lc0ex-speedups-20260908
# If libabsl-dev is installed system-wide, add --force-fallback-for=absl_flat_hash_map to every meson setup.
NET=/path/to/net.pb.gz          # the network the program was built for
PROG=/path/to/program.lc0ex
FENS=/path/to/fens.txt          # one FEN per line, ~10k real positions
```

The scripts `run_datagen.sh`, `search_eq2.py`, `games2pgn.py` and `gen_openings.py` are at the end of this file. In
`run_datagen.sh`, change the first line of the command to the fleet's real backend (`--backend=lc0ex-cuda
--backend-opts=...` and `--weights=$NET`).

### Datagen (fleet)

**1. Build the fleet's binary and the merged one with the fleet's options.**

```bash
FLEETOPTS="-Dbuildtype=release -Db_lto=true -Dcc_cuda=89 -Dcudnn=false -Dcutlass=true -Dgtest=false -Dlc0ex-runtime=true -Dnative_cuda=true -Dnvcc=true"
git worktree add ../fleet-old ce44d88fda92f624ec318eef1546419f41a0c20a
git worktree add ../fleet-new origin/cloud/b3-fleet-datagen-on-new-base
for t in fleet-old fleet-new; do (cd ../$t && meson setup build/fleet $FLEETOPTS && meson compile -C build/fleet); done
# Test-only seeded copies (never commit):
for t in fleet-old fleet-new; do
  git worktree add ../$t-seeded $(git -C ../$t rev-parse HEAD)
  (cd ../$t-seeded && git apply /path/to/test_only_seed_random.patch && meson setup build/fleet $FLEETOPTS && meson compile -C build/fleet)
done
```

**2. Same seed, same openings, one GPU.** Run old and new on the same GPU one after the other, not at the same time.

```bash
# (a) The recipe with the three random draws off: real binaries, as in 2c.
X="--forced-exploration-visits=0 --temp-endgame=0 --resign-playthrough=0"
./run_datagen.sh ../fleet-old/build/fleet/lc0 gpu_old_a $X
./run_datagen.sh ../fleet-old/build/fleet/lc0 gpu_old_b $X      # determinism of the GPU path first
./run_datagen.sh ../fleet-new/build/fleet/lc0 gpu_new_a $X
# (b) The full recipe, seeded builds, as in the seeded runs above.
for s in 20261015 7; do
  LC0_TEST_RANDOM_SEED=$s ./run_datagen.sh ../fleet-old-seeded/build/fleet/lc0 gpu_seed${s}_old
  LC0_TEST_RANDOM_SEED=$s ./run_datagen.sh ../fleet-new-seeded/build/fleet/lc0 gpu_seed${s}_new
done
# Compare:
for p in "gpu_old_a gpu_old_b" "gpu_old_a gpu_new_a" "gpu_seed20261015_old gpu_seed20261015_new" "gpu_seed7_old gpu_seed7_new"; do
  set -- $p; sed 's/ player1 [a-z]*//' $1/games.txt > /tmp/a; sed 's/ player1 [a-z]*//' $2/games.txt > /tmp/b
  cmp -s /tmp/a /tmp/b && echo "$1 vs $2 games identical" || echo "$1 vs $2 games DIFFER"
  echo "  training files differing: $(for f in $1/td/*.gz; do cmp -s $f $2/td/$(basename $f) || echo x; done | wc -l) of $(ls $1/td | wc -l)"
done
```

- **Expected.** All identical if `lc0ex-cuda` rows do not depend on their batch companions (bitwise-checked on 09-12, per
  the comment in `memcache.cc`).
- **If old vs old already differs**, repeat with `--batch-merge-max-batch=0`, which switches batch merging off, so that every
  search batch is evaluated alone.
- **If old vs new differs but old vs old does not**, bisect the first-parent history of the new base between d8ce482 and
  113221d. The candidates by area are in "Upstream commits that change behaviour" above. Only `lc0ex-cuda`-side code would
  be a surprise, because `backends/lc0ex-cuda/` is identical on both trees.

**3. 24-hour soak:** the fleet command verbatim, with the real paths, `--games=-1`, the merged binary, one GPU.

```bash
timeout 24h /usr/bin/time -v ../fleet-new/build/fleet/lc0 selfplay <the recipe-39 command, unchanged> \
    > soak_new.out 2> soak_new.err; echo "exit $?"
grep -c '^gameready' soak_new.out                           # games, compare with the old binary's rate on this GPU
grep -c 'MemCache: a computation reached' soak_new.err      # the invisible split (29b91af); expected rare, not fatal
grep -iE 'exception|overflow|error' soak_new.err | sort | uniq -c
grep 'Maximum resident set size' soak_new.err               # RSS; also sample `ps -o rss` hourly for growth
ls $XDG_CACHE_HOME/lc0/data-*/ | wc -l                      # training files; feed a sample to the trainer's V8 reader
```

**Pass criteria:**
- no crash (in particular no `AtomicVector overflow`);
- games per hour and nps within noise of the old binary on the same GPU;
- no RSS growth after warm-up;
- the trainer's reader accepts the files (V8, `v8_meta[:,0] == 2`, recipe 39).

### Serving (match engine)

**1. Build old (f905a99) and new with the usual serving options.** For example, as for B1+B2: `-Dbuildtype=release
-Dlc0ex-runtime=true -Dgtest=false`.

- **f905a99's `backendcompare` has no `--dump-test`.** Swap in 2607c3d's `backendcompare.cc` in the old tree: it is
  f905a99's file plus `--dump-test` and nothing else, and needs no child-Q code. On the VM, the old binary built this way
  writes the same dump as the merged one, and goes back byte-identical once the file is restored.

```bash
git worktree add ../serving-old f905a99dfe874c4199bbacfa467c43233cff1020
git worktree add ../serving-new origin/cloud/b3-serving-on-new-base
(cd ../serving-old && git show 2607c3d:src/tools/backendcompare.cc > src/tools/backendcompare.cc)   # test-only
for t in serving-old serving-new; do (cd ../$t && meson setup build/s -Dbuildtype=release -Dlc0ex-runtime=true -Dgtest=false && meson compile -C build/s); done
OLD=../serving-old/build/s/lc0; NEW=../serving-new/build/s/lc0
```

**2. `backendcompare` of the merged engine against the old one, on the same `.lc0ex` program.** The dump holds q, d, m and
the legal-move policy as float32 per position, the same format in both builds.

```bash
for bs in 64 7 33; do          # 7 and 33 are between rungs: the padding path
  for b in OLD NEW; do
    ${!b} backendcompare --weights=$NET --fens=$FENS --max-positions=10000 --batch-size=$bs \
        --backend=lc0ex-cuda --backend-opts="lc0ex=$PROG,gpu=0" \
        --ref-backend=lc0ex-cuda --ref-backend-opts="lc0ex=$PROG,gpu=0" --dump-test=dump_${b}_$bs.bin | tail -12
  done
  cmp dump_OLD_$bs.bin dump_NEW_$bs.bin && echo "batch $bs: IDENTICAL"
done
# With a blend weights file ($BLEND_NET), both the UCI route and the backend-option route:
for b in OLD NEW; do
  ${!b} backendcompare --weights=$BLEND_NET --fens=$FENS --max-positions=10000 --batch-size=64 --policy-blend-1=0.3 \
      --backend=lc0ex-cuda --backend-opts="lc0ex=$PROG" --ref-backend=lc0ex-cuda --ref-backend-opts="lc0ex=$PROG" \
      --dump-test=dump_${b}_blend_uci.bin > /dev/null
  ${!b} backendcompare --weights=$BLEND_NET --fens=$FENS --max-positions=10000 --batch-size=64 \
      --backend=lc0ex-cuda --backend-opts="lc0ex=$PROG,policy_blend_0=0.6,policy_blend_1=0.4" \
      --ref-backend=lc0ex-cuda --ref-backend-opts="lc0ex=$PROG" --dump-test=dump_${b}_blend_opt.bin > /dev/null
done
cmp dump_OLD_blend_uci.bin dump_NEW_blend_uci.bin; cmp dump_OLD_blend_opt.bin dump_NEW_blend_opt.bin
```

- **Expected: identical dumps** when `lc0ex-cuda` is deterministic.
- **If they differ**, quantify per field with numpy:
  - the dump is records of `3 + n_legal` float32;
  - the stderr of the two runs shows whether "Converting weights to ONNX first." happened. If `$NET` has no ONNX model,
    rerun with a net that has one, to separate the converter from the runtime.
- **If `$PROG` has a child-Q head,** the new binary's NN cache stores cq / cs (more memory). Run the same commands with
  `--nncache=...` at the match setting to check memory.

**3. Fixed-node search equality on the GPU.** Use the TCEC options with 1 thread for a deterministic comparison, then with
the match's thread count to compare only nps and stability.

```bash
COMMON="dag-preview --weights=$NET --backend=lc0ex-cuda --backend-opts=lc0ex=$PROG,gpu=0 --cpuct=2.15 --cpuct-base=45569 \
  --cpuct-factor=3.973 --policy-softmax-temp=1.36 --fpu-value=0.58 --smart-pruning-factor=0 --minibatch-size=32"
python3 - "$OLD" "$NEW" "$COMMON" > cfg_gpu.json <<'EOF'
import json, sys
old, new, common = sys.argv[1], sys.argv[2], sys.argv[3].split()
print(json.dumps({"old": [old, common + ["--threads=1"]], "new": [new, common + ["--threads=1"]],
                  "old_again": [old, common + ["--threads=1"]]}))
EOF
python3 search_eq2.py "$(cat cfg_gpu.json)" fens6.txt 20000 > se_gpu.json   # fens6.txt: the 6 FENs of Test 3
```

- **Expected:** `old vs new: 6/6 identical` when `old vs old_again` is 6/6.
- **Then pondering.** If the match uses `Ponder`, play one short ponder game per binary; #2409 changed that path.

## Scripts and patch used on the VM

<details><summary><code>run_datagen.sh</code> (Test 2; extra flags are appended)</summary>

```bash
#!/bin/bash
# usage: run_datagen.sh <lc0 binary> <out dir> [extra flags...]
# The fleet's recipe-39 command with --backend=random,childq=true and the
# randomness turned off: --noise-epsilon=0 --temperature=0, sequential
# openings from openings20.pgn, --parallelism=1, --games=20.
bin=$1; out=$2; shift 2
rm -rf "$out"; mkdir -p "$out/cache"
s=$(date +%s.%N)
XDG_CACHE_HOME="$out/cache" "$bin" selfplay --backend=random --backend-opts=childq=true \
 --training --parallelism=1 --minibatch-size=32 --batch-merge-max-batch=64 --batch-merge-wait-us=200 \
 --batch-merge-threads=4 --games=20 --visits=800 --cpuct=0.65 --cpuct-at-root=1.5 --root-has-own-cpuct-params=true \
 --temperature-simulated-cpuct=2.53 --resign-percentage=1.0 --resign-playthrough=20 --smart-pruning-factor=0.0 --temperature=0 \
 --temp-endgame=0.60 --tempdecay-moves=60 --tempdecay-delay-moves=20 --temp-cutoff-move=40 --temp-visit-offset=-0.8 \
 --fpu-strategy=reduction --fpu-value=0.23 --fpu-strategy-at-root=absolute --fpu-value-at-root=1.0 --policy-softmax-temp=1.45 \
 --resign-wdlstyle=true --noise-epsilon=0 --noise-alpha=0.12 --sticky-endgames=true --moves-left-max-effect=0.3 \
 --moves-left-threshold=0.0 --moves-left-slope=0.007 --moves-left-quadratic-factor=0.85 --moves-left-scaled-factor=0.15 \
 --moves-left-constant-factor=0.0 --task-workers=0 --resign-earliest-move=0 --forced-exploration-visits=220 \
 --forced-exploration-max-policy=7 --policy-target-blend-lambda=0.0 --policy-target-blend-tau=0.07 --policy-target-tail-eps=0.04 \
 --policy-target-tail-kappa=0.0 --policy-target-tail-floor=0.00125 --policy-target-tail-shrink=4.0 --training-data-v7=true \
 --training-data-v8=true --training-data-recipe-id=39 --training-data-child-q=true --training-data-child-q-min-visits=1 \
 --policy-target-draw-steer-beta=0.5 --policy-target-draw-steer-tau-d=0.25 --no-use-policy-post-processing \
 --policy-target-grill9=false --policy-target-prune-forced=true --policy-target-hybrid-tail=true --openings-pgn=/home/user/b3data/openings20.pgn \
 --openings-mode=sequential --selfplay-search=childq-classic --childq-mode=off --forced-exploration-sharp-checkpoint=200 \
 --forced-exploration-sharp-visits=440 --forced-exploration-sharp-max-q=0.5 --forced-exploration-sharp-min-gap=0.1503 \
 --policy-target-deforce-self-consistent=true --nncache-rule50-key-from=50 --stop-at-iteration-boundary \
 --forced-exploration-estimate-clamp --forced-exploration-sharp-mode=mu_s --forced-exploration-sharp-max-mu=1.5 \
 --forced-exploration-sharp-min-s=0.5 "$@" > "$out/stdout.txt" 2> "$out/stderr.txt"
rc=$?
e=$(date +%s.%N)
# Training files, renamed away from the random data-XXXXXXXXXXXX directory.
mkdir -p "$out/td"
for f in "$out"/cache/lc0/data-*/game_*.gz; do [ -e "$f" ] && cp "$f" "$out/td/"; done
grep '^gameready' "$out/stdout.txt" | sed 's/ trainingfile [^ ]*//' > "$out/games.txt"
printf "rc=%s time=%.1fs games=%s tdfiles=%s\n" $rc $(echo "$e - $s" | bc) $(wc -l < "$out/games.txt") $(ls "$out/td" | wc -l)
```

`run_fleet_verbatim.sh` (2d) is the same command with the recipe's own values: `--parallelism=24 --temperature=0.9
--noise-epsilon=0.1 --openings-mode=shuffled`, plus `--games=4`.

</details>

<details><summary>search flags of Test 2b (<code>childq-classic</code>, after <code>--threads=1 --backend=random --backend-opts=childq=true</code>)</summary>

```
--minibatch-size=32 --cpuct=0.65 --cpuct-at-root=1.5 --root-has-own-cpuct-params=true --temperature-simulated-cpuct=2.53
--smart-pruning-factor=0.0 --temperature=0 --temp-endgame=0.60 --tempdecay-moves=60 --tempdecay-delay-moves=20
--temp-cutoff-move=40 --temp-visit-offset=-0.8 --fpu-strategy=reduction --fpu-value=0.23 --fpu-strategy-at-root=absolute
--fpu-value-at-root=1.0 --policy-softmax-temp=1.45 --noise-epsilon=0 --noise-alpha=0.12 --sticky-endgames=true
--moves-left-max-effect=0.3 --moves-left-threshold=0.0 --moves-left-slope=0.007 --moves-left-quadratic-factor=0.85
--moves-left-scaled-factor=0.15 --moves-left-constant-factor=0.0 --task-workers=0 --forced-exploration-visits=220
--forced-exploration-max-policy=7 --childq-mode=off --forced-exploration-sharp-checkpoint=200
--forced-exploration-sharp-visits=440 --forced-exploration-sharp-max-q=0.5 --forced-exploration-sharp-min-gap=0.1503
--nncache-rule50-key-from=50 --stop-at-iteration-boundary --forced-exploration-estimate-clamp
--forced-exploration-sharp-mode=mu_s --forced-exploration-sharp-max-mu=1.5 --forced-exploration-sharp-min-s=0.5
--policy-target-blend-lambda=0.0 --training-data-child-q=true
```

</details>

<details><summary><code>test_only_seed_random.patch</code> (never committed; applied to both trees for the seeded runs)</summary>

```diff
--- a/src/utils/random.cc
+++ b/src/utils/random.cc
@@ -26,11 +26,18 @@
 */
 
 #include "random.h"
+#include <cstdlib>
 #include <random>
+#include <string>
 
 namespace lczero {
 
-Random::Random() : gen_(std::random_device()()) {}
+// TEST ONLY (B3 datagen equality, never committed): LC0_TEST_RANDOM_SEED
+// seeds the global generator so that two builds can be compared game by game.
+Random::Random() : gen_(std::random_device()()) {
+  const char* seed = std::getenv("LC0_TEST_RANDOM_SEED");
+  if (seed) gen_.seed(std::stoul(seed));
+}
 
 Random& Random::Get() {
   static Random rand;
```

</details>

<details><summary><code>search_eq2.py</code> (Tests 2b and 3)</summary>

```python
"""go nodes N on every FEN of a file, for several lc0 configurations; prints
JSON with the bestmove line and the last info line (time / nps removed).

usage: search_eq2.py '<json {name: [binary, [args...]]}>' <fens file> <nodes>
"""
import json
import re
import subprocess
import sys
import time


def run(binary: str, args: list[str], fen: str, nodes: int) -> dict:
    proc = subprocess.Popen([binary, *args], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    assert proc.stdin and proc.stdout
    position = "position startpos" if fen == "startpos" else f"position fen {fen}"
    proc.stdin.write(f"uci\nisready\nucinewgame\n{position}\ngo nodes {nodes}\n")
    proc.stdin.flush()
    last_info = best = ""
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("info depth"):
            last_info = line
        if line.startswith("bestmove"):
            best = line
            break
        if line.startswith("error"):
            best = line
            proc.kill()
            break
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    _, err = proc.communicate(timeout=120)
    if not best:
        print(f"no bestmove from {binary} {args} on {fen}: {err[-2000:]}",
              file=sys.stderr)
    return {"bestmove": best, "info": re.sub(r" (time|nps) \d+", "", last_info)}


def main() -> None:
    configs = json.loads(sys.argv[1])
    fens = [l.strip() for l in open(sys.argv[2]) if l.strip()]
    nodes = int(sys.argv[3])
    out = {}
    for name, (binary, args) in configs.items():
        t0 = time.time()
        out[name] = [run(binary, args, fen, nodes) for fen in fens]
        print(f"{name}: {len(fens)} searches in {time.time() - t0:.1f}s",
              file=sys.stderr)
    names = list(configs)
    ref = names[0]
    for name in names[1:]:
        same = sum(a == b for a, b in zip(out[ref], out[name]))
        print(f"{ref} vs {name}: {same}/{len(fens)} identical (bestmove + info)",
              file=sys.stderr)
    print(json.dumps({"fens": fens, "results": out}, indent=1))


main()
```

</details>

<details><summary><code>gen_openings.py</code> (<code>openings20.pgn</code>: <code>gen_openings.py 20 6</code>) and <code>games2pgn.py</code></summary>

```python
"""20 fixed openings: 6 random legal plies each from the start position."""
import random
import sys

import chess
import chess.pgn

count, plies = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(20261015)
seen = set()
games = []
while len(games) < count:
    board = chess.Board()
    for _ in range(plies):
        board.push(rng.choice(list(board.legal_moves)))
    if board.fen() in seen:
        continue
    seen.add(board.fen())
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = f"opening {len(games)}"
    games.append(game)
print("\n\n".join(str(g) for g in games))
```

```python
"""gameready lines (selfplay stdout) -> PGN, one game per line, in game id order."""
import sys

import chess
import chess.pgn

RESULTS = {"whitewon": "1-0", "blackwon": "0-1", "draw": "1/2-1/2"}
games = []
for line in open(sys.argv[1]):
    tok = line.split()
    if not tok or tok[0] != "gameready":
        continue
    gid = int(tok[tok.index("gameid") + 1])
    result = RESULTS.get(tok[tok.index("result") + 1], "*") if "result" in tok else "*"
    moves = tok[tok.index("moves") + 1:] if "moves" in tok else []
    if "from_fen" in moves:
        moves = moves[:moves.index("from_fen")]
    board = chess.Board()
    for m in moves:
        board.push_uci(m)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = f"game {gid}"
    game.headers["Result"] = result
    games.append((gid, str(game)))
print("\n\n".join(g for _, g in sorted(games)))
```

`fens20.txt` is `startpos` plus the first 19 lines of B1+B2's `fens_1000.txt` (made by `gen_fens.py` in B1+B2's REPORT).

</details>
