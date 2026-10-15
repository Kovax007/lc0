# B1+B2 report: upstream master + demux on the Backend API, child-Q through demux

Branch `cloud/b1-b2-demux-backend-api` (base `lc0ex-master`) in `Kovax007/lc0`.
Work was done on a CPU VM (4 cores, 15 GB RAM, Ubuntu 24.04, gcc 13.3, meson 1.3.2) with no GPU, on 2026-10-14
between 21:59 and 22:40 UTC.

## Summary

- **B1.** Merged upstream `master` (dab9898f). The merge was clean.
- **B2.**
  - Merged #2479 (f7d5378a). Clean.
  - Merged #2482 (9618d825). It conflicted in `network_demux.cc` and `wrapper.cc`, as expected; both are resolved.
  - #2482 is **not** merged upstream: upstream `master` is still dab9898f, which does not contain 9618d825.
- **Demux now accepts `lc0ex-cuda` children.** Before/after, on builds with `-Dlc0ex-runtime=true`:
  - the pre-merge `lc0ex-master` build fails with `Unknown backend: lc0ex-cuda`;
  - the merged build reaches `lc0ex-cuda`'s own factory. On the VM that factory stops at
    `The lc0ex-cuda backend requires a network file.`, because the VM has no network file.
- **Child-Q passes through demux unchanged.**
  - Under #2482, demux hands each caller's `EvalResultPtr`, `cq` / `cs` spans included, to the child that evaluates that
    sample, so the child writes straight into that sample's buffers. No per-sample offset arithmetic is left.
  - I added the one missing piece: `has_childq` aggregation.
  - Checks against one `random,childq=true` backend:
    - the new gtest gives exact equality (q, d, m, policy, cq, cs);
    - `backendcompare` over 1,000 positions reports 0 for every metric;
    - search: best move and final `info` line are identical on 4 FENs.
- **Mixed children** (some with the child-Q head, some without): demux reports `has_childq = false` and warns on stderr.
  It also never forwards `cq` / `cs` to any child, so callers get no child-Q at all rather than a sample-by-sample mix.
  This matches what the old-API demux did, through `NetworkCapabilities::Merge()`.
- **Nothing else changes.**
  - Searches on the pre-merge and merged builds are identical: `random,childq=true` alone, and demux.
  - `wrapper.cc` keeps the fork's child-Q path byte for byte.
  - The fork's delta on top of "upstream master + #2479 + #2482" is exactly the fork's own changes, plus 16 lines in
    `network_demux.cc`, the new test, the `backendcompare` change, and one meson test target.
- **Two things you need to know before building on the GPU box:**
  - **abseil.** #2482's `src/utils/optionsdict.cc` does not compile against Ubuntu 24.04's system `libabsl-dev`. Use the
    repo's abseil wrap (see [Build note](#build-note-abseil)).
  - **Inherited option changes.** #2482 itself changes some demux options and behaviour, for example per-child `threads=`
    becomes `demux_threads=` (see [Behaviour changes that come with #2482](#behaviour-changes-that-come-with-2482-not-from-the-conflict-resolution)).
    I did not revert them, because they are upstream's design.

## Heads used

| ref | expected | used | date | note |
|---|---|---|---|---|
| `lc0ex-master` (Kovax007/lc0) | e309c6af | **e309c6afbe6e** | 2026-09-28 | upstream 1227b4c8 + 2 lc0ex commits |
| upstream `master` | dab9898f | **dab9898f33f6** | 2026-10-09 | adds af8ae6a (#2459), dab9898 (#2476) |
| upstream `pull/2479/head` | f7d5378a | **f7d5378a56d5** | 2026-10-08 | 1 commit on af8ae6a |
| upstream `pull/2482/head` | 9618d825 | **9618d825941f** | 2026-10-10 | 7 commits on af8ae6a, not merged upstream |

The branch head after my commits is listed in the PR. Commits on the branch, oldest first:

```
1039e22 Merge upstream lc0 master (dab9898f) into lc0ex-master                    (B1)
464022f Merge upstream PR #2479 (f7d5378a): remove dead LoadNetwork / BackendConfiguration
7de114c Merge upstream PR #2482 (9618d825): migrate demux to the Backend API      (both conflicts resolved here)
63e5740 demux: child-Q through the Backend-API demux                             (has_childq + DemuxChildQTest)
495f631 backendcompare: report child-Q error
(next)  REPORT_cloud_b1_b2_demux.md
```

I merged rather than cherry-picked, so the upstream commits keep their identity, and a later upstream merge (squash or
not) meets identical content. The merge of #2482 contains only the conflict resolution. The child-Q logic for the new
demux is a separate commit, so it can be reviewed on its own.

## Conflicts and how I resolved them

### 1. `src/neural/backends/network_demux.cc`

Conflict hunk (ours = `lc0ex-master`, theirs = #2482), inside `class DemuxingComputation`:

```diff
  void ComputeBlocking() override;

<<<<<<< HEAD
  int GetBatchSize() const override { return planes_.size(); }
  float GetQVal(int sample) const override { auto [parent, offset] = GetParent(sample); ... }
  float GetDVal(int sample) const override { ... }
  float GetMVal(int sample) const override { ... }
  float GetPVal(int sample, int move_id) const override { ... }
  float GetChildQVal(int sample, int move_id) const override {
    auto [parent, offset] = GetParent(sample);
    if (!parent) return 0;
    return parent->GetChildQVal(offset, move_id);
  }
  float GetChildSigmaVal(int sample, int move_id) const override {
    auto [parent, offset] = GetParent(sample);
    if (!parent) return -1;
    return parent->GetChildSigmaVal(offset, move_id);
  }
=======
>>>>>>> upstream/pr2482
  void NotifyComplete() {
```

**Resolution: I took #2482's side, so the file is byte-identical to 9618d825 after the merge commit.**

- The `HEAD` side holds the old-API `NetworkComputation` getters (upstream's five plus the fork's two). #2482 turns
  `DemuxingComputation` into a `BackendComputation`, so these getters no longer exist.
- Their job, "each sample's values come from the child that evaluated it, at its own offset", is now done structurally:
  `DemuxingComputation::AddInput(pos, result)` passes the caller's `EvalResultPtr` unchanged to
  `computations_[child]->AddInput(pos, result)`.
- `EvalResultPtr` carries the `cq` / `cs` spans (fork commit e309c6a), so the child writes child-Q straight into the
  caller's per-sample buffers. There is no planes array, no offset and no copy left to get wrong.

**Follow-up commit 63e5740** adds what #2482 could not know about (+16 lines):

```cpp
// AggregateDemuxAttributes():
      .has_childq = true,                    // in the initializer
    ...
    // Child-Q only when every child fills it, as the old-API demux did through
    // NetworkCapabilities::Merge().
    result.has_childq &= attr.has_childq;    // in the per-child loop

// DemuxingBackend::DemuxingBackend(), after aggregation:
  if (!attrs_.has_childq && std::any_of(... child has_childq ...)) {
    CERR << "demux: some child backends have no child-Q head; child-Q is "
            "disabled for all of them.";
  }

// DemuxingComputation::AddInput(), before forwarding:
  // The child writes cq / cs straight into the caller's spans. Without child-Q
  // on every child, no child gets them, so no sample is filled by one child and
  // left untouched by another.
  if (!backend_->attrs_.has_childq) result.cq = result.cs = {};
```

**Why mixed children report false instead of refusing.**

- Old-API demux had the same rule, via `NetworkCapabilities::Merge()` (`if (!other.has_childq) has_childq = false;`). Under
  requirement (c), the migrated demux should not start rejecting configurations the old one accepted.
- It is consistent with how demux already aggregates `has_mlh` and `has_wdl`.
- The warning makes the misconfiguration visible.
- Stripping the spans makes the result uniform across samples even if a caller asks for child-Q despite
  `has_childq = false`. (Nothing in the tree does: `Backend::EvaluateBatch` and the NN cache request child-Q only when
  `has_childq` is true.)
- If you would rather refuse, it is a one-line change: throw in the constructor instead of the `CERR`.

### 2. `src/neural/wrapper.cc`

Conflict hunk, in `NetworkAsBackendComputation::ComputeBlocking()`:

```diff
      const EvalResultPtr& result = entries_[i].result;
<<<<<<< HEAD
      if (result.q) *result.q = computation_->GetQVal(i);
      if (result.d) *result.d = computation_->GetDVal(i);
      if (result.m) *result.m = computation_->GetMVal(i);
      if (!result.p.empty()) SoftmaxPolicy(result.p, computation_.get(), i);
      if (!result.cq.empty()) FillChildQ(result, computation_.get(), i);
    }
  }

  void FillChildQ(const EvalResultPtr& result,
                  const NetworkComputation* computation, int idx) {
    ... (fork's body: raw head outputs at MoveToNNIndex(move, transform))
=======
      if (result.q) *result.q = computation->GetQVal(i);
      if (result.d) *result.d = computation->GetDVal(i);
      if (result.m) *result.m = computation->GetMVal(i);
      if (!result.p.empty()) SoftmaxPolicy(result.p, computation.get(), i);
>>>>>>> upstream/pr2482
```

**Resolution: I kept both sides.**

- From #2482: the `NetworkComputation` is now a local created in `ComputeBlocking()` (`computation`), instead of the
  `computation_` member created in the constructor.
- From the fork: `attrs_.has_childq = caps.has_childq` (it merged without conflict) and the `FillChildQ` call and
  function, unchanged. The call now passes `computation.get()`.

After the merge commit, `git diff upstream/pr2482 -- src/neural/wrapper.cc` is exactly the fork's original +14 lines.

### Non-conflicting overlaps (checked)

- **`src/neural/register.h`:** the fork had already made the same `REGISTER_BACKEND` fix as #2482
  (`SearchFactory::Register` → `BackendManager::Register`), so git merged the identical change.
- **`.gitignore`:** both sides added `.juggler/`. The result keeps #2482's `builddir` and the fork's trailing newline.
- **`meson.build`:** merged cleanly. #2482's `DemuxTest` target sits next to the fork's lc0ex additions.
- **#2479:** nothing in the fork used `NetworkFactory::LoadNetwork` or `BackendConfiguration` (grep of `src/`).

## Exact demux syntax (confirmed from the code)

```
--backend=demux --backend-opts="[batch_step=N,][shared scalars...,]NAME1(backend=B1[,demux_threads=T][,child opts...]),NAME2(...)..."
```

How `DemuxingBackend::DemuxingBackend` builds the child options:

- Every **subdict** is one child. Its backend is `backend=` if given, otherwise the subdict name. Children are created
  in subdict-name order (the subdicts are held in a `std::map`).
- The child's backend-opts string is built in four steps:
  1. take the parent's **scalar** options, minus `batch_step`;
  2. merge the child's subdict over them (the subdict wins);
  3. drop `backend` and `demux_threads`;
  4. serialize the result with `OptionsDict::Serialize()`.
- The child is created with `BackendManager::Get()->CreateFromParams(child_opts)`, where `child_opts` has the top-level
  options as parent. So the child inherits `--weights`, `--policy-softmax-temp`, `--history-fill` and the rest.
- This lookup goes through the Backend API, so every backend is reachable: `REGISTER_BACKEND` ones such as `lc0ex-cuda`,
  and old `REGISTER_NETWORK` ones, which register in both registries.
- Top-level `batch_step` (default 1): samples are routed round-robin in groups of `batch_step`, and the starting child
  rotates with every new computation.
- Per-child `demux_threads` (default: the child's `suggested_num_search_threads`, which is 2 for `lc0ex-cuda`) sets
  the number of worker threads per child.

For `lc0ex-cuda` on two GPUs:

```
--backend=demux --backend-opts="lc0ex=/tmp/BT4-1024x15x32h-swa-6147500-sm120.lc0ex,g0(backend=lc0ex-cuda,gpu=0),g1(backend=lc0ex-cuda,gpu=1)"
```

That gives each child the backend-opts `"gpu"=0,"lc0ex"="/tmp/BT4-...lc0ex"` (and `gpu=1` for `g1`). `lc0ex-cuda`'s
factory parses that and calls `CheckAllOptionsRead()`. No demux-only key reaches it, so it does not throw.

- Shared `lc0ex-cuda` options such as `concurrency=2` or `graph=dag` can go at the top level (all children) or in one
  subdict.
- To keep each search gather whole on one GPU (like `multi`'s `mode=roundrobin`), add `batch_step=<top rung of the
  program>`.
- The DemuxChildQ test `CreatesBackendApiChildren` uses this exact shape with a stand-in backend that parses options the
  way `Lc0exCudaBackendFactory` does.

## Behaviour changes that come with #2482 (not from the conflict resolution)

I kept these as upstream wrote them. Existing demux command lines may need updating:

1. **Per-child `threads=` is now `demux_threads=`.** An old-style `threads=` in a child subdict is no longer consumed by
   demux. It reaches the child, and children that don't know it fail. Seen on the VM:
   `a(backend=random,threads=2)`:
   - pre-merge: works;
   - merged: `Unknown integer option: random.threads`.
2. **Demux with no child subdicts now throws** `demux needs child backends.` Before, it ran the default backend (or the
   root-level `backend=`) on the root options. Seen on the VM:
   - pre-merge `--backend-opts='childq=true'`: `Creating backend [blas]... The blas backend requires a network file.`
   - merged: `demux needs child backends.`
3. **Routing.**
   - Before: each batch was split into contiguous, near-equal chunks across all children. The step came from the
     children's `GetPreferredBatchStep()`.
   - Now: samples go round-robin in groups of `batch_step`, a top-level option with default 1. With the default, one
     gather of B is split into N interleaved parts of about B/N, which on the lc0ex rung ladder means smaller programs per
     GPU. See the benchmark commands below.
4. **Batch-size attributes.**
   - `maximum_batch_size` is now derived from the children: `N * (min child max / step) * step + (min child max % step)`.
     The old-API wrapper used a fixed 1024.
   - `recommended_batch_size` is `min(N * min child recommended, max)`.
   - `suggested_num_search_threads` is 1, as before.
5. **`wrapper.cc`:** old-API networks wrapped as Backends now create their `NetworkComputation` in `ComputeBlocking()`,
   that is, on the thread that computes. Before, it was created when the `BackendComputation` was constructed. Results
   are unaffected; the pre/post search comparison below is identical.
6. Only `demux` moved to the Backend API. **`multiplexing` and `roundrobin` are still old-API** and still cannot take
   `lc0ex-cuda` children. `multi` is unchanged; note that it takes `has_childq` from its first device without
   aggregating (out of scope, unchanged).

## Build note: abseil

#2482 adds `absl::StrCat("\"", str, "\"")` with `str` a `std::string_view` (`src/utils/optionsdict.cc:340-341`). Ubuntu
24.04's `libabsl-dev` (20220623) is built with `ABSL_OPTION_USE_STD_STRING_VIEW 0`, so `absl::string_view` is not
`std::string_view` there, and the build fails:

```
../../src/utils/optionsdict.cc:340:62: error: no matching function for call to 'StrCat(const char [2], std::string_view&, const char [2])'
```

- **The fix is to build without the system package.** I removed `libabsl-dev`, and meson then used
  `subprojects/abseil-cpp.wrap` (20240722.0), which is what upstream CI uses. Both builds, pre-merge and merged, then
  used the same abseil.
- If the GPU box has `libabsl-dev` installed, add `--force-fallback-for=absl_flat_hash_map` to `meson setup` (or to
  `build.sh`'s arguments), or remove the package.
- The pre-merge `lc0ex-master` builds against the system package, so this is a regression a box with `libabsl-dev` would
  hit.
- I did not patch #2482's code, to avoid merge conflicts with upstream. The one-line portable fix would be
  `absl::StrCat("\"", absl::string_view(str.data(), str.size()), "\"")`, or plain `std::string` concatenation. Worth
  reporting on #2482.

## Commands run and results

All builds are meson `release` (LTO on, per the project's defaults). `M` is the merged branch build, `B` the pre-merge
`lc0ex-master` (e309c6a) build in a separate worktree.

### Setup, B1, B2

```bash
git clone --depth 50 --branch lc0ex-master https://github.com/Kovax007/lc0 /home/user/lc0
git remote add upstream https://github.com/LeelaChessZero/lc0
git fetch --depth 80 upstream master 'refs/pull/2479/head:refs/remotes/upstream/pr2479' \
                                     'refs/pull/2482/head:refs/remotes/upstream/pr2482'
git checkout -b cloud/b1-b2-demux-backend-api
git merge --no-ff upstream/master        # clean: 3 files changed, 2 insertions(+), 302 deletions(-)
git merge --no-ff upstream/pr2479        # clean: 2 files changed, 1 insertion(+), 58 deletions(-)
git merge --no-ff --no-commit upstream/pr2482
# CONFLICT (content): Merge conflict in src/neural/backends/network_demux.cc
# CONFLICT (content): Merge conflict in src/neural/wrapper.cc
git checkout --theirs src/neural/backends/network_demux.cc    # see "Conflicts" above
# wrapper.cc: hand-merged as described above
```

### Test 1: build and `meson test`

```bash
apt-get install meson libopenblas-dev libgtest-dev libeigen3-dev    # meson 1.3.2, gtest 1.14.0, OpenBLAS 0.3.26
meson setup build/release --buildtype release -Dblas=true -Dgtest=true
meson compile -C build/release -j 4
meson test -C build/release
```

```
 1/10 FP16                OK              0.01s
 2/10 HashCat             OK              0.01s
 3/10 PositionTest        OK              0.11s
 4/10 OptionsParserTest   OK              0.11s     (includes #2482's 4 new OptionsDict tests)
 5/10 SyzygyTest          OK              0.11s
 6/10 EncodePositionForNN OK              0.11s
 7/10 DemuxTest           OK              0.11s     (#2482's network_demux_test: 2 tests)
 8/10 DemuxChildQTest     OK              0.11s     (new, below)
 9/10 EngineTest          OK              0.11s
10/10 ChessBoard          OK              2.41s
Ok:                 10
Fail:               0
real	0m2.729s
```

Backends in the merged `-Dblas=true` build:
`blas eigen trivial random check roundrobin recordreplay multiplexing demux multi`.

### Tests 2 and 3: new gtest `DemuxChildQTest` (`src/neural/backends/network_demux_childq_test.cc`)

```
[ RUN      ] DemuxChildQ.RandomChildrenMatchSingleBackend
[       OK ] DemuxChildQ.RandomChildrenMatchSingleBackend (29 ms)
[ RUN      ] DemuxChildQ.EverySampleComesWhollyFromOneChild
[       OK ] DemuxChildQ.EverySampleComesWhollyFromOneChild (17 ms)
[ RUN      ] DemuxChildQ.MixedChildrenReportNoChildQ
demux: some child backends have no child-Q head; child-Q is disabled for all of them.   (x4, expected)
[       OK ] DemuxChildQ.MixedChildrenReportNoChildQ (22 ms)
[ RUN      ] DemuxChildQ.CreatesBackendApiChildren
[       OK ] DemuxChildQ.CreatesBackendApiChildren (0 ms)
[  PASSED  ] 4 tests.
```

What each test covers:

- **`RandomChildrenMatchSingleBackend`** (test 2).
  - Setup: demux over **2 and 3** `random` children with `childq=true`, against one `random,childq=true`.
  - Grid: batch sizes **1, 2, 7, 64, 257**, `batch_step` **1, 3, 64**.
  - Positions: four FENs plus deterministic random walks of up to 22 plies, so histories differ.
  - Checks: **exact** float equality of q, d, m, the policy vector, cq and cs for every sample. It also asserts that
    every `cs` is in [0.05, 0.30], so the values come from the synthetic head and not from the "no head" defaults
    (0 / -1).
- **`EverySampleComesWhollyFromOneChild`.**
  - Setup: three children with different `seed`s (so they give different numbers), same batch grid.
  - Checks: every sample matches exactly one seed's reference in **all** fields, child-Q included. The number of children
    used equals `min(groups, 3)`.
  - This is the "child-Q comes back from the child that evaluated it" property, tested directly.
- **`MixedChildrenReportNoChildQ`** (the mixed case I chose).
  - Setup: `childq=true,a(backend=random),b(backend=random,childq=false)` and a 1-without/2-with variant; `batch_step`
    1 and 3; all 5 batch sizes.
  - Checks:
    - `has_childq == false`;
    - `EvaluateBatch` returns empty cq / cs, with q / d / m / policy equal to the reference;
    - when a caller passes cq / cs spans anyway, they stay untouched for every sample;
    - demux over two non-child-Q children also reports false.
- **`CreatesBackendApiChildren`** (test 3, the `lc0ex-cuda` route).
  - Setup: a test-only backend registered with **`REGISTER_BACKEND`** (static registration in the test binary).
    - It is absent from `NetworkFactory::GetBackendsList()`, which is where the old demux looked, and present in
      `BackendManager::GetBackendNames()`.
    - Its factory reads options the way `Lc0exCudaBackendFactory` does: required `lc0ex`, `gpu`, then
      `CheckAllOptionsRead()`.
  - Demux is created with
    `batch_step=S,lc0ex=/models/net.lc0ex,g0(backend=demux-childq-registered-test,gpu=0),g1(backend=...,gpu=1,demux_threads=2)`,
    `--weights=parent-weights.pb.gz`, for S = 1 and 2.
  - Checks:
    - both children are created, each with `lc0ex=/models/net.lc0ex`, its own `gpu`, and the parent's weights path;
    - nothing leaks: `backend`, `demux_threads` and `batch_step` never reach a child, otherwise `CheckAllOptionsRead()`
      would throw;
    - `has_childq` is true;
    - each sample's cq comes from the child that set its q, and cs encodes the sample and the move.

**Mutation check of the new test.** I broke demux three ways; each was caught, and I restored the file afterwards with
`git checkout`:

| mutation in `network_demux.cc` | result |
|---|---|
| never forward cq / cs (`result.cq = result.cs = {};` unconditionally) | 3 of 4 tests FAIL (all but Mixed) |
| drop `result.has_childq &= attr.has_childq` | `MixedChildrenReportNoChildQ` FAILS |
| keep cq / cs for mixed children (drop the strip) | `MixedChildrenReportNoChildQ` FAILS |

**The real `lc0ex-cuda` lookup path**, in builds with `-Dlc0ex-runtime=true` (see Test 6):

```bash
lc0 backendcompare --backend=demux \
  --backend-opts='lc0ex=/nonexistent/net.lc0ex,a(backend=lc0ex-cuda,gpu=0),b(backend=lc0ex-cuda,gpu=1)' \
  --ref-backend=random --fens=fens_1000.txt
```

```
B (lc0ex-master, -Dlc0ex-runtime=true):  Unknown backend: lc0ex-cuda
M (merged,       -Dlc0ex-runtime=true):  The lc0ex-cuda backend requires a network file.
```

The second message comes from inside `Lc0exCudaBackendFactory::Create`, so demux found and called `lc0ex-cuda`. The VM
has no network or `.lc0ex` file to go further.

### Test 4: search equality

Every run is a fresh process: `uci`, `isready`, `ucinewgame`, `position ...`, `go nodes 2000`, wait for `bestmove`. The
common flags were:

```
--threads=1 --minibatch-size=32 --smart-pruning-factor=0
```

`--smart-pruning-factor=0` is there for determinism. lc0's smart-pruning stopper waits
`kSmartPruningToleranceMs = 200` of wall-clock time after the first eval (`stoppers.cc:190`), so with pruning on, a
node-limited search can stop at a time-dependent node count. Compared: the `bestmove` line, and the last `info depth`
line with `time` and `nps` removed.

Configurations:

| name | binary | backend flags |
|---|---|---|
| merged_random | M | `--backend=random --backend-opts=childq=true` |
| merged_random_again | M | same, second run (determinism) |
| merged_demux2 | M | `--backend=demux --backend-opts=childq=true,a(backend=random),b(backend=random)` |
| merged_demux3_step4 | M | `--backend=demux --backend-opts=batch_step=4,childq=true,a(backend=random),b(backend=random),c(backend=random)` |
| base_random | B | `--backend=random --backend-opts=childq=true` |
| base_demux2 | B | `--backend=demux --backend-opts=childq=true,a(backend=random),b(backend=random)` (old-API demux) |

Result: all six configurations agree on all four positions (24 searches in 4 s):

```
== startpos   bestmove g1h3 ponder h7h6   info depth 4 seldepth 8 nodes 2012 score cp 9 tbhits 0 pv g1h3 h7h6 c2c4 d7d6 d2d3 b7b5 h3f4 g7g5
== kiwipete   bestmove g2g3 ponder h8h4   info depth 4 seldepth 7 nodes 2012 score cp 16 tbhits 0 pv g2g3 h8h4 c3b1 a6c4 d2g5 h4g4 f3f6
== perft4     bestmove b4c5 ponder g7h6   info depth 4 seldepth 9 nodes 2031 score cp 21 tbhits 0 pv b4c5 g7h6 d1c1 a3c3 c1e1 a8a7
== endgame    bestmove e2e4 ponder h5d5   info depth 5 seldepth 8 nodes 2016 score cp 7 tbhits 0 pv e2e4 h5d5 e4d5 c7c6 b5c6 h4g4 b4b5 g4f5
   (each line: merged_random, merged_random_again, merged_demux2, merged_demux3_step4, base_random, base_demux2 -> SAME)
```

The FENs:

- kiwipete = `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`
- perft4 = `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1`
- endgame = `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1`

Positive control: demux with one child at `seed=1` (`childq=true,a(backend=random),b(backend=random,seed=1)`) gives
**DIFF** on all four positions (for example startpos `bestmove b1a3`), so the comparison does detect a wrong child.

Caveat: search on `lc0ex-master` does not read child-Q yet. These runs exercise child-Q only through the NN cache, which
requests and stores it for every entry once `has_childq` is true. They do not exercise a search that uses it.

### Test 5: `backendcompare`, `random,childq=true` against demux of two, 1,000 positions

The 1,000 positions come from deterministic random playouts (python-chess, seed 20261014, a random ply from 4 to 120,
finished positions skipped).

```bash
lc0 backendcompare --backend=demux --backend-opts="childq=true,a(backend=random),b(backend=random)" \
    --ref-backend=random --ref-backend-opts="childq=true" \
    --fens=fens_1000.txt --max-positions=1000 --batch-size=32      # and again with --batch-size=7
```

```
# test backend      : demux  opts='childq=true,a(backend=random),b(backend=random)'
# reference backend : random  opts='childq=true'
# positions compared: 1000  (skipped 0)
metric                            mean         p50         p90         p99         max
--------------------------------------------------------------------------------------
policy KL(ref||test)          0.000000    0.000000    0.000000    0.000000    0.000000
policy L1                     0.000000    0.000000    0.000000    0.000000    0.000000
|q_test - q_ref|              0.000000    0.000000    0.000000    0.000000    0.000000
|d_test - d_ref|              0.000000    0.000000    0.000000    0.000000    0.000000
|m_test - m_ref|              0.000000    0.000000    0.000000    0.000000    0.000000
|cq_test - cq_ref| (move)     0.000000    0.000000    0.000000    0.000000    0.000000
|cs_test - cs_ref| (move)     0.000000    0.000000    0.000000    0.000000    0.000000
top-1 agreement          : 100.0000 %
ref best move in test top-3 : 100.0000 %
```

`--batch-size=7` gives the same table: all zeros, 100 %.

Supporting checks:

- **Exactness.**
  - The table prints 6 decimals. The `--dump-test` files (q, d, m and policy as float32) for demux-of-two and for single
    `random` are **byte-identical**: 135,328 bytes each, sha256 `42da8f14…6052bb` for both.
  - cq / cs exactness is asserted by the gtest above.
- **Positive control.** With the reference at `childq=true,seed=1`:
  - |cq| p99 is 1.804 and max 1.997; |cs| p99 is 0.2255 and max 0.2493;
  - top-1 agreement is 4 %.
  - So the new rows do measure something.
- **Reference without child-Q.** The tool prints `child-Q not compared: test has_childq=yes, ref has_childq=no`.

**The `backendcompare` change (495f631).**

- When both backends have child-Q, it adds two rows, `|cq_test - cq_ref| (move)` and `|cs_test - cs_ref| (move)`, over
  every legal move of every position, with mean, p50, p90, p99 and max.
- Otherwise it prints which side lacks child-Q.
- The rest of the output and the `--dump-test` format are unchanged.

### Test 6 (optional): build with `-Dlc0ex-runtime=true -Dplain_cuda=false`

```bash
apt-get install --no-install-recommends nvidia-cuda-dev      # 12.0.146: /usr/include/cuda.h, libcuda.so (+ stubs/libcuda.so)
meson setup build/lc0ex --buildtype release -Dlc0ex-runtime=true -Dplain_cuda=false -Dblas=true -Dgtest=false
meson compile -C build/lc0ex -j 4
```

```
Library cuda found: YES
    lc0ex-runtime: true
    plain_cuda   : false
[267/267] Linking target lc0
```

- **It builds and links on the VM.** Merged branch: 1 min 45 s. The pre-merge branch builds the same way.
- The `Backend` option of the merged binary lists
  `blas eigen trivial lc0ex-cuda random check roundrobin recordreplay multiplexing demux multi`.
- **Not runnable here:** there is no GPU, network or `.lc0ex` file. The before/after lookup check above is as far as the
  VM goes.
- **Linking used `/usr/lib/x86_64-linux-gnu/libcuda.so`**, the userspace library that `nvidia-cuda-dev` pulled in through
  `libnvidia-compute-535`. A stub is at `/usr/lib/x86_64-linux-gnu/stubs/libcuda.so` if a box has no driver package. The
  `cuda.h` was from CUDA 12.0; the lc0ex runtime compiled against it without errors.

## Time per step

| step | time |
|---|---|
| clone `lc0ex-master` (depth 50) + fetch upstream master, #2479, #2482 | ~1 min |
| reading the fork's child-Q commit, #2482 and the backends | ~2 min |
| B1 merge (clean) | < 1 s |
| #2479 merge (clean) | < 1 s |
| #2482 merge and resolution of both conflicts | ~2 min (22:01:38 to 22:03:18) |
| demux `has_childq` change and new gtest | ~3 min |
| first merged build: failed on system abseil, then diagnosis | ~1.5 min |
| merged build, `-Dblas=true -Dgtest=true`, with the abseil wrap (setup 3 s, compile 2 min 19 s at `-j4`) | ~2.5 min |
| pre-merge build, `-Dblas=true` (at `-j2`, alongside other work) | 3 min 05 s |
| `meson test` (10 tests) | 2.7 s |
| test fix (`InitializeMagicBitboards()` in the new test's `main`) and rerun | ~1 min |
| `backendcompare` extension, rebuild, runs and controls | ~3 min |
| search equality: 6 configs × 4 FENs, plus control | 4 s + 2 s (script ~3 min) |
| mutation testing of the gtest (3 mutations) | ~1.5 min |
| optional step: `nvidia-cuda-dev` install 67 s, merged lc0ex build 1 min 45 s, pre-merge lc0ex build 2 min 10 s | ~5 min |
| REPORT, PR | ~15 min |

## What I could not test on the VM

- Anything that runs `lc0ex-cuda`: CUDA initialisation, `.lc0ex` loading, the network-fingerprint check, graph
  inference, and decoding of the real child-Q head (FP16 / FP32 rows, `sqrt` of the variance) through demux.
- Multi-GPU behaviour of demux with `lc0ex-cuda` children:
  - `ComputeBlocking()` runs on demux worker threads, not on the search thread. The runtime calls `cuCtxSetCurrent` on
    each call, and `multi`'s split mode already relies on that, but I could not run it.
  - The interaction of `demux_threads` (default 2 per `lc0ex-cuda` child) with `concurrency`.
  - Throughput.
- Whether `lc0ex-cuda` is bit-for-bit deterministic across runs and GPUs. That decides whether the GPU `backendcompare`
  below should show exact zeros or a small floor.
- A search that consumes child-Q: `lc0ex-master`'s search does not read it yet.
- The ONNX backend's child-Q (`network_onnx.cc`): onnxruntime is not installed here. It is untouched by this branch and
  reaches demux through `wrapper.cc` like any old-API network.

## Commands for a maintainer on a multi-GPU box

Get the branch into the `lczero-triton` checkout and build it, following `AGENTS.md`:

```bash
git -C submodules/lc0 fetch https://github.com/Kovax007/lc0 cloud/b1-b2-demux-backend-api
git -C submodules/lc0 checkout FETCH_HEAD
# If libabsl-dev is installed system-wide, add --force-fallback-for=absl_flat_hash_map (see "Build note").
./submodules/lc0/build.sh release -Dlc0=true -Dlc0ex-runtime=true -Dbuild_backends=false -Dgtest=false

LC0=./submodules/lc0/build/release/lc0
NET=/home/crem/dev/lc0/build/release/BT4-1024x15x32h-swa-6147500.pb.gz
PROG=/tmp/BT4-1024x15x32h-swa-6147500-sm120.lc0ex
CAP=169            # largest compiled batch (top rung) of $PROG
FENS=/path/to/fens.txt   # one FEN per line, ~10k real positions
```

### 0. Preload check: demux over 2 `lc0ex-cuda` children

```bash
printf 'uci\nquit\n' | $LC0 --config= --preload --weights=$NET --backend=demux \
    --backend-opts="lc0ex=$PROG,g0(backend=lc0ex-cuda,gpu=0),g1(backend=lc0ex-cuda,gpu=1)" \
    --logfile='<stderr>'
```

Expected: both children load, with no `demux: some child backends have no child-Q head` line if `$PROG` has the
child-Q head.

### 1. `backendcompare`: one `lc0ex-cuda` against demux over N children (q / d / m / policy / cq / cs)

The cq / cs rows appear only when both sides report `has_childq`.

**(a) Cross-GPU floor.** The same program on GPU 1 against GPU 0. This is the best that any multi-GPU setup can match:

```bash
$LC0 backendcompare --weights=$NET --fens=$FENS --max-positions=10000 --batch-size=64 \
    --backend=lc0ex-cuda --backend-opts="lc0ex=$PROG,gpu=1" \
    --ref-backend=lc0ex-cuda --ref-backend-opts="lc0ex=$PROG,gpu=0"
```

**(b) Demux, whole batches.** With `batch_step=64`, every 64-position batch goes whole to one GPU, alternating, and runs
the same program rung as the reference. Expect the same numbers as (a): exact zeros on identical GPUs if the runtime is
deterministic. **This is the decisive check of routing and child-Q plumbing.**

```bash
$LC0 backendcompare --weights=$NET --fens=$FENS --max-positions=10000 --batch-size=64 \
    --backend=demux \
    --backend-opts="lc0ex=$PROG,batch_step=64,g0(backend=lc0ex-cuda,gpu=0),g1(backend=lc0ex-cuda,gpu=1)" \
    --ref-backend=lc0ex-cuda --ref-backend-opts="lc0ex=$PROG,gpu=0"
# N=4: ...,g0(backend=lc0ex-cuda,gpu=0),g1(...gpu=1),g2(...gpu=2),g3(...gpu=3)
```

**(c) Demux, split batches.** The default `batch_step=1` interleaves each batch across the GPUs (64/N each), so the
children run smaller rungs than the reference. The result is informative: small rung-to-rung differences are expected,
top-1 should stay near 100 %, and cq / cs errors should be about the size of the q error.

```bash
$LC0 backendcompare --weights=$NET --fens=$FENS --max-positions=10000 --batch-size=64 \
    --backend=demux \
    --backend-opts="lc0ex=$PROG,g0(backend=lc0ex-cuda,gpu=0),g1(backend=lc0ex-cuda,gpu=1)" \
    --ref-backend=lc0ex-cuda --ref-backend-opts="lc0ex=$PROG,gpu=0"
```

### 2. `backendbench`: demux over N, against N separate single-GPU processes, and against `multi`

Use the same sweep everywhere. Shown for N=2; for N=4 add `g2`, `g3` / `gpus=0:1:2:3` and double `--threads`.

```bash
SWEEP="--batches=200 --start-batch-size=16 --batch-step=16"
```

**N separate single-GPU processes.** Run them concurrently and add the nps of the two outputs row by row:

```bash
for g in 0 1; do
  $LC0 backendbench --weights=$NET --backend=lc0ex-cuda --backend-opts="lc0ex=$PROG,gpu=$g" \
      --threads=2 $SWEEP --max-batch-size=$CAP > bench_single_gpu$g.txt &
done; wait
```

**Demux, whole-gather round robin** (`batch_step=$CAP`). Each computation goes whole to one GPU, which is comparable to
`multi mode=roundrobin`. It needs ≥ N concurrent computations, hence `--threads=2N`:

```bash
$LC0 backendbench --weights=$NET --backend=demux \
    --backend-opts="lc0ex=$PROG,batch_step=$CAP,g0(backend=lc0ex-cuda,gpu=0),g1(backend=lc0ex-cuda,gpu=1)" \
    --threads=4 $SWEEP --max-batch-size=$CAP > bench_demux_rr.txt
```

**Demux, split** (default `batch_step=1`). Each batch of B is interleaved, B/N per GPU, which is comparable to
`multi mode=split`. Sweep up to N×CAP:

```bash
$LC0 backendbench --weights=$NET --backend=demux \
    --backend-opts="lc0ex=$PROG,g0(backend=lc0ex-cuda,gpu=0),g1(backend=lc0ex-cuda,gpu=1)" \
    --threads=2 $SWEEP --max-batch-size=$((2*CAP)) > bench_demux_split.txt
```

**The fork's `multi` backend**, same two modes:

```bash
$LC0 backendbench --weights=$NET --backend=multi \
    --backend-opts="backend=lc0ex-cuda,gpus=0:1,mode=roundrobin,opts=lc0ex=$PROG" \
    --threads=4 $SWEEP --max-batch-size=$CAP > bench_multi_rr.txt
$LC0 backendbench --weights=$NET --backend=multi \
    --backend-opts="backend=lc0ex-cuda,gpus=0:1,mode=split,opts=lc0ex=$PROG" \
    --threads=2 $SWEEP --max-batch-size=$((2*CAP)) > bench_multi_split.txt
```

Knobs worth one extra run each:

- `concurrency=2` at the top level of the demux options (it applies to every child), or `opts=lc0ex=$PROG;concurrency=2`
  for `multi`;
- `demux_threads=1` or `demux_threads=3` per child subdict.

### 3. Optional: search equality on GPU

The same procedure as Test 4. If `lc0ex-cuda` is deterministic, the best move and final `info` line should match
exactly; `batch_step` = minibatch keeps every gather on one GPU, on the same rung as the single-GPU run:

```bash
for B in "--backend=lc0ex-cuda --backend-opts=lc0ex=$PROG,gpu=0" \
         "--backend=demux --backend-opts=lc0ex=$PROG,batch_step=64,g0(backend=lc0ex-cuda,gpu=0),g1(backend=lc0ex-cuda,gpu=1)"; do
  (printf 'uci\nisready\nucinewgame\nposition startpos\ngo nodes 20000\n'; sleep 30; printf 'quit\n') | \
    $LC0 --weights=$NET --threads=1 --minibatch-size=64 --smart-pruning-factor=0 $B | grep -E '^(info depth|bestmove)' | tail -2
done
```

## Scripts used on the VM

<details><summary><code>gen_fens.py</code> (1,000 positions for Test 5)</summary>

```python
"""Deterministic FEN list: random playouts from the start position, one FEN
sampled per game at a random ply, skipping finished positions."""
import random
import sys

import chess

count = int(sys.argv[1])
rng = random.Random(20261014)
fens: list[str] = []
seen: set[str] = set()
while len(fens) < count:
    board = chess.Board()
    target = rng.randint(4, 120)
    for _ in range(target):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    if board.is_game_over():
        continue
    fen = board.fen()
    if fen in seen:
        continue
    seen.add(fen)
    fens.append(fen)
print("\n".join(fens))
```

</details>

<details><summary><code>search_eq.py</code> (Test 4)</summary>

```python
"""Runs `go nodes N` per FEN for several lc0 configurations and compares the
best move and the final info line (time and nps fields removed)."""
import json
import re
import subprocess
import sys
import time

FENS = {
    "startpos": None,
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "perft4": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
}
COMMON = ["--threads=1", "--minibatch-size=32", "--smart-pruning-factor=0"]


def run(binary: str, backend: list[str], fen: str | None, nodes: int) -> dict:
    proc = subprocess.Popen(
        [binary, *COMMON, *backend],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    assert proc.stdin and proc.stdout
    position = "position startpos" if fen is None else f"position fen {fen}"
    proc.stdin.write(f"uci\nisready\nucinewgame\n{position}\ngo nodes {nodes}\n")
    proc.stdin.flush()
    last_info = ""
    best = ""
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("info depth"):
            last_info = line
        if line.startswith("bestmove"):
            best = line
            break
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=60)
    stable = re.sub(r" (time|nps) \d+", "", last_info)
    return {"bestmove": best, "info": stable, "raw": last_info}


def main() -> None:
    configs = json.loads(sys.argv[1])
    nodes = int(sys.argv[2])
    results: dict[str, dict[str, dict]] = {}
    for name, (binary, backend) in configs.items():
        results[name] = {}
        for fen_name, fen in FENS.items():
            t0 = time.time()
            results[name][fen_name] = run(binary, backend, fen, nodes)
            results[name][fen_name]["seconds"] = round(time.time() - t0, 2)
    print(json.dumps(results, indent=1))


main()
```

</details>
