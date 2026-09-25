# lc0ex runtime — work in progress

A branch on top of **`fa893575`** (`wip-20260825`) of `mooskagh/lc0`. This is the *runtime* half of
the lc0ex speedup work: it runs the artifact. The *builder* half — CUTLASS GEMMs, segment
chunking, autotune reuse — is a matching branch on `mooskagh/lczero-triton`, and everything there
is baked into the `.lc0ex` file at build time. **You need both.**

All numbers below were measured on **RTX 4090s (sm_89)**.

## What is in the branch

| file | what it does |
|---|---|
| `neural/backends/lc0ex-cuda/runtime/lc0ex_cuda.{cc,h}` | Execution-slot pool — a semaphore admits `concurrency` threads, each getting its own stream and buffers. CUDA-graph capture and instantiation (`graph=dag\|linear\|off`). `LC0EX_BATCH_HIST=1` records the batch sizes search actually asks for. |
| `neural/backends/lc0ex-cuda/network_lc0ex_cuda.cc` | Backend options: `gpu`, `lc0ex`, `concurrency`, `graph`. |
| `neural/backends/lc0ex-cuda/runtime/runtime.h` | Graph-mode enum. |
| `neural/backends/backend_multi.cc` | `multi` composite backend — fan one search out over several devices, `mode=roundrobin`. |
| `tools/backendcompare.{cc,h}` | Compare two backends position by position: policy KL, top-1 agreement, value error. This is the correctness gate for every change above. |
| `meson.build`, `main.cc` | Register the two new units. |

## Build

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH        # 12.9+
meson setup build/rel -Dcc_cuda=120 -Dnative_cuda=true -Dlc0ex-runtime=true \
      -Dplain_cuda=true -Dbuildtype=release
ninja -C build/rel lc0
```
Add `-Db_lto=true` for a shipping build. Set `-Dcc_cuda` to your own architecture (120 = sm_120).

⚠ There is no `-Dcutlass` or `-Dnvcc` option on this base. lc0ex's CUTLASS route is Python-side and
compiled into the artifact, so it needs nothing from lc0's build.

## Run — one rule, on one GPU and on many

**One slot per device, minibatch 128**, with an artifact built at the chunk size for your L2.
Measured on 4090s: **23,742 nps** on one GPU, **75,138 nps** on four — the latter **+5.5 %** over
the unchunked control at the same slot count.

```bash
--backend=lc0ex-cuda \
--backend-opts="gpu=0,lc0ex=bt4.lc0ex,concurrency=1,graph=dag" \
--threads=2 --minibatch-size=128
```

Multi-GPU. Note `concurrency=1`, and the quoting — the option lexer types a bare `0` as an integer
and refuses a second `=` in an unquoted value:

```bash
--backend=multi \
--backend-opts="backend=lc0ex-cuda,gpus='0:1',mode=roundrobin,opts='lc0ex=bt4.lc0ex;concurrency=1;graph=dag'" \
--threads=<GPUs+1> --minibatch-size=128
```

⛔ **`multi` defaults to `concurrency=4` per device — set it to 1 explicitly.** Leaving the default
is what made the multi-GPU case look like it wanted a different rule from the single-GPU case. It
does not: the two collapse into one rule once the slot count matches.

⚠ `concurrency` is a **cap, not an occupancy** — it bounds how many search threads may be inside
the backend at once, it does not create work to fill them.

## Gating

```bash
lc0 backendcompare --backend=lc0ex-cuda --backend-opts=... \
                   --ref-backend=lc0ex-cuda --ref-backend-opts=<unchunked artifact> \
                   --weights=<net> --ref-weights=<net> --fens=<file>
```
* Against the **unchunked** artifact, `c = 64` on a 4090 is **bit-identical**: policy KL 0.000000,
  top-1 100 %. Below that expect in-class, not exact.
* Self-determinism: the same artifact against itself, x3, must be 0.000000.
* ⚠ `--fens=<file>` is **required**.
* ⚠ **Do not gate by comparing artifact sha256s.** Two builds of an identical tree differ in ~12 of
  93 cubins while the emitted graph is byte-identical — the *compilation* is not reproducible, the
  *graph* is.
* ⚠ `cuda-fp16` is exact at 16/32/48/64/96 and first deviates at **112**, so it cannot arbitrate
  above 96; gate against lc0ex itself there.

⚠ `LC0EX_BATCH_HIST` prints from `std::atexit`, which `SIGTERM` skips — a run killed by `timeout`
reports nothing. Give the run a finite work limit instead.
