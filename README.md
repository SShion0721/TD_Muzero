# TD_MuZero

C++17 / LibTorch MuZero research project for an 11x11 tower-defense environment.

The C++ engine is the canonical environment. The repository contains deterministic game rules, observation/action contracts, MCTS, self-play, replay storage, target construction, LibTorch network code, compatibility manifests, tests, benchmarks and a native Windows debug GUI.

This README describes the current `master` state and separates completed infrastructure from unfinished training work.

## Current status

Current development line:

```text
Observation V2
Network architecture v3
Replay/index format v3
Direct K-step sequence sampling under validation
```

### Completed and validated

- deterministic 11x11 tower-defense engine and public 727-action ABI;
- Observation V2 with 40 spatial planes;
- spatial action encoding for MuZero dynamics;
- spatial policy and legality heads;
- MCTS capacity checks, finite-output contracts and single-player backup semantics;
- reusable node pool, contiguous edge pool and reusable search scratch buffers;
- optional batched recurrent leaf inference;
- atomic replay writer lifecycle and corruption detection;
- versioned compatibility metadata for replay, environment, observation, action and network contracts;
- transition shard v3 with explicit truncated-cutoff bootstrap state;
- hand-checked near-cutoff n-step target tests;
- normal and LibTorch validation for replay v3 and generator smoke;
- zero-dependency Win32/GDI environment debug GUI.

### Implemented but still requiring the current validation gate

`DirectSequenceReplayDataset` reads only the records needed for K-step and n-step targets instead of deserializing complete games. It supports:

- a uniform index over transition roots and persisted truncated cutoff roots;
- local bootstrap records for windows ending before the real episode end;
- deduplication of overlapping and duplicate windows;
- field-by-field comparison with the whole-game reference path;
- diagnostics for requested/unique records, physical reads, bytes, games and shards;
- `bench_sequence_replay` checksum and throughput reporting.

The remaining D2.2 gate requires full normal and LibTorch suites plus direct/reference parity on generated v3 data.

### Not complete

The repository is not yet a finished end-to-end MuZero trainer. Remaining work includes:

1. coalescing adjacent v3 step payloads into fewer physical reads;
2. connecting the legality loss to the trainer;
3. implementing a complete LibTorch optimizer/checkpoint training loop;
4. validating real-network self-play and trained-versus-baseline strength;
5. only then performing longer training and performance optimization.

Do not describe the project as a trained production agent until these gates are complete.

## Environment semantics

### Board and initial state

- board: `11x11`;
- spawn: `(0, 5)`;
- base: `(10, 5)`;
- initial money: `200`;
- initial base HP: `100`;
- initial wave: `1`;
- each public action advances one second;
- invalid build/upgrade/sell receives a penalty and still advances the second;
- wait is always legal.

### Action space

The public action space remains 727 IDs:

| IDs | Meaning |
|---:|---|
| `0..120` | build Basic |
| `121..241` | build Sniper |
| `242..362` | build AOE |
| `363..483` | build Slow |
| `484..604` | upgrade |
| `605..725` | sell |
| `726` | wait one second |

Cell order is row-major:

```text
cell = y * 11 + x
```

### One-second update order

1. update spawn timer and spawn a queued enemy when ready;
2. move enemies;
3. process leaks and base damage;
4. perform terminal checks;
5. advance tower cooldowns and fire;
6. remove killed enemies and add money/reward;
7. generate the next wave when the current wave is exhausted;
8. advance environment time.

The engine semantics are locked by deterministic tests and golden traces. A neural-network or replay refactor must not silently change them.

## Observation V2

Observation V2 has 40 planes and includes:

- blocked cells, spawn and base;
- tower types, levels and cooldowns;
- placeability and distance-to-base information;
- enemy density, HP, speed, slow state and route direction;
- exact occupied cells;
- money, base HP, wave and spawn timer;
- pending-enemy counts and aggregate statistics;
- next pending enemy attributes.

Old 20-channel data is incompatible and must not be mixed with V2 replay or checkpoints.

## Network architecture v3

```text
observation [B, 40, 11, 11]
  -> representation network
  -> latent [B, 32, 11, 11]
```

Dynamics receives seven spatial action planes:

```text
Basic / Sniper / AOE / Slow / Upgrade / Sell / Wait
```

Prediction produces:

```text
Value
6 x 11 x 11 spatial policy logits + 1 wait logit
6 x 11 x 11 spatial legality logits + 1 wait logit
```

The external 727-action mapping is unchanged.

## MCTS

Current search infrastructure includes:

- exact preflight node-capacity calculation;
- non-finite network-output rejection;
- single-player reward/value backup without opponent sign inversion;
- reusable `NodePool`;
- contiguous reusable edge storage;
- scratch-buffer reuse after warm-up;
- deterministic root/action ordering tests;
- optional `recurrent_batch_size` for unique-leaf recurrent inference;
- evaluator-call, sample, occupancy and collision-stop diagnostics.

`recurrent_batch_size=1` preserves the sequential scheduling path. Larger batches intentionally change within-batch selection order and require separate strength validation.

## Replay format v3

Replay v3 stores transition-indexed records and explicit provenance. Each game records enough information to distinguish:

- terminal completion;
- truncation at a training or self-play cutoff;
- the persisted bootstrap state after truncation;
- wave mode and semantic versions;
- observation, action, reward, value, policy and legal mask contracts.

Important rules:

- terminal games cannot carry a cutoff bootstrap;
- truncated games must carry the required cutoff state;
- truncated targets accumulate all available rewards and bootstrap at the actual cutoff;
- legacy v2 replay is rejected by current training rather than guessed into v3 semantics;
- index and shard metadata must agree.

## Build

Source root for the C++ project:

```text
cpp/
```

### Core-only build

This mode does not require LibTorch and uses test/dummy evaluators where appropriate.

```bash
cmake -S cpp -B cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTDMZ_ENABLE_TORCH=OFF
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

### LibTorch build

Use one coherent Visual Studio, CUDA and LibTorch toolchain. Pass the PyTorch CMake prefix explicitly.

```bash
cmake -S cpp -B cpp/build_torch -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTDMZ_ENABLE_TORCH=ON \
  -DCMAKE_PREFIX_PATH=<torch-cmake-prefix>
cmake --build cpp/build_torch --parallel
ctest --test-dir cpp/build_torch --output-on-failure
```

Do not run the full suite before all targets are built. With Ninja, test executables are under `build_torch/tests/`, not a `tests/Release/` subdirectory.

## Useful tests and benchmarks

Depending on the selected build mode, relevant targets include:

```text
test_golden_trace
test_network_shape
test_mcts_libtorch_smoke
test_selfplay_torch
test_training_replay
test_sequence_replay
test_direct_sequence_replay
bench_engine
bench_mcts
bench_observation
bench_selfplay
bench_sequence_replay
```

Benchmark output is useful only when the commit, build type, toolchain, model/evaluator and parameters are recorded. Historical throughput numbers should not be copied into the README as current guarantees after replay or search architecture changes.

## Native debug GUI

On Windows, the optional Win32/GDI target visualizes the canonical C++ environment and supports build, upgrade, sell, wait, auto-run, reset and seed changes. It is a debugging surface, not a second environment implementation.

## Project structure

```text
cpp/src/engine/              canonical environment
cpp/src/mcts/                MuZero search
cpp/src/replay/              replay writer, reader and sequence sampling
cpp/src/training/            targets and training-side utilities
cpp/src/network/             LibTorch representation/dynamics/prediction
cpp/tests/                   semantic and regression tests
cpp/benchmarks/              performance and I/O benchmarks
cpp/tools/                   self-play and debugging tools
docs/tdmuzero_semantics_repair_plan.md
```

## Development rules

- environment rules, observation planes, action ABI, replay format and network architecture are versioned contracts;
- old replay/checkpoints require explicit compatibility handling and must not be silently accepted;
- whole-game and direct sequence samplers must remain field-for-field equivalent for the same references;
- terminal and truncated trajectories must remain distinct;
- performance work is accepted only after deterministic semantics and tests pass;
- README status follows validated commits and the repair plan, not generated agent notes or local command fragments.
