# TD_MuZero C++ correctness and performance repair plan

Date: 2026-07-10
Canonical implementation: C++ only
Target branch: `master`

The old Python engine and Python MuZero implementation are legacy references and are not parity constraints for this plan. Rule corrections are allowed to invalidate old Python behavior, old golden hashes, and old self-play data.

## Repair rules

1. Correctness before throughput.
2. One independently testable concern per commit group.
3. Keep deterministic ordering unless a rule correction explicitly requires a semantic change.
4. Update Golden Trace hashes only after reproducing the previous hashes with the validation model.
5. Do not mix environment-rule, network-architecture, replay-format, and trainer changes in one patch.
6. Any rule or observation change increments the data compatibility boundary; old replay data must not be silently mixed with new data.

## Phase A — environment time and combat semantics

### A1. Enemy movement and slow timing — completed

- Consume the full `speed * dt` distance across multiple waypoints.
- Remove repeated `erase(begin())` calls inside the waypoint loop; erase the consumed prefix once per step.
- Apply slow for its full time interval.
- Split a large `dt` at slow expiry so the remaining interval uses base speed.
- Prevent weaker overlapping slows from increasing the current slowed speed.
- Add direct regression tests for fast movement, full slow duration, and slow expiry inside a large timestep.
- Refresh deterministic Golden Trace hashes.

Compatibility impact: old trajectories generated with the waypoint-capped movement rule are obsolete.

### A2. Combat and removal cleanup — completed

- Exclude `hp <= 0` enemies from normal targeting, bucket targeting, off-board fallback, and AOE damage.
- Replace reached-base ID collection plus repeated `find_if`/`erase` with one stable compaction pass.
- Replace dead-enemy ID collection plus repeated `find_if`/`erase` with one stable compaction pass.
- Preserve survivor order so target tie-breaking remains deterministic.
- Add a regression proving that a later tower does not spend cooldown on an enemy killed earlier in the same tick.

Golden Trace remains unchanged after A2 because the locked scripts do not trigger the corrected overkill case.

### A3. General timestep contract — pending

Choose and enforce one contract:

- discrete engine: only integral one-second ticks are accepted; or
- continuous engine: spawn timers and tower cooldowns consume timer overshoot and may trigger multiple events in one call.

The current RL path uses one-second actions, but the public `step_time(float dt)` API must not claim continuous semantics unless all timers satisfy timestep equivalence.

## Phase B — MCTS correctness

### B0. Configuration and evaluator contracts — completed

- Implement root Dirichlet noise using an instance-local seeded RNG.
- Expose root priors for regression testing.
- Seed self-play MCTS from the game seed.
- Reject unsupported multiplayer backup mode instead of silently applying single-player signs.
- Validate legal root actions, evaluator batch dimensions, policy width, and finite value/reward/logits.
- Add deterministic root-noise, invalid-output, duplicate-action, and multiplayer-rejection tests.
- Reject unsupported LibTorch initial root batches greater than one until explicit root node IDs exist.
- Validate LibTorch recurrent vector lengths and output tensor shapes before indexing.

### B1. Latent legality and search semantics — pending

- Use legality prediction for latent-node expansion, or remove the unused head.
- Add explicit NaN/Inf evaluator tests for both initial and recurrent calls.
- Add node-pool boundary and value-sign tests.

### B2. MCTS layout and batching — pending

- Reuse per-search path and evaluator scratch buffers.
- Replace per-node child vectors with a contiguous edge pool after semantic tests pass.
- Batch leaf inference before further GPU micro-optimization.

## Phase C — replay and binary data path

### C0. Compatibility-preserving replay fixes — completed

- Replace the zero-valued tensor-size sentinel with an explicit initialized state.
- Detect an empty first sample followed by a different non-empty tensor shape.
- Resolve shard paths relative to the index file when the path is not valid from the process working directory.
- Close cached readers before Windows test cleanup removes temporary files.
- Sample all batch references first, group by global game, and deserialize each sampled game once per batch.
- Preserve the previous game/step sampling distribution and original batch output order.
- Write directly into pre-sized flat batch buffers instead of appending temporary sampled steps.
- Add a physical game-read counter and report it separately from packed payload throughput.

For a batch of 2048 drawn from 128 games, the compatibility path now performs at most 128 whole-game reads in that batch instead of 2048. A transition-level format is still required for large datasets where most sampled games are unique.

### C1. Writer and reader hardening — completed

- Add an explicit async writer `Open -> Closing -> Closed/Failed` state machine.
- Reject writes as soon as closing begins and serialize concurrent `close()` calls.
- Propagate worker failures consistently to waiting producers and closers.
- Validate file size, offset-table size, first and monotonic offsets, per-history boundaries, tensor sizes, stream-size limits, and multiplication overflow before allocation.
- Reject invalid flags, non-finite step scalars, malformed empty histories, truncated payloads, and oversized tensor headers.
- Write shards to unique temporary paths in the destination directory, flush and close them completely, then atomically publish them.
- Use POSIX same-directory rename and Windows `MoveFileExW` with replacement and write-through semantics.
- Remove incomplete temporary shards after failure.
- Add regressions for atomic visibility, concurrent close, write rejection after closing, incomplete close cleanup, invalid offsets, truncation, and oversized headers.

Compatibility impact: the version-1 on-disk layout is unchanged. Previously accepted malformed or trailing-byte files are now rejected.

### C2. Transition-level replay format — pending

- Define the sampling contract explicitly: uniform games versus uniform transitions.
- Add a transition-level offset index so sampling one step does not deserialize an entire game.
- Add `read_step_into_batch` with preallocated batch memory.
- Group sampled reads by shard/offset to reduce random seeks.
- Report physical bytes read separately from packed batch bytes.

## Phase D — C++ network architecture

- Replace scalar/global action encoding with structured action type plus spatial target planes.
- Produce spatial policy and legality logits for build/upgrade/sell actions, plus a separate wait logit.
- Keep the public 727-action flat representation only as an interface flattening step.
- Add semantic tests: coordinate sensitivity, translation behavior, local latent response, and legality-mask integration.

This phase changes checkpoint compatibility and must use a new architecture identifier.

## Phase E — persistence, reproducibility, and build hygiene

### E0. Core build reproducibility — completed

- Replace source `GLOB` usage with explicit core and neural-network source lists.
- Replace global include directories and global Torch CXX flags with target-scoped usage requirements.
- Add Linux and Windows Ninja-based C++17 core CI with full normal CTest.

### E1. Persistence and release validation — pending

- Save complete network, environment, observation, action-space, reward-transform, and rule-version configuration with checkpoints.
- Add replay format and environment rule version fields.
- Establish a current locked toolchain.
- Add a separate optional LibTorch CI job after a reproducible Torch package source is selected.
- Record benchmark deltas after the first CI-confirmed build.

## Validation gates

Every completed phase should provide:

- focused unit/regression tests;
- full normal CTest result;
- Golden Trace result when environment semantics change;
- before/after benchmark using the same workload;
- explicit replay/checkpoint compatibility statement;
- commit SHA recorded in the phase notes.

The C1 pull-request gate passed the complete Torch-disabled CTest suite on both `ubuntu-latest` and `windows-latest`. Push-triggered status visibility remains limited through the connected API, so CI claims in this document are restricted to runs whose jobs were explicitly inspected.

## Current completed commits

### Environment and combat

- `87b0651f6a49cd0921f6449d18453f7952896699` — continuous enemy movement and slow timing
- `5da6500acb23cd3fd46a5a93b477a4e0c51c3c45` — movement and slow regression tests
- `7e891597d2e9e8be7c6fd394f9650cf506fa1ba9` — refreshed Golden Trace constants
- `10843db915ddd6b2362a3f43d84d8cd07c6dad29` — dead-enemy filtering and stable removal compaction
- `74667a780c322ea51cd58d92f737066ff9368679` — dead-enemy targeting regression

### Observation

- `7ec0f64cdcbe474947defe01db07ad15361ee937` — content-based static observation cache identity and cooldown clamping
- `fe1267b8a382e0ba45b428c54d67a62aaff96f12` — observation cache and cooldown-bound regressions

### Replay

- `45ffb2a09d0449e5ce26532c6379b6e0bcd7208c` — explicit tensor-size initialization and relative shard paths
- `1392f32e31e8e9a4da43bee26cdf3aac3060008d` — replay tensor-shape and relative-path regressions
- `6132aee61795884a4c6c3b979d6400ce49dbaece` — close replay readers before Windows cleanup
- `94b47ea9542015949004483b895ccbd54360b862` — replay physical-read counter
- `ce05ac203f85cb0ecfde717c8e625d484c0913d0` — group batch samples by game and write directly into flat buffers
- `4dfd06ad7fd324eb99a0bdfe06f207cffe7978ec` — verify one whole-game read per sampled game
- `702be27a0100815148e86c64e49bc0cfd3bec770` — report physical game reads in the batch benchmark
- `83bc98069671b429ede79fcaf9d0ab07d2b986f3` — async writer lifecycle, atomic shard publication, and bounded reader validation

### MCTS and LibTorch contracts

- `4508cd741063f0ae6ebc20efac09071f3c97a9de` — seeded MCTS configuration and explicit single-player support
- `994018efa3ccf246aa6a9e55eda044739bcdc74f` — expose root priors
- `7babd71c8b0229f0f142604133f7c098286dbda6` — MCTS root-noise and validation interface
- `13e5d663a3aa3c71ca4229dbbea61da5b1d34c3f` — root noise, legal-action checks, and evaluator-output validation
- `6fc4ce7a232dc79d4d70a694ffbd2b9d6dbd68f4` — seed self-play MCTS from game seed
- `e24b3a8d524a8b5a99d88e1bd4e1392f26e355de` — MCTS regressions
- `e0eb4fb3fe0c603cfeb3cd2c7acb354382cce505` — LibTorch evaluator input/output validation
- `9e98c578a8a17bb074739127dc4bc4eed3fefc3d` — LibTorch evaluator contract tests

### Build and CI

- `0daa6a17cbb54fd92bd402b66bf8594744f6377f` — Linux and Windows C++ core CI
- `cbb0caad816367461ff159c4dcf90f5d7b64e5f9c` — explicit source lists and target-scoped build settings

New Golden Trace values after A1:

- combined: `0x7b079a8fe1032058`
- wait-only: `0xc8a91dc7092499da`
- mixed-build: `0xddc910cdb7fb9549`
- invalid-and-slow: `0x0b8a7c366424e6a0`
