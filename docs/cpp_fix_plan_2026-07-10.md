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

### A2. Combat and removal cleanup — next

- Exclude `hp <= 0` enemies from target selection and AOE damage.
- Replace ID collection plus repeated `find_if`/`erase` with one stable compaction pass.
- Preserve survivor order so target tie-breaking remains deterministic.
- Add regression tests proving that later towers do not spend cooldown on already-dead enemies.
- Benchmark high-enemy-count removal separately from tower targeting.

### A3. General timestep contract

Choose and enforce one contract:

- discrete engine: only integral one-second ticks are accepted; or
- continuous engine: spawn timers and tower cooldowns consume timer overshoot and may trigger multiple events in one call.

The current RL path uses one-second actions, but the public `step_time(float dt)` API must not claim continuous semantics unless all timers satisfy timestep equivalence.

## Phase B — MCTS correctness

- Implement or remove silent no-op options: root noise and player-perspective mode.
- Use legality prediction for latent-node expansion, or remove the unused head.
- Validate evaluator output sizes and finite logits before expansion.
- Add tests for root noise, NaN/Inf handling, node-pool exhaustion, and value-sign behavior.
- Reuse per-search scratch buffers and replace per-node child vectors with a contiguous edge pool only after semantic tests pass.

## Phase C — replay and binary data path

- Add file-size, offset, step-count, and tensor-size validation.
- Write shards to temporary paths and atomically publish completed files.
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
- Batch leaf inference before attempting further GPU micro-optimizations.

This phase changes checkpoint compatibility and must use a new architecture identifier.

## Phase E — persistence, reproducibility, and build hygiene

- Save complete network, environment, observation, action-space, reward-transform, and rule-version configuration with checkpoints.
- Add replay format and environment rule version fields.
- Replace global CMake flags/includes with target-scoped settings.
- Replace source `GLOB` usage or add explicit configure dependencies.
- Establish a current locked toolchain instead of relying on the legacy Python 3.7/Torch 1.10 lock file.
- Add a normal C++ CI build with CTest and a separate optional LibTorch job.

## Validation gates

Every completed phase should provide:

- focused unit/regression tests;
- full normal CTest result;
- Golden Trace result when environment semantics change;
- before/after benchmark using the same workload;
- explicit replay/checkpoint compatibility statement;
- commit SHA recorded in the phase notes.

## Current completed commits

- `87b0651f6a49cd0921f6449d18453f7952896699` — continuous enemy movement and slow timing
- `5da6500acb23cd3fd46a5a93b477a4e0c51c3c45` — movement and slow regression tests
- `7e891597d2e9e8be7c6fd394f9650cf506fa1ba9` — refreshed Golden Trace constants

New Golden Trace values after A1:

- combined: `0x7b079a8fe1032058`
- wait-only: `0xc8a91dc7092499da`
- mixed-build: `0xddc910cdb7fb9549`
- invalid-and-slow: `0x0b8a7c366424e6a0`
