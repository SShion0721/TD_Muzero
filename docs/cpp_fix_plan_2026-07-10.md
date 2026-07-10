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

- Implement or remove silent no-op options: root noise and player-perspective mode.
- Use legality prediction for latent-node expansion, or remove the unused head.
- Validate evaluator output sizes and finite logits before expansion.
- Add tests for root noise, NaN/Inf handling, node-pool exhaustion, and value-sign behavior.
- Reuse per-search scratch buffers and replace per-node child vectors with a contiguous edge pool only after semantic tests pass.

## Phase C — replay and binary data path

### C0. Compatibility-preserving replay fixes — completed

- Replace the zero-valued tensor-size sentinel with an explicit initialized state.
- Detect an empty first sample followed by a different non-empty tensor shape.
- Resolve shard paths relative to the index file when the path is not valid from the process working directory.
- Close cached readers before Windows test cleanup removes temporary files.

### C1. Writer and reader hardening — pending

- Give the async writer an explicit `Open -> Closing -> Closed/Failed` state machine.
- Reject writes as soon as closing begins and serialize concurrent `close()` calls.
- Add file-size, offset, step-count, tensor-size, and multiplication-overflow validation.
- Write shards to temporary paths and atomically publish completed files.

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

### Environment and combat

- `87b0651f6a49cd0921f6449d18453f7952896699` — continuous enemy movement and slow timing
- `5da6500acb23cd3fd46a5a93b477a4e0c51c3c45` — movement and slow regression tests
- `7e891597d2e9e8be7c6fd394f9650cf506fa1ba9` — refreshed Golden Trace constants
- `10843db915ddd6b2362a3f43d84d8cd07c6dad29` — dead-enemy filtering and stable removal compaction
- `74667a780c322ea51cd58d92f737066ff9368679` — dead-enemy targeting regression

### Observation

- `7ec0f64cdcbe474947defe01db07ad15361ee937` — content-based static observation cache identity and cooldown clamping
- `fe1267b8a382e0ba45b428c54d67a62aaff96f12` — observation cache and cooldown-bound regressions

### Replay compatibility fixes

- `45ffb2a09d0449e5ce26532c6379b6e0bcd7208c` — explicit tensor-size initialization and relative shard paths
- `1392f32e31e8e9a4da43bee26cdf3aac3060008d` — replay tensor-shape and relative-path regressions
- `6132aee61795884a4c6c3b979d6400ce49dbaece` — close replay readers before Windows cleanup

New Golden Trace values after A1:

- combined: `0x7b079a8fe1032058`
- wait-only: `0xc8a91dc7092499da`
- mixed-build: `0xddc910cdb7fb9549`
- invalid-and-slow: `0x0b8a7c366424e6a0`
