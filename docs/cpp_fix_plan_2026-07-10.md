# TD_MuZero C++ correctness and performance repair plan

Date: 2026-07-10  
Canonical implementation: C++ only  
Target branch: `master`

The old Python engine and Python MuZero implementation are legacy references, not parity constraints. Rule corrections may invalidate old Python behavior, Golden Trace hashes, replay data, and checkpoints, but compatibility boundaries must be explicit.

## Repair rules

1. Correctness before throughput.
2. One independently testable phase per `master` commit.
3. Implementation, tests, benchmark changes, and plan updates stay in the same squash commit.
4. Preserve deterministic ordering unless a deliberate rule correction changes semantics.
5. Do not mix environment rules, replay format, network architecture, and trainer changes in one phase.
6. Any rule or observation change establishes a new data compatibility boundary.
7. Every completed phase must pass focused regressions and full Torch-disabled CTest on Ubuntu and Windows.

## Phase A — environment time and combat semantics

### A1. Enemy movement and slow timing — completed

- Consume the full `speed * dt` distance across multiple waypoints.
- Remove repeated front erases while traversing paths.
- Apply slow for its complete duration and split movement at slow expiry.
- Prevent weaker overlapping slows from increasing movement speed.
- Lock behavior with movement and slow-duration regressions.

Compatibility: trajectories generated under the old waypoint-capped movement rule are obsolete.

### A2. Combat and removal cleanup — completed

- Exclude dead enemies from normal, bucket, fallback, and AOE targeting.
- Replace repeated enemy lookup/erase operations with stable compaction passes.
- Preserve survivor order and deterministic target tie-breaking.
- Prove later towers do not consume cooldown on enemies killed earlier in the same tick.

### A3. General timestep contract — completed

- The engine now uses discrete one-second ticks.
- `step_time(dt)` accepts only finite, non-negative integer seconds.
- A multi-second call executes repeated one-second ticks and stops early on terminal state.
- `step_time(0)` is a no-op that preserves the current terminal flag.
- Fractional, negative, NaN, infinite, and out-of-range durations are rejected.
- `step_wait()` uses the same tick implementation and rejects negative waits.
- Full public engine state is compared between batched multi-tick execution and repeated one-tick execution, including follow-up evolution.

Compatibility: standard RL actions remain one second and retain their previous behavior. Callers using fractional `dt` must migrate to explicit discrete ticks.

## Phase B — MCTS correctness and performance

### B0. Configuration and evaluator contracts — completed

- Seeded instance-local root Dirichlet noise.
- Deterministic self-play MCTS seeding from the game seed.
- Explicit single-player backup support; unsupported multiplayer backup is rejected.
- Validation of legal roots, evaluator dimensions, policy width, latent IDs, and finite values/rewards/logits.
- Root-prior and invalid-output regressions.

### B1. Latent legality and search semantics — partially completed

Completed:

- Reject non-positive node-pool capacities.
- Precompute exact search node demand before initial inference.
- Fail undersized searches before mutating the tree or evaluator latent store.
- Prevent partial child allocation.
- Test NaN and infinity in initial and recurrent inference.
- Lock exact-capacity, one-slot-short, and single-player `reward + discount * value` behavior.

Pending architecture decision:

- A legality head exists in the prediction network, but the evaluator and MCTS do not expose or consume it.
- No canonical C++ trainer currently supervises or calibrates latent legality.
- Legality integration or head removal is deferred to Phase D because either choice changes checkpoint compatibility.

### B2. MCTS layout and batching — pending

- Reuse search path, action, and evaluator scratch buffers.
- Measure and remove per-simulation dynamic allocation.
- Replace per-node child vectors with a contiguous edge pool after semantic parity tests.
- Batch leaf recurrent inference before lower-level GPU micro-optimization.

## Phase C — replay and binary data path

### C0. Compatibility-preserving replay fixes — completed

- Explicit tensor-shape initialization instead of a zero-size sentinel.
- Relative shard path resolution from the index location.
- Reader lifetime cleanup for Windows test deletion.
- Batch references grouped by game so each sampled game is deserialized once per batch.
- Direct writes into pre-sized flat batch buffers.
- Separate physical game-read and packed-payload metrics.

Validated local result: a 2048-sample batch over a four-game test dataset performed four physical game reads, or 512 samples per read.

### C1. Writer and reader hardening — completed

- Explicit async writer lifecycle: `Open -> Closing -> Closed/Failed`.
- Serialized concurrent close and immediate rejection after closing starts.
- Atomic same-directory shard publication.
- Cleanup of incomplete temporary shards.
- Bounds, offset, tensor-size, overflow, finite-value, flag, truncation, and trailing-byte validation.
- Corruption, concurrency, and failure-cleanup regressions.

Compatibility: version-1 layout is unchanged, but malformed files previously accepted may now be rejected.

### C2. Transition-level replay format — pending

- Preserve and document the sampling distribution explicitly.
- Add transition offsets so one sampled step does not deserialize an entire game.
- Implement `read_step_into_batch()` with preallocated memory.
- Group reads by shard and physical offset.
- Report physical bytes read, packed bytes, and read amplification separately.
- Keep replay v1 readable while writing a versioned replay v2 format.

## Phase D — C++ network architecture — pending

- Replace scalar/global action encoding with structured action type and spatial target planes.
- Produce spatial policy and legality logits for build, upgrade, and sell actions, plus an independent wait logit.
- Keep the public flat 727-action representation only as an interface mapping.
- Add coordinate sensitivity, translation, local latent response, action mapping, and legality integration tests.
- Introduce a new network architecture identifier and checkpoint boundary.

## Phase E — persistence, reproducibility, and build hygiene

### E0. Core build reproducibility — completed

- Explicit core and neural-network source lists.
- Target-scoped includes, compile options, and Torch flags.
- Ninja C++17 core CI on Ubuntu and Windows with full Torch-disabled CTest.

### E1. Persistence and release validation — pending

- Store network, environment, observation, action-space, reward-transform, and rule versions with checkpoints.
- Version replay format and environment rules.
- Reject incompatible replay/checkpoint/runtime combinations instead of silently mixing them.
- Lock a current toolchain.
- Add optional reproducible LibTorch CI.
- Record benchmark deltas with fixed workloads.

## Validation gates

Each completed phase must provide:

- focused unit and regression tests;
- full Torch-disabled CTest on Ubuntu and Windows;
- Golden Trace verification when environment semantics can affect normal one-second actions;
- before/after benchmark for performance work;
- explicit replay and checkpoint compatibility notes.

## Compact history policy

- `dd9cecced9bd0b6a5acfa46c06b9b2eef9b21a32` is the compact baseline containing all work completed before A3.
- Each subsequent independently validated phase adds one squash-equivalent commit to `master`.
- Temporary feature-branch commits and maintenance workflows are not retained in `master` history.
- The planned final history remains well below 30 commits.

## Current execution order

1. E1 compatibility and version metadata.
2. C2 transition-indexed replay v2.
3. B2 MCTS scratch reuse, contiguous edges, and batched leaf inference.
4. D spatial action, policy, and legality architecture.
5. Final locked release validation and benchmark report.
