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

- The engine uses discrete one-second ticks.
- `step_time(dt)` accepts only finite, non-negative integer seconds.
- Multi-second calls execute repeated one-second ticks and stop on terminal state.
- `step_time(0)` is a no-op that preserves the terminal flag.
- Fractional, negative, NaN, infinite, and out-of-range durations are rejected.
- `step_wait()` uses the same tick implementation and rejects negative waits.
- Full public state is compared between batched and repeated one-tick execution.

Compatibility: standard RL actions remain one second. Fractional-`dt` callers must migrate to explicit ticks.

## Phase B — MCTS correctness and performance

### B0. Configuration and evaluator contracts — completed

- Seeded instance-local root Dirichlet noise.
- Deterministic self-play MCTS seeding from the game seed.
- Explicit single-player backup support; unsupported multiplayer backup is rejected.
- Validation of legal roots, evaluator dimensions, policy width, latent IDs, and finite outputs.
- Root-prior and invalid-output regressions.

### B1. Latent legality and search semantics — partially completed

Completed:

- Reject non-positive node-pool capacities.
- Precompute exact search node demand before initial inference.
- Fail undersized searches before mutating the tree or evaluator latent store.
- Prevent partial child allocation.
- Test NaN and infinity in initial and recurrent inference.
- Lock exact-capacity, one-slot-short, and `reward + discount * value` behavior.

Pending architecture decision:

- A legality head exists, but evaluator and MCTS do not expose or consume it.
- No canonical C++ trainer currently supervises or calibrates latent legality.
- Integration or removal is deferred to Phase D because it changes checkpoints.

### B2.1. MCTS node and scratch reuse — completed

- Retain one `NodePool` per `MCTS` instance instead of allocating its backing store for every search.
- Reuse previously created node objects while resetting scalar search state.
- Preserve each node's `actions` and `children` vector capacity between searches.
- Reuse legal-action validation, initial-observation, recurrent-input, search-path, top-k, softmax, and root-noise buffers.
- Keep inactive node IDs invalid after pool clear while retaining their storage for the next search.
- Expose node creation/reuse, node-buffer growth, scratch-capacity growth, and maximum-depth diagnostics.
- Warm the benchmark once and report allocation-growth events only for the timed repeated-search interval.

Validation:

- Repeated deterministic searches compare action, root value, root actions, priors, visits, full policy, node count, branching, and depth field-for-field.
- After warm-up, identical searches create zero node objects and trigger zero node-buffer or scratch-capacity growth events.
- Full Torch-disabled CTest passes on GitHub-hosted Ubuntu and Windows runners.

### B2.2. Contiguous edge storage — pending

- Replace per-node `actions` and `children` vectors with a contiguous edge pool.
- Preserve action order, child lookup, tie-breaking, visit counts, Q values, and root outputs exactly.
- Compare the old and new layouts under deterministic evaluator traces before removing the vector layout.

### B2.3. Batched leaf inference — pending

- Select multiple leaves without corrupting virtual tree state.
- Submit one recurrent inference batch for the selected leaves.
- Expand and back up each result deterministically.
- Measure evaluator calls, batch occupancy, simulations per second, and search-strength parity.

## Phase C — replay and binary data path

### C0. Compatibility-preserving replay fixes — completed

- Explicit tensor-shape initialization instead of a zero-size sentinel.
- Relative shard paths resolved from the index location.
- Reader lifetime cleanup for Windows test deletion.
- Batch references grouped by game so each sampled game is deserialized once.
- Direct writes into pre-sized flat batch buffers.
- Separate physical game-read and packed-payload metrics.

Validated local result: a 2048-sample batch over a four-game test dataset performed four whole-game reads, or 512 samples per read.

### C1. Writer and reader hardening — completed

- Explicit async writer lifecycle: `Open -> Closing -> Closed/Failed`.
- Serialized concurrent close and immediate rejection after closing starts.
- Atomic same-directory shard publication and incomplete-temp cleanup.
- Bounds, offset, tensor-size, overflow, finite-value, flag, truncation, and trailing-byte validation.
- Corruption, concurrency, and failure-cleanup regressions.

Compatibility: replay binary v1 remains readable, while malformed files previously accepted may now be rejected.

### C2. Transition-level replay format — completed

- Advance the binary replay format to version 2 while retaining v1 read support.
- Store payloads sequentially, followed by compact game and transition index tables.
- Persist one physical offset and size for every transition.
- Preserve the exact sampling distribution: uniform game, then uniform step within that game.
- Sort batch requests by shard and physical offset.
- Read observations, policy targets, legal masks, values, rewards, actions, and done flags directly into preallocated batch rows.
- Deduplicate repeated requests for the same transition inside a batch.
- Reject mixed v1/v2 shard sets and index/shard version mismatches.
- Report exact contiguous-range reads, physical bytes, packed bytes, unique games/shards, and read amplification for v2.
- Keep whole-game v1 loading as a compatibility fallback with explicitly non-exact byte metrics.

Validation:

- v2 round-trip, atomic publication, async concurrent close, future-version rejection, bad-offset rejection, truncation rejection, and shape mismatch tests.
- Fixed-seed v1/v2 batch outputs are compared field-for-field to lock sampling semantics.
- Full Torch-disabled CTest passes on GitHub-hosted Ubuntu and Windows runners.

Compatibility:

- Replay index format remains version 2.
- New self-play generation writes replay binary v2 and documents `uniform_game_then_uniform_step`.
- Existing index-v2 datasets declaring replay binary v1 remain readable.
- Legacy v1 or unversioned indexes still require explicit reviewed opt-in.

## Phase D — C++ network architecture — pending

- Replace scalar/global action encoding with structured action type and spatial target planes.
- Produce spatial policy and legality logits for build, upgrade, and sell, plus an independent wait logit.
- Keep flat 727-action representation only as an interface mapping.
- Add coordinate sensitivity, translation, local latent response, action mapping, and legality integration tests.
- Introduce a new network architecture identifier and checkpoint boundary.

## Phase E — persistence, reproducibility, and observability

### E0. Core build reproducibility — completed

- Explicit core and neural-network source lists.
- Target-scoped includes, compile options, and Torch flags.
- Ninja C++17 core CI on Ubuntu and Windows with full Torch-disabled CTest.

### E0.5. Interactive engine observability — completed

- Optional zero-dependency Win32/GDI GUI driving the canonical `TDEngine`.
- Board, placeability, towers, enemies, HP, slow state, economy, waves, timers, legality, rewards, and recent actions.
- Inspect, build, upgrade, sell, one/multi-tick stepping, auto-run, reset, seeds, and fixed/budgeted waves.
- Linux builds remain unchanged; Windows CI compiles and links the GUI.

### E1a. Compatibility metadata and manifests — completed

- Canonical replay, environment-rule, observation, action-space, reward-transform, and network-architecture versions.
- Replay index version 2 with complete compatibility metadata.
- Legacy or incompatible indexes rejected by default.
- Field-specific actual/expected compatibility errors.
- Checkpoint manifest contract for dimensions, versions, training step, seed, and optimizer-state presence.

The canonical C++ path still has no complete model/optimizer serialization implementation; the manifest defines the contract without claiming persistence exists.

### E1b. Trainer persistence and release toolchain — pending

- Integrate manifests with canonical model and optimizer save/load.
- Save and restore RNG and training state for reproducible continuation.
- Lock supported compiler, CMake, LibTorch, CUDA, and cuDNN versions.
- Add optional reproducible LibTorch CI.
- Record benchmark deltas with fixed workloads.

## Validation gates

Each completed phase must provide:

- focused unit and regression tests;
- full Torch-disabled CTest on Ubuntu and Windows;
- Golden Trace verification when environment semantics can affect one-second actions;
- before/after benchmark for performance work;
- explicit replay and checkpoint compatibility notes.

## Compact history policy

- `dd9cecced9bd0b6a5acfa46c06b9b2eef9b21a32` is the compact baseline containing all work completed before A3.
- Each subsequent independently validated phase adds one squash-equivalent commit to `master`.
- Temporary feature-branch commits are not retained in `master` history.
- The planned final history remains well below 30 commits.

## Current execution order

1. Implement B2.2 contiguous edge storage with deterministic layout parity.
2. Implement B2.3 batched leaf recurrent inference.
3. Implement D spatial action, policy, and legality architecture.
4. Complete E1b trainer persistence, toolchain locking, and optional LibTorch CI.
5. Run final locked release validation and benchmark reporting.
