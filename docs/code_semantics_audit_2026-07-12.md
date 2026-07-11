# TD_MuZero C++ semantics audit — 2026-07-12

## Scope and method

This was a static, cross-file audit of every production translation unit registered by
`cpp/CMakeLists.txt`, their public headers, the test registration file, and the app/test
call sites found through repository-wide code search. The baseline before this audit had
already passed the complete normal and LibTorch CTest suites.

The audit separates three categories:

1. **confirmed defects** — behavior contradicts the current public semantics;
2. **planned semantic debt** — known missing MuZero functionality already assigned to a
   later phase;
3. **heuristic limitations** — intentional approximations that must not be mistaken for
   exact game or training semantics.

Passing tests are evidence for covered behavior, not proof that uncovered behavior is
correct.

## Executive result

The engine/action/observation core is substantially consistent with the accepted 11x11,
727-action, one-second-step rules. The largest remaining correctness risks are no longer
basic pathfinding or observation construction; they are **episode provenance, replay
semantics, terminal-state absorption, and latent legality**.

The audit also confirmed that the old neural architecture destroyed action locality:
Dynamics broadcast four scalars over the whole board and Prediction globally pooled the
latent before producing all 727 logits. Phase B therefore proceeded immediately after
this audit and replaced both paths with spatial semantics.

## Confirmed defects fixed in this pass

### 1. Unsupported wait durations silently aliased to Wait1

`encode_action()` previously mapped every `ActionType::Wait1` object to action 726 even
when `wait_steps != 1`. This could silently collapse an unsupported action into the
public one-second action.

**Fix:** `Action::type` now has a safe default, `wait_steps` must equal one, and action ABI
static assertions lock 726/727.

### 2. Invalid tower enums produced zero-cost towers

`tower_stats()` previously returned a zero-initialized stat block after an unmatched
`switch`. A corrupted or unchecked enum cast could therefore construct an invalid,
zero-cost tower.

**Fix:** unmatched tower types now throw `std::invalid_argument`.

### 3. Enemy movement accepted non-finite time

`Enemy::step()` treated NaN as ordinary input because comparisons against NaN are false.
That could contaminate coordinates and all downstream observations.

**Fix:** negative, NaN, and infinite `dt` are rejected; zero remains a no-op.

### 4. DefenseCapacity disagreed with the engine cooldown interval

The engine uses a half-open tick interval `[0, dt)`, but the estimator used floor-based
formulas that both missed interior shots and counted shots exactly on the window end.
Examples included an AOE tower with cooldown 1.5 over two seconds and a ready Basic tower
over an exact integer boundary.

**Fix:** both new-tower and existing-tower capacity use the same half-open schedule:
shot times `delay + n * cooldown` are counted only when strictly less than the window.

### 5. External self-play environments recorded the config seed

`SelfPlayRunner::run(TDEngine&, ...)` stored `cfg.seed` even when the supplied engine had
been constructed/reset with a different seed.

**Fix:** `TDEngine::seed()` is now authoritative for trajectory provenance and for the
MCTS seed in that run.

### 6. Neural action locality was absent

The old action encoder produced a small global vector and expanded it uniformly over the
board. The old policy/legality heads pooled the latent to 1x1 before generating 727
logits. Two actions differing only by target cell therefore entered Dynamics through a
weak coordinate MLP rather than an exact local intervention, while Prediction had to
reconstruct every spatial decision from one global vector.

**Fix (Phase B):** Dynamics now receives seven action planes; Prediction emits six
11x11 policy/legality planes plus one wait scalar. The flat 727-action ABI is unchanged.

## Open correctness blockers

### P0 — terminal states are not fully absorbing through every public mutation API

`step_wait()`/`step_one_tick()` stop after game over, but `step_action()` performs
build/upgrade/sell before entering the tick path. Direct calls to `place_tower()`,
`upgrade_tower()`, and `sell_tower()` can also mutate a terminal engine.

Required repair:

- make terminal `step_action()` return `{0, true}` before decoding/mutating;
- make direct economy mutations fail after terminal;
- define terminal legal actions explicitly, preferably Wait-only for the existing
  “Wait is always legal” contract;
- add state-equality regression tests after terminal actions.

This changes public engine behavior and therefore should increment the environment rule
version when implemented.

### P0 — wave-generation mode is hidden transition state

`use_budgeted_waves` changes the next-wave transition but is not encoded in Observation
V2 or current replay compatibility metadata. Fixed and budgeted environments can expose
the same visible state and then generate different futures.

Until an explicit mode field is added, fixed and budgeted data must use separate output
directories/indexes and must never be mixed in one training dataset. Runtime mode changes
inside a training episode should be forbidden.

### P0 — direct transition-shard loading can bypass the compatibility fence

`ReplayDataset::from_index_json()` validates the current observation/action metadata, but
`ReplayDataset(vector<path>)` only checks that v2 shards agree with each other. A set of
old 20-channel transition shards can therefore be opened directly without an index.

Required repair:

- keep `TransitionShardReader` generic for diagnostics;
- make training-facing `ReplayDataset` require current `kObservationSize` and 727-sized
  policy/legal tensors even without an index;
- add an explicit rejection test for old dimensions.

### P0 — replay values are not MuZero targets

`ReplayBatch.values` currently receives `TrajectoryStep.root_value`. That is a search
statistic at the same state, not the required n-step return target. The batch is also
single-step rather than K-step.

This is assigned to Phase D and blocks meaningful training.

### P0 — terminal and truncated episodes are conflated

`GameHistory` has `terminal`, and each step has `done`, but a max-step cutoff has no
separate `truncated` marker. A trainer cannot decide whether bootstrapping is valid.

This is assigned to Phase D.

### P1 — legality logits stop at `NetworkOutput`

The network computes legality logits, but `LibTorchEvaluator` drops them and
`INetworkEvaluator::EvalOutput` has no legality field. Root actions remain exact because
the engine mask is used, but latent expansion selects policy top-k across all 727
outputs.

This is assigned to Phase C. Untrained legality must never hard-prune latent actions.

## Subsystem review

### Core action and board tables

- Flat mapping is internally consistent: four build planes, upgrade, sell, wait.
- Exhaustive encode/decode round-trip remains the primary ABI gate.
- Board cell indexing is consistently row-major (`y * 11 + x`).
- Precomputed action tables preserve ascending deterministic root action order.
- Bitboard bounds and the 121-cell split across two 64-bit words are consistent.

### Pathfinding and placement

- Fixed-array BFS is correct for the fixed 11x11 grid and does not allocate per node.
- Cached base distance/next-step fields are invalidated by grid version.
- Tarjan-style spawn-to-base cut detection correctly separates structural placement
  from temporary enemy occupancy.
- Placement legality rechecks enemy occupancy, money, spawn/base exclusion, and path
  preservation.
- Header dependency on integer `std::abs` is now explicit.

### Enemy and tower timing

- Enemy movement consumes full continuous travel distance across multiple waypoints.
- Slow expiry correctly splits a large `dt` into slowed and base-speed intervals.
- Tower cooldown scheduling follows a half-open interval and does not create a boundary
  shot.
- Dead enemies are filtered during retargeting.
- The per-tick shot cap is a safety assertion; current cooldown floor keeps it far above
  valid behavior.

### Engine step order and rewards

The implemented order matches the locked contract:

1. spawn;
2. enemy movement;
3. base leak removal/damage;
4. terminal check;
5. tower attacks;
6. kill removal/reward;
7. wave transition;
8. time increment.

The fixed `-50` leak reward, integer `max_hp / 10` base damage, kill reward as both money
and reward, `+100` wave reward, and `-1000` terminal reward are coherent but are policy
choices, not derived invariants. They must remain versioned with the environment/reward
schema if changed.

### Observation V2

- Shape is 40x11x11 and all exposed values are bounded/finite.
- Continuous enemy quantities use bilinear splatting; exact temporary occupancy uses the
  engine's rounded-cell rule.
- Absolute HP/max HP, reward, leak damage, speed, slow time, route distance/direction,
  pending totals, and next-enemy fields remove the major V1 aliases.
- Structural placeability is not confused with temporary occupancy.
- Static-cache identity is based on exact blocked/tower state, not only an engine pointer.
- V1-named wrappers now produce V2 and are source-compatibility aliases only; serialized
  metadata is authoritative.

Observation normalization constants are engineering bounds. Values above those bounds
saturate and can alias. Current wave/enemy generators stay within practical ranges, but
training telemetry should record saturation rates.

### Defense and attack budget

- DefenseCapacity is a heuristic upper/operational estimate, not an exact combat solver.
- Path coverage uses geometric cells, not enemy arrival-time distributions or target
  contention.
- AOE/slow multipliers are hand-set approximations.
- Spendable-money search is greedy, selects at most one upgrade per existing tower, and
  does not recompute path/candidate interactions after each hypothetical build.
- Budget caps and unlock rules are deterministic and finite-clamped.
- Elite classification currently recognizes archetypes by hard-coded speed values; this
  is safe for current archetypes but brittle if their speeds become configurable.

These limitations are acceptable for wave shaping only and must not be described as an
exact defense capacity.

### MCTS

- Root expansion uses the exact engine mask; illegal root actions get no edge or policy
  mass.
- Duplicate/out-of-range root actions and malformed evaluator outputs are rejected.
- Node/edge pool capacity is checked before atomic expansion.
- Single-player backup signs are internally consistent; multiplayer mode is explicitly
  rejected.
- Batched recurrent inference reserves unique leaves and removes temporary virtual
  visits before evaluation.
- Root policy targets are normalized visit counts, not network priors.
- Root noise implementation is present but remains disabled by default.
- Latent top-k legality remains the Phase C blocker described above.

### Neural network

Architecture v3 now has:

- Representation: 40-channel observation to spatial latent;
- Dynamics: latent plus seven spatial action planes;
- Prediction trunk: spatial latent processing;
- Value: global pooling;
- Policy: six 11x11 planes plus wait scalar;
- Legality: six 11x11 planes plus wait scalar.

The flatten order is channel-major and matches the existing action ABI exactly. Wait uses
a full constant input plane in Dynamics and one scalar output in Prediction.

The current action encoder builds a CPU tensor and copies it to the latent device. This
is semantically correct but may become a GPU inference bottleneck. Optimize only after
profiling, for example with cached device tensors or a batched native scatter.

### Self-play and replay

- Root observations and exact legal masks are stored at the pre-action state.
- Reward/done are stored from the resulting transition.
- Policy target is the searched root visit distribution.
- DummyNetwork generators remain smoke/performance infrastructure, not training data
  quality evidence.
- Sampling is currently uniform-game then uniform-step, which overweights short games
  relative to uniform-position sampling.
- Binary readers perform strong size/overflow/truncation checks.
- Compatibility is primarily carried by replay indexes/checkpoint manifests rather than
  embedded in every legacy history payload.

### Persistence and build

- Checkpoint manifest v2 records spatial `action_planes` and `policy_planes` and rejects
  architecture v2 checkpoints.
- Observation/action/reward/environment/network versions are compared independently.
- The normal build remains LibTorch-free.
- The Torch build requires one coherent MSVC/CUDA/LibTorch toolchain; the reproducible
  Windows procedure and observed failure modes are documented in `README.md`.

## Next mandatory order

1. Validate the current Phase B commits with focused and full normal/Torch tests.
2. Repair terminal absorption and increment the environment rule version.
3. Add wave-mode provenance and direct-shard compatibility enforcement.
4. Implement Phase C legality transport/loss/metrics without hard pruning.
5. Implement Phase D K-step batches, n-step targets, and terminal/truncated semantics.
6. Only then implement the trainer, real-checkpoint self-play, and exploration.
