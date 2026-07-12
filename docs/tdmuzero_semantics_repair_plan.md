# TD MuZero semantics repair plan

This plan replaces the earlier performance-first sequence. Correct state semantics and
training targets are mandatory gates before exploration tuning or large-scale self-play.

## Phase A — compatibility fence and Observation V2

Status: completed and validated.

1. Increment environment, observation, and network compatibility versions.
2. Reject old 20-channel replay/checkpoints from the corrected 40-channel pipeline.
3. Encode pending-wave state: next enemy, pending totals, count, and speed summary.
4. Encode enemy absolute HP, max HP, reward, leak damage, route distance, and direction.
5. Bilinearly splat continuous enemy positions instead of rounding all dynamic features.
6. Separate structural placeability from temporary enemy occupancy.
7. Replace negative/unbounded linear scalars with bounded or logarithmic transforms.
8. Preserve an exact legacy Python observation only as a migration oracle.

Validated gates:

- all observation values are finite and in [0, 1];
- reusable and allocating outputs are bit-identical;
- different absolute-HP states no longer alias;
- continuous positions affect adjacent cells;
- pending queue order affects the next-enemy features;
- old schema/checkpoint metadata is rejected;
- normal and LibTorch CTest suites pass;
- Golden Trace remains unchanged because observations do not change engine rules.

## Phase B — spatial action and prediction semantics

Status: completed and validated.

1. Replace the four broadcast action scalars with seven spatial action planes:
   four build types, upgrade, sell, and wait.
2. Inject target-cell information locally into Dynamics.
3. Replace pooled 727-way policy and legality heads with six 11x11 action planes plus
   one wait scalar.
4. Keep Value globally pooled.
5. Increment network architecture version to 3 and checkpoint manifest version to 2.

Validated gates:

- exhaustive flat-action/spatial-plane checks for all 727 actions;
- local target-cell one-hot encoding, with a dedicated full wait plane;
- exact channel-major policy/legality flattening into the existing 727 IDs;
- unchanged public action-space ABI;
- updated checkpoint metadata for `action_planes` and `policy_planes`;
- normal suite passed 21/21;
- LibTorch suite passed 24/24;
- `test_network_shape`, `test_mcts_libtorch_smoke`, and `test_selfplay_torch` passed;
- Observation V2 allocating/reused checksums remained identical.

## Pre-Phase C semantic hardening

### Terminal absorption

Status: completed and validated.

Completed behavior:

- terminal `step_action()` returns `{0, true}` before decoding or mutation;
- terminal `step_wait()` and `step_time()` do not advance time or repeat rewards;
- build, upgrade, sell, and wave-mode mutation are disabled after terminal;
- terminal legal actions are exactly Wait1, preserving the non-empty mask contract;
- environment rule version incremented to 4 for this transition change;
- regression tests snapshot all public engine state and verify repeated terminal calls
  are exact no-ops;
- the former monolithic engine implementation was split into state, board, action, and
  tick translation units without changing the public API;
- complete normal and LibTorch suites passed after the change.

### Wave-mode and training replay provenance fence

Status: completed and validated.

Completed behavior:

1. `WaveMode` is an explicit `Unknown/Fixed/Budgeted` type.
2. A self-play `GameHistory` records the actual environment mode, like its actual seed.
3. Wave mode can be changed only before the first mutation/time step and only while
   regenerating the pending wave; it is immutable once the episode begins.
4. Reset starts a new episode and reopens pre-start mode selection.
5. Environment rule version is 5.
6. `TrainingReplayDataset` is the training-facing fence:
   - requires explicit fixed/budgeted provenance for direct shard sets;
   - reads the existing `budgeted` index field or a future `wave_mode` string;
   - rejects missing provenance;
   - requires transition-indexed v2 replay;
   - requires exact current dimensions: 40x11x11 observation and 727 policy/legal masks;
   - rejects old 20-channel direct transition shards.
7. Generic `ReplayDataset` and low-level shard readers remain available for diagnostics
   and migration, but future trainers must use `TrainingReplayDataset`.
8. Focused, complete normal, and complete LibTorch suites passed after the change.

## Phase C — legality-aware evaluator path

Status: C1 transport implemented; local validation required.

### C1 — evaluator transport and validation

Implemented:

1. `EvalOutput` now carries `[B][727]` legality logits.
2. `INetworkEvaluator` validates value, reward, policy, and legality at the common public
   interface boundary for both initial and recurrent inference.
3. Invalid batch counts, invalid policy/legality widths, NaN, and infinity are rejected
   before MCTS consumes the output.
4. `DummyNetwork` returns deterministic finite legality logits.
5. `LibTorchEvaluator` copies the network legality tensor to CPU output and independently
   validates tensor definition, shape, batch agreement, and finite values.
6. Batched custom evaluators and MCTS contract tests now provide legality logits.
7. `test_legality_transport` verifies that opposite legality predictions produce exactly
   the same search output while gating remains disabled.
8. `test_mcts_libtorch_smoke` verifies the real LibTorch evaluator exposes 727 finite
   legality logits.

Deliberately unchanged:

- root expansion still uses only the engine's exact legal mask;
- latent expansion still uses policy top-k only;
- legality logits do not modify priors, candidates, visits, or action selection;
- no soft gating or hard pruning is enabled;
- network/checkpoint architecture versions do not change because architecture v3 already
  contained the legality head.

C1 validation gate:

- `test_mcts`, `test_legality_transport`, and `test_mcts_batched_inference` pass;
- `test_mcts_libtorch_smoke` and `test_selfplay_torch` pass;
- complete normal and LibTorch suites pass;
- Golden Trace and deterministic DummyNetwork search outputs remain unchanged.

### C2 — legality targets, loss utilities, and metrics

Status: blocked on the C1 validation gate.

Next implementation:

1. Build legality targets from the engine's exact root masks.
2. Add numerically stable binary-cross-entropy-with-logits utilities.
3. Record legal recall, illegal precision, false-negative rate, false-positive rate,
   Wait recall, predicted legal count, and true legal count.
4. Keep all search gating disabled while the head is untrained.
5. Permit soft latent gating only behind an explicit default-off feature flag after
   trainer integration and metric validation.
6. Permit hard pruning only after legal recall is at least 99.5% and Wait recall is 100%.

Phase C gate:

- root legality always comes from the real engine;
- no illegal root action receives prior, visit count, or policy mass;
- untrained legality cannot hard-prune latent actions.

## Phase D — K-step replay and correct targets

Status: not started.

1. Sample positions uniformly rather than games uniformly.
2. Produce K-step action sequences and K+1 prediction targets.
3. Build n-step value targets from real rewards plus a future bootstrap value.
4. Never use `root_value(t)` directly as its own value target.
5. Distinguish terminal from truncated episodes.
6. Add valid masks for sequence tails.
7. Store K+1 legal masks for the legality loss.

Initial configuration:

- unroll K = 5;
- n-step return = 10;
- discount = 0.997.

Gate:

- hand-computed target tests pass for terminal, truncated, and tail-padded samples;
- one-step and K-step readers agree on shared fields;
- trainer consumes contiguous batches without whole-game deserialization.

## Phase E — trainer and real-network self-play

Status: not started.

1. Implement the C++/LibTorch training loop over K-step batches.
2. Save model, optimizer, RNG, training step, config, and compatibility metadata.
3. Load real checkpoints in self-play workers.
4. Keep DummyNetwork only for deterministic smoke/performance tests.
5. Separate train, evaluation, and benchmark commands.

Gate:

- overfit a tiny deterministic dataset;
- resume produces the same next update as uninterrupted training;
- real-network self-play produces valid shards;
- checkpoint/schema mismatch fails before inference or training.

## Phase F — exploration

Status: blocked by Phases C–E.

Only after Phases A–E:

1. enable root Dirichlet noise in self-play only;
2. sample executed actions from visit counts with a wave-based temperature schedule;
3. retain the complete visit distribution as the policy target;
4. keep evaluation deterministic;
5. log root entropy, action entropy, legal count, and unique actions.

Gate:

- same seed produces the same trajectory;
- different seeds produce meaningful diversity;
- invalid action count remains zero;
- evaluation mode remains deterministic.

## Remaining audit blockers before training

1. replay targets remain single-step and `root_value(t)` is not a valid final value label;
2. terminal and truncated episodes are still conflated;
3. legality targets/loss/metrics are not yet connected to a trainer;
4. the low-level transition shard binary itself does not yet embed wave mode; the
   training fence therefore treats the index/direct constructor as authoritative.

Already fixed:

- invalid tower types throw instead of creating zero-cost towers;
- unsupported multi-second wait actions are rejected by the 727-action encoder;
- enemy movement rejects negative and non-finite time inputs;
- DefenseCapacity shot counting follows the engine's half-open cooldown interval;
- externally supplied self-play environments record their actual seed and wave mode;
- action and pathfinding headers declare their dependencies explicitly;
- terminal states are fully absorbing across public mutation and stepping APIs;
- training-facing replay requires explicit wave mode and exact current tensor sizes;
- legality logits now traverse the complete evaluator boundary without affecting search.

## Validation discipline

Every phase requires:

1. focused unit tests;
2. full `ctest`;
3. Golden Trace check when engine rules are unchanged;
4. observation, engine, MCTS, replay, and self-play benchmarks as applicable;
5. no large data generation before compatibility and semantic gates pass.
