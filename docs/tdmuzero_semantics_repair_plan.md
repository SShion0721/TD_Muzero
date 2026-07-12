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

### C1 — evaluator transport and validation

Status: completed and validated.

Completed behavior:

1. `EvalOutput` carries `[B][727]` legality logits.
2. `INetworkEvaluator` validates value, reward, policy, and legality at the common public
   interface boundary for both initial and recurrent inference.
3. Invalid batch counts, invalid policy/legality widths, NaN, and infinity are rejected
   before MCTS consumes the output.
4. `DummyNetwork` returns deterministic finite legality logits.
5. `LibTorchEvaluator` copies the network legality tensor to CPU output and independently
   validates tensor definition, shape, batch agreement, and finite values.
6. Batched custom evaluators and MCTS contract tests provide legality logits.
7. `test_legality_transport` verifies that opposite legality predictions produce exactly
   the same search output while gating remains disabled.
8. `test_mcts_libtorch_smoke` verifies the real LibTorch evaluator exposes 727 finite
   legality logits.
9. Focused, complete normal, and complete LibTorch suites passed.

### C2 — exact targets, BCE loss, and metrics

Status: implemented; local validation required.

Implemented behavior:

1. `make_legality_targets()` converts exact engine masks to floating `[B,727]` BCE
   targets without changing action order.
2. Target construction rejects incorrect dimensions, non-binary masks, and masks that
   incorrectly mark Wait1 illegal.
3. `LegalityMetricsAccumulator` records:
   - true/predicted legal and illegal counts;
   - true positive, false positive, true negative, and false negative counts;
   - legal recall and legal precision;
   - illegal precision;
   - false-negative and false-positive rates;
   - Wait recall.
4. The default decision boundary is logit 0, equivalent to sigmoid probability 0.5.
5. `hard_pruning_ready()` encodes the minimum reporting gate of legal recall >= 99.5%
   and Wait recall == 100%, but it does not enable or modify MCTS pruning.
6. `legality_bce_with_logits()` implements the numerically stable formula
   `max(x,0) - x*y + log1p(exp(-abs(x)))` over `[B,727]` tensors.
7. Integer exact masks are accepted by the loss and converted to the logits' device and
   floating dtype.
8. `test_legality_training` covers exact conversion, corruption rejection, confusion
   counts, metric rates, batch accumulation, and the reporting gate.
9. `test_legality_loss` covers log(2) at zero logits, confident predictions, finite
   backward gradients, integer targets, and shape rejection.

Deliberately unchanged through C2:

- root expansion uses only the engine's exact legal mask;
- latent expansion uses policy top-k only;
- legality logits do not modify priors, candidates, visits, or action selection;
- no soft gating or hard pruning is enabled;
- no compatibility version changes because the environment and network architecture are
  unchanged.

C2 validation gate:

- `test_legality_training` passes in normal and LibTorch builds;
- `test_legality_loss` passes in the LibTorch build;
- C1 transport/MCTS tests continue to pass;
- complete normal and LibTorch suites pass;
- Golden Trace and deterministic search results remain unchanged.

### C3 — trainer integration and measured legality quality

Status: blocked by Phase D replay targets and the Phase E trainer.

Required later:

1. Feed K+1 exact legal masks from sequence replay to the legality head.
2. Add legality BCE to the joint training objective with an explicit configurable weight.
3. Report the C2 metrics separately on train and held-out validation states.
4. Examine class imbalance before introducing positive/negative weighting.
5. Permit soft latent gating only behind an explicit default-off feature flag after
   measured validation.
6. Permit hard pruning only after legal recall is at least 99.5% and Wait recall is 100%,
   with a regression test proving root exact masks remain authoritative.

Phase C invariant:

- root legality always comes from the real engine;
- no illegal root action receives prior, visit count, or policy mass;
- untrained or insufficiently validated legality cannot prune latent actions.

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
3. C2 legality targets/loss/metrics are not yet connected to a sequence trainer;
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
- legality logits traverse the complete evaluator boundary without affecting search;
- exact legality targets, metrics, and stable BCE loss utilities are available.

## Validation discipline

Every phase requires:

1. focused unit tests;
2. full `ctest`;
3. Golden Trace check when engine rules are unchanged;
4. observation, engine, MCTS, replay, and self-play benchmarks as applicable;
5. no large data generation before compatibility and semantic gates pass.
