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

Status: implemented; local validation required.

Completed behavior:

- terminal `step_action()` returns `{0, true}` before decoding or mutation;
- terminal `step_wait()` and `step_time()` do not advance time or repeat rewards;
- build, upgrade, sell, and wave-mode mutation are disabled after terminal;
- terminal legal actions are exactly Wait1, preserving the non-empty mask contract;
- environment rule version incremented to 4;
- regression test snapshots all public engine state and verifies repeated terminal calls
  are exact no-ops;
- the former monolithic engine implementation was split into state, board, action, and
  tick translation units without changing the public API.

Validation gate:

- focused timing/checkpoint tests pass;
- complete normal and LibTorch CTest suites pass;
- Golden Trace remains unchanged for traces that do not act after terminal.

### Remaining provenance fence

Status: next.

1. Add explicit fixed/budgeted wave-mode provenance to generated replay indexes.
2. Reject mixed wave modes in one training dataset.
3. Prevent runtime wave-mode changes once an episode has begun.
4. Make direct transition-shard dataset construction enforce current observation,
   policy, and legal-mask dimensions even without an index manifest.
5. Add rejection tests for old 20-channel direct shards and mixed wave modes.

## Phase C — legality-aware evaluator path

Status: not started.

1. Extend EvalOutput with legality logits.
2. Copy legality logits through LibTorchEvaluator.
3. Train legality with the engine's exact root masks.
4. Record legal recall, illegal precision, false-negative rate, and Wait recall.
5. Use soft legality gating in latent nodes only after the auxiliary head is trained.
6. Permit hard pruning only after legal recall is at least 99.5% and Wait recall is 100%.

Gate:

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

1. direct transition-shard construction must enforce current tensor dimensions even
   without an index manifest;
2. fixed-wave and budgeted-wave datasets need explicit provenance and must not be mixed;
3. replay targets remain single-step and `root_value(t)` is not a valid final value label;
4. terminal and truncated episodes are still conflated;
5. latent-node legality is unavailable until Phase C.

Already fixed:

- invalid tower types throw instead of creating zero-cost towers;
- unsupported multi-second wait actions are rejected by the 727-action encoder;
- enemy movement rejects negative and non-finite time inputs;
- DefenseCapacity shot counting follows the engine's half-open cooldown interval;
- externally supplied self-play environments record their actual seed;
- action and pathfinding headers declare their dependencies explicitly;
- terminal states are fully absorbing across public mutation and stepping APIs.

## Validation discipline

Every phase requires:

1. focused unit tests;
2. full `ctest`;
3. Golden Trace check when engine rules are unchanged;
4. observation, engine, MCTS, replay, and self-play benchmarks as applicable;
5. no large data generation before compatibility and semantic gates pass.
