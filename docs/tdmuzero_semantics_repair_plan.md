# TD MuZero semantics repair plan

Correct state semantics and training targets are mandatory gates before exploration or
large-scale self-play.

## Phase A — compatibility fence and Observation V2

Status: completed and validated.

Completed:

- 40x11x11 Observation V2;
- pending-wave and next-enemy state;
- absolute HP, reward, leak, speed, route distance/direction;
- bilinear continuous-position splatting;
- structural placeability separated from temporary occupancy;
- bounded finite normalization;
- old 20-channel data rejected by the training compatibility fence.

## Phase B — spatial action and prediction semantics

Status: completed and validated.

Completed:

- seven spatial action planes in Dynamics;
- six 11x11 Policy planes plus one Wait scalar;
- six 11x11 Legality planes plus one Wait scalar;
- global pooling retained only for Value/Reward;
- exact unchanged 727-action ABI;
- architecture version 3 and checkpoint manifest version 2.

## Pre-Phase C semantic hardening

Status: completed and validated.

Completed:

- terminal states are fully absorbing;
- terminal legal mask is Wait-only;
- fixed/budgeted wave mode is immutable after episode start;
- self-play records actual seed and wave mode;
- training replay requires explicit wave-mode provenance and exact 40/727/727 sizes;
- environment rule version is 5.

## Phase C — legality-aware evaluator path

### C1 — evaluator transport

Status: completed and validated.

- `EvalOutput` carries `[B,727]` legality logits;
- common evaluator boundaries reject bad shape, batch, NaN, and infinity;
- LibTorch and Dummy evaluators expose legality;
- opposite legality predictions produce identical MCTS output while gating is disabled.

### C2 — targets, loss, and metrics

Status: completed and validated.

- exact masks convert to floating BCE targets;
- stable BCE-with-logits is available;
- legal recall/precision, illegal precision, FNR/FPR, Wait recall, and counts are recorded;
- the reporting gate is legal recall >= 99.5% and Wait recall == 100%;
- this gate does not enable pruning.

### C3 — trainer integration

Status: blocked by Phase D and Phase E.

Later requirements:

- feed K+1 exact legal masks to the head;
- add configurable legality loss weight;
- report train and held-out validation metrics;
- keep soft gating default-off;
- prohibit hard pruning until the measured gate passes.

Invariant throughout Phase C:

- root legality always comes from the real engine;
- illegal root actions receive no prior, visit, or policy mass;
- untrained legality never affects latent expansion.

## Phase D — K-step replay and correct targets

### D1 — reference sequence semantics

Status: implemented; local validation required.

Implemented:

1. `GameHistory` distinguishes `terminal` and `truncated` endings.
2. `SelfPlayRunner` marks max-step cutoffs as truncated instead of terminal.
3. Transition shard v2 truncation is losslessly derived at the training boundary from
   `!terminal && step_count == max_steps`; no old diagnostic reader is reinterpreted.
4. `SequenceTargetConfig` defaults to:
   - unroll K = 5;
   - TD horizon n = 10;
   - discount = 0.997.
5. `KStepSample` contains:
   - one initial observation;
   - K actions and reward targets;
   - K+1 policy, value, and exact legality targets;
   - transition/state/value valid masks;
   - terminal/truncated provenance.
6. Value targets use real discounted rewards plus `root_value(t+n)` only as a future
   bootstrap. `root_value(t)` is never copied as its own target.
7. Terminal horizons stop without bootstrap.
8. Truncated tails without a persisted post-cutoff value are marked value-invalid rather
   than treated as terminal.
9. `UniformPositionIndex` gives every stored root state equal probability, independent of
   game length.
10. `KStepReplaySampler` packs contiguous training buffers and deduplicates whole-game
    reads within a sampled batch.
11. Validation rejects incomplete episodes, terminal/truncated conflicts, non-contiguous
    steps, bad actions, malformed policies, and corrupt exact masks.

D1 validation gate:

- hand-computed terminal n-step targets pass;
- hand-computed truncated targets pass;
- terminal and truncated tail padding masks pass;
- short and long games map to one position-uniform index;
- K-step packed dimensions and checksums pass;
- `test_selfplay`, `test_training_replay`, and all prior suites remain green.

### D2 — direct sequence I/O and persisted cutoff bootstrap

Status: next after D1 validation.

Required:

1. Add direct contiguous range reads for the maximum of `K+1` and `n+1` states.
2. Avoid whole-game deserialization in the training sampler.
3. Persist explicit terminal/truncated metadata in the next transition-shard format.
4. Persist a post-cutoff observation/value for truncated games, or define an equivalent
   bootstrap record, so truncated tail value targets need not be discarded.
5. Add I/O counters and compare direct sequence reads against the D1 reference sampler.
6. Prove one-step and direct K-step readers agree on all shared fields.

Phase D completion gate:

- hand-computed target tests pass;
- direct and reference sequence samplers are bit-identical for the same sample refs;
- trainer consumes contiguous K-step batches without whole-game deserialization;
- no truncated state is silently treated as terminal.

## Phase E — trainer and real-network self-play

Status: not started.

Required:

- C++/LibTorch joint training over K-step batches;
- Policy, Value, Reward, and Legality losses with valid masks;
- model, optimizer, RNG, step, config, and compatibility checkpoint state;
- tiny deterministic overfit;
- exact resume equivalence;
- real-checkpoint self-play;
- DummyNetwork retained only for smoke/performance tests.

## Phase F — exploration

Status: blocked by Phases D and E.

Only after the complete training loop is correct:

- root Dirichlet noise in self-play only;
- visit-count temperature action sampling;
- deterministic evaluation mode;
- root/action entropy and diversity logging.

## Remaining blockers before meaningful training

1. D1 still reads whole games; D2 must add direct sequence I/O.
2. Transition shard v2 does not explicitly store truncated or post-cutoff bootstrap state.
3. C2 legality loss is not connected to a trainer.
4. No complete LibTorch optimizer/checkpoint training loop exists yet.

## Validation discipline

Every phase requires:

1. focused unit tests;
2. full normal and LibTorch `ctest`;
3. Golden Trace when engine rules are unchanged;
4. replay/self-play benchmarks where applicable;
5. no large data generation before semantic gates pass.
