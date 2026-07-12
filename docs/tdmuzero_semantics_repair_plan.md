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

Status: completed and validated.

Completed:

1. `GameHistory` distinguishes `terminal` and `truncated` endings.
2. `SelfPlayRunner` marks max-step cutoffs as truncated instead of terminal.
3. `SequenceTargetConfig` defaults to:
   - unroll K = 5;
   - TD horizon n = 10;
   - discount = 0.997.
4. `KStepSample` contains:
   - one initial observation;
   - K actions and reward targets;
   - K+1 policy, value, and exact legality targets;
   - transition/state/value valid masks;
   - terminal/truncated provenance.
5. Value targets use real discounted rewards plus a future root value only as bootstrap.
   `root_value(t)` is never copied as its own target.
6. Terminal horizons stop without bootstrap.
7. Position-uniform sampling removes the old short-game over-weighting.
8. `KStepReplaySampler` packs contiguous buffers and deduplicates whole-game reads within
   a sampled batch.
9. Validation rejects incomplete episodes, terminal/truncated conflicts, malformed policy,
   invalid action/mask pairs, and non-finite state data.
10. Focused, complete normal, and complete LibTorch suites passed after D1.

### D2.1 — transition shard v3 and explicit cutoff bootstrap

Status: implemented; local validation required.

Implemented:

1. Replay binary format and replay index version are now 3.
2. Every v3 game entry persists:
   - terminal;
   - truncated;
   - fixed/budgeted wave mode;
   - step count and max steps;
   - optional cutoff-bootstrap presence and payload location.
3. A truncated self-play game can run one additional root search after its last transition
   and persist the real cutoff root:
   - observation;
   - root value;
   - policy target;
   - exact legal mask.
4. Truncated tail targets now accumulate every available reward and bootstrap at the actual
   cutoff state, even when fewer than `n` transitions remain.
5. The cutoff root can appear as the final valid K+1 prediction state.
6. Terminal games are forbidden from carrying a cutoff bootstrap.
7. Shard and index/direct wave-mode provenance must agree.
8. Legacy transition shard v2 data is explicitly rejected for current training rather than
   having terminal/truncated/bootstrap semantics inferred.
9. The current dummy self-play shard generator emits v3 data and counts the extra cutoff
   search in its simulation totals.

D2.1 validation gate:

- v3 terminal/truncated/wave-mode metadata round-trips;
- cutoff bootstrap tensors and value round-trip;
- direct step and direct bootstrap reads match source histories;
- hand-computed near-cutoff n-step targets pass;
- current training replay rejects mode disagreement and missing cutoff bootstrap;
- full normal and LibTorch suites pass;
- tiny v3 generator smoke produces a readable index and shards.

### D2.2 — direct sequence I/O

Status: blocked on D2.1 validation.

Next implementation:

1. Expose game metadata and bootstrap records through the generic replay layer.
2. Add direct range reads for the maximum of `K+1` prediction states and `n` reward steps.
3. Avoid whole-game deserialization in the training sampler.
4. Include persisted cutoff roots in the global position-uniform index.
5. Sort and deduplicate overlapping physical ranges inside a batch.
6. Compare direct sequence samples against the D1 whole-game reference path field by field.
7. Add sequence I/O counters and throughput benchmarks.

Phase D completion gate:

- hand-computed target tests pass;
- direct and reference sequence samplers are identical for the same sample refs;
- trainer consumes contiguous K-step batches without whole-game deserialization;
- no truncated state is silently treated as terminal;
- direct sequence I/O is measurably below whole-game I/O on long trajectories.

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

1. D2.1 requires local validation.
2. D2.2 must replace whole-game reads with direct sequence I/O.
3. C2 legality loss is not connected to a trainer.
4. No complete LibTorch optimizer/checkpoint training loop exists yet.

## Validation discipline

Every phase requires:

1. focused unit tests;
2. full normal and LibTorch `ctest`;
3. Golden Trace when engine rules are unchanged;
4. replay/self-play benchmarks where applicable;
5. no large data generation before semantic gates pass.
