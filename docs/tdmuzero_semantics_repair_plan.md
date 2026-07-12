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

Status: completed and validated.

Completed:

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
4. Truncated tail targets accumulate every available reward and bootstrap at the actual
   cutoff state, even when fewer than `n` transitions remain.
5. The cutoff root can appear as the final valid K+1 prediction state.
6. Terminal games are forbidden from carrying a cutoff bootstrap.
7. Shard and index/direct wave-mode provenance must agree.
8. Legacy transition shard v2 data is explicitly rejected for current training.
9. The dummy self-play shard generator emits v3 data and counts the extra cutoff search.
10. Focused, complete normal, complete LibTorch, and v3 generator smoke tests passed.

### D2.2 — direct sequence sampling

Status: implemented; local validation required.

Implemented:

1. `DirectSequenceReplayDataset` opens v3 transition shards independently of the generic
   whole-game reader.
2. The direct training path reads only the step records needed by the union of K-step and
   n-step targets. It never calls `ReplayDataset::read_game()`.
3. `DirectPositionIndex` includes every transition root and each persisted truncated cutoff
   root with equal sampling probability.
4. A direct window ending before the real episode end uses the next stored root as a local
   bootstrap record, preserving the same target mathematics as the whole-game oracle.
5. Overlapping and duplicate windows are merged per game; every unique step record and
   cutoff bootstrap is read at most once per batch.
6. `build_reference_k_step_batch` remains as the whole-game oracle for parity tests.
7. `test_direct_sequence_replay` compares all packed fields for:
   - middle-of-game synthetic windows;
   - real terminal tails;
   - real truncated tails;
   - the standalone persisted cutoff root;
   - exact duplicates and overlapping windows;
   - multiple shards.
8. Per-batch diagnostics report requested/unique records, physical reads/bytes, and unique
   games/shards.
9. `bench_sequence_replay` reports sequence throughput, deduplication, read amplification,
   checksums, and a first-batch reference parity gate.

D2.2 validation gate:

- direct/reference samples agree field by field;
- direct sampling leaves generic `game_read_count` at zero;
- cutoff roots are present in the uniform position index;
- overlapping windows reduce unique records read;
- full normal and LibTorch suites pass;
- `bench_sequence_replay` runs on v3 smoke data with equal reference/direct checksum.

### D2.3 — physical range coalescing

Status: next after D2.2 validation.

The current D2.2 path deduplicates logical step records but the low-level v3 reader still
issues one physical read per unique step payload. D2.3 will:

1. add one-seek contiguous step-range reads in `TransitionShardReader`;
2. decode a merged range from one in-memory byte buffer;
3. preserve exact I/O accounting;
4. compare per-step and coalesced readers byte-for-byte;
5. require a measurable reduction in physical read operations on long trajectories.

Phase D completion gate:

- hand-computed target tests pass;
- direct and reference sequence samplers are identical for the same sample refs;
- trainer consumes contiguous K-step batches without whole-game deserialization;
- no truncated state is silently treated as terminal;
- coalesced direct sequence I/O is measurably below whole-game I/O on long trajectories.

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

1. D2.2 requires local validation.
2. D2.3 must coalesce unique adjacent step payloads into fewer physical reads.
3. C2 legality loss is not connected to a trainer.
4. No complete LibTorch optimizer/checkpoint training loop exists yet.

## Validation discipline

Every phase requires:

1. focused unit tests;
2. full normal and LibTorch `ctest`;
3. Golden Trace when engine rules are unchanged;
4. replay/self-play benchmarks where applicable;
5. no large data generation before semantic gates pass.
