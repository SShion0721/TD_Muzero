# TD_MuZero

C++ / LibTorch MuZero tower-defense research project.

The current direction is to move the old Python prototype toward a fast C++ engine, C++ MCTS, LibTorch network inference, C++ self-play trajectory generation, and later replay / trainer support. The C++ engine is now treated as the canonical engine once its rules are accepted.

## Current status

### Completed phases

- Phase 1 / 1.5: C++ tower-defense core cleanup
  - `tdmz_core` is pure C++17 and does not depend on LibTorch.
  - Board/action constants are unified around an 11x11 board and a 727-action space.
  - Only `Wait1` is currently used.
  - Non-11x11 boards are rejected explicitly.

- Phase 2 / 2.1: C++ MCTS
  - `MCTS` works through an `INetworkEvaluator` abstraction.
  - Root expansion uses real legal actions.
  - Latent expansion uses policy top-k.
  - NodePool max-node limit is enforced.
  - Root value comes from the searched root node.

- Phase 3: LibTorch forward network
  - Added representation, dynamics, prediction, action encoder, and full MuZero network modules.
  - Added `LibTorchEvaluator` with latent state storage keyed by MCTS node ids.
  - Added shape tests and MCTS + LibTorch smoke tests.

- Phase 4: C++ self-play trajectory writer
  - Added `SelfPlayRunner`, `GameHistory`, `TrajectoryStep`, and JSONL writer.
  - A self-play record stores observation, action, reward, root value, policy target, legal mask, done flag, and state metadata.
  - Added dummy self-play generator app.

- Phase 4.1: Game logic regression tests
  - Added tests for path length, tower economy, invalid action penalty, spawning, movement, base damage, tower cooldown, slow effect, build legality, and non-11x11 rejection.

- Phase 4.1C / 4.2A-D: first engine rule and cache optimization pass
  - Tower max level is now 5.
  - `Tower::can_upgrade()` prevents upgrades beyond max level.
  - `legal_actions()` no longer emits upgrade actions for max-level towers.
  - Observation tower-level channel is normalized by `kTowerMaxLevel`.
  - Added engine version counters and perf counters.
  - Added cached `compute_placeable_mask()` keyed by grid version.
  - Added cached `legal_actions()` keyed by grid / tower / money versions.

- Phase 4.2E-H: Stockfish/Amazons-style table and path optimization pass
  - Added `Bitboard128` for the 121-cell board.
  - Added precomputed board geometry tables: cell coordinates, 4-neighbor lists, bitboards, Manhattan distances, squared distances, tower range masks, and action tables.
  - Replaced generic `priority_queue + map` A* pathfinding with fixed-array BFS.
  - Added cached base-distance and next-step-to-base fields keyed by `grid_version`.
  - Enemy pathing to the base now uses the cached base-distance field when possible.
  - `observation_v1` now reuses cached distance-to-base values.
  - `compute_placeable_mask()` now uses one DFS/Tarjan-style spawn-base cut-cell pass instead of testing every empty cell with a separate pathfind.
  - `legal_actions()` now uses precomputed action ids instead of recomputing flat ids.
  - Added `test_board_tables` for bitboard, geometry, action table, range mask, and BFS regression checks.
  - Validation PASS: normal CTest `5/5` passed, Torch CTest `8/8` passed.
  - Latest benchmark: normal `bench_engine` about 354k steps/s, Torch `bench_engine` about 359k steps/s; `bench_mcts` remains about 61k-62k simulations/s.

- Phase 4.2K: golden trace semantic lock
  - Added deterministic golden trace infrastructure with fixed seeds and fixed action scripts.
  - Added `test_golden_trace` as a semantic lock test with fixed expected hashes.
  - Added `export_golden_traces` app to export JSONL traces and hash summaries.
  - Locked cases:
    - `wait_only_seed0`: `0x385b69812eea12a1`, 16 steps.
    - `mixed_build_seed1`: `0xf16bcd753375ebc1`, 10 steps.
    - `invalid_and_slow_seed42`: `0xd142454b6573e9d4`, 10 steps.
    - combined hash: `0x595648689a6a7435`.
  - Validation PASS: normal CTest `6/6` passed, Torch CTest `9/9` passed.

- Phase 4.3A: static DefenseCapacity estimator
  - Added `DefenseCapacityConfig`, `DefenseCapacityResult`, and `estimate_defense_capacity()`.
  - Estimates current tower theoretical damage capacity over a finite wave window.
  - Estimates spendable-money capacity by greedily selecting the best level-1 tower damage/cost option.
  - Uses `virtual_base_hp`, not real `base_hp`, for leak capacity to avoid a defender lowering future budgets by intentionally leaking.
  - Added `test_defense_capacity` and `export_defense_capacity`.
  - Validation PASS: normal CTest `7/7` passed, Torch CTest `10/10` passed.

- Phase 4.3B: path-aware greedy DefenseCapacity
  - Added current main-path extraction and path bitboard coverage through `Bitboard128`.
  - Uses `BoardTables::range_mask[type][level][cell] & path_bb` to discount towers that do not cover the current spawn-to-base path.
  - Keeps static tower capacity as a debug upper-bound field while using path-aware capacity for the actual budget.
  - Added differential candidate valuation for builds and upgrades.
  - Greedily selects build / upgrade candidates by `cap_gain / cost`.
  - Added diagnostics for path length, static/current capacity delta, build/upgrade candidate counts, selected build/upgrade cap, and best value-per-cost.
  - Validation PASS: normal CTest `7/7` passed, Torch CTest `10/10` passed.

- Phase 4.3C: AttackBudget API
  - Added `AttackBudgetConfig`, `AttackBudgetResult`, and `estimate_attack_budget()`.
  - Wraps DefenseCapacity into a next-wave attack HP budget.
  - Adds a wave-stage cap: `wave_base_hp + wave_linear_hp * wave + wave_quadratic_hp * wave^2`.
  - Final `allowed_attack_hp` is the minimum of defense-derived capacity and the wave-stage cap when wave cap is enabled.
  - Adds archetype ratio caps for regular, fast, tank, and boss budgets.
  - Adds tank unlock and boss unlock rules; boss is restricted to odd waves by default.
  - Added `test_attack_budget` and `export_attack_budget`.
  - Validation PASS: normal CTest `8/8` passed, Torch CTest `11/11` passed.

- Phase 4.3D / 4.3D2: budgeted wave generator and elite budget absorption
  - Added `BudgetedWaveConfig`, `BudgetedWaveResult`, and `generate_budgeted_wave()`.
  - Converts `AttackBudgetResult` into deterministic `EnemySpec` waves without changing the engine's fixed wave path.
  - Uses regular / fast / tank / boss ratios, tank/boss unlocks, and archetype HP caps from AttackBudget.
  - Adds slot-aware greedy generation: regular backbone first, then boss/tank, then regular budget, then fast fillers.
  - Adds elite scaling to absorb remaining budget when `max_enemy_count` prevents adding more enemies.
  - Elite scaling prioritizes Regular, then Tank, then Boss; Fast is not elite-scaled to avoid high-speed high-HP enemies.
  - Added diagnostics: `elite_bonus_hp`, `elite_scaled_count`, and `max_enemy_hp`.
  - Added `test_budgeted_wave_generator` and `export_budgeted_wave`.
  - Validation PASS: normal CTest `9/9` passed.

- Phase 4.3E: engine wave generation mode switch
  - Integrated budgeted wave generation into `TDEngine` behind a runtime feature flag.
  - Default constructor still uses fixed waves, preserving existing engine behavior and golden trace hashes.
  - Added `TDEngine(int width, int height, uint64_t seed, bool use_budgeted_waves)`.
  - Added `use_budgeted_waves()`, `set_use_budgeted_waves(enabled, regenerate_pending_wave)`, and `pending_spawn_total_hp()`.
  - Split wave generation into `get_fixed_wave_enemies()`, `get_budgeted_wave_enemies()`, and `get_wave_enemies()`.
  - Added `test_engine_wave_modes` and `export_engine_wave_modes`.
  - Validation PASS: fixed mode keeps wave-1 count `9` and total HP `281.2`; budgeted mode produces count `21` and total HP `515`; runtime toggling regenerates the pending queue correctly.
  - Golden trace semantic lock remains unchanged: combined hash `0x595648689a6a7435`.

- Phase 4.3F: fixed-wave versus budgeted-wave A/B smoke
  - Added `export_wave_mode_ab` diagnostic app.
  - Runs seeds `0..4` in both fixed and budgeted modes with the same scripted economy policy.
  - Records steps, total reward, final wave, base HP, money, game-over status, tower count, active enemies, pending enemies, and enemy HP totals.
  - Added shuffle for budgeted waves inside `TDEngine::get_budgeted_wave_enemies()` so seed affects spawn order.
  - Validation PASS: budgeted mode is no longer seed-invariant and is materially smoother than fixed mode in the smoke script.
  - Latest A/B smoke summary:
    - Fixed mode: steps `55..69`, final wave always `3`, total reward about `-1275..-1165`, game over early.
    - Budgeted mode: steps `123..207`, final wave `5..8`, total reward `-484..396`, game over later.
    - Budgeted mode reaches higher waves and produces more varied outcomes after wave shuffle is enabled.

## Current gameplay rules

- Board size: fixed 11x11.
- Spawn: `(0, 5)`.
- Base: `(10, 5)`.
- Initial money: 200.
- Initial base HP: 100.
- Initial wave: 1.
- Each action advances game time by 1 second.
- Invalid build / upgrade / sell actions receive a -5 reward penalty and still advance time.
- Wait is always legal.
- Default wave generation mode is still fixed PvE waves; budgeted wave generation is available behind an explicit engine flag.

### Action space

The action space has 727 flat actions:

- `0..483`: build actions, four tower types times 121 cells.
- `484..604`: upgrade actions.
- `605..725`: sell actions.
- `726`: wait one second.

### Towers

Current tower stats:

| Tower | Cost | Damage | Range | Cooldown | AOE radius | Slow factor | Slow duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic | 50 | 10 | 2.5 | 1.0 | 0.0 | 1.0 | 0.0 |
| Sniper | 100 | 50 | 5.5 | 3.0 | 0.0 | 1.0 | 0.0 |
| AOE | 150 | 15 | 2.0 | 1.5 | 2.0 | 1.0 | 0.0 |
| Slow | 75 | 2 | 3.0 | 1.0 | 0.0 | 0.4 | 2.0 |

Upgrade rule:

- Max level: 5.
- Upgrade increases damage by 1.5x.
- Upgrade increases range by 1.1x.
- Upgrade reduces cooldown by 0.9x, with a floor of 0.1.
- Upgrade cost increases by 1.5x each upgrade.

### Step order

Each simulation second runs in this order:

1. Spawn one enemy if `spawn_timer <= 0` and the current wave still has queued enemies.
2. Move enemies along their existing paths.
3. Remove enemies that reached the base and apply base damage.
4. End the game if base HP is zero or lower.
5. Tick tower cooldowns and let ready towers attack the nearest enemy in range.
6. Remove dead enemies and grant money / reward.
7. If the current wave is fully cleared, advance to the next wave and set a 3-second spawn delay.
8. Advance `time` by `dt`.

### Current fixed wave generation

Fixed PvE wave generation remains the default:

- `base_hp = 20 + wave * 15 + wave^2 * 2`
- Swarm: `wave * 2`, HP `base_hp * 0.3`, speed `2.8`, reward `5`
- Regular: `5 + wave * 2`, HP `base_hp`, speed `1.5`, reward `10`
- Tank: from wave 3, count `wave`, HP `base_hp * 3.5`, speed `0.8`, reward `30`
- Boss: from odd waves >= 5, count `1 + wave / 10`, HP `base_hp * 10`, speed `0.6`, reward `100`

### Budgeted wave mode

Budgeted wave mode is available only when explicitly enabled:

```cpp
TDEngine env(11, 11, 0, true);
// or
env.set_use_budgeted_waves(true, true);
```

Budgeted mode uses:

- path-aware DefenseCapacity
- AttackBudget wave-stage cap and archetype caps
- deterministic BudgetedWaveGenerator
- RNG shuffle of the generated queue for seed-dependent spawn order
- elite scaling for unused budget when enemy-count slots are saturated

## Current test targets

Normal build should include:

- `test_engine`
- `test_board_tables`
- `test_game_logic`
- `test_golden_trace`
- `test_defense_capacity`
- `test_attack_budget`
- `test_budgeted_wave_generator`
- `test_engine_wave_modes`
- `test_mcts`
- `test_selfplay`

Torch build should include the above plus:

- `test_network_shape`
- `test_mcts_libtorch_smoke`
- `test_selfplay_torch`

## Build and test commands

From `cpp/`:

```cmd
cmake -B build -S . -G "Visual Studio 17 2022"
cmake --build build --config Release -j 10
ctest --test-dir build -C Release --output-on-failure
```

Torch build:

```cmd
set PATH=E:\Micromamba\envs\mamba_env\Lib\site-packages\torch\lib;%PATH%
cmake -B build_torch -S . -G "Visual Studio 17 2022" -DTDMZ_ENABLE_TORCH=ON -DTorch_DIR="E:/Micromamba/envs/mamba_env/Lib/site-packages/torch/share/cmake/Torch"
cmake --build build_torch --config Release -j 10
ctest --test-dir build_torch -C Release --output-on-failure
```

Benchmarks and exports:

```cmd
.\build\Release\bench_engine.exe
.\build\Release\bench_mcts.exe
.\build\Release\generate_selfplay_dummy.exe
.\build\Release\export_golden_traces.exe golden_trace_current.jsonl
.\build\Release\export_defense_capacity.exe
.\build\Release\export_attack_budget.exe
.\build\Release\export_budgeted_wave.exe
.\build\Release\export_engine_wave_modes.exe
.\build\Release\export_wave_mode_ab.exe

.\build_torch\Release\bench_engine.exe
.\build_torch\Release\bench_mcts.exe
```

Latest local validation after Wave Mode A/B 4.3F:

- Normal test targets: `10`.
- Golden trace combined hash: `0x595648689a6a7435`.
- Fixed-wave A/B smoke:
  - seed 0: steps `69`, reward `-1220`, final wave `3`, base HP `-21`.
  - seed 1: steps `66`, reward `-1165`, final wave `3`, base HP `-21`.
  - seed 2: steps `55`, reward `-1275`, final wave `3`, base HP `0`.
  - seed 3: steps `60`, reward `-1185`, final wave `3`, base HP `-16`.
  - seed 4: steps `57`, reward `-1210`, final wave `3`, base HP `-1`.
- Budgeted-wave A/B smoke after shuffle fix:
  - seed 0: steps `129`, reward `-434`, final wave `5`, base HP `-13`.
  - seed 1: steps `207`, reward `396`, final wave `8`, base HP `-21`.
  - seed 2: steps `147`, reward `77`, final wave `6`, base HP `-7`.
  - seed 3: steps `123`, reward `-484`, final wave `5`, base HP `-8`.
  - seed 4: steps `142`, reward `-158`, final wave `6`, base HP `-16`.

## Current optimization state

Implemented:

- `Bitboard128` for 121-cell board occupancy.
- `BoardTables` singleton with precomputed geometry, action ids, and range masks.
- Fixed-array BFS pathfinding with `dist[121]`, `parent[121]`, and `queue[121]`.
- Cached base-distance and next-step-to-base fields keyed by `grid_version`.
- Placeable-mask cache keyed by `grid_version`.
- Placeable-mask recomputation via one spawn-base cut-cell DFS instead of per-cell pathfinding.
- Legal-actions cache keyed by `grid_version`, `tower_version`, and `money_version`.
- Observation distance-to-base reuse through the engine distance cache.
- Golden trace semantic lock with fixed hashes.
- Static DefenseCapacity estimator with current tower damage, spendable-money damage, virtual-base leak cap, raw HP cap, and allowed attack HP.
- Path-aware greedy DefenseCapacity with path bitboard coverage, static/current differential diagnostics, and build/upgrade candidate gain-per-cost selection.
- AttackBudget API with wave-stage cap, enemy archetype ratio caps, tank unlock, and boss unlock rules.
- Budgeted wave generator with ratio allocation, unlock handling, deterministic slot-aware fill, and elite budget absorption.
- Runtime engine wave-generation switch between fixed waves and budgeted waves.
- Wave-mode A/B smoke diagnostics.
- Perf counters:
  - `pathfind_calls`
  - `placeable_recompute`
  - `legal_recompute`
  - `base_distance_recompute`

Not yet implemented:

- Config serialization for wave-generation mode.
- Multi-upgrade lookahead or sequential candidate refresh after a chosen upgrade.
- Exact tower attack acceleration using range masks and enemy cell buckets.
- Observation full static-channel cache.
- Debug GUI / trace viewer.

## Engineering rule

When a phase is completed and validated, record the completed work, validation result, and latest benchmark/test result in this README before moving to the next phase.

If the user reports that tests passed without details, treat the phase as validated by default and update this README accordingly.

## Next optimization plan

### Phase 4.2I: tower attack acceleration

Budgeted waves increase episode length and can increase active/pending enemy pressure, so the tower attack loop becomes more important. Use precomputed `range_mask[type][level][cell]` as a coarse filter before exact float-distance checks:

- Maintain enemy rounded-cell buckets.
- For each tower, intersect its range mask with occupied enemy cells.
- Only run exact distance checks on candidate enemies in covered cells.
- Keep damage semantics unchanged and validate with golden traces.

### Phase 4.2J: observation static-channel cache

Static and semi-static channels can be cached by version:

- grid blocked
- spawn/base
- tower type
- tower level
- distance-to-base
- placeable mask

Dynamic channels are still rewritten every step:

- tower cooldown
- enemy HP/density/speed/slow
- base HP, money, wave, spawn timer, to-spawn count

### Phase 4.4: debug GUI / trace viewer

A lightweight debug UI should show:

- 11x11 board
- tower type / level / cooldown
- enemy position / HP / speed
- money, base HP, wave, time
- legal actions
- defense capacity and allowed attack HP
- per-wave damage and kill stats

Initial UI can be Python Tkinter reading C++ JSON traces. Keep C++ as the canonical engine.
