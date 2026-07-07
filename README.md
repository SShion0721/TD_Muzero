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
  - Latest `export_defense_capacity` output:
    - `empty_seed0`: current `0`, spendable `600`, leak `990`, raw `1590`, allowed `1431`, placeable `119`, selected `4 x Basic`.
    - `one_basic_seed0`: current `160`, spendable `450`, leak `990`, raw `1600`, allowed `1440`, placeable `118`, selected `3 x Basic`.
    - `basic_sniper_seed0`: current `460`, spendable `150`, leak `990`, raw `1600`, allowed `1440`, placeable `117`, selected `1 x Basic`.

- Phase 4.3B: path-aware greedy DefenseCapacity
  - Added current main-path extraction and path bitboard coverage through `Bitboard128`.
  - Uses `BoardTables::range_mask[type][level][cell] & path_bb` to discount towers that do not cover the current spawn-to-base path.
  - Keeps static tower capacity as a debug upper-bound field while using path-aware capacity for the actual budget.
  - Added differential candidate valuation for builds and upgrades:
    - build gain: path-aware level-1 tower capacity at a placeable cell.
    - upgrade gain: path-aware upgraded tower capacity minus current tower capacity.
  - Greedily selects build / upgrade candidates by `cap_gain / cost`, with one build per cell and one upgrade per existing tower in this approximation pass.
  - Added diagnostics for path length, static/current capacity delta, build/upgrade candidate counts, selected build/upgrade cap, and best value-per-cost.
  - Validation PASS: normal CTest `7/7` passed, Torch CTest `10/10` passed.
  - Latest `export_defense_capacity` output:
    - `empty_seed0`: current `0`, spendable `600`, leak `990`, raw `1590`, allowed `1431`, path_len `10`, build candidates `296`, selected `4 x Basic`.
    - `off_path_basic_seed0`: current `0`, static `160`, spendable `450`, raw `1440`, allowed `1296`; off-path tower is correctly discounted to zero current capacity.
    - `one_path_basic_seed0`: current `160`, static `160`, spendable `450`, raw `1600`, allowed `1440`, upgrade candidates `1`.
    - `path_basic_sniper_seed0`: current `460`, static `460`, spendable `150`, raw `1600`, allowed `1440`, upgrade candidates `2`.
    - `static_path_basic_sniper_seed0`: current `460`, spendable `150`, raw `1600`, allowed `1440`, build candidates `468`, showing the static mode keeps many more candidate cells than path-aware mode.

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

### Current wave generation

Current fixed PvE wave generation:

- `base_hp = 20 + wave * 15 + wave^2 * 2`
- Swarm: `wave * 2`, HP `base_hp * 0.3`, speed `2.8`, reward `5`
- Regular: `5 + wave * 2`, HP `base_hp`, speed `1.5`, reward `10`
- Tank: from wave 3, count `wave`, HP `base_hp * 3.5`, speed `0.8`, reward `30`
- Boss: from odd waves >= 5, count `1 + wave / 10`, HP `base_hp * 10`, speed `0.6`, reward `100`

## Current test targets

Normal build should include:

- `test_engine`
- `test_board_tables`
- `test_game_logic`
- `test_golden_trace`
- `test_defense_capacity`
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

.\build_torch\Release\bench_engine.exe
.\build_torch\Release\bench_mcts.exe
```

Latest local validation after DefenseCapacity 4.3B:

- Normal CTest: `7/7` passed.
- Torch CTest: `10/10` passed.
- Golden trace combined hash: `0x595648689a6a7435`.
- `bench_engine`: `354620` steps/s.
- Torch `bench_engine`: `359361` steps/s.
- `bench_mcts`: `62427.5` simulations/s.
- Torch `bench_mcts`: `61515.4` simulations/s.
- `generate_selfplay_dummy`: `steps=58`, `total_reward=-2150`.
- `export_defense_capacity`: off-path Basic is discounted from static `160` to current `0`; path-aware build candidates are reduced from `468` static candidates to `288-296` useful path candidates in the shown cases.

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
- Perf counters:
  - `pathfind_calls`
  - `placeable_recompute`
  - `legal_recompute`
  - `base_distance_recompute`

Not yet implemented:

- Attack budget integration into actual wave / attacker generation.
- Multi-upgrade lookahead or sequential candidate refresh after a chosen upgrade.
- Exact tower attack acceleration using range masks and enemy cell buckets.
- Observation full static-channel cache.
- Debug GUI / trace viewer.

## Engineering rule

When a phase is completed and validated, record the completed work, validation result, and latest benchmark/test result in this README before moving to the next phase.

If the user reports that tests passed without details, treat the phase as validated by default and update this README accordingly.

## Next optimization plan

### Phase 4.3C: attack budget API

Turn DefenseCapacity into an explicit attack budget interface:

```text
AllowedAttackHP <= DefenseDamageCap + LeakCap
```

First integration:

- Given current `TDEngine`, return `allowed_attack_hp` for the next wave.
- Add clamps by wave index so early waves cannot receive late-game monster quality.
- Add tests that increasing defense increases attack budget and lowering virtual base HP lowers budget.
- Keep 4.3A/4.3B diagnostics available for GUI/debug and balancing.

### Phase 4.3D: attacker generation under budget

Use the attack budget to generate a bounded enemy set:

- Convert `allowed_attack_hp` into a wave budget.
- Assign budget across regular / swarm / tank / boss archetypes.
- Clamp special enemies by unlock wave and ratio limits.
- Add deterministic tests for budget non-overflow and monotonic scaling with defense capacity.

### Phase 4.2I: tower attack acceleration

Use precomputed `range_mask[type][level][cell]` as a coarse filter before exact float-distance checks:

- Maintain enemy rounded-cell buckets.
- For each tower, intersect its range mask with occupied enemy cells.
- Only run exact distance checks on candidate enemies in covered cells.

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
