# Engine Debug GUI

`tdmz_engine_debug_gui` is a small Windows-only diagnostic front end for the canonical C++ `TDEngine`.
It uses the Win32/GDI APIs already available on Windows and has no third-party GUI dependency.

## Build

```powershell
cmake -S cpp -B cpp/build_gui -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DTDMZ_ENABLE_TORCH=OFF `
    -DTDMZ_BUILD_ENGINE_DEBUG_GUI=ON

cmake --build cpp/build_gui --parallel 15
```

Run:

```powershell
.\cpp\build_gui\tdmz_engine_debug_gui.exe
```

The target is only created on Windows. Set `-DTDMZ_BUILD_ENGINE_DEBUG_GUI=OFF` to skip it.

## Controls

- `Inspect` / `I`: select a cell without advancing the engine.
- `Basic` / `1`: build a Basic tower on the clicked cell.
- `Sniper` / `2`: build a Sniper tower.
- `AOE` / `3`: build an AOE tower.
- `Slow` / `4`: build a Slow tower.
- `Upgrade` / `U`: upgrade the tower on the clicked cell.
- `Sell` / `S`: sell the tower on the clicked cell.
- `Wait 1` / `W`: advance one discrete engine tick.
- `Wait 10` / `T`: advance ten discrete ticks.
- `Auto` / `Space`: advance one tick every 250 ms until stopped or terminal.
- `Reset` / `R`: reset the same seed.
- `Next seed` / `N`: reset with the next seed.
- `Waves`: switch between fixed and budgeted wave generation and reset.

## Display

- Light green cells are currently placeable according to `compute_placeable_mask()`.
- Blue and red cells mark spawn and base.
- Towers show their type letter and level.
- Enemies show interpolated position, remaining HP, and slow state.
- The right panel shows money, base HP, wave, time, spawn timer, enemy counts, selected tool, selected cell, and recent actions.

## Invalid actions

The GUI deliberately sends clicked build/upgrade/sell actions through `step_action()` even when the current legal mask rejects them. The log records `legal=false`; the engine then applies its normal invalid-action penalty and still advances one tick. This makes rule mismatches visible instead of silently blocking them in the GUI.

## Suggested manual checks

1. Compare a legal and illegal build and verify money, reward, and time changes.
2. Place two towers that can target the same weak enemy and verify the second tower does not spend cooldown on an already-dead target.
3. Place a Slow tower and inspect movement before, during, and after slow expiry.
4. Use `Wait 10`, reset, then perform ten `Wait 1` actions and compare the visible state.
5. Switch fixed/budgeted waves and compare pending enemies and progression across seeds.
6. Let enemies reach the base and verify damage, reward, cleanup, and terminal behavior.

The GUI is an observability tool, not a replacement for deterministic unit tests or Golden Trace checks.
