# TD_MuZero

C++17 / LibTorch MuZero research project for an 11x11 tower-defense environment.

The C++ engine is the canonical environment. The repository contains the engine, exact
root-mask MCTS, self-play/replay infrastructure, a LibTorch MuZero network, compatibility
manifests, deterministic semantic locks, and performance benchmarks.

## Current status

### Validated baseline

The Observation V2 baseline was validated with both normal and LibTorch builds:

- complete normal CTest suite passed;
- complete LibTorch CTest suite passed;
- Observation V2 allocating/reused checksums matched;
- Golden Trace combined hash remained `0x9150f0decb0cff52`;
- PyTorch `2.5.1+cu121`, CUDA 12.1, and MSVC 19.39 were verified on Windows.

### Current HEAD

Phase B has now been implemented and requires local revalidation after pulling:

- Observation V2: 40x11x11;
- Dynamics action input: seven spatial planes;
- Policy head: six 11x11 planes plus one wait scalar;
- Legality head: six 11x11 planes plus one wait scalar;
- public action ABI unchanged at 727 actions;
- network architecture version: 3;
- checkpoint manifest version: 2.

The detailed sequence is in
[`docs/tdmuzero_semantics_repair_plan.md`](docs/tdmuzero_semantics_repair_plan.md).
The cross-file semantic review is in
[`docs/code_semantics_audit_2026-07-12.md`](docs/code_semantics_audit_2026-07-12.md).

## Environment and action semantics

### Board and initial state

- board: 11x11;
- spawn: `(0, 5)`;
- base: `(10, 5)`;
- initial money: 200;
- initial base HP: 100;
- initial wave: 1;
- each public action advances one second;
- invalid build/upgrade/sell: `-5` reward and the second still advances;
- Wait1 is always legal.

### Action space

The action space is fixed at 727 IDs:

| IDs | Meaning |
|---:|---|
| `0..120` | build Basic |
| `121..241` | build Sniper |
| `242..362` | build AOE |
| `363..483` | build Slow |
| `484..604` | upgrade |
| `605..725` | sell |
| `726` | wait one second |

Cell order is row-major: `cell = y * 11 + x`.

### One-second engine order

1. decrease spawn timer and spawn one queued enemy when ready;
2. move enemies continuously;
3. remove leaks and damage the base;
4. apply terminal check;
5. advance tower cooldowns and fire using half-open timing `[0, 1)`;
6. remove killed enemies and add money/reward;
7. generate the next wave when active and pending enemies are empty;
8. advance environment time.

## Observation V2

Observation V2 contains 40 planes. It includes:

- blocked cells, spawn, base;
- four tower-type planes, level, cooldown;
- structural placeability and distance to base;
- enemy density, HP ratio, absolute HP/max HP, speed, slow duration;
- enemy reward, leak damage, remaining route distance and direction;
- exact enemy-occupied cells;
- base HP, money, wave and spawn timer;
- pending count, total HP/reward/leak, speed summaries;
- next pending enemy HP/speed/reward/leak.

Continuous enemy quantities are bilinearly splatted. Temporary build occupancy still uses
the engine's exact rounded-cell rule. All values are finite and bounded to `[0, 1]`.

Old 20-channel data must not be mixed with V2 data.

## Network architecture v3

### Representation

```text
observation [B,40,11,11]
  -> Conv 40->64, ReLU
  -> Conv 64->32, ReLU
  -> latent [B,32,11,11]
```

### Spatial action encoding

Dynamics receives seven action planes:

```text
0 Basic build
1 Sniper build
2 AOE build
3 Slow build
4 Upgrade
5 Sell
6 Wait
```

Build/upgrade/sell is one-hot at the target cell. Wait fills the complete wait plane.
This keeps the target-cell intervention local instead of broadcasting coordinate scalars
across the board.

### Dynamics

```text
latent + action planes
  -> spatial convolutional transition
  -> next latent
  -> globally pooled reward head
```

### Prediction

```text
latent
  -> spatial trunk
  -> globally pooled Value
  -> 6x11x11 Policy + 1 Wait scalar
  -> 6x11x11 Legality + 1 Wait scalar
```

Spatial logits are flattened channel-major, so the public 727-action mapping remains
exactly unchanged.

## Build modes

There are two separate CMake builds.

### Core-only build

This build has no LibTorch dependency. MCTS/self-play benchmarks use `DummyNetwork`.

```powershell
cd E:\Desktop\TD_MuZero\cpp

cmake -S . -B build -G Ninja `
    "-DTDMZ_ENABLE_TORCH=OFF" `
    "-DTDMZ_BUILD_ENGINE_DEBUG_GUI=OFF" `
    "-DCMAKE_BUILD_TYPE=Release"

cmake --build build --parallel 10
ctest --test-dir build --output-on-failure -j 8
```

### LibTorch CUDA build

This build compiles `tdmz_nn`, the LibTorch evaluator and the three Torch-specific tests.
It uses the dedicated environment:

```text
E:\Micromamba\envs\tdmz_cuda121
```

Verified toolchain:

```text
Visual Studio 2022 / MSVC 19.39
CUDA compiler 12.1.105
PyTorch 2.5.1+cu121
Ninja single-config Release build
```

## Creating the clean CUDA 12.1 environment

The old shared `mamba_env` must not be used as a CUDA compiler root. It contained a mixed
12.1/12.4/13.2 development toolchain.

```powershell
$Mamba = "C:\Users\34574\AppData\Local\micromamba\micromamba.exe"
$TDEnv = "E:\Micromamba\envs\tdmz_cuda121"
$env:MAMBA_ROOT_PREFIX = "E:\Micromamba"

& $Mamba create -y -p $TDEnv `
    --strict-channel-priority `
    -c "nvidia/label/cuda-12.1.1" `
    -c conda-forge `
    "python=3.11" `
    "pip" `
    "cmake" `
    "ninja" `
    "cuda"

& "$TDEnv\python.exe" -m pip install `
    "torch==2.5.1" `
    --index-url https://download.pytorch.org/whl/cu121

& "$TDEnv\python.exe" -m pip install numpy
```

Verify before configuring CMake:

```powershell
& "$TDEnv\python.exe" -c `
    "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.utils.cmake_prefix_path)"

& "$TDEnv\bin\nvcc.exe" --version
```

Expected essentials:

```text
torch 2.5.1+cu121
Torch CUDA 12.1
CUDA available True
nvcc 12.1.105
```

## Reproducible Windows LibTorch build

Run this in a fresh PowerShell session.

### 1. Load one coherent VS2022 environment

```powershell
$VsRoot = "E:\Microsoft Visual Studio2022\Microsoft Visual Studio2022Community"
$VcVars = "$VsRoot\VC\Auxiliary\Build\vcvars64.bat"

Get-Process mspdbsrv -ErrorAction SilentlyContinue |
    Stop-Process -Force

cmd /s /c "`"$VcVars`" && set" |
    ForEach-Object {
        $pair = $_ -split "=", 2
        if ($pair.Count -eq 2) {
            Set-Item -Path "Env:$($pair[0])" -Value $pair[1]
        }
    }

where.exe cl
where.exe link
```

The first compiler must be the VS2022 14.39 x64 compiler, not Strawberry/MinGW and not a
newer incompatible MSVC installation.

### 2. Select the clean CUDA/PyTorch environment

```powershell
$TDEnv = "E:\Micromamba\envs\tdmz_cuda121"
$Python = "$TDEnv\python.exe"
$TorchPrefix = & $Python -c "import torch; print(torch.utils.cmake_prefix_path)"

$env:CUDA_PATH = $TDEnv
$env:CUDA_HOME = $TDEnv
$env:CUDACXX = "$TDEnv\bin\nvcc.exe"
$env:CC = "cl"
$env:CXX = "cl"

$env:PATH = "$TDEnv\Lib\site-packages\torch\lib;" +
            "$TDEnv\bin;" +
            "$TDEnv\Library\bin;" +
            $env:PATH

where.exe cl
where.exe nvcc
```

The first `nvcc` must be:

```text
E:\Micromamba\envs\tdmz_cuda121\bin\nvcc.exe
```

### 3. Configure with VS CMake/Ninja

```powershell
$CMake = "$VsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$Ninja = "$VsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$CTest = "$VsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"

$TDEnvCMake = $TDEnv.Replace('\', '/')
$TorchPrefixCMake = $TorchPrefix.Replace('\', '/')
$NinjaCMake = $Ninja.Replace('\', '/')

cd E:\Desktop\TD_MuZero\cpp
Remove-Item -Recurse -Force .\build_torch -ErrorAction SilentlyContinue

& $CMake -S . -B build_torch -G Ninja `
    "-DTDMZ_ENABLE_TORCH=ON" `
    "-DTDMZ_BUILD_ENGINE_DEBUG_GUI=OFF" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release" `
    "-DCMAKE_PREFIX_PATH=$TorchPrefixCMake" `
    "-DCUDAToolkit_ROOT=$TDEnvCMake" `
    "-DCMAKE_MAKE_PROGRAM=$NinjaCMake" `
    "-DCMAKE_CXX_COMPILER=cl" `
    "-DCMAKE_CUDA_COMPILER=$TDEnvCMake/bin/nvcc.exe" `
    "-DCMAKE_CUDA_HOST_COMPILER=cl"
```

### 4. Build everything before running full CTest

```powershell
& $CMake --build build_torch --parallel 10
& $CTest --test-dir build_torch --output-on-failure -j 8
```

Ninja is single-config. Test executables are normally here:

```text
build_torch\tests\test_network_shape.exe
build_torch\tests\test_mcts_libtorch_smoke.exe
build_torch\tests\test_selfplay_torch.exe
```

There is no `tests\Release\` directory in this configuration.

## Focused validation after Phase B

```powershell
cd E:\Desktop\TD_MuZero\cpp
git pull

cmake --build build --parallel 10
ctest --test-dir build --output-on-failure -j 8

& $CMake --build build_torch --parallel 10
& $CTest --test-dir build_torch --output-on-failure -j 8

.\build_torch\tests\test_network_shape.exe
.\build_torch\tests\test_mcts_libtorch_smoke.exe
.\build_torch\tests\test_selfplay_torch.exe
```

Also rerun the semantic lock and benchmarks:

```powershell
.\build\tests\test_golden_trace.exe
.\build\bench_observation.exe
.\build\bench_selfplay.exe
```

Observation and self-play checksums may change only when their serialized/network
semantics change intentionally. The engine Golden Trace must remain unchanged for Phase B.

## Windows build failures encountered

### `Error: could not load cache`

`cmake --build build_torch` was called before `build_torch/CMakeCache.txt` existed, or the
build directory had been deleted. Run the configure command again.

### CMake selects `C:/Strawberry/c/bin/c++.exe`

The shell was not initialized with VS2022. Load `vcvars64.bat` and explicitly use the VS
CMake/Ninja and `cl`.

### `nvcc fatal: Cannot find compiler 'cl.exe' in PATH`

CUDA was found before the Visual Studio developer environment was loaded. Import
`vcvars64.bat`, then verify `where.exe cl` before running CMake.

### `fatal error C1902: 程序数据库管理器不匹配`

Different MSVC/PDB tool versions were mixed, or CMake's compiler probe used Debug `/Zi`.
Use a fresh shell, kill stale `mspdbsrv`, load one VS2022 environment, and configure with:

```text
-DCMAKE_TRY_COMPILE_CONFIGURATION=Release
```

### `Invalid character escape '\M'`

A Windows backslash path was written into a generated CMake file. Use forward-slash CMake
paths and compiler names such as `cl` rather than a backslash-heavy absolute compiler
path.

### `nvlink fatal ... cudadevrt.lib ... newer than toolkit (124 vs 121)`

The shared environment had:

```text
nvcc/nvlink 12.1
cudadevrt.lib 12.4
cuda-version 13.2
```

Runtime PyTorch and ONNX projects could still work because they used precompiled DLLs,
but native CUDA compilation exposed the mismatch. Use the dedicated strict-channel CUDA
12.1 environment; do not copy individual CUDA libraries by hand.

### Full CTest reports many `Could not find executable` / `Not Run`

Only three named Torch targets had been compiled. CTest registers all normal and Torch
tests, so build the whole tree first:

```powershell
& $CMake --build build_torch --parallel 10
```

### PyTorch warns that NumPy is missing

This warning does not block LibTorch CMake discovery, but install NumPy to remove it:

```powershell
& "$TDEnv\python.exe" -m pip install numpy
```

## TensorRT policy

Adding `E:\libs\TensorRT-10.16.1.11\bin` to `PATH` only makes DLLs discoverable. It does
not make LibTorch or ONNX Runtime automatically use TensorRT.

TensorRT is intentionally not part of the current training build:

```text
training and correctness: LibTorch CUDA
future high-throughput self-play inference: ONNX/TensorRT after the network and trainer freeze
```

A future inference backend should export separate initial/recurrent ONNX graphs, register
TensorRT explicitly, retain CUDA fallback, use FP16 and engine/timing caches, and benchmark
batch sizes before adoption.

## Repository layout

```text
cpp/
  include/tdmz/
    core/          engine, actions, observation, compatibility
    balance/       defense capacity and budgeted waves
    mcts/          pools, masks, search and evaluator interface
    nn/            LibTorch representation/dynamics/prediction
    persistence/   checkpoint manifest
    selfplay/      history, writers, transition shards and replay
  src/             implementations
  tests/           normal and LibTorch regression tests
  apps/            exporters, generators and benchmarks

docs/
  tdmuzero_semantics_repair_plan.md
  code_semantics_audit_2026-07-12.md
```

## Mandatory next order

1. validate Phase B on the clean normal and LibTorch builds;
2. make terminal states fully absorbing and version that rule change;
3. add wave-mode provenance and close direct-shard compatibility bypasses;
4. implement Phase C legality transport/loss/metrics;
5. implement Phase D K-step replay, n-step returns and terminal/truncated semantics;
6. implement the trainer and real-checkpoint self-play;
7. enable exploration only after the complete training loop is correct.
