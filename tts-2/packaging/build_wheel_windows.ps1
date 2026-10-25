# Build the Windows kitten-inference-tts2-cu12 wheel (py3-none-win_amd64).
#   powershell -File tts-2\packaging\build_wheel_windows.ps1 [-Out DIR] [-Jobs N] [-Version 0.2.0.post1]
# Run it from a Visual Studio 2022 x64 developer environment (vcvars64; the CI job uses ilammy/msvc-dev-cmd) with
# CUDA 12.8 (CUDA_PATH), CMake 3.28+, Ninja and Python 3 on PATH.
# Steps: the text dependencies with the static C runtime (tools/build_text_deps.py --msvc-static-runtime),
# `cmake --preset wheel-windows`, the GPU code check against the default build (sass_manifest.py; it needs cuobjdump
# from CUDA 12.8), install into a stage folder, make_wheel.py, then a check that the DLL and programs
# import only Windows system DLLs (static C runtime, static CUDA runtime; the NVIDIA driver's nvcuda.dll is loaded at
# run time), so they run on a clean Windows without the Visual C++ redistributable.
param(
  [string]$Out = "",
  [int]$Jobs = 0,
  [string]$Version = ""   # default: the engine version (make_wheel.py)
)
$ErrorActionPreference = "Stop"
$Src = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Out) { $Out = Join-Path $Src "dist" }
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Set-Location $Src
function Run([string]$exe, [string[]]$argv) {
  & $exe @argv
  if ($LASTEXITCODE -ne 0) { throw "$exe $($argv -join ' ') failed with exit code $LASTEXITCODE" }
}
$t0 = Get-Date
Run "nvcc" @("--version")
if (-not $env:KT2N_TEXT_DEPS) {
  Run "python" @("tools/build_text_deps.py", "--prefix", "build-deps/text-mt", "--jobs", "$Jobs", "--msvc-static-runtime")
  $env:KT2N_TEXT_DEPS = (Join-Path $Src "build-deps/text-mt")
}
$t1 = Get-Date
Run "cmake" @("--preset", "wheel-windows")
Run "cmake" @("--build", "--preset", "wheel-windows", "--parallel", "$Jobs")
# The GPU code must be the default build's, function for function (packaging/sass_manifest.py): files whose code came
# out differently are compiled again. On Windows nearly every compile of the CUDA-core decode kernels varies in a few of
# them (4 of 36 CI compiles had all 78 identical), so the differing object is compiled and checked on its own up to 40
# times before each relink (about 25 s each; at most 6 x 40 compiles, about 2.5 hours, within the job's 6 hours).
# KT2N_SASS_CHECK=0 skips this (development builds only).
if ($env:KT2N_SASS_CHECK -ne "0") {
  Run "python" @("packaging/sass_manifest.py", "check", "build/wheel-windows/bin/kt2n_engine.dll", "--reference",
                 "packaging/sass-reference.txt", "--build-dir", "build/wheel-windows", "--attempts", "6",
                 "--object-attempts", "40", "--",
                 "cmake", "--build", "--preset", "wheel-windows", "--parallel", "$Jobs")
} else {
  "WARNING: KT2N_SASS_CHECK=0: the GPU code was not compared with the default build's; do not release this wheel"
}
$t2 = Get-Date
$Stage = Join-Path $Src "build/wheel-windows-stage"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
Run "cmake" @("--install", "build/wheel-windows", "--prefix", $Stage)
$makeArgs = @("packaging/make_wheel.py", "--stage", $Stage, "--plat", "win_amd64", "--out", $Out)
if ($Version) { $makeArgs += @("--version", $Version) }
Run "python" $makeArgs

# Imports: Windows system DLLs only.
$allowed = @("KERNEL32.dll", "ADVAPI32.dll", "WINMM.dll", "SHELL32.dll", "USER32.dll", "ole32.dll", "OLEAUT32.dll",
             "WS2_32.dll", "bcrypt.dll", "SHLWAPI.dll", "VERSION.dll", "dbghelp.dll", "kt2n_engine.dll")
$report = @()
foreach ($f in @("bin/kt2n_engine.dll", "bin/kitten-tts-native.exe", "bin/kt2n_prepare_text_data.exe")) {
  $path = Join-Path $Stage $f
  # /dependents lists the imported and the delay-loaded DLLs
  $all = (& dumpbin /nologo /dependents $path) | Where-Object { $_ -match '^\s+\S+\.dll\s*$' } |
    ForEach-Object { $_.Trim() } | Sort-Object -Unique
  $bad = $all | Where-Object { $allowed -notcontains $_ }
  $report += "${f}: $($all -join ', ')"
  if ($bad) { throw "$f imports non-system DLLs: $($bad -join ', ')" }
}
$report | Tee-Object -FilePath (Join-Path $Out "dll-imports.txt")
$whl = Get-ChildItem (Join-Path $Out "kitten_inference_tts2_cu12-*-py3-none-win_amd64.whl") | Sort-Object LastWriteTime | Select-Object -Last 1
"timings: text deps $([int]($t1 - $t0).TotalSeconds) s, engine build $([int]($t2 - $t1).TotalSeconds) s, total $([int]((Get-Date) - $t0).TotalSeconds) s"
"$($whl.FullName) $($whl.Length) bytes; kt2n_engine.dll $((Get-Item (Join-Path $Stage 'bin/kt2n_engine.dll')).Length) bytes"
