param(
    [string]$CudaHome = "C:\Users\hight\.codex\cuda-12.8\Library",
    [string]$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    [string]$OutDir = "C:\Users\hight\.codex\nvcc_smoke"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath (Join-Path $CudaHome "bin\nvcc.exe"))) {
    throw "nvcc.exe was not found under CUDA_HOME: $CudaHome"
}
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "VsDevCmd.bat path does not exist: $VsDevCmd"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$source = Join-Path $OutDir "minimal.cu"
$object = Join-Path $OutDir "minimal.obj"
@"
#include <cuda_runtime.h>

__global__ void add_one_kernel(float* x, long long n) {
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += 1.0f;
  }
}

extern "C" void launch_add_one(float* x, long long n) {
  add_one_kernel<<<(n + 255) / 256, 256>>>(x, n);
}
"@ | Set-Content -LiteralPath $source -Encoding UTF8

$dump = & cmd.exe /s /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && set"
foreach ($line in $dump) {
    if ($line -match "^(.*?)=(.*)$") {
        Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
}

$env:CUDA_HOME = $CudaHome
$env:CUDA_PATH = $CudaHome
$env:PATH = (Join-Path $CudaHome "bin") + ";" + $env:PATH

& (Join-Path $CudaHome "bin\nvcc.exe") -allow-unsupported-compiler -std=c++17 -arch=sm_100 -c $source -o $object
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $object)) {
    throw "nvcc did not create object: $object"
}
Get-Item -LiteralPath $object | Select-Object FullName, Length
