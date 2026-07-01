param(
    [string]$CudaHome = "C:\Users\hight\.codex\cuda-12.8\Library",
    [string]$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    [string]$TorchExtensionsDir = "C:\Users\hight\.codex\torch_extensions_vs2022",
    [string]$TorchCudaArchList = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CudaHome)) {
    throw "CUDA_HOME path does not exist: $CudaHome"
}
if (-not (Test-Path -LiteralPath (Join-Path $CudaHome "bin\nvcc.exe"))) {
    throw "nvcc.exe was not found under CUDA_HOME: $CudaHome"
}
$condaCudartLib = Join-Path $CudaHome "lib\cudart.lib"
$torchExpectedCudaLibDir = Join-Path $CudaHome "lib\x64"
$torchExpectedCudartLib = Join-Path $torchExpectedCudaLibDir "cudart.lib"
if ((Test-Path -LiteralPath $condaCudartLib) -and -not (Test-Path -LiteralPath $torchExpectedCudartLib)) {
    New-Item -ItemType Directory -Force -Path $torchExpectedCudaLibDir | Out-Null
    Copy-Item -LiteralPath $condaCudartLib -Destination $torchExpectedCudartLib -Force
}
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "VsDevCmd.bat path does not exist: $VsDevCmd"
}

$dump = & cmd.exe /s /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && set"
foreach ($line in $dump) {
    if ($line -match "^(.*?)=(.*)$") {
        Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
}

$env:CUDA_HOME = $CudaHome
$env:CUDA_PATH = $CudaHome
$env:PATH = (Join-Path $CudaHome "bin") + ";" + $env:PATH
$env:TORCH_EXTENSIONS_DIR = $TorchExtensionsDir
$env:MAX_JOBS = "1"
if ($TorchCudaArchList) {
    $env:TORCH_CUDA_ARCH_LIST = $TorchCudaArchList
}
New-Item -ItemType Directory -Force -Path $TorchExtensionsDir | Out-Null
$smokeBuildDir = Join-Path $TorchExtensionsDir "cortex3_cuda128_smoke"
if (Test-Path -LiteralPath $smokeBuildDir) {
    Remove-Item -LiteralPath $smokeBuildDir -Recurse -Force
}

& python tools\cuda_extension_smoke.py
exit $LASTEXITCODE
