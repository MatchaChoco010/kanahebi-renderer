param(
    [Parameter(Mandatory=$false)]
    [int]$Jobs = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$VENV_DIR     = ".venv"
$USD_REPO_DIR = "external/OpenUSD"
$BUILD_DIR    = "build/OpenUSD"
$INSTALL_DIR  = "dependencies/OpenUSD"
$PATCH_FILE = Join-Path $PSScriptRoot "build-usd.blender-4.5.patch"

if ($Jobs -gt 0) {
    $JOBS = $Jobs
} else {
    $JOBS = [int]$env:NUMBER_OF_PROCESSORS
    if (-not $JOBS -or $JOBS -le 0) { $JOBS = 4 }
}

if (-not (Test-Path -Path $VENV_DIR -PathType Container)) {
  Write-Host "[INFO] Creating virtual environment..."
  & python -m venv $VENV_DIR
} else {
  Write-Host "[INFO] Virtual environment already exists. Skipping creation."
}

. (Join-Path $VENV_DIR "Scripts/Activate.ps1")

Write-Host "[INFO] Installing Python dependencies..."
& python -m pip install --upgrade pip
& python -m pip install -U PyOpenGL PySide6

$PYTHON_EXEC = (Get-Command python).Source
$PYTHON_VERSION = & python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
$PYTHON_INCLUDE_DIR = & python -c "import sysconfig; print(sysconfig.get_path('include'))"
$PYTHON_LIBRARY_CODE = @'
import sys, sysconfig, os
libdir = sysconfig.get_config_var("LIBDIR")
if not libdir or not os.path.exists(libdir):
    libdir = os.path.join(sys.base_prefix, "libs")
ver = sysconfig.get_python_version().replace(".", "")
name = f"python{ver}.lib"
path = os.path.join(libdir, name)
print(path if os.path.exists(path) else "")
'@
$PYTHON_LIBRARY = & python -c $PYTHON_LIBRARY_CODE

Write-Host "[INFO] Using Python from venv:"
Write-Host ("  Executable: {0}" -f $PYTHON_EXEC)
Write-Host ("  Include:    {0}" -f $PYTHON_INCLUDE_DIR)
Write-Host ("  Library:    {0}" -f $PYTHON_LIBRARY)
Write-Host ""

Push-Location $USD_REPO_DIR
& git checkout .
& git checkout "v25.02"
& git apply --ignore-whitespace --whitespace=nowarn $PATCH_FILE
Pop-Location

$vcvarsPath = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$output = & cmd.exe /c "`"$vcvarsPath`" && set"
foreach ($line in $output) {
    if ($line -match '^(.*?)=(.*)$') {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2] -Force
    }
}

Write-Host "[INFO] Building OpenUSD with $JOBS jobs..."
$PXR_INTERNAL_NAMESPACE="pxrBlender_v25_02"
$buildScript = Join-Path $USD_REPO_DIR "build_scripts/build_usd.py"
$usdArgs = @(
  '-u'
  $buildScript
  '--build', $BUILD_DIR
  '--build-monolithic'
  '--usd-imaging'
  '--onetbb'
  '--no-python'
  '--no-openvdb'
  '--no-materialx'
  '--no-examples'
  "-j$JOBS"
  "--build-args", "USD,-DPXR_SET_INTERNAL_NAMESPACE=$PXR_INTERNAL_NAMESPACE"
  '--build-python-info',
    $PYTHON_EXEC,
    $PYTHON_INCLUDE_DIR,
    $PYTHON_LIBRARY,
    $PYTHON_VERSION,
  $INSTALL_DIR
)

& python @usdArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "[INFO] Build completed successfully."
} else {
    Write-Error "Build failed with exit code $LASTEXITCODE"
}
