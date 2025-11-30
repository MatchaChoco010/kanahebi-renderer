Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$USD_REPO_DIR = "external/OpenUSD"

# checkout USD
Push-Location $USD_REPO_DIR
& git checkout .
& git checkout "v25.11"
Pop-Location
