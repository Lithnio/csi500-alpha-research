[CmdletBinding()]
param(
    [ValidateSet("Plan", "Run")]
    [string]$Mode = "Plan",

    [string]$AnnualConfig = "",

    [ValidateRange(1, 8)]
    [int]$Workers = 2,

    [int[]]$Year = @(),

    [string[]]$Trial = @(),

    [switch]$Force
)

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExecutable = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Project Python executable not found: $pythonExecutable"
}

if (-not $AnnualConfig) {
    $AnnualConfig = Join-Path $repositoryRoot "configs\annual\factor_family_v4_postsolve.yaml"
}
elseif (-not [System.IO.Path]::IsPathRooted($AnnualConfig)) {
    $AnnualConfig = Join-Path $repositoryRoot $AnnualConfig
}
if (-not (Test-Path -LiteralPath $AnnualConfig -PathType Leaf)) {
    throw "Annual configuration not found: $AnnualConfig"
}
$resolvedAnnualConfig = (Resolve-Path -LiteralPath $AnnualConfig).Path

if ($Mode -eq "Plan") {
    & $pythonExecutable -m csi500_alpha plan-annual-study --annual $resolvedAnnualConfig
    exit $LASTEXITCODE
}

$cliArguments = @(
    "-m",
    "csi500_alpha",
    "run-annual-study",
    "--annual",
    $resolvedAnnualConfig,
    "--workers",
    $Workers
)
foreach ($selectedYear in $Year) {
    $cliArguments += @("--year", $selectedYear)
}
foreach ($selectedTrial in $Trial) {
    $cliArguments += @("--trial", $selectedTrial)
}
if ($Force) {
    $cliArguments += "--force"
}

& $pythonExecutable @cliArguments
exit $LASTEXITCODE
