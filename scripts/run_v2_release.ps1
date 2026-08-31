[CmdletBinding()]
param(
    [ValidateSet("Plan", "Run")]
    [string]$Mode = "Plan",

    [ValidateRange(1, 8)]
    [int]$Workers = 2,

    [int[]]$Year = @(),

    [switch]$Force
)

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$annualScript = Join-Path $PSScriptRoot "run_annual_study.ps1"
$releaseConfig = Join-Path $repositoryRoot "configs\annual\a32_turnover_budgeted_residual_v1_canonical.yaml"

$arguments = @{
    Mode = $Mode
    AnnualConfig = $releaseConfig
    Workers = $Workers
}
if ($Year.Count -gt 0) {
    $arguments["Year"] = $Year
}
if ($Force) {
    $arguments["Force"] = $true
}

& $annualScript @arguments
exit $LASTEXITCODE
