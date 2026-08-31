[CmdletBinding()]
param(
    [string]$Spec = "configs/factor_audit_v2.yaml",
    [switch]$PlanOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $pythonExecutable = $venvPython
}
else {
    $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
}

if ([System.IO.Path]::IsPathRooted($Spec)) {
    $specPath = [System.IO.Path]::GetFullPath($Spec)
}
else {
    $specPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Spec))
}
if (-not (Test-Path -LiteralPath $specPath -PathType Leaf)) {
    throw "Factor-audit configuration does not exist: $specPath"
}

function Invoke-ProjectCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $pythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Push-Location $repoRoot
try {
    Write-Host "Repository: $repoRoot"
    Write-Host "Python:     $pythonExecutable"
    Write-Host "Audit:      $specPath"

    Invoke-ProjectCommand `
        -Label "Offline factor-audit plan" `
        -Arguments @("-m", "csi500_alpha", "plan-factor-audit", "--spec", $specPath)

    if ($PlanOnly) {
        Write-Host ""
        Write-Host "Plan-only mode completed; no data was materialized." -ForegroundColor Green
        exit 0
    }

    Invoke-ProjectCommand `
        -Label "Point-in-time factor quality and return audit" `
        -Arguments @("-m", "csi500_alpha", "run-factor-audit", "--spec", $specPath)

    Write-Host ""
    Write-Host "Factor audit completed successfully." -ForegroundColor Green
}
finally {
    Pop-Location
}
