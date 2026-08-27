[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$SkipProbe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$configPath = Join-Path $repoRoot "configs\final_holdout_2026.yaml"
$marketScript = Join-Path $repoRoot "scripts\download_tushare.ps1"
$eligibilityScript = Join-Path $repoRoot "scripts\download_eligibility.ps1"

Push-Location $repoRoot
try {
    if (-not $Execute) {
        & $marketScript -Config $configPath -PlanOnly
        if ($LASTEXITCODE -ne 0) {
            throw "Final snapshot plan failed with exit code $LASTEXITCODE"
        }
        Write-Host ""
        Write-Host "Plan completed; no Tushare request was sent." -ForegroundColor Green
        Write-Host "The request estimate is an upper bound before Raw-cache reuse;"
        Write-Host "existing 2016-2025 request files remain reusable."
        Write-Host "Use -Execute from a clean committed worktree to build the snapshot."
        exit 0
    }

    $gitHead = & git -C $repoRoot rev-parse --verify HEAD
    if ($LASTEXITCODE -ne 0 -or -not $gitHead) {
        throw "Final snapshot execution requires a recorded Git commit."
    }
    $gitStatus = & git -C $repoRoot status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect Git status."
    }
    if ($gitStatus) {
        throw "Final snapshot execution requires a clean committed worktree."
    }

    $marketParameters = @{
        Config = $configPath
    }
    if ($SkipProbe) {
        $marketParameters["SkipProbe"] = $true
    }
    & $marketScript @marketParameters
    if ($LASTEXITCODE -ne 0) {
        throw "Final market snapshot failed with exit code $LASTEXITCODE"
    }

    & $eligibilityScript `
        -Config $configPath `
        -SkipProbe
    if ($LASTEXITCODE -ne 0) {
        throw "Final eligibility snapshot failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Final 2016-2026 Silver snapshot passed data quality gates." -ForegroundColor Green
    Write-Host "Next run validation only:"
    Write-Host ".\.venv\Scripts\python.exe -m csi500_alpha run-workflow --config configs\final_holdout_2026.yaml"
    Write-Host "Do not open frozen_test until validation artifacts have been audited."
}
finally {
    Pop-Location
}
