[CmdletBinding()]
param(
    [string]$Config = "configs/full.yaml",
    [switch]$PlanOnly,
    [switch]$SkipProbe,
    [string]$RefreshNamesFrom,
    [switch]$Force
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

if ([System.IO.Path]::IsPathRooted($Config)) {
    $configPath = [System.IO.Path]::GetFullPath($Config)
}
else {
    $configPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Config))
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Config file does not exist: $configPath"
}

function Invoke-ProjectCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
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
    Write-Host "Config:     $configPath"

    Invoke-ProjectCommand `
        -Label "Offline eligibility supplement plan" `
        -Arguments @("-m", "csi500_alpha", "plan-eligibility", "--config", $configPath)

    if ($PlanOnly) {
        Write-Host ""
        Write-Host "Plan-only mode completed; no Tushare request was sent." -ForegroundColor Green
        exit 0
    }

    $doctorArguments = @("-m", "csi500_alpha", "doctor", "--config", $configPath)
    if (-not $SkipProbe) {
        $doctorArguments += "--probe"
    }
    Invoke-ProjectCommand `
        -Label "Environment, credential, and connectivity check" `
        -Arguments $doctorArguments

    $downloadArguments = @(
        "-m",
        "csi500_alpha",
        "download-eligibility",
        "--config",
        $configPath
    )
    if ($Force) {
        Write-Warning "Force mode bypasses valid request caches."
        $downloadArguments += "--force"
    }
    if ($RefreshNamesFrom) {
        $downloadArguments += @("--refresh-names-from", $RefreshNamesFrom)
    }
    Invoke-ProjectCommand `
        -Label "Resumable historical-name and resumption download" `
        -Arguments $downloadArguments

    Write-Host ""
    Write-Host "Eligibility supplements completed successfully." -ForegroundColor Green
    Write-Host "If interrupted later, rerun this command without -Force."
}
finally {
    Pop-Location
}
