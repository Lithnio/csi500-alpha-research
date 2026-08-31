[CmdletBinding()]
param(
    [string]$Spec = "configs/financial_core.yaml",
    [string]$BaseConfig = "configs/full.yaml",
    [switch]$PlanOnly,
    [switch]$SkipProbe,
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

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}

$specPath = Resolve-ProjectPath $Spec
$baseConfigPath = Resolve-ProjectPath $BaseConfig
foreach ($path in @($specPath, $baseConfigPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Configuration file does not exist: $path"
    }
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
    Write-Host "Repository:  $repoRoot"
    Write-Host "Python:      $pythonExecutable"
    Write-Host "Financial:   $specPath"
    Write-Host "Base config: $baseConfigPath"

    Invoke-ProjectCommand `
        -Label "Offline financial-data plan" `
        -Arguments @("-m", "csi500_alpha", "plan-financial", "--spec", $specPath)

    if ($PlanOnly) {
        Write-Host ""
        Write-Host "Plan-only mode completed; no Tushare request was sent." -ForegroundColor Green
        exit 0
    }

    $doctorArguments = @("-m", "csi500_alpha", "doctor", "--config", $baseConfigPath)
    if (-not $SkipProbe) {
        $doctorArguments += "--probe"
    }
    Invoke-ProjectCommand `
        -Label "Environment, credential, and connectivity check" `
        -Arguments $doctorArguments

    $downloadArguments = @(
        "-m",
        "csi500_alpha",
        "download-financial",
        "--spec",
        $specPath
    )
    if ($Force) {
        Write-Warning "Force mode bypasses valid request caches."
        $downloadArguments += "--force"
    }
    Invoke-ProjectCommand `
        -Label "Resumable financial download and point-in-time validation" `
        -Arguments $downloadArguments

    Write-Host ""
    Write-Host "Financial dataset completed successfully." -ForegroundColor Green
    Write-Host "If interrupted later, rerun this command without -Force."
}
finally {
    Pop-Location
}
