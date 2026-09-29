[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  throw "uv is required and must be available on PATH."
}

foreach ($project in @("automind", "api")) {
  $projectPath = Join-Path $scriptRoot $project
  if (-not (Test-Path (Join-Path $projectPath "pyproject.toml"))) {
    throw "Missing project configuration: $projectPath"
  }

  Write-Host "Syncing $project with its lockfile..." -ForegroundColor Cyan
  Push-Location $projectPath
  try {
    uv sync --locked --dev
    if ($LASTEXITCODE -ne 0) {
      throw "uv sync failed for $project (exit code $LASTEXITCODE)."
    }
  } finally {
    Pop-Location
  }
}

Write-Host "Core environments are ready." -ForegroundColor Green
