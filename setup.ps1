# Script to create Python virtual environments for projects
# Uses uv for package installation
# Requires Python 3.10

$projects = @("automind", "api")
# $pythonCmd = "python"
$pythonVersion = "3.10"
$pythonFound = $false
        
if (Get-Command "python$pythonVersion" -ErrorAction SilentlyContinue)
{
  # $pythonCmd = "python$pythonVersion"
  $pythonFound = $true
} elseif (Get-Command "py" -ErrorAction SilentlyContinue)
{
  # $pythonCmd = "py -$pythonVersion"
  $pythonFound = $true
}

function Initialize-VirtualEnvironment
{
  param (
    [string]$ProjectName
  )

  Write-Host "Setting up environment for $ProjectName..." -ForegroundColor Cyan

  # Check if project directory exists, create if it doesn't
  if (-not (Test-Path $ProjectName))
  {
    Write-Host "Creating directory for $ProjectName..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $ProjectName | Out-Null
  }

  # Navigate to project directory
  Push-Location $ProjectName

  # Check if .venv already exists
  if (Test-Path ".venv")
  {
    Write-Host ".venv already exists for $ProjectName. Skipping creation." -ForegroundColor Yellow
  } else
  {
    # Create virtual environment with Python 3.10
    Write-Host "Creating .venv with Python $pythonVersion..." -ForegroundColor Green
        

    # Create the virtual environment
    # Invoke-Expression "$pythonCmd -m venv .venv"
    Invoke-Expression "uv venv --python $pythonVersion .venv"
        
    if (-not (Test-Path ".venv"))
    {
      Write-Host "Failed to create virtual environment for $ProjectName. Please ensure Python $pythonVersion is installed." -ForegroundColor Red
      Pop-Location
      return
    }
  }

  # Activate the virtual environment
  if ($IsWindows -or $PSVersionTable.PSVersion.Major -lt 6)
  {
    # Windows PowerShell
    & .\.venv\Scripts\activate 
  } else
  {
    # PowerShell Core on Linux/MacOS
    & ./.venv/Scripts/activate 
  }

  # Install uv if not already installed
  Write-Host "Installing/ensuring uv is available..." -ForegroundColor Green
    
  if (-not (Get-Command "uv" -ErrorAction SilentlyContinue))
  {
    pip install uv
  }

  # Check for requirements.txt and install packages if it exists
  # if (Test-Path "requirements.txt") {
  #     Write-Host "Installing packages from requirements.txt using uv..." -ForegroundColor Green
  #     uv pip install -r requirements.txt
  # }
  # else {
  #     Write-Host "No requirements.txt found for $ProjectName." -ForegroundColor Yellow
  #
  #     # Create empty requirements.txt file
  #     "" | Out-File -FilePath "requirements.txt"
  #     Write-Host "Created empty requirements.txt file." -ForegroundColor Yellow
  # }

  uv sync

  # Deactivate virtual environment
  if ($IsWindows -or $PSVersionTable.PSVersion.Major -lt 6)
  {
    deactivate
  } else
  {
    # On some systems/configurations
    if (Get-Command "deactivate" -ErrorAction SilentlyContinue)
    {
      deactivate
    }
  }

  # Return to original directory
  Pop-Location
    
  Write-Host "Setup complete for $ProjectName." -ForegroundColor Green
  Write-Host "------------------------------------" -ForegroundColor Gray
}

# Main script execution
Write-Host "Starting virtual environment setup for projects..." -ForegroundColor Magenta


if (-not $pythonFound)
{
  Write-Host "Warning: Python $pythonVersion was not found. Please ensure it's installed and in your PATH." -ForegroundColor Red
  Write-Host "Attempting to continue, but the script may fail..." -ForegroundColor Yellow
}

# Setup environments for each project
foreach ($project in $projects)
{
  Initialize-VirtualEnvironment -ProjectName $project
}

Write-Host "All virtual environments have been set up!" -ForegroundColor Magenta
