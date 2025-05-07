# Script to create Python virtual environments for projects
# Uses uv for package installation
# Requires Python 3.10

$projects = @("automind", "api")
$pythonVersion = "3.10"

function Setup-VirtualEnvironment {
    param (
        [string]$projectName
    )

    Write-Host "Setting up environment for $projectName..." -ForegroundColor Cyan

    # Check if project directory exists, create if it doesn't
    if (-not (Test-Path $projectName)) {
        Write-Host "Creating directory for $projectName..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $projectName | Out-Null
    }

    # Navigate to project directory
    Push-Location $projectName

    # Check if .venv already exists
    if (Test-Path ".venv") {
        Write-Host ".venv already exists for $projectName. Skipping creation." -ForegroundColor Yellow
    }
    else {
        # Create virtual environment with Python 3.10
        Write-Host "Creating .venv with Python $pythonVersion..." -ForegroundColor Green
        
        # Try to find Python 3.10
        $pythonCmd = "python"
        
        # Check if we can find Python 3.10 specifically
        if (Get-Command "python$pythonVersion" -ErrorAction SilentlyContinue) {
            $pythonCmd = "python$pythonVersion"
        }
        elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
            # On Windows, try using the py launcher with version specification
            $pythonCmd = "py -$pythonVersion"
        }

        # Create the virtual environment
        # Invoke-Expression "$pythonCmd -m venv .venv"
        Invoke-Expression "uv venv --python $pythonVersion .venv"
        
        if (-not (Test-Path ".venv")) {
            Write-Host "Failed to create virtual environment for $projectName. Please ensure Python $pythonVersion is installed." -ForegroundColor Red
            Pop-Location
            return
        }
    }

    # Activate the virtual environment
    if ($IsWindows -or $PSVersionTable.PSVersion.Major -lt 6) {
        # Windows PowerShell
        & .\.venv\Scripts\activate 
    }
    else {
        # PowerShell Core on Linux/MacOS
        & ./.venv/Scripts/activate 
    }

    # Install uv if not already installed
    Write-Host "Installing/ensuring uv is available..." -ForegroundColor Green
    
    if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
        pip install uv
    }

    # Check for requirements.txt and install packages if it exists
    # if (Test-Path "requirements.txt") {
    #     Write-Host "Installing packages from requirements.txt using uv..." -ForegroundColor Green
    #     uv pip install -r requirements.txt
    # }
    # else {
    #     Write-Host "No requirements.txt found for $projectName." -ForegroundColor Yellow
    #
    #     # Create empty requirements.txt file
    #     "" | Out-File -FilePath "requirements.txt"
    #     Write-Host "Created empty requirements.txt file." -ForegroundColor Yellow
    # }

    uv sync

    # Deactivate virtual environment
    if ($IsWindows -or $PSVersionTable.PSVersion.Major -lt 6) {
        deactivate
    }
    else {
        # On some systems/configurations
        if (Get-Command "deactivate" -ErrorAction SilentlyContinue) {
            deactivate
        }
    }

    # Return to original directory
    Pop-Location
    
    Write-Host "Setup complete for $projectName." -ForegroundColor Green
    Write-Host "------------------------------------" -ForegroundColor Gray
}

# Main script execution
Write-Host "Starting virtual environment setup for projects..." -ForegroundColor Magenta

# Check if Python 3.10 is available
$pythonFound = $false

if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $version = python --version | Out-String
    if ($version -match "Python 3\.10\.") {
        $pythonFound = $true
    }
}

if (Get-Command "python3.10" -ErrorAction SilentlyContinue) {
    $pythonFound = $true
}

if (Get-Command "py" -ErrorAction SilentlyContinue) {
    $version = py -3.10 --version 2>$null | Out-String
    if ($version -match "Python 3\.10\.") {
        $pythonFound = $true
    }
}

if (-not $pythonFound) {
    Write-Host "Warning: Python $pythonVersion was not found. Please ensure it's installed and in your PATH." -ForegroundColor Red
    Write-Host "Attempting to continue, but the script may fail..." -ForegroundColor Yellow
}

# Setup environments for each project
foreach ($project in $projects) {
    Setup-VirtualEnvironment -projectName $project
}

Write-Host "All virtual environments have been set up!" -ForegroundColor Magenta
