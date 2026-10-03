$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create virtual environment' }
}
$requirementsFile = if (Test-Path 'requirements-lock-windows.txt') { 'requirements-lock-windows.txt' } else { 'requirements.txt' }
& ./.venv/Scripts/python.exe -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
Write-Host 'Ready: ./.venv/Scripts/python.exe -m churn'
