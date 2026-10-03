$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& ./.venv/Scripts/python.exe -m churn
if ($LASTEXITCODE -ne 0) { throw 'Pipeline failed' }
& ./.venv/Scripts/python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
