$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Consola Abierta · LLM Engineering · Lab 03" -ForegroundColor Cyan
Write-Host "Python / PyTorch CPU training experiment"
python -m py_compile .\src\run_lab.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -u .\src\run_lab.py
exit $LASTEXITCODE
