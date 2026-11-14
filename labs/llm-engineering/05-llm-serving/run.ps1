$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m py_compile .\src\run_lab.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -u .\src\run_lab.py
exit $LASTEXITCODE
