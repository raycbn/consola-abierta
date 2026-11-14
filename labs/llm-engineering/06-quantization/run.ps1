$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    python .\src\run_lab.py
    exit $LASTEXITCODE
}

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    py -3 .\src\run_lab.py
    exit $LASTEXITCODE
}

throw "Python 3.10+ was not found. Install Python, then run this script again."
