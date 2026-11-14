$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python .\src\run_lab.py --output .\reports
if ($LASTEXITCODE -ne 0) { throw "Benchmark trace analysis failed with exit code $LASTEXITCODE" }
Write-Host "Lab complete. Review .\reports\lab-report.md and .\reports\metrics.json"
