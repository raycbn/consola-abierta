$ErrorActionPreference = 'Stop'
$python = Get-Command python -ErrorAction Stop
Write-Output "Python: $(& $python.Source --version 2>&1)"
& $python.Source -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('XPU:', bool(getattr(torch, 'xpu', None) and torch.xpu.is_available()))"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch import check failed.' }
& $python.Source -u .\src\run_lab.py
if ($LASTEXITCODE -ne 0) { throw "Self-Attention lab failed with exit code $LASTEXITCODE." }
