param(
    [int]$Steps = 600
)
$ErrorActionPreference = 'Stop'
$python = Get-Command python -ErrorAction Stop
$version = & $python.Source --version 2>&1
Write-Output "Python: $version"
& $python.Source -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('XPU:', bool(getattr(torch, 'xpu', None) and torch.xpu.is_available()))"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch import check failed.' }
& $python.Source -u .\src\train_tiny_gpt.py --steps $Steps
if ($LASTEXITCODE -ne 0) { throw "TinyGPT lab failed with exit code $LASTEXITCODE." }
