from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "recovery-runbook-preflight-02-lab.zip"
INCLUDE = [
    Path("README.md"),
    Path("runbook_preflight.py"),
    Path("samples/test-failover.json"),
    Path("samples/failback.json"),
    Path("tests/test_runbook_preflight.py"),
]
with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
    for relative in INCLUDE:
        archive.write(ROOT / relative, Path("recovery-runbook-preflight-02-lab") / relative)
print(f"Built {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
