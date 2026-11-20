from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent
archive = root / "recovery-test-scorecard-01-lab.zip"
files = [
    root / "README.md",
    root / "recovery_scorecard.py",
    root / "sample-test.json",
    root / "tests" / "test_recovery_scorecard.py",
]
with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zf:
    for file in files:
        zf.write(file, file.relative_to(root))
print(f"Created {archive} ({archive.stat().st_size} bytes)")
