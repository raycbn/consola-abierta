from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "dr-programme-coverage-auditor-04-lab.zip"
INCLUDE = [
    Path("README.md"),
    Path("programme_audit.py"),
    Path("samples/services.csv"),
    Path("samples/exercises.csv"),
    Path("samples/actions.csv"),
    Path("samples/exceptions.csv"),
    Path("tests/test_programme_audit.py"),
]
with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
    for relative in INCLUDE:
        archive.write(ROOT / relative, Path("dr-programme-coverage-auditor-04-lab") / relative)
print(f"Built {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
