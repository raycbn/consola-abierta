from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "recovery-evidence-pack-03-lab.zip"
INCLUDE = [
    Path("README.md"), Path("evidence_pack.py"), Path("samples/exercise.json"),
    Path("samples/evidence/EV-001-chronology.csv"),
    Path("samples/evidence/EV-002-network-review.txt"),
    Path("samples/evidence/EV-003-service-acceptance.txt"),
    Path("tests/test_evidence_pack.py"), Path("templates/aar-ip-template.md"),
    Path("templates/evidence-register-template.csv"),
]
with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
    for relative in INCLUDE:
        archive.write(ROOT / relative, Path("recovery-evidence-pack-03-lab") / relative)
print(f"Built {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
