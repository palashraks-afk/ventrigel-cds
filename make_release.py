"""
Build the source archive that gets attached to the Zenodo record.

    python make_release.py

A Zenodo record should be self-contained: someone who downloads it in ten years,
after the GitHub repository has moved or vanished, should be able to reproduce
every number in the paper from what is in the archive. This builds that archive
and then proves the property rather than assuming it, by extracting to a clean
directory and running the test suite and the manuscript checker from inside it.

What is included, and why
-------------------------

``deprecated/`` is included deliberately. The correction notice claims the v1
classifier's label was a deterministic function of its own input features, and
says that is checkable in under a minute by anyone with the code open. Shipping
the archive without that code would make the claim unverifiable, which is the
opposite of what a correction is for.

What is excluded, and why
-------------------------

The trial's own PDFs are not redistributed here. The JACC article is open access
under CC BY-NC-ND and the repository does include it, but re-hosting a
third-party paper inside an archive published under someone else's name is a
different act from keeping it alongside the code for convenience. It is cited
instead, with its DOI, in ``SOURCES.md``.

Serialized model binaries are excluded because they are large and reproducible
from the scripts. Virtual environments, caches and LaTeX intermediates are
excluded for the obvious reasons.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

VERSION = "2.0"
ROOT = Path(__file__).parent
OUT = ROOT / "submission" / f"ventrigel-cds-v{VERSION}-source.zip"

SKIP_DIRS = {
    ".git", "venv", "__pycache__", ".github", ".ipynb_checkpoints",
    "submission", ".claude",
    "Ventri PDF", "Supplemental Material",  # third-party source documents
}
SKIP_EXT = {".pyc", ".joblib", ".aux", ".log", ".out", ".synctex.gz", ".fls", ".fdb_latexmk"}
SKIP_FILES = {"Supplemental Material.pdf", OUT.name}


def build() -> list[str]:
    OUT.parent.mkdir(exist_ok=True)
    included: list[str] = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if fn in SKIP_FILES or p.suffix in SKIP_EXT:
                    continue
                rel = p.relative_to(ROOT)
                z.write(p, Path(f"ventrigel-cds-v{VERSION}") / rel)
                included.append(str(rel))
    return included


def verify() -> bool:
    """Extract to a clean directory and run the checks from inside it."""
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(OUT) as zf:
            zf.extractall(td)
        root = Path(td) / f"ventrigel-cds-v{VERSION}"

        required = [
            "run_analysis.py", "make_figures.py", "test_ventrigel.py",
            "check_manuscript.py", "requirements.txt", "LICENSE", "README.md",
            "CORRECTION.md", "SOURCES.md", "paper/ventrigel_enrichment.tex",
            "paper/ventrigel_enrichment.pdf", "ventrigel/literature.py",
            "deprecated/synthetic_generator.py",
        ]
        missing = [f for f in required if not (root / f).exists()]
        if missing:
            print("  MISSING from archive:", ", ".join(missing))
            return False
        print(f"  all {len(required)} required paths present")

        ok = True
        for script, label in (
            ("test_ventrigel.py", "test suite"),
            ("check_manuscript.py", "manuscript checker"),
        ):
            r = subprocess.run(
                [sys.executable, script], cwd=root, capture_output=True, text=True, timeout=1800
            )
            last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
            print(f"  {label}: {last}")
            ok = ok and r.returncode == 0
        return ok


def main() -> int:
    print(f"Building {OUT.name}")
    files = build()
    size = OUT.stat().st_size
    digest = hashlib.md5(OUT.read_bytes()).hexdigest()
    print(f"  {len(files)} files, {size / 1e6:.2f} MB")
    print(f"  md5: {digest}")

    print("\nVerifying the archive is self-sufficient:")
    ok = verify()
    print("\n" + ("Archive is reproducible standalone." if ok else "ARCHIVE FAILED VERIFICATION."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
