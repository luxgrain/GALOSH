"""STRUCTURAL LaTeX check for the paper sources (no LaTeX toolchain needed).
Scope: \\input files exist, \\ref targets defined, \\cite keys present, figures
exist, environments balanced. It does NOT verify table numbers against the
benchmark JSONs -- use verify_table_numbers.py for that.
\\input and \\includegraphics paths are resolved relative to the main .tex
file's directory, so this runs from anywhere:
  python benchmark/scripts/check_paper_consistency.py docs/paper/galosh_paper.tex"""
import re
import sys
from pathlib import Path

main_file = Path(sys.argv[1] if len(sys.argv) > 1 else "galosh_paper.tex")
base = main_file.parent
main = main_file.read_text(encoding="utf-8")
full = main
ok = True
for m in re.finditer(r"\\input\{([^}]+)\}", main):
    p = base / m.group(1)
    print(f"input {p}: {'OK' if p.exists() else 'MISSING'}")
    if p.exists():
        full += p.read_text(encoding="utf-8")
    else:
        ok = False

labels = set(re.findall(r"\\label\{([^}]+)\}", full))
refs = set(re.findall(r"\\ref\{([^}]+)\}", full))
cites = {c.strip() for cs in re.findall(r"\\cite\{([^}]+)\}", full) for c in cs.split(",")}
bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", full))
issues = {
    "undefined refs": sorted(refs - labels),
    "unused labels": sorted(labels - refs),
    "missing bib": sorted(cites - bibs),
    "uncited bib": sorted(bibs - cites),
}
for k, v in issues.items():
    print(f"{k:15s}: {v if v else 'none'}")
    if v and k in ("undefined refs", "missing bib"):
        ok = False

for m in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", full):
    exists = (base / m.group(1)).exists()
    print(f"figure {m.group(1)}: {'OK' if exists else 'MISSING'}")
    ok = ok and exists

for env in ["table", "tabular", "figure\\*", "abstract", "itemize", "document"]:
    b = len(re.findall(r"\\begin\{" + env + r"\}", full))
    e = len(re.findall(r"\\end\{" + env + r"\}", full))
    print(f"env {env:9s}: begin={b} end={e} {'OK' if b == e else 'MISMATCH'}")
    ok = ok and b == e

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
