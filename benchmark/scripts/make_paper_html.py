"""Generate the self-contained review HTMLs (EN + JA) from the paper sources.

pandoc drops any tabular wrapped in \\resizebox (the wrapper the PDF build needs
to keep the 8-column tables inside \\columnwidth), so this script inlines the
\\input table files with the resizebox wrapper STRIPPED into a temporary .tex
and feeds that to pandoc. Figures are embedded (base64). Run from anywhere.

  python benchmark/scripts/make_paper_html.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[2])))
PAPER = ROOT / "docs" / "paper"
PANDOC = os.environ.get(
    "PANDOC",
    r"C:\Users\luxgrain\AppData\Local\Microsoft\WinGet\Packages"
    r"\JohnMacFarlane.Pandoc_Microsoft.Winget.Source_8wekyb3d8bbwe\pandoc-3.10\pandoc"
    if os.name == "nt" else "pandoc",
)


def strip_resizebox(s):
    s = s.replace("\\resizebox{\\columnwidth}{!}{%\n", "")
    s = s.replace("\\end{tabular}}", "\\end{tabular}")
    return s


def build(main_tex, out_html, lang=None):
    src = (PAPER / main_tex).read_text(encoding="utf-8")

    def inline(m):
        return strip_resizebox((PAPER / m.group(1)).read_text(encoding="utf-8"))

    src = re.sub(r"\\input\{([^}]+)\}", inline, src)
    tmp = PAPER / f"_html_{main_tex}"
    tmp.write_text(src, encoding="utf-8")
    cmd = [PANDOC, str(tmp), "-o", str(PAPER / out_html),
           "--standalone", "--embed-resources", "--mathml"]
    if lang:
        cmd += ["--metadata", f"lang={lang}"]
    r = subprocess.run(cmd, cwd=str(PAPER), capture_output=True)
    tmp.unlink(missing_ok=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", "replace")[-500:])
        raise SystemExit(1)
    print(f"{out_html} OK")


if __name__ == "__main__":
    build("galosh_paper.tex", "galosh_paper.html")
    build("galosh_paper_ja.tex", "galosh_paper_ja.html", lang="ja")
