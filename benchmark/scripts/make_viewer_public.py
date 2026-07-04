"""Build the PUBLIC comparison-viewer bundle (benchmark/viewer_public/).

The bundle is what the gh-pages branch serves at https://luxgrain.github.io/GALOSH/:
  index.html            viewer_raw.html with the full-res links stripped
                        (full-res PNG artifacts are not published) and a
                        header link back to the repo
  results_raw/cmp/**    the full comparison set re-encoded as JPEG quality 92,
                        chroma 4:4:4 (denoising comparisons live and die on
                        chroma, so no 4:2:0), sized as in the viewer JSON
  .nojekyll             serve paths verbatim

Sources: the existing cmp JPEGs define the file set + display size; pixels are
re-encoded from the full-res PNG artifacts (GALOSH_SIDD_ARTIFACTS /
GALOSH_RAWNIND_ARTIFACTS). Cached: existing outputs are skipped, so reruns
only encode new scenes.

  python benchmark/scripts/make_viewer_public.py
"""
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

GALOSH = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[2])))
CMP = GALOSH / "benchmark" / "results_raw" / "cmp"
OUT = GALOSH / "benchmark" / "viewer_public"
SRC = {
    "sidd_medium": Path(os.environ.get("GALOSH_SIDD_ARTIFACTS",
                        str(GALOSH / "benchmark" / "results_raw_sidd" / "_artifacts"))),
    "rawnind": Path(os.environ.get("GALOSH_RAWNIND_ARTIFACTS",
                    "benchmark/results_raw_rawnind/_artifacts")),
}
QUALITY = 92          # ~visually transparent at 1600 px; 95 costs ~1.6x size
SUBSAMPLING = 0       # 4:4:4 -- keep chroma at full resolution


def encode_one(job):
    ds, scene, name, w, h = job
    src = SRC[ds] / scene / f"{name}.png"
    dst = OUT / "results_raw" / "cmp" / ds / scene / f"{name}.jpg"
    if dst.exists():
        return "cached"
    if not src.exists():
        return f"MISSING {ds}/{scene}/{name}"
    try:
        im = Image.open(src).convert("RGB")
        if im.size != (w, h):
            im = im.resize((w, h), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=QUALITY, subsampling=SUBSAMPLING, optimize=True)
        return "ok"
    except Exception as e:
        return f"ERR {ds}/{scene}/{name}: {str(e)[:80]}"


def build_index():
    src = (GALOSH / "benchmark" / "viewer_raw.html").read_text(encoding="utf-8")
    anchor = '<a class="open" href="${im.full}" target="_blank">full‑res ↗</a>'
    src = src.replace(anchor, "")
    h1_end = src.find("</h1>")
    inject = ('<div class="hint">interactive comparison viewer — '
              '<a href="https://github.com/luxgrain/GALOSH" '
              'style="color:#8ab4ff">github.com/luxgrain/GALOSH</a> · '
              'JPEG q92, chroma 4:4:4; full-res PNG artifacts are regenerable '
              'from the repo benchmarks</div>')
    src = src[:h1_end + 5] + inject + src[h1_end + 5:]
    (OUT / "index.html").write_text(src, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print("index.html + .nojekyll written")


def main():
    jobs = []
    for ds in SRC:
        for jp in (CMP / ds).rglob("*.jpg"):
            stem = jp.stem
            if stem.endswith("_c") or stem.endswith("_t"):
                continue   # figure-crop leftovers; the viewer never references them
            with Image.open(jp) as im:
                w, h = im.size
            jobs.append((ds, jp.parent.name, stem, w, h))
    print(f"jobs: {len(jobs)}", flush=True)
    ok = cached = 0
    bad = []
    with ProcessPoolExecutor(max_workers=20) as ex:
        for f in as_completed([ex.submit(encode_one, j) for j in jobs]):
            r = f.result()
            if r == "ok":
                ok += 1
            elif r == "cached":
                cached += 1
            else:
                bad.append(r)
    print(f"encode: ok={ok} cached={cached} bad={len(bad)}")
    for b in bad[:10]:
        print(" ", b)
    build_index()
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
