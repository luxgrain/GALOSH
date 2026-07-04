"""Build the PUBLIC comparison-viewer bundle (benchmark/viewer_public/).

Published layout (three GitHub Pages sites, to stay under the 1 GB/site limit):
  https://luxgrain.github.io/GALOSH/                       <- gh-pages branch:
      index.html + results_raw/cmp/** (raw sets, ~880 MB)
  https://luxgrain.github.io/GALOSH-viewer-sidd-srgb/      <- asset repo (SIDD sRGB)
  https://luxgrain.github.io/GALOSH-viewer-rawnind-srgb/   <- asset repo (RawNIND sRGB)
The sRGB <img> URLs in index.html are absolute to the asset repos; raw ones
are relative. All images: JPEG quality 92, chroma 4:4:4 (denoising comparisons
live and die on chroma, so no 4:2:0).

Local outputs under benchmark/viewer_public/:
  index.html                        4-dataset viewer (full-res links stripped)
  results_raw/cmp/<ds>/<scene>/     raw sets (file set + sizes mirror the
                                    legacy cmp thumbnails; pixels re-encoded
                                    from the full-res PNG artifacts)
  results_srgb/cmp/<ds>/<scene>/    sRGB sets (from results_srgb_* PNGs;
                                    SIDD full frames resized to 1600 px wide)

Cached: existing JPEGs are skipped, so reruns only encode new scenes.
  python benchmark/scripts/make_viewer_public.py
"""
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageFile
# high-entropy (noisy) frames overflow the default JPEG encode buffer with optimize=True
ImageFile.MAXBLOCK = 64 * 1024 * 1024

GALOSH = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[2])))
CMP = GALOSH / "benchmark" / "results_raw" / "cmp"
OUT = GALOSH / "benchmark" / "viewer_public"
QUALITY, SUBSAMPLING = 92, 0

RAW_SRC = {
    "sidd_medium": Path(os.environ.get("GALOSH_SIDD_ARTIFACTS",
                        str(GALOSH / "benchmark" / "results_raw_sidd" / "_artifacts"))),
    "rawnind": Path(os.environ.get("GALOSH_RAWNIND_ARTIFACTS",
                    str(GALOSH / "benchmark" / "results_raw_rawnind" / "_artifacts"))),
}
SRGB_SETS = {
    "sidd_srgb": (GALOSH / "benchmark" / "results_srgb_sidd_full", 1600),
    "rawnind_srgb": (GALOSH / "benchmark" / "results_srgb_rawnind", None),
}
SRGB_METHODS = [
    ("_noisy", "Noisy (input)", "noisy"),
    ("_gt", "Ground truth", "gt"),
    ("galosh_yuv_gpu_fp32", "GALOSH-YUV (ours)", "galosh_yuv_gpu_fp32"),
    ("cbm3d", "CBM3D", "cbm3d"),
    ("color_nlm", "Color-NLM", "color_nlm"),
    ("guided", "Guided filter", "guided"),
    ("nafnet", "NAFNet-SIDD (DL)", "nafnet"),
    ("scunet", "SCUNet-real (DL)", "scunet"),
    ("restormer", "Restormer-SIDD (DL)", "restormer"),
]
ASSET_BASE = {
    "sidd_srgb": "https://luxgrain.github.io/GALOSH-viewer-sidd-srgb/",
    "rawnind_srgb": "https://luxgrain.github.io/GALOSH-viewer-rawnind-srgb/",
}
DS_NAMES = {
    "sidd_medium": "SIDD Medium — raw (full frame)",
    "rawnind": "RawNIND — raw (512² crops)",
    "sidd_srgb": "SIDD Medium — sRGB (full frame)",
    "rawnind_srgb": "RawNIND — sRGB (512² crops)",
}


def encode_one(job):
    src, dst, maxw, wh = job
    if dst.exists():
        return "cached"
    if not src.exists():
        return f"MISSING {src}"
    try:
        im = Image.open(src).convert("RGB")
        if wh and im.size != wh:
            im = im.resize(wh, Image.LANCZOS)
        elif maxw and im.size[0] > maxw:
            im = im.resize((maxw, round(im.size[1] * maxw / im.size[0])), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=QUALITY, subsampling=SUBSAMPLING, optimize=True)
        return "ok"
    except Exception as e:
        return f"ERR {dst}: {str(e)[:80]}"


def raw_jobs():
    for ds, art in RAW_SRC.items():
        for jp in (CMP / ds).rglob("*.jpg"):
            stem = jp.stem
            if stem.endswith("_c") or stem.endswith("_t"):
                continue   # figure-crop leftovers; the viewer never references them
            with Image.open(jp) as im:
                wh = im.size
            yield (art / jp.parent.name / f"{stem}.png",
                   OUT / "results_raw" / "cmp" / ds / jp.parent.name / f"{stem}.jpg",
                   None, wh)


def srgb_jobs():
    for ds, (root, maxw) in SRGB_SETS.items():
        for scene in sorted(p.stem for p in (root / "gt").glob("*.png")):
            for _, _, fn in SRGB_METHODS:
                yield (root / fn / f"{scene}.png",
                       OUT / "results_srgb" / "cmp" / ds / scene / f"{fn}.jpg",
                       maxw, None)


def build_index():
    src = (GALOSH / "benchmark" / "viewer_raw.html").read_text(encoding="utf-8")
    start = src.find("const D=")
    jstart = start + len("const D=")
    D, jlen = json.JSONDecoder().raw_decode(src[jstart:])
    # Rebuild the raw datasets from the encoded directories: the legacy JSON
    # capped RawNIND raw at 196 (high-ISO) scenes, but the published viewer
    # serves the FULL sets. Method order + labels are taken from the legacy JSON.
    label_order = {ds["key"]: [(im["k"], im["label"]) for im in ds["scenes"][0]["imgs"]]
                   for ds in D["datasets"]}
    rebuilt = []
    for key in ("sidd_medium", "rawnind"):
        root = OUT / "results_raw" / "cmp" / key
        scenes = []
        for sd in sorted(p for p in root.iterdir() if p.is_dir()):
            gt = sd / "_gt.jpg"
            if not gt.exists():
                continue
            with Image.open(gt) as im:
                w, h = im.size
            imgs = [{"k": k, "label": lbl,
                     "img": f"results_raw/cmp/{key}/{sd.name}/{k}.jpg"}
                    for k, lbl in label_order[key] if (sd / f"{k}.jpg").exists()]
            scenes.append({"scene": sd.name, "w": w, "h": h, "imgs": imgs})
        rebuilt.append({"name": DS_NAMES[key], "key": key, "scenes": scenes})
    D["datasets"] = rebuilt
    for key in SRGB_SETS:
        root = OUT / "results_srgb" / "cmp" / key
        scenes = []
        for sd in sorted(p for p in root.iterdir() if p.is_dir()):
            gt = sd / "gt.jpg"
            if not gt.exists():
                continue
            with Image.open(gt) as im:
                w, h = im.size
            imgs = [{"k": k, "label": lbl,
                     "img": f"{ASSET_BASE[key]}{sd.name}/{fn}.jpg"}
                    for k, lbl, fn in SRGB_METHODS if (sd / f"{fn}.jpg").exists()]
            scenes.append({"scene": sd.name, "w": w, "h": h, "imgs": imgs})
        D["datasets"].append({"name": DS_NAMES[key], "key": key, "scenes": scenes})
    out = src[:start] + "const D=" + json.dumps(D, ensure_ascii=False) + src[jstart + jlen:]
    # published bundle has no full-res assets: strip the link and its usage hint
    out = out.replace('<a class="open" href="${im.full}" target="_blank">full‑res ↗</a>', "")
    out = out.replace(" · click tile = full-res PNG", "")
    # the source viewer is raw-only; the published one is multi-domain
    out = out.replace("GALOSH RAW — synchronized", "GALOSH — synchronized")
    h1_end = out.find("</h1>")
    inject = ('<div class="hint">interactive comparison viewer (raw + sRGB, both '
              'benchmarks) — <a href="https://github.com/luxgrain/GALOSH" '
              'style="color:#8ab4ff">github.com/luxgrain/GALOSH</a> · JPEG q92, '
              'chroma 4:4:4; full-res PNG artifacts are regenerable from the repo '
              'benchmarks</div>')
    out = out[:h1_end + 5] + inject + out[h1_end + 5:]
    (OUT / "index.html").write_text(out, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"index.html written ({len(D['datasets'])} datasets)")


def main():
    jobs = list(raw_jobs()) + list(srgb_jobs())
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
