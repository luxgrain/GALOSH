"""Render RawNIND denoised RAW outputs via simple-camera-pipeline,
compute LPIPS/DISTS/NIQE on rendered sRGB.

Per-scene metadata loaded from rawnind_full/__metadata__/<scene>.json
(stored by bench_rawnind.py with cfa_pattern, WB, CCM extracted via rawpy).
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch

os.environ["PYTHONIOENCODING"] = "utf-8"

GALOSH = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[2])))
SCP = GALOSH / "benchmark" / "external" / "simple-camera-pipeline"
sys.path.insert(0, str(SCP))
from python.pipeline import run_pipeline_v2

OUT = Path(os.environ.get("GALOSH_RAWNIND_BENCH", r"E:\img_dataset\rawnind_bench"))
RESULTS = OUT / "_metrics.json"
RESULTS_CLASSICAL = OUT / "_metrics_classical.json"
LOG = OUT / "logs" / "_rawnind_render.log"

METHODS = ["galosh_raw_cpu",
           "galosh_raw_cpu_oracle",
           "galosh_raw_gpu_32_s1",
           "galosh_raw_gpu_32_s2",
           "galosh_raw_gpu_32_s3",
           "bm3d_cfa_oracle", "nlm_cfa_oracle"]

PARAMS = {
    "input_stage": "normal",
    "output_stage": "gamma",
    "demosaic_type": "menon2007",
    "save_dtype": np.uint8,
}

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def render(raw, meta_dict):
    pipeline_meta = {
        "black_level": meta_dict["black_level"],
        "white_level": meta_dict["white_level"],
        "cfa_pattern": meta_dict["cfa_pattern"],
        "as_shot_neutral": meta_dict["as_shot_neutral"],
        "color_matrix_1": meta_dict["color_matrix_1"],
        "color_matrix_2": meta_dict["color_matrix_2"],
        "orientation": meta_dict["orientation"],
    }
    img = run_pipeline_v2(raw.astype(np.float64), params=PARAMS,
                          metadata=pipeline_meta, fix_orient=False)
    return np.clip(img, 0, 1).astype(np.float32)

def save_png(img_f32, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    arr = (np.clip(img_f32, 0, 1) * 255.0).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(str(out_path), arr)

def parse_tag(tag):
    """e.g. 'sceneABC__ISO1600' → 'sceneABC'"""
    return tag.rsplit("__ISO", 1)[0]

def step1_render():
    log("Step 1: render GT + denoised outputs")
    meta_dir = OUT / "__metadata__"
    gt_dir = OUT / "__gt_raw__"
    gt_render_dir = OUT / "__gt_raw_render__"
    nz_dir = OUT / "__noisy_raw__"
    nz_render_dir = OUT / "__noisy_raw_render__"
    gt_render_dir.mkdir(parents=True, exist_ok=True)
    nz_render_dir.mkdir(parents=True, exist_ok=True)

    # Build scene → meta cache
    scene_meta = {}
    for f in meta_dir.glob("*.json"):
        with open(f) as fp:
            scene_meta[f.stem] = json.load(fp)
    log(f"  loaded {len(scene_meta)} scene metadata")

    # Render GT (1 per scene, deduped)
    for scene, meta in scene_meta.items():
        gt_p = gt_dir / f"{scene}.npy"
        out_p = gt_render_dir / f"{scene}.png"
        if not gt_p.exists() or out_p.exists(): continue
        try:
            gt = np.load(gt_p).astype(np.float32)
            save_png(render(gt, meta), out_p)
        except Exception as e:
            log(f"  GT {scene}: {e}")

    # Render noisy + each method per tag
    n_total = 0
    for nz_npy in sorted(nz_dir.glob("*.npy")):
        tag = nz_npy.stem
        scene = parse_tag(tag)
        meta = scene_meta.get(scene)
        if meta is None: continue
        # Noisy render
        nz_out = nz_render_dir / f"{tag}.png"
        if not nz_out.exists():
            try:
                save_png(render(np.load(nz_npy).astype(np.float32), meta), nz_out)
            except Exception as e:
                log(f"  noisy {tag}: {e}")
        # Method renders
        for m in METHODS:
            in_p = OUT / m / f"{tag}.npy"
            out_p = OUT / f"{m}_render" / f"{tag}.png"
            if not in_p.exists() or out_p.exists(): continue
            try:
                save_png(render(np.load(in_p).astype(np.float32), meta), out_p)
                n_total += 1
            except Exception as e:
                log(f"  {m}/{tag}: {e}")
    log(f"  rendered {n_total} new method outputs")


def step2_perceptual():
    log("Step 2: compute LPIPS / DISTS / NIQE")
    import lpips, pyiqa
    from skimage.io import imread
    dev = torch.device("cuda")
    lpips_fn = lpips.LPIPS(net="alex").eval().to(dev)
    dists_fn = pyiqa.create_metric("dists", device=dev)
    niqe_fn = pyiqa.create_metric("niqe", device=dev)

    metrics = {}
    if RESULTS.exists():
        with open(RESULTS) as f: metrics = json.load(f)
    if RESULTS_CLASSICAL.exists():
        with open(RESULTS_CLASSICAL) as f: cls = json.load(f)
        for k, v in cls.items():
            metrics.setdefault(k, {}).update(v)
        log(f"  merged {len(cls)} classical method keys from _metrics_classical.json")

    gt_render_dir = OUT / "__gt_raw_render__"
    for m in METHODS:
        m_render = OUT / f"{m}_render"
        if not m_render.exists(): continue
        metrics.setdefault(m, {}).setdefault("per_image_rendered", {})
        log(f"  {m}: computing perceptual...")
        n_done = 0
        for den_p in sorted(m_render.glob("*.png")):
            tag = den_p.stem
            scene = parse_tag(tag)
            gt_p = gt_render_dir / f"{scene}.png"
            if not gt_p.exists(): continue
            entry = metrics[m]["per_image_rendered"].get(tag, {})
            if all(k in entry for k in ("lpips", "dists", "niqe")):
                continue
            try:
                gt = imread(str(gt_p)).astype(np.float32) / 255.0
                den = imread(str(den_p)).astype(np.float32) / 255.0
                ta = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
                tb = torch.from_numpy(den.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
                with torch.no_grad():
                    if "lpips" not in entry:
                        entry["lpips"] = float(lpips_fn(ta * 2 - 1, tb * 2 - 1).item())
                    if "dists" not in entry:
                        entry["dists"] = float(dists_fn(ta, tb).item())
                    if "niqe" not in entry:
                        entry["niqe"] = float(niqe_fn(tb).item())
                metrics[m]["per_image_rendered"][tag] = entry
                n_done += 1
            except Exception as e:
                log(f"    ERR {tag}: {e}")
            if n_done % 100 == 0 and n_done:
                with open(RESULTS, "w") as f:
                    json.dump(metrics, f, indent=2, default=str)
        log(f"  {m}: {n_done} new")
        with open(RESULTS, "w") as f:
            json.dump(metrics, f, indent=2, default=str)


def main():
    log("=" * 60)
    log("RawNIND rendering + perceptual metrics")
    log("=" * 60)
    step1_render()
    step2_perceptual()
    log("DONE")


if __name__ == "__main__":
    main()
