# One-shot: repoint scripts + viewer at the reorganized tree (2026-07-04):
#   benchmark/raw_v2_results        -> benchmark/results_raw   (result JSONs/logs/cmp)
#   paper sources                   -> docs/paper
#   benchmark/sidd_medium_v2        -> benchmark/results_raw_sidd
#   benchmark/yuv_srgb_results_full -> benchmark/results_srgb_sidd_full
#   benchmark/yuv_srgb_results      -> benchmark/results_srgb_sidd_1024
#   benchmark/rawnind_srgb_results  -> benchmark/results_srgb_rawnind
#   benchmark/yuv_srgb_viewer       -> benchmark/viewer_srgb
#   benchmark/sidd_medium (tmp)     -> benchmark/_tmp
import io
import os

EDITS = {
 "benchmark/scripts/bench_yuv_srgb.py": [
   ('OUT_NAME   = {"sidd": "yuv_srgb_results", "rawnind": "rawnind_srgb_results"}',
    'OUT_NAME   = {"sidd": "results_srgb_sidd_1024", "rawnind": "results_srgb_rawnind"}'),
   ('TMP        = GALOSH / "benchmark" / "_yuv_bench_tmp"',
    'TMP        = GALOSH / "benchmark" / "_tmp"'),
   ('OUT = GALOSH / "benchmark" / (OUT_NAME[args.dataset] + ("_full" if not args.crop else ""))',
    'OUT = GALOSH / "benchmark" / ("results_srgb_sidd_full" if (args.dataset == "sidd" and not args.crop)\n'
    '                              else OUT_NAME[args.dataset])'),
 ],
 "benchmark/scripts/make_yuv_qualitative.py": [
   ('DIRS = {"sidd": ROOT / "yuv_srgb_results_full",       # full-frame run (paper table source)\n'
    '        "sidd_crop": ROOT / "yuv_srgb_results",       # legacy central-1024^2 run\n'
    '        "rawnind": ROOT / "rawnind_srgb_results"}',
    'DIRS = {"sidd": ROOT / "results_srgb_sidd_full",      # full-frame run (paper table source)\n'
    '        "sidd_crop": ROOT / "results_srgb_sidd_1024", # legacy central-1024^2 run\n'
    '        "rawnind": ROOT / "results_srgb_rawnind"}'),
   ('OUT  = ROOT / "raw_v2_results"   # paper directory (figures live next to the .tex)',
    'OUT  = ROOT.parent / "docs" / "paper"   # figures live next to the .tex'),
 ],
 "benchmark/scripts/make_qualitative_figure.py": [
   (r'Path(r"C:\Users\luxgrain\GALOSH\benchmark\sidd_medium_v2")',
    r'Path(r"C:\Users\luxgrain\GALOSH\benchmark\results_raw_sidd")'),
 ],
 "benchmark/scripts/bench_raw_v2_campaign.py": [
   ('GALOSH / "benchmark" / "sidd_medium_v2"',
    'GALOSH / "benchmark" / "results_raw_sidd"'),
 ],
 "benchmark/scripts/bench_sidd_medium.py": [
   ('OUTDIR     = BASE / "sidd_medium"', 'OUTDIR     = BASE / "_tmp"'),
   ('RESULTS    = BASE / "results"', 'RESULTS    = BASE / "_tmp"'),
 ],
 "benchmark/scripts/retry_failed_galosh.py": [
   ('"benchmark" / "rawnind_srgb_results"', '"benchmark" / "results_srgb_rawnind"'),
 ],
 "benchmark/scripts/merge_rawnind.py": [
   ("rawnind_srgb_results", "results_srgb_rawnind"),
 ],
 "benchmark/scripts/_sigma_source_exp.py": [
   ('Path("benchmark/rawnind_srgb_results")', 'Path("benchmark/results_srgb_rawnind")'),
 ],
 "benchmark/scripts/_revst_gpu_time.py": [
   ('"raw_v2_results" / f"_revst_gpu_time_{a.dataset}.json"',
    '"results_raw" / f"_revst_gpu_time_{a.dataset}.json"'),
 ],
 "benchmark/scripts/_fill_cpu_times.py": [
   ('"benchmark" / "raw_v2_results" / "_cpu_time_fill.json"',
    '"benchmark" / "results_raw" / "_cpu_time_fill.json"'),
 ],
 "benchmark/scripts/make_yuv_viewer.py": [
   ("yuv_srgb_results", "results_srgb_sidd_1024"),
   ("rawnind_srgb_results", "results_srgb_rawnind"),
   ("yuv_srgb_viewer", "viewer_srgb"),
 ],
 "benchmark/viewer_raw.html": [
   ("benchmark/raw_v2_results", "results_raw"),
   ("benchmark/sidd_medium_v2", "results_raw_sidd"),
 ],
}

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for f, edits in EDITS.items():
    s = io.open(f, encoding="utf-8").read()
    n = 0
    for old, new in edits:
        c = s.count(old)
        if c == 0:
            print(f"  !! NOT FOUND in {f}: {old[:60]!r}")
            continue
        s = s.replace(old, new)
        n += c
    io.open(f, "w", encoding="utf-8", newline="\n").write(s)
    print(f"{f}: {n} replacements")
