"""Per-Phase verification harness: dump CPU + GPU o32 intermediates,
compute per-Phase PSNR/diff to find where Phase 3-10 diverges."""
import os, subprocess, time
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ.get("PATH", "")
GALOSH = Path(os.environ.get("GALOSH_ROOT", str(Path(__file__).resolve().parents[1])))
EXE_CPU = GALOSH / "standalone" / "galosh_raw_cpu.exe"
EXE_GPU = GALOSH / "standalone" / "galosh_raw_gpu.exe"

import tempfile
_TMP = Path(os.environ.get("GALOSH_TMP", tempfile.gettempdir()))
CPU_DIR = _TMP / "verify_cpu"; CPU_DIR.mkdir(parents=True, exist_ok=True)
GPU_DIR = _TMP / "verify_gpu"; GPU_DIR.mkdir(parents=True, exist_ok=True)

# input frame: raw float32 Bayer, W x H (set GALOSH_VERIFY_INPUT to point at one)
INPUT = os.environ.get("GALOSH_VERIFY_INPUT", str(_TMP / "real_in.bin"))
W, H = 256, 256

# CPU run
env_cpu = os.environ.copy()
env_cpu["GALOSH_DUMP_DIR"] = str(CPU_DIR)
print("[harness] running CPU --variant=o ...")
r = subprocess.run([str(EXE_CPU), INPUT, str(CPU_DIR / "_out.bin"), str(W), str(H),
                    "galosh", "1.0", "1.0", "1.0", "0", "0", "--variant=o"],
                   env=env_cpu, capture_output=True, timeout=120)
if r.returncode != 0:
    print("CPU FAILED:", r.stderr.decode()[:500])
    raise SystemExit

# GPU run
env_gpu = os.environ.copy()
env_gpu["GALOSH_DUMP_DIR"] = str(GPU_DIR)
print("[harness] running GPU --variant=o32 ...")
r = subprocess.run([str(EXE_GPU), INPUT, str(GPU_DIR / "_out.bin"), str(W), str(H),
                    "1.0", "1.0", "1.0", "0", "0", "0", "--variant=o32"],
                   env=env_gpu, capture_output=True, timeout=120)
if r.returncode != 0:
    print("GPU FAILED:", r.stderr.decode()[:500])
    raise SystemExit

# Compare
phases = [
    ("p2_in_gat",      "Phase 2 end (in_gat full-res)",       W*H),
    ("p3_L_cs",        "Phase 3 forward L stride1",            W*H),
    ("p4_C1_h",        "Phase 4 chroma C1 (half-res)",         (W//2)*(H//2)),
    ("p4_C2_h",        "Phase 4 chroma C2 (half-res)",         (W//2)*(H//2)),
    ("p4_C3_h",        "Phase 4 chroma C3 (half-res)",         (W//2)*(H//2)),
    ("p5_pilot",       "Phase 5 Pass1 pilot (BayesShrink)",    W*H),
    ("p5_L_cs_den",    "Phase 5 Pass2 L_cs_den (Wiener)",      W*H),
    ("p6_L_pixel",     "Phase 6 L_pixel (full-res)",           W*H),
    ("p6_L_h_den",     "Phase 6 L_h_den (half-res)",           (W//2)*(H//2)),
    ("p7_C1_loess_h",  "Phase 7 C1_loess_h (LOESS half)",      (W//2)*(H//2)),
    ("p7_C1_q_up",     "Phase 7 C1_q_up (K16 q->h)",           (W//2)*(H//2)),
    ("p7_C1_e_up",     "Phase 7 C1_e_up (K16 e->q->h)",        (W//2)*(H//2)),
    ("p8_C1_h_den",    "Phase 8 C1_h_den (smoothstep blend)",  (W//2)*(H//2)),
    ("p9_C1_aligned",  "Phase 9 C1_aligned (final K16 full)",  W*H),
    ("p10_output",     "Phase 10 final denoised output",       W*H),
]

print(f"\n{'Phase':<35}  {'PSNR':>8}  {'max|d|':>9}  {'mean|d|':>9}  {'GPU mean':>9}  {'CPU mean':>9}")
print("-" * 100)
first_bad_phase = None
for name, label, n_floats in phases:
    cpu_path = CPU_DIR / f"{name}.bin"
    gpu_path = GPU_DIR / f"{name}.bin"
    if not cpu_path.exists() or not gpu_path.exists():
        print(f"{label:<35}  MISSING ({cpu_path.exists()},{gpu_path.exists()})")
        continue
    cpu = np.fromfile(cpu_path, dtype=np.float32)
    gpu = np.fromfile(gpu_path, dtype=np.float32)
    if cpu.size != n_floats or gpu.size != n_floats:
        print(f"{label:<35}  SIZE MISMATCH cpu={cpu.size} gpu={gpu.size} expected={n_floats}")
        continue
    diff = gpu - cpu
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    val_range = max(float(cpu.max()) - float(cpu.min()), 1e-6)
    # PSNR with data range = max-min of CPU
    psnr = 10 * np.log10((val_range * val_range) / max(mse, 1e-30))
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())
    flag = ""
    if psnr < 40 and first_bad_phase is None:
        flag = "  <-- FIRST DIVERGENCE"
        first_bad_phase = name
    elif psnr < 60:
        flag = "  (degraded)"
    print(f"{label:<35}  {psnr:>8.2f}  {max_abs:>9.4f}  {mean_abs:>9.5f}  {gpu.mean():>9.4f}  {cpu.mean():>9.4f}{flag}")

print()
if first_bad_phase:
    print(f"BUG located: first phase with PSNR < 40 = {first_bad_phase}")
else:
    print("All phases match within float precision (>= 40 dB).")
