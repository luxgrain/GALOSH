#!/bin/bash
# 2026-07-17 noise-estimator ablation A/B/D full run (54 cells x 3 arms).
# [2026-07-19 CPU-only mode] 4070 Ti is busy (Topaz) -> LPIPS/DISTS/NIQE on
# CPU (GALOSH_METRICS_DEVICE=cpu), CPU is idle -> jobs=8, OMP capped at 6
# threads/process so 8 workers share the 32C box sanely.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONIOENCODING=utf-8
export GALOSH_METRICS_DEVICE=cpu
export OMP_NUM_THREADS=6
python benchmark/scripts/_noiseest_ablation.py --jobs 8 \
  > benchmark/results_noiseest_ablation/_campaign.log 2>&1
touch benchmark/_noiseest_20260717.done
