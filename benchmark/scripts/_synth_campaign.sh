#!/bin/bash
# 2026-07-17 generator-fidelity campaign: CRVD clean + pgnoise-v2 synthetic
# noise (CRVD-measured per-ISO a,b), full unified 420 method set.
# Real-noise twin = the existing results_crvd 420 table; diff = fidelity.
# 3 parallel jobs (within the 4-job GPU lesson).
cd "$(dirname "$0")/../.." || exit 1
S=benchmark/scripts
CR=benchmark/results_crvd
python $S/bench_crvd.py --mode 420 --scenes 1,2,3,4 --synth imx385 \
  --tag c1 > "$CR/_synth_c1.log" 2>&1 &
python $S/bench_crvd.py --mode 420 --scenes 5,6,7,8 --synth imx385 \
  --tag c2 > "$CR/_synth_c2.log" 2>&1 &
python $S/bench_crvd.py --mode 420 --scenes 9,10,11 --synth imx385 \
  --tag c3 > "$CR/_synth_c3.log" 2>&1 &
wait
touch benchmark/_synth_20260717.done
