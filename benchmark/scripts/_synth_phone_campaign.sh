#!/bin/bash
# 2026-07-17 arm-2: GENERIC phone curve, sigma-matched per CRVD ISO
# (_synth_phone_match.json).  Launch AFTER the imx385 arm finishes.
cd "$(dirname "$0")/../.." || exit 1
S=benchmark/scripts
CR=benchmark/results_crvd
python $S/bench_crvd.py --mode 420 --scenes 1,2,3,4 --synth phone-match \
  --tag p1 > "$CR/_synthphone_p1.log" 2>&1 &
python $S/bench_crvd.py --mode 420 --scenes 5,6,7,8 --synth phone-match \
  --tag p2 > "$CR/_synthphone_p2.log" 2>&1 &
python $S/bench_crvd.py --mode 420 --scenes 9,10,11 --synth phone-match \
  --tag p3 > "$CR/_synthphone_p3.log" 2>&1 &
wait
touch benchmark/_synthphone_20260717.done
