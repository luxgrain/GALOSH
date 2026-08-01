#!/bin/bash
# 2026-07-16 method-unification campaign v2 (relaunch after reboot).
# v1 ran 9 concurrent groups and saturated the GPU (11.4/12GB VRAM, zero
# cells completed in 35 min) -> v2 = 4 concurrent groups (the proven
# parallelism of the original campaigns).  Content is identical:
# V-BM3D family + hqdn3d on the 444 lanes (pg core+cmp, awgn) + CRVD-444.
cd "$(dirname "$0")/../.." || exit 1
S=benchmark/scripts
PG=benchmark/results_set8_pgnoise
AW=benchmark/results_set8_awgn
CR=benchmark/results_crvd

pg_pair() {  # $1=seqs $2=idx
  python $S/bench_set8_pgnoise.py --mode 444 --seqs "$1" \
    --methods vbm3d,vbm3db,hqdn3d --limit-frames 20 \
    --tag "c444_vb$2" > "$PG/_c444_vb$2.log" 2>&1
  python $S/bench_set8_pgnoise.py --mode 444 --seqs "$1" \
    --methods vbm3d,vbm3db,vbm3dbg,hqdn3d --limit-frames 20 \
    --compress-crf 23 --tag "m444_vb$2" > "$PG/_m444_vb$2.log" 2>&1
}

( pg_pair tractor,touchdown 1; pg_pair park_joy,sunflower 2 ) &
( pg_pair hypersmooth,motorbike 3; pg_pair rafting,snowboard 4 ) &
(
  python $S/bench_set8_baselines.py --mode 444 \
    --seqs tractor,park_joy,hypersmooth,rafting --sigmas 10,20,30,40,50 \
    --methods vbm3d,vbm3db,hqdn3d --tag vb1 > "$AW/_444_vb1.log" 2>&1
  python $S/bench_set8_baselines.py --mode 444 \
    --seqs touchdown,sunflower,motorbike,snowboard --sigmas 10,20,30,40,50 \
    --methods vbm3d,vbm3db,hqdn3d --tag vb2 > "$AW/_444_vb2.log" 2>&1
) &
(
  python $S/bench_crvd.py --mode 444 --scenes 1,2,3,4 --tag c1 \
    > "$CR/_444_c1.log" 2>&1
  python $S/bench_crvd.py --mode 444 --scenes 5,6,7,8 --tag c2 \
    > "$CR/_444_c2.log" 2>&1
  python $S/bench_crvd.py --mode 444 --scenes 9,10,11 --tag c3 \
    > "$CR/_444_c3.log" 2>&1
) &

wait
touch benchmark/_unify_20260716.done
