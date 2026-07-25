#!/bin/bash
# 2026-07-26 sigma_C radius rule + one-sided MAP-ridge eps GALOSH-row rerun.
#
# Scope: the 4 GALOSH rows (cpu-fit/cpu-hold/vk-fit/vk-hold) on the 420
# lanes ONLY (awgn-420, pg core-420, pg cmp-420, crvd-420) after flipping
# the canonical defaults (galosh_yuv420.h GALOSH_YUV420_SIGC_T /
# EPS_ANCHOR / EPS_HI; regression envs GALOSH_YUV420_RADIUS_SRC=lin /
# GALOSH_YUV420_EPS_SRC=const).  The 444 lanes are untouched by the change
# (both rules live inside the 420 half-res chroma branch) and are NOT rerun.
# vk rows on the Intel Arc A310 = same device as the 2026-07-19 envelope
# baseline (clean delta attribution); metrics on CPU.
#
# Shard/merge conventions (same as _envelope_rerun_campaign.sh):
#   awgn : bench_set8_video untagged FULL rerun rewrites _metrics_420.json
#          completely (holds only galosh+noisy — correct).
#   pg   : tag must match ^c420_.+ / ^m420_.+ and sort AFTER c420_zzenv /
#          m420_zzenv -> zzsigc ("zzs" > "zze").
#   crvd : tag f1 (> e1 > c1..c4 in sorted order; deep method merge).
# Pre-rerun values snapshotted in _ARCHIVE/pre_sigc_20260726/.
#
# Frame counts mirror the envelope rerun: awgn=full (DERF_CAP 85),
# pg=--limit-frames 20, crvd=NFR 7 (script constant).  PNG stays ON.
cd "$(dirname "$0")/../.." || exit 1
S=benchmark/scripts
AW=benchmark/results_set8_awgn
PG=benchmark/results_set8_pgnoise
CR=benchmark/results_crvd
export GALOSH_METRICS_DEVICE=cpu
export GALOSH_VK_DEVICE=A310
export OMP_NUM_THREADS=8
export PYTHONIOENCODING=utf-8
GAL=galosh-cpu-fit,galosh-cpu-hold,galosh-vk-fit,galosh-vk-hold

python $S/bench_set8_video.py --mode 420 \
  > "$AW/_sigc420_rerun.log" 2>&1 &
J1=$!
python $S/bench_set8_pgnoise.py --mode 420 --methods "$GAL" \
  --limit-frames 20 --tag c420_zzsigc > "$PG/_c420_zzsigc.log" 2>&1 &
J2=$!
python $S/bench_set8_pgnoise.py --mode 420 --methods "$GAL" \
  --limit-frames 20 --compress-crf 23 --tag m420_zzsigc \
  > "$PG/_m420_zzsigc.log" 2>&1 &
J3=$!
# crvd is launched FIRST (standalone smoke) by the operator; keep it here
# for a from-scratch rerun of the whole campaign.
if [ "${SKIP_CRVD:-0}" != "1" ]; then
  python $S/bench_crvd.py --mode 420 --methods "$GAL" --tag f1 \
    > "$CR/_420_f1.log" 2>&1 &
  J4=$!
else
  J4=""
fi

wait $J1 $J2 $J3 $J4

# reports + viewers (idempotent; deep merges pick up the new shards)
python $S/report_set8.py          > benchmark/_sigc_report_set8.log 2>&1
python $S/report_set8_pgnoise.py  > benchmark/_sigc_report_pg.log 2>&1
python $S/report_crvd.py          > benchmark/_sigc_report_crvd.log 2>&1
python $S/make_set8_viewer.py         >> benchmark/_sigc_report_set8.log 2>&1
python $S/make_set8_pgnoise_viewer.py >> benchmark/_sigc_report_pg.log 2>&1
python $S/make_crvd_viewer.py         >> benchmark/_sigc_report_crvd.log 2>&1

touch benchmark/_sigc_rerun_20260726.done
