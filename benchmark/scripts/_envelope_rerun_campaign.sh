#!/bin/bash
# 2026-07-19 envelope-canonical GALOSH-row rerun (4070Ti hands-off).
#
# Scope: the 4 GALOSH rows (cpu-fit/cpu-hold/vk-fit/vk-hold) on all 4
# datasets x 2 lanes, after the estimator switch (envelope canonical,
# ablation +0.39dB/-0.056 LPIPS vs MAD) + the 420 GPU adapter clip fix.
# Everything runs WITHOUT the 4070 Ti: vk rows on the Intel Arc A310
# (same device the original campaign used — deterministic sort picked
# Arc), metrics (LPIPS/DISTS/NIQE) on CPU via GALOSH_METRICS_DEVICE=cpu.
#
# Shard/merge conventions (verified against the report scripts):
#   awgn : bench_set8_video untagged FULL rerun rewrites _metrics_{420,444}
#          .json completely (those files hold only galosh+noisy — correct).
#   pg   : tags must match ^c(420|444)_.+ (core) / ^m(420|444)_.+ (cmp) and
#          sort AFTER the per-seq shards -> zzenv suffix tag.
#   crvd : tag e1 (> c1..c4, bgcopy in sorted order; deep method merge).
# Pre-rerun values snapshotted in _ARCHIVE/pre_envelope_20260719/.
#
# Frame counts mirror the original runs: awgn=full (DERF_CAP 85),
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
  > "$AW/_env420_rerun.log" 2>&1 &
J1=$!
python $S/bench_set8_video.py --mode 444 \
  > "$AW/_env444_rerun.log" 2>&1 &
J2=$!
python $S/bench_set8_pgnoise.py --mode 420 --methods "$GAL" \
  --limit-frames 20 --tag c420_zzenv > "$PG/_c420_zzenv.log" 2>&1 &
J3=$!
python $S/bench_set8_pgnoise.py --mode 420 --methods "$GAL" \
  --limit-frames 20 --compress-crf 23 --tag m420_zzenv \
  > "$PG/_m420_zzenv.log" 2>&1 &
J4=$!
python $S/bench_set8_pgnoise.py --mode 444 --methods "$GAL" \
  --limit-frames 20 --tag c444_zzenv > "$PG/_c444_zzenv.log" 2>&1 &
J5=$!
(
  python $S/bench_set8_pgnoise.py --mode 444 --methods "$GAL" \
    --limit-frames 20 --compress-crf 23 --tag m444_zzenv \
    > "$PG/_m444_zzenv.log" 2>&1
  python $S/bench_crvd.py --mode 420 --methods "$GAL" --tag e1 \
    > "$CR/_420_e1.log" 2>&1
  python $S/bench_crvd.py --mode 444 --methods "$GAL" --tag e1 \
    > "$CR/_444_e1.log" 2>&1
) &
J6=$!

wait $J1 $J2 $J3 $J4 $J5 $J6

# reports + viewers (idempotent; deep merges pick up the new shards)
python $S/report_set8.py          > benchmark/_env_report_set8.log 2>&1
python $S/report_set8_pgnoise.py  > benchmark/_env_report_pg.log 2>&1
python $S/report_crvd.py          > benchmark/_env_report_crvd.log 2>&1
python $S/make_set8_viewer.py         >> benchmark/_env_report_set8.log 2>&1
python $S/make_set8_pgnoise_viewer.py >> benchmark/_env_report_pg.log 2>&1
python $S/make_crvd_viewer.py         >> benchmark/_env_report_crvd.log 2>&1

touch benchmark/_envelope_rerun_20260719.done
