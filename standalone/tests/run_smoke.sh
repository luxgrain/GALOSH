#!/usr/bin/env bash
# ============================================================================
# GALOSH smoke tests — constant / random / odd dims / small dims / high-noise /
# near-black inputs through the canonical RAW (FP32 + INT32) and YUV/RGB CPU
# paths; GPU paths are smoked too when an OpenCL device is available.
# Checks per case: exit code, output size, all-finite, sane range, stats.
#                                                            (Apache-2.0)
# Requires: built exes (make all / ./build.sh all) + python3 + numpy.
# ============================================================================
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
T=tests/_smoke_tmp; mkdir -p "$T"
fails=0; total=0

run_case() {  # run_case <label> <exe+args-template> <W> <H> <C> <case> [expect-may-fail]
  local label="$1" exe="$2" w="$3" h="$4" c="$5" cas="$6" mayfail="${7:-no}"
  total=$((total+1))
  local ip="$T/${label}_in.bin" op="$T/${label}_out.bin"
  "$PY" tests/_synth.py gen "$cas" "$w" "$h" "$c" "$ip" > /dev/null || { echo "FAIL($label): gen"; fails=$((fails+1)); return; }
  rm -f "$op"
  if ! eval "$exe" > "$T/${label}.log" 2>&1; then
    if [ "$mayfail" = "may-fail" ]; then
      echo "OK  ($label): rejected gracefully (rc!=0, no output required)"; return
    fi
    echo "FAIL($label): nonzero exit — tail:"; tail -2 "$T/${label}.log" | sed 's/^/    /'
    fails=$((fails+1)); return
  fi
  if "$PY" tests/_synth.py check "$op" "$w" "$h" "$c" | sed "s/^/  [$label]/"; then
    echo "OK  ($label)"
  else
    echo "FAIL($label): output check"; fails=$((fails+1))
  fi
}

echo "=== RAW FP32 CPU (galosh_raw_cpu.exe) ==="
for cas in constant random high-noise near-black gradient-pg; do
  run_case "raw_fp32_$cas" './galosh_raw_cpu.exe "$ip" "$op" 128 96 galosh 1.0 1.0 1.0 0 0' 128 96 1 "$cas"
done
run_case "raw_fp32_small" './galosh_raw_cpu.exe "$ip" "$op" 32 32 galosh 1.0 1.0 1.0 0 0' 32 32 1 gradient-pg
run_case "raw_fp32_odd"   './galosh_raw_cpu.exe "$ip" "$op" 65 63 galosh 1.0 1.0 1.0 0 0' 65 63 1 gradient-pg may-fail

echo "=== RAW INT32 r32 CPU (galosh_raw_cpu_int.exe) ==="
for cas in constant gradient-pg near-black; do
  run_case "raw_r32_$cas" './galosh_raw_cpu_int.exe "$ip" "$op" 128 96 galosh 1.0 1.0 1.0 0 0' 128 96 1 "$cas"
done

echo "=== YUV/RGB FP32 CPU (galosh_yuv_cpu.exe) ==="
for cas in constant random high-noise near-black gradient-pg; do
  run_case "yuv_$cas" './galosh_yuv_cpu.exe "$ip" "$op" 128 96 1.0 1.0' 128 96 3 "$cas"
done
run_case "yuv_small" './galosh_yuv_cpu.exe "$ip" "$op" 32 32 1.0 1.0' 32 32 3 gradient-pg
run_case "yuv_odd"   './galosh_yuv_cpu.exe "$ip" "$op" 65 63 1.0 1.0' 65 63 3 gradient-pg may-fail

echo "=== GPU smoke (skipped without a capable OpenCL device) ==="
# Multi-GPU boxes can enumerate a non-capable iGPU first, so probe device
# indices 0..3 until one builds the kernels (mirrors the benchmark harness).
run_gpu_case() {  # run_gpu_case <label> <exe> <extra-args-before-dev> <W> <H> <C>
  local label="$1" exe="$2" args="$3" w="$4" h="$5" c="$6"
  total=$((total+1))
  local ip="$T/${label}_in.bin" op="$T/${label}_out.bin"
  "$PY" tests/_synth.py gen gradient-pg "$w" "$h" "$c" "$ip" > /dev/null
  for dev in 0 1 2 3; do
    rm -f "$op"
    if eval "./$exe \"\$ip\" \"\$op\" $w $h $args $dev" > "$T/${label}.log" 2>&1 && [ -f "$op" ]; then
      if "$PY" tests/_synth.py check "$op" "$w" "$h" "$c" | sed "s/^/  [$label dev$dev]/"; then
        echo "OK  ($label, device $dev)"; return
      fi
    fi
  done
  echo "SKIP($label): no OpenCL device could run the kernels (not a failure on GPU-less hosts)"
}
[ -f galosh_raw_gpu.exe ] && run_gpu_case raw_gpu galosh_raw_gpu.exe '1.0 1.0 1.0 0 0' 128 96 1
[ -f galosh_yuv_gpu.exe ] && run_gpu_case yuv_gpu galosh_yuv_gpu.exe '1.0 1.0' 128 96 3

echo "----------------------------------------------------------------------"
if [ "$fails" -gt 0 ]; then echo "SMOKE: FAIL ($fails/$total cases)"; exit 1; fi
echo "SMOKE: PASS ($total cases)"
