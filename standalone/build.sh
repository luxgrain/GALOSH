#!/usr/bin/env bash
# ============================================================================
# GALOSH — portable build script (no-make fallback / restricted-TEMP hosts).
#                                                             (code: Apache-2.0)
# Same canonical targets as the Makefile, but guarantees a writable TMPDIR for
# the compiler (some CI / sandboxed hosts force the child TEMP to a read-only
# location, which breaks `make`). On a normal machine `make` works too.
#
#   ./build.sh              # = all: RAW + YUV CPU references
#   ./build.sh raw|yuv|gpu|all|test|bench-small|check-no-int64
#
# Toolchain: C99 + OpenMP; GPU targets also need OpenCL (headers + libOpenCL).
# On Windows use the MSYS2 ucrt64 gcc:  export PATH=/c/msys64/ucrt64/bin:$PATH
# ============================================================================
set -eu
cd "$(dirname "$0")"

# guarantee a writable temp dir for the compiler
if [ -z "${TMPDIR:-}" ] || [ ! -w "${TMPDIR:-/nonexistent}" ]; then
  for d in /tmp /c/tmp "$PWD/.buildtmp"; do
    if mkdir -p "$d" 2>/dev/null && [ -w "$d" ]; then export TMPDIR="$d" TMP="$d" TEMP="$d"; break; fi
  done
fi

CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O3 -fopenmp -std=gnu11}"
LM="-lm"; LCL="-lOpenCL"

_cc() { echo "  $CC $CFLAGS $1 -o $2 ${3:-}"; $CC $CFLAGS "$1" -o "$2" ${3:-} $LM; }

build_raw() { _cc galosh_raw_cpu.c     galosh_raw_cpu.exe     ""; _cc galosh_raw_cpu_int.c galosh_raw_cpu_int.exe ""; }
build_yuv() { _cc galosh_yuv_cpu.c     galosh_yuv_cpu.exe     ""; }
build_gpu() { _cc galosh_raw_gpu.c     galosh_raw_gpu.exe     "$LCL"; _cc galosh_yuv_gpu.c galosh_yuv_gpu.exe "$LCL"; }

case "${1:-all}" in
  raw)             build_raw ;;
  yuv)             build_yuv ;;
  gpu)             build_gpu ;;
  all)             build_raw; build_yuv ;;
  test)            build_raw; build_yuv; bash tests/run_smoke.sh ;;
  bench-small)     build_raw; build_yuv; bash tests/run_bench_small.sh ;;
  check-no-int64)  bash check_no_int64.sh ;;
  *) echo "usage: $0 {raw|yuv|gpu|all|test|bench-small|check-no-int64}" >&2; exit 2 ;;
esac
echo "build.sh: ${1:-all} done"
