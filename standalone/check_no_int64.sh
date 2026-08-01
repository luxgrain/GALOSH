#!/usr/bin/env bash
# ============================================================================
# check-no-int64 : fail if the INT / fixed-point SHIPPING path uses a native
# 64-bit (or floating) type.                                   (Apache-2.0)
# ----------------------------------------------------------------------------
# SCOPE (locked): the INT16 / INT32 fixed-point shipping path ONLY. A real ISP
# datapath is natively fixed-point, so the shipping INT path must express any
# value that needs >32 bits with paired-INT32 / mul-hi / split-multiply, never a
# native 64-bit type.
#
# INTENTIONALLY OUT OF SCOPE (not scanned):
#   * the FP32 offline REFERENCE  (galosh_cpu.h, galosh_raw_cpu.c,
#     galosh_yuv_cpu.c, and the o32 FP32 kernels in galosh.cl) -- double is the
#     reference numeric type there, by design;
#   * diagnostic HP probes (*_hp.{c,h}) -- these deliberately use int64/__int128
#     for high-precision diagnostics and are isolated by filename;
#   * tests, the benchmark harness, host-side file I/O, and profiling/timing.
#
# Per-line exemptions inside a scanned file (profiling / timing / pixel-count
# I/O, all off the algorithm datapath) are recognised by pattern or by an
# explicit  /* no64-exempt: <reason> */  marker on the line.
# ============================================================================
set -u
cd "$(dirname "$0")"

# ---- INT fixed-point SHIPPING files (the scan set) ----
SHIP=(
  galosh_cpu_int.h
  galosh_raw_cpu_int.c
  galosh_int_p0.cl galosh_int_p1.cl galosh_int_p2.cl galosh_int_p3.cl
  galosh_int_p4.cl galosh_int_p5.cl galosh_int_p6.cl galosh_int_p7.cl
  galosh_int_p8.cl galosh_int_p10.cl galosh_int_i16.cl
)

# forbidden native 64-bit / floating tokens (C sources)
PAT_C='int64_t|uint64_t|long long|unsigned long long|long double|__int128|\bdouble\b'
# OpenCL: `long`/`ulong` are 64-bit; `double` needs cl_khr_fp64
PAT_CL='\blong\b|\bulong\b|\bdouble\b'

# per-line exemptions: profiling counters, timing, and explicit markers
EXEMPT='clock\(\)|CLOCKS_PER_SEC|g_n_mac|g_n_sf|no64-exempt'

viol=0
for f in "${SHIP[@]}"; do
  [ -f "$f" ] || { echo "  (skip missing $f)"; continue; }
  case "$f" in *.cl) pat="$PAT_CL";; *) pat="$PAT_C";; esac
  while IFS= read -r line; do
    ln=${line%%:*}; text=${line#*:}
    # strip comments (// and inline /* */) and block-comment body lines, then
    # re-test: if the token only lived in a comment it is not a real use.
    code=$(printf '%s' "$text" | sed -E 's#//.*$##; s#/\*[^*]*\*/##g; s#^[[:space:]]*\*.*$##')
    printf '%s' "$code" | grep -qE "$pat" || continue
    printf '%s' "$text" | grep -qE "$EXEMPT" && continue
    printf '  VIOLATION %s:%s: %s\n' "$f" "$ln" "$(printf '%s' "$text" | sed 's/^[[:space:]]*//')"
    viol=$((viol+1))
  done < <(grep -nE "$pat" "$f")
done

echo "----------------------------------------------------------------------"
if [ "$viol" -gt 0 ]; then
  echo "check-no-int64: FAIL — $viol violation(s) in the INT fixed-point shipping path"
  exit 1
fi
echo "check-no-int64: PASS — INT fixed-point shipping path is free of native 64-bit types"
exit 0
