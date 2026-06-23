/* ============================================================================
 *  galosh_cpu_int.h
 *
 *  Integer (fixed-point) primitives for the GALOSH-RAW INT pipeline.
 *
 *  Stage 1 = r32 (CPU INT32 reference, Q11.20 storage, INT64 NEVER used).
 *  Stage 2 = r16 (CPU INT16, per-variable Q-format, future).
 *
 *  Design intent:
 *    - Pure INT32 ALU.  No INT64 type appears in any source line.  Where
 *      a 64-bit intermediate is mathematically required (= mul-hi, multi-
 *      term Gauss-Hermite quadrature, IRLS 5x5 elimination), we emulate
 *      it via paired (hi32, lo32) INT32 words and split-multiply
 *      ("mul_hi pattern"), so the codebase stays portable to ISP silicon
 *      whose ALU is 32-bit only.
 *    - Q-format = Q11.20 (1 sign + 11 integer + 20 fractional, scale = 2^20).
 *      Range ±2048.0 in real space, precision 1/2^20 ≈ 9.5e-7 (uniform).
 *      Comfortably covers all intermediate values in the GALOSH-RAW pipeline
 *      (max observed = LUT D ≈ 270 in normalized GAT space).
 *    - Verified target: vs CPU FP32 reference (galosh_raw_cpu.c --variant=o)
 *      ≥ 40 dB PSNR per Phase, ≥ 35 dB end-to-end on s00_p00 (worst-case).
 *
 *  Not yet covered by this header (lives in galosh_raw_cpu_int.c):
 *    - Phase 0 noise estimation pipeline (histograms, Huber WLS)
 *    - Phase 1-10 main loop
 *    - Per-CFA dark IRLS (5x5 fxp_acc Gauss elimination)
 *    - LOESS / K16 joint bilateral upsample
 *
 *  NOT INCLUDED in this header:
 *    - Anything specific to OpenCL / GPU.  GPU r16 port lives in galosh.cl.
 *    - FP32 helpers (already in galosh_cpu.h).  This header does not include
 *      galosh_cpu.h to avoid accidental FP fallback inside an integer
 *      function.
 *
 *  Reference for review:
 *    - Q-format design discussion: 2026-05-10 conversation transcript
 *    - INT64-avoidance rationale: ISP silicon 32-bit ALU portability
 * ========================================================================== */
#ifndef GALOSH_CPU_INT_H
#define GALOSH_CPU_INT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ============================================================================
 * Q-format definitions (Q11.20)
 * ========================================================================== */

typedef int32_t fxp32;                /* Q11.20 fixed-point */

#define FXP_FRAC       20
#define FXP_ONE        ((fxp32)1 << FXP_FRAC)            /* = 1048576 */
#define FXP_HALF       ((fxp32)1 << (FXP_FRAC - 1))      /* = 524288  */
#define FXP_FRAC_MASK  (FXP_ONE - 1)
#define FXP_MAX_INT    ((fxp32)0x7FFFFFFF)               /* +max */
#define FXP_MIN_INT    ((fxp32)0x80000000)               /* -max */

/* Range check macros (debug only).  Real range covers ±2047.999999. */
#define FXP_MAX_REAL   2047.999f
#define FXP_MIN_REAL  -2048.0f

/* Convert FP32 ↔ Q11.20.
 *
 * Note: FP32 → fxp32 uses round-to-nearest-even by virtue of (int) cast on
 * the rounded float value (we add 0.5f after adjusting sign).  This matches
 * the FP32 reference's behaviour for typical values; for values exceeding
 * FXP_MAX_REAL, saturation happens silently (range is checked to be safe
 * for our pipeline). */
static inline fxp32 fxp_from_float(float f) {
  /* Saturate first to prevent UB on large values. */
  if(f >=  FXP_MAX_REAL) return FXP_MAX_INT;
  if(f <=  FXP_MIN_REAL) return FXP_MIN_INT;
  return (fxp32)(f * (float)FXP_ONE + (f >= 0.0f ? 0.5f : -0.5f));
}

static inline float fxp_to_float(fxp32 x) {
  return (float)x * (1.0f / (float)FXP_ONE);
}

/* ============================================================================
 * Multi-precision accumulator (fxp_acc) — INT64 emulation via paired INT32.
 *
 * Represents a signed 64-bit integer as { hi: int32, lo: uint32 }.
 *   value = (int64)hi * 2^32 + (uint32)lo, sign-extended through hi.
 *
 * Used for places that mathematically need >32-bit precision:
 *   - Gauss-Hermite quadrature in LUT build (10-term ∑ weight × value)
 *   - IRLS 5x5 matrix Gauss elimination (entry × entry × cofactor chain)
 *   - Sum-of-squares accumulators across multi-block reductions
 *
 * NOT used for Q11.20 single-pair multiplication; that uses fxp_mul (split).
 * ========================================================================== */

typedef struct fxp_acc {
  int32_t  hi;
  uint32_t lo;
} fxp_acc;

static inline fxp_acc fxp_acc_zero(void) { fxp_acc a = { 0, 0u }; return a; }

/* Set acc = (int32 sign-extended).  Lo = value as unsigned, hi = -1 if neg. */
static inline fxp_acc fxp_acc_from_int32(int32_t v) {
  fxp_acc a;
  a.lo = (uint32_t)v;
  a.hi = (v < 0) ? -1 : 0;
  return a;
}

/* acc += v   (signed add of int32 sign-extended into 64-bit).
 * Carry propagates from lo overflow into hi. */
static inline void fxp_acc_add_i32(fxp_acc *a, int32_t v) {
  uint32_t lo_old = a->lo;
  uint32_t v_lo = (uint32_t)v;
  a->lo = lo_old + v_lo;
  /* Sign-extend v into upper word and add carry. */
  int32_t v_hi = (v < 0) ? -1 : 0;
  uint32_t carry = (a->lo < lo_old) ? 1u : 0u;
  a->hi += v_hi + (int32_t)carry;
}

/* acc -= v */
static inline void fxp_acc_sub_i32(fxp_acc *a, int32_t v) {
  uint32_t lo_old = a->lo;
  uint32_t v_lo = (uint32_t)v;
  a->lo = lo_old - v_lo;
  int32_t v_hi = (v < 0) ? -1 : 0;
  uint32_t borrow = (a->lo > lo_old) ? 1u : 0u;
  a->hi -= v_hi + (int32_t)borrow;
}

/* acc += a*b   (full 64-bit mul-add via 16-bit split, INT64 NEVER used).
 *
 * a, b are INT32.  Their product is mathematically a 64-bit signed integer.
 * We compute it via 4 unsigned 16-bit sub-products (each fits in uint32),
 * combine into a (hi, lo) unsigned 64-bit, then negate if signs disagree.
 *
 * Caveat: -INT32_MIN is undefined; we assume callers don't pass INT32_MIN.
 * For our pipeline this is safe (typical magnitudes < 2^28).
 */
static inline void fxp_acc_madd(fxp_acc *acc, int32_t a, int32_t b) {
  int sign = ((a < 0) ^ (b < 0)) ? -1 : 1;
  uint32_t ua = (uint32_t)((a < 0) ? -a : a);
  uint32_t ub = (uint32_t)((b < 0) ? -b : b);

  uint32_t a_hi = ua >> 16, a_lo = ua & 0xFFFFu;
  uint32_t b_hi = ub >> 16, b_lo = ub & 0xFFFFu;

  uint32_t hh = a_hi * b_hi;       /* max 0x7FFF * 0xFFFF ≈ 2^31, fits uint32 */
  uint32_t hl = a_hi * b_lo;       /* fits */
  uint32_t lh = a_lo * b_hi;       /* fits */
  uint32_t ll = a_lo * b_lo;       /* fits: max 0xFFFE0001 */

  /* Combine: full_64 = (hh << 32) + ((hl + lh) << 16) + ll. */
  uint32_t mid = hl + lh;                            /* may wrap uint32 */
  uint32_t mid_carry_to_hi = (mid < hl) ? 1u : 0u;   /* carry-out of hl+lh */

  uint32_t lo_part = ll + (mid << 16);
  uint32_t lo_carry = (lo_part < ll) ? 1u : 0u;

  uint32_t hi_part = hh + (mid >> 16) + (mid_carry_to_hi << 16) + lo_carry;

  if(sign < 0) {
    /* Negate the (hi, lo) pair (two's complement of 64-bit value). */
    lo_part = ~lo_part + 1u;
    hi_part = ~hi_part + ((lo_part == 0u) ? 1u : 0u);
  }

  /* Add to acc. */
  uint32_t old_lo = acc->lo;
  acc->lo = old_lo + lo_part;
  uint32_t add_carry = (acc->lo < old_lo) ? 1u : 0u;
  acc->hi += (int32_t)hi_part + (int32_t)add_carry;
}

/* Convert acc → fxp32 WITHOUT shifting — for accumulators of Q-format
 * additions (vs sum-of-products via madd which require >> FXP_FRAC).
 *
 * Saturates to int32 if hi != sign-extension of lo.  Use this when summing
 * a list of Q11.20 values (e.g., block-mean accumulator), NOT when summing
 * products (use fxp_acc_to_fxp32 for that pattern). */
static inline fxp32 fxp_acc_extract_q20(const fxp_acc *acc) {
  int32_t expected_hi = ((int32_t)acc->lo < 0) ? -1 : 0;
  if(acc->hi != expected_hi) return (acc->hi < 0) ? FXP_MIN_INT : FXP_MAX_INT;
  return (fxp32)acc->lo;
}

/* Convert acc → fxp32 by right-shifting by FXP_FRAC and saturating to int32.
 *
 * The acc holds the full 64-bit product (in raw scaled units).  After
 * shifting right by FXP_FRAC = 20, the result should fit in int32 for
 * any value within the Q11.20 representable range.  Saturate otherwise. */
static inline fxp32 fxp_acc_to_fxp32(const fxp_acc *acc) {
  /* Compose the shifted value:
   *   result_lo = acc.lo >> FXP_FRAC | (acc.hi << (32 - FXP_FRAC)).
   *   result_hi = acc.hi >> FXP_FRAC.
   * For final fxp32 we need result_hi to be 0 (positive) or -1 (negative)
   * (= proper sign-extension of the 32-bit result), otherwise saturate. */
  uint32_t lo_shift = (acc->lo >> FXP_FRAC) | ((uint32_t)acc->hi << (32 - FXP_FRAC));
  int32_t  hi_shift = acc->hi >> FXP_FRAC;       /* arithmetic shift */
  int32_t  expected_hi = ((int32_t)lo_shift < 0) ? -1 : 0;
  if(hi_shift != expected_hi) {
    /* Overflow — saturate. */
    return (acc->hi < 0) ? FXP_MIN_INT : FXP_MAX_INT;
  }
  return (fxp32)lo_shift;
}

/* Forward declarations (definitions below). */
static inline fxp32 fxp_div_q20(fxp32 num, fxp32 den);

/* ============================================================================
 * fxp_acc_div_i32  — divide paired-int32 accumulator by an int32 divisor.
 *
 * Computes (hi:lo) / div, returning an int32 result.  No Q-format shifting
 * applied — the result has the same scaling as the input acc relative to
 * the divisor.  Used in WLS to compute mean_x = Σ(w_int × x_q20) / Σw_int,
 * where the Q11.20 raw is carried inside the acc (so the returned int32
 * is Q11.20 of the mean).
 *
 * Why this exists:
 *   Σ(w_int × FXP_ONE) over many bins overflows int32 (= silent saturate in
 *   fxp_acc_extract_q20).  Solution: keep weight sum as plain int32 (which
 *   fits up to 2^31 = 2 billion blocks), keep weighted-value sums as
 *   fxp_acc (64-bit), and divide via paired long division.  Output mean
 *   fits int32 by construction of the WLS problem.
 *
 * Algorithm: 64-bit bit-by-bit long division, INT32 ALU only.
 *
 * paired-int32 (hi:lo) を int32 div で除算。WLS の Σw 飽和問題を回避するため、
 * 重み合計を plain int32 / 重み×値合計を fxp_acc で保持し、本ヘルパーで除算。
 * ========================================================================== */
static inline int32_t fxp_acc_div_i32(const fxp_acc *acc, int32_t div) {
  if(div == 0) return (acc->hi < 0) ? FXP_MIN_INT : FXP_MAX_INT;
  int sign = (((acc->hi < 0) ? 1 : 0) ^ ((div < 0) ? 1 : 0)) ? -1 : 1;

  /* Absolute value of paired-int32 (two's complement negation). */
  uint32_t hi_abs, lo_abs;
  if(acc->hi < 0) {
    hi_abs = ~(uint32_t)acc->hi;
    lo_abs = ~acc->lo + 1u;
    if(lo_abs == 0) hi_abs += 1;
  } else {
    hi_abs = (uint32_t)acc->hi;
    lo_abs = acc->lo;
  }
  uint32_t div_abs = (div < 0) ? (~(uint32_t)div + 1u) : (uint32_t)div;

  /* Long division: process bits 63..0 of dividend (hi:lo). */
  uint32_t r = 0, q_hi = 0, q_lo = 0;
  for(int i = 0; i < 64; i++) {
    /* Shift (hi:lo) left, extract top bit. */
    uint32_t top = hi_abs >> 31;
    hi_abs = (hi_abs << 1) | (lo_abs >> 31);
    lo_abs = lo_abs << 1;
    /* r << 1 may temporarily overflow uint32; OK because r < div < 2^31. */
    r = (r << 1) | top;
    /* Shift quotient left. */
    q_hi = (q_hi << 1) | (q_lo >> 31);
    q_lo = q_lo << 1;
    if(r >= div_abs) {
      r -= div_abs;
      q_lo |= 1u;
    }
  }
  /* Saturate if quotient exceeds int32. */
  if(q_hi != 0 || q_lo > (uint32_t)FXP_MAX_INT) {
    return (sign < 0) ? FXP_MIN_INT : FXP_MAX_INT;
  }
  return (sign < 0) ? -(int32_t)q_lo : (int32_t)q_lo;
}

/* ============================================================================
 * fxp_acc_div_acc  — divide paired-int32 numerator by paired-int32 denominator,
 *                    returning Q11.20 result.
 *
 * Computes (num_acc × 2^20) / den_acc as int32 (Q11.20).  Used in WLS to
 * compute α = Σ(w·xc·yc) / Σ(w·xc²), where both sums are 64-bit and the
 * ratio is the slope in real space (= multiplied by 2^20 for Q11.20).
 *
 * Algorithm: shift both numerator and denominator down by the same amount
 * (preserving the ratio) until both fit int32, then use the existing
 * fxp_div_q20 long division.  The shift count is determined by the larger
 * absolute value.
 *
 * Cost: ~12-20 op shift-detect + one fxp_div_q20 call (~500 ops).  Once per
 * WLS iteration, negligible.
 * ========================================================================== */
static inline fxp32 fxp_acc_div_acc(const fxp_acc *num, const fxp_acc *den) {
  /* Take abs of both. */
  fxp_acc na = *num, da = *den;
  int sign = (((na.hi < 0) ? 1 : 0) ^ ((da.hi < 0) ? 1 : 0)) ? -1 : 1;
  if(na.hi < 0) {
    uint32_t lo2 = ~na.lo + 1u;
    na.hi = (int32_t)(~(uint32_t)na.hi + (lo2 == 0 ? 1 : 0));
    na.lo = lo2;
  }
  if(da.hi < 0) {
    uint32_t lo2 = ~da.lo + 1u;
    da.hi = (int32_t)(~(uint32_t)da.hi + (lo2 == 0 ? 1 : 0));
    da.lo = lo2;
  }
  /* Find shift K so both fit int32 (top bit of (hi:lo) at position ≤30). */
  int k = 0;
  while(k < 63) {
    uint32_t na_hi_u = (uint32_t)na.hi;
    uint32_t da_hi_u = (uint32_t)da.hi;
    /* Both fit int32 if hi == 0 and lo's top bit clear. */
    if(na_hi_u == 0 && da_hi_u == 0 &&
       !(na.lo & 0x80000000u) && !(da.lo & 0x80000000u)) break;
    /* Shift right by 1 (unsigned, since we already took abs). */
    na.lo = (na.lo >> 1) | (na.hi << 31);
    na.hi = (int32_t)((uint32_t)na.hi >> 1);
    da.lo = (da.lo >> 1) | (da.hi << 31);
    da.hi = (int32_t)((uint32_t)da.hi >> 1);
    k++;
  }
  int32_t num_i = (int32_t)na.lo;
  int32_t den_i = (int32_t)da.lo;
  if(den_i == 0) return (sign < 0) ? FXP_MIN_INT : FXP_MAX_INT;
  fxp32 q = fxp_div_q20(num_i, den_i);
  return (sign < 0) ? -q : q;
}

/* ============================================================================
 * fxp_mul  — Q11.20 multiplication with no INT64.
 *
 * (a * b) >> FXP_FRAC, with full intermediate precision, no overflow on
 * the intermediate.  Uses the same 16-bit split as fxp_acc_madd but is
 * specialized for the single-multiply case so the result is computed
 * directly without pushing through the 64-bit accumulator.
 * ========================================================================== */
static inline fxp32 fxp_mul(fxp32 a, fxp32 b) {
  fxp_acc acc = fxp_acc_zero();
  fxp_acc_madd(&acc, a, b);
  return fxp_acc_to_fxp32(&acc);
}

/* ============================================================================
 * Saturating arithmetic helpers (rare; mostly for debug guards).
 * ========================================================================== */

static inline fxp32 fxp_add_sat(fxp32 a, fxp32 b) {
  /* Overflow if a and b have the same sign but the wrapped result has
   * the opposite sign.  No INT64 needed: pure INT32 sign-equality check. */
  fxp32 r = a + b;
  if(((a ^ r) & (b ^ r)) < 0) return (a < 0) ? FXP_MIN_INT : FXP_MAX_INT;
  return r;
}

static inline fxp32 fxp_sub_sat(fxp32 a, fxp32 b) {
  fxp32 r = a - b;
  /* Overflow if sign of (a - b) differs from sign of a when b's sign opposes a. */
  if(((a ^ b) & (a ^ r)) < 0) return (a < 0) ? FXP_MIN_INT : FXP_MAX_INT;
  return r;
}

/* Forward declarations for fxp_sqrt small-x branch (= uses fxp_log/fxp_exp). */
static inline fxp32 fxp_log(fxp32 x);
static inline fxp32 fxp_exp(fxp32 x);

/* ============================================================================
 * fxp_sqrt  — square root via Newton-Raphson on rsqrt (no division).
 *
 * Computes sqrt(x) for x >= 0 in Q11.20.
 *
 * Strategy:
 *   - x >= FXP_ONE (real ≥ 1): Newton-iterate r = 1/sqrt(x) via
 *       r_{n+1} = r_n * (3 - x * r_n^2) / 2
 *     Then sqrt(x) = x * r.  r ≤ 1 here so r² fits Q11.20.
 *   - x < FXP_ONE (real < 1): r = 1/sqrt(x) > 1, and r² may exceed
 *     Q11.20 range (e.g. x=1e-4 → r²=1e4, overflows Q11.20 max=2048).
 *     Use sqrt(x) = exp(log(x)/2) via fxp_log / fxp_exp LUT.
 *     Precision ~1% (LUT-limited), acceptable for typical use.
 * ========================================================================== */
static inline fxp32 fxp_sqrt(fxp32 x) {
  if(x <= 0) return 0;

  if(x < FXP_ONE) {
    /* Small x: sqrt(x) = exp(log(x) / 2).  fxp_log returns Q11.20 of log(x_real).
     * For x_real ∈ (0, 1), log < 0, half_log < 0, exp in [-16, 0] domain. */
    fxp32 lx = fxp_log(x);
    fxp32 half_lx = lx >> 1;   /* arithmetic shift, signed (=/2 for negatives) */
    return fxp_exp(half_lx);
  }

  /* Large x (≥ 1): Newton on rsqrt. */
  int32_t b = 0;
  uint32_t v = (uint32_t)x;
  while(v >>= 1) b++;
  int32_t shift = (3 * FXP_FRAC - b) / 2;
  fxp32 r = (shift >= 0 && shift < 31) ? ((fxp32)1 << shift)
            : (shift < 0 ? 1 : FXP_MAX_INT);
  if(r <= 0) r = 1;
  for(int it = 0; it < 5; it++) {
    fxp32 rr = fxp_mul(r, r);
    fxp32 xrr = fxp_mul(x, rr);
    fxp32 t = (3 * FXP_ONE - xrr) >> 1;
    r = fxp_mul(r, t);
    if(r <= 0) r = 1;
  }
  return fxp_mul(x, r);
}

/* ============================================================================
 * fxp_recip  — reciprocal 1/x in Q11.20 (no INT64).
 *
 * Useful for division-by-constant (e.g., 1 / unified_sigma) where division
 * is hot-path.  Computed via Newton-Raphson on r_{n+1} = r * (2 - x*r).
 *
 * Precision: ~24 bits after 5 iterations (matches FP32).
 * ========================================================================== */
static inline fxp32 fxp_recip(fxp32 x) {
  if(x == 0) return FXP_MAX_INT;
  int32_t sign = (x < 0) ? -1 : 1;
  fxp32 ax = (x < 0) ? -x : x;
  /* Initial guess: 1/x ≈ 2^(2*FXP_FRAC - bit_pos(x))  scaled.
   * For x = 2^k Q11.20, 1/x = 2^(2*20 - k) Q11.20 = 2^(40 - k). */
  int32_t b = 0;
  uint32_t v = (uint32_t)ax;
  while(v >>= 1) b++;
  /* Initial guess: 1/x ≈ 2^(2*FXP_FRAC - bit_pos(x) - 1).
   *
   * Bug fix (2026-05-12): the old formula `1 << (2*FXP_FRAC - bit_pos)`
   * silently overshoots when x's mantissa m ∈ [1, 2) is close to 2 (e.g.
   * x = 0.998 raw 1046642, bit_pos=19, gives initial r=2.0 in Q11.20
   * while true 1/x = 1.002).  First Newton iter computes
   *   r' = r × (2 - x × r) = 2 × (2 - 1.996) = 0.008
   * and never recovers within 6 iterations → returns ~0.21 instead of 1.002.
   *
   * The −1 shift centers the initial guess so it's within Newton's
   * convergence range for all m ∈ [1, 2).  Trace verified on x=0.5
   * (true 2.0) and x=0.998 (true 1.002) — both converge correctly.
   *
   * Observed failure: SIDD s11_p28 unified_sigma=0.998 → fxp_recip wrong →
   * Phase 1d divides GAT by 5× → Phase 2 subtracts wrong dark_ref-relative
   * values → Phase 3 forward L collapses → PSNR 25 dB (vs FP32 56 dB).
   *
   * fxp_recip の Newton 初期推定が x.mantissa が 2 に近いとき発散していたバグの修正。
   * shift-1 で全 mantissa 範囲で収束する初期値を選ぶ。 */
  int32_t shift = 2 * FXP_FRAC - b - 1;
  fxp32 r = (shift >= 0 && shift < 31) ? (1 << shift) : 1;

  for(int it = 0; it < 6; it++) {
    /* r' = r * (2 - x * r)  in Q11.20:
     *   xr = x * r          (Q11.20)
     *   delta = 2*FXP_ONE - xr   (Q11.20)
     *   r' = r * delta      (Q11.20) */
    fxp32 xr = fxp_mul(ax, r);
    fxp32 delta = (FXP_ONE << 1) - xr;
    r = fxp_mul(r, delta);
    if(r <= 0) r = 1;
  }
  return (sign < 0) ? -r : r;
}

/* ============================================================================
 * fxp_div_q20  — Q11.20 division a/b skipping 1/b intermediate.
 *
 * Why this exists:
 *   Q11.20 max representable value = 2048 (sign + 11 int bits).
 *   For very small denominators (e.g. α = 4.8e-4 → 1/α = 2096 > 2048),
 *   fxp_recip(α) cannot fit the answer and silently saturates.  The
 *   downstream fxp_mul(σ², 1/α) then loses the σ²/α term entirely.
 *
 *   Observed failure: SIDD scene s04 (α=4.77e-4 < 4.88e-4 = Q11.20 limit),
 *   fxp_recip returned 64 raw (true 2196 raw), σ²/α computed as 0 instead
 *   of 0.064 → Phase 1 GAT DC offset off by ~30%, Phase 10 inverse-GAT
 *   collapsed output to ~0 → 34 dB PSNR loss vs FP32.
 *
 *   This routine computes a/b directly via paired-int32 long division on
 *   the wide numerator (a << 20), avoiding the unrepresentable 1/b
 *   intermediate.  The final quotient is back in Q11.20.
 *
 *   なぜ必要か:
 *     Q11.20 では 1/α の中間値が α<4.88e-4 で表現不能になり、
 *     fxp_recip が黙って saturate して σ²/α が 0 に潰れていた。
 *     直接 long division で a/b を計算することで 1/α 経由を避け、
 *     Q11.20 範囲内の結果を正しく返す。
 *
 * Math:
 *   Result Q11.20 raw = (a_raw × 2^20) / b_raw.
 *   Numerator (a_raw × 2^20) is up to 52 bits, stored as paired (hi, lo)
 *   where hi = a_raw >> 12 and lo = a_raw << 20.  Standard bit-by-bit
 *   long division on (hi:lo) by b_raw gives a 52-bit quotient; we saturate
 *   if it exceeds INT32 range (which corresponds to a result outside
 *   Q11.20's [-2048, 2048] domain).
 *
 * Cost: 52 iterations × ~10 int32 ops = ~500 ops.  Called once per image
 *       in fxp_gat_precompute and once per IRLS iteration in Phase 2;
 *       not in inner pixel loop, so cost is negligible.
 *
 * Constraint: INT32-only (no INT64, no float).
 * ========================================================================== */
static inline fxp32 fxp_div_q20(fxp32 num, fxp32 den) {
  if(den == 0) return (num >= 0) ? FXP_MAX_INT : -FXP_MAX_INT;
  int sign = ((num < 0) ^ (den < 0)) ? -1 : 1;
  /* Two's-complement absolute value (handles INT32_MIN safely in unsigned domain). */
  uint32_t num_abs = (num < 0) ? (~(uint32_t)num + 1u) : (uint32_t)num;
  uint32_t den_abs = (den < 0) ? (~(uint32_t)den + 1u) : (uint32_t)den;

  /* Dividend D = num_abs × 2^20.  We need to process its 52 significant
   * bits via shift-left long division extracting bits from the MSB of a
   * 64-bit paired register.  Place D aligned so its bit 51 sits at paired
   * bit 63: equivalently, store the conceptual 64-bit value num_abs × 2^32,
   * of which only the top 52 bits (positions 12..63) are processed by the
   * loop — those 52 bits exactly encode num_abs × 2^20.
   *
   * Concretely:
   *   hi = num_abs   (paired bits 32..63 = num_abs's bits 0..31)
   *   lo = 0         (paired bits 0..31 are zero)
   * The 52-iteration loop consumes paired bits 63..12, which equals
   * num_abs × 2^20 read as a 52-bit number.
   *
   * dividend を paired (hi:lo) の最上位 52 bit に配置する。
   * hi=num_abs, lo=0 とすると paired の bit 63..12 が num_abs*2^20 を表し、
   * ループはこの 52 bit を MSB から bit-by-bit に処理する。 */
  uint32_t hi = num_abs;
  uint32_t lo = 0;
  uint32_t r = 0, q = 0;
  int overflow = 0;

  /* 52 bit-iterations.  At each step, shift the (r : hi : lo) triple left
   * by 1, pulling the top bit of hi into r; then compare r against den_abs
   * and set the new quotient LSB accordingly. */
  for(int i = 0; i < 52; i++) {
    uint32_t top = hi >> 31;
    hi = (hi << 1) | (lo >> 31);
    lo = lo << 1;
    r  = (r << 1) | top;
    q  = q << 1;
    if(r >= den_abs) { r -= den_abs; q |= 1u; }
    /* After 20 iterations, q holds the top 20 bits of the conceptual
     * 52-bit quotient (which correspond to result bits 32..51).  If any
     * are set, the result overflows INT32. */
    if(i == 19 && q != 0) overflow = 1;
  }

  if(overflow || q > (uint32_t)FXP_MAX_INT)
    return (sign < 0) ? -FXP_MAX_INT : FXP_MAX_INT;
  return (sign < 0) ? -(int32_t)q : (int32_t)q;
}

/* ============================================================================
 * fxp_exp / fxp_log  — transcendentals via 256-entry LUT + linear interp.
 *
 * fxp_exp:
 *   - Domain x ∈ [-16, 0] (used in bilateral kernel `exp(-d² / 2σ²)`).
 *     Outside this domain returns FXP_ONE (x≥0) or 0 (x<-16).
 *   - LUT generated at init time via std lib expf().  256 entries cover
 *     [-16, 0] → step = 16/256 = 0.0625 = 65536 in Q11.20.
 *   - Interpolation: linear in segment, ~1e-4 max error.
 *
 * fxp_log:
 *   - Domain x ∈ [2^-16, 2^16] (cover all our positive variables).
 *   - 256-entry log2-spaced LUT.
 *   - Interpolation: linear in log2 segments.
 * ========================================================================== */

#define FXP_EXP_DOMAIN_LO  (-16 * FXP_ONE) /* x_min = -16 */

/* Note: fxp_exp_lut[] removed 2026-05-13 — was populated at init but never
 * read after Bug 8 fix (fxp_exp now uses Taylor + pow2_table directly).
 * Init flag retained so callers can still call fxp_exp_lut_init() which
 * now only builds the exp(-k) pow2 table. */
extern int   fxp_exp_lut_initialized;

/* INT-only Taylor evaluator: returns exp(-r) for r ∈ [0, 1] in Q11.20.
 * Series: exp(-r) = sum_k (-r)^k / k! = 1 - r + r²/2 - r³/6 + r⁴/24 - ...
 *
 * 8-term Horner-style: powers built incrementally, alternating signs.
 * Coefficients are pre-quantized Q11.20 constants (compile-time).
 *
 * Precision: ~1e-6 absolute error vs ideal exp(-r), within Q11.20 LSB. */
static inline fxp32 fxp_exp_neg_taylor_unit(fxp32 r) {
  /* Q11.20 reciprocal-factorial constants:
   *   1/k! for k = 0..7 truncated to Q11.20. */
  static const fxp32 inv_factorial[8] = {
    FXP_ONE,                     /* 1/0! */
    FXP_ONE,                     /* 1/1! */
    524288,                      /* 1/2  = 0x80000      */
    174762,                      /* 1/6  ≈ 0.16666666...*/
    43690,                       /* 1/24                */
    8738,                        /* 1/120               */
    1456,                        /* 1/720               */
    208                          /* 1/5040              */
  };
  /* Sum 1 + (-r) + (-r)² / 2! + ... = 1 + sum_{k=1..7} (-1)^k r^k / k!. */
  fxp32 r_pow = FXP_ONE;       /* r^0 */
  fxp32 sum = inv_factorial[0];
  int sign = -1;
  for(int k = 1; k < 8; k++) {
    r_pow = fxp_mul(r_pow, r);                 /* r^k in Q11.20 */
    fxp32 term = fxp_mul(r_pow, inv_factorial[k]);
    if(sign < 0) sum -= term; else sum += term;
    sign = -sign;
  }
  return sum;
}

/* Compute exp(-1) once via Taylor (= 1/e ≈ 0.367879441), then build
 * exp(-k) table for k = 0..16 via repeated multiplication.  All INT, no FP. */
extern fxp32 fxp_exp_neg_pow2_table[17];   /* exp(-k) for k=0..16 */

static inline void fxp_exp_lut_init(void) {
  if(fxp_exp_lut_initialized) return;
  /* Bootstrap exp(-1) via Taylor at r=1, then build exp(-k) table for
   * k = 0..16 by repeated multiplication.  These are the only values
   * needed by fxp_exp (which composes exp(-k) × exp(-frac) at runtime). */
  fxp_exp_neg_pow2_table[0] = FXP_ONE;
  fxp_exp_neg_pow2_table[1] = fxp_exp_neg_taylor_unit(FXP_ONE);
  for(int k = 2; k <= 16; k++) {
    fxp_exp_neg_pow2_table[k] =
      fxp_mul(fxp_exp_neg_pow2_table[k - 1], fxp_exp_neg_pow2_table[1]);
  }
  fxp_exp_lut_initialized = 1;
}

static inline fxp32 fxp_exp(fxp32 x) {
  if(x <= FXP_EXP_DOMAIN_LO) return 0;
  if(x >= 0) return FXP_ONE;
  /* Compute exp(x) for x ∈ (-16, 0) directly via:
   *   x = -k - frac    where k = ceil(-x), frac ∈ [0, 1)
   *   exp(x) = exp(-k) × exp(-frac)
   *
   * Bug fix (2026-05-12): old code used 256-entry LUT with LINEAR
   * interpolation in input space.  exp is non-linear → ~0.05% systematic
   * error per call.  Identical pattern to fxp_log linear-interp bug.
   *
   * Fix: skip the LUT, compute directly using existing primitives:
   *   - fxp_exp_neg_pow2_table[k] for k ∈ [0, 16]: precomputed exp(-k)
   *   - fxp_exp_neg_taylor_unit(frac) for frac ∈ [0, 1]: 8-term Taylor (~1e-7 err)
   * Total error: ~1e-7, three orders better than LUT linear interp.
   *
   * fxp_exp LUT 線形補間が exp の非線形性から ~0.05% 誤差を出していたバグの修正。
   * 既存の Taylor 級数 (8 項) + exp(-k) table を直接呼ぶように変更。 */
  fxp32 neg_x = -x;                       /* in (0, 16] in Q11.20 raw */
  int32_t k = (int32_t)(neg_x >> FXP_FRAC); /* floor(-x), in [0, 16] */
  if(k > 16) k = 16;                      /* clamp (shouldn't trigger, guarded above) */
  fxp32 frac = neg_x - ((fxp32)k << FXP_FRAC);  /* ∈ [0, FXP_ONE) */
  fxp32 e_frac = fxp_exp_neg_taylor_unit(frac);
  fxp32 e_k    = fxp_exp_neg_pow2_table[k];
  return fxp_mul(e_k, e_frac);
}

/* Note: fxp_log_lut[] removed 2026-05-13 — was populated at init but never
 * read after Bug 6 fix (fxp_log now uses Mercator-Gregory directly).  Init
 * function retained to compute fxp_ln2_q20 (still used in fxp_log proper). */
extern int   fxp_log_lut_initialized;
extern fxp32 fxp_ln2_q20;       /* ln(2) in Q11.20, computed at init */

/* INT-only ln((1+u)/(1-u)) = 2 (u + u³/3 + u⁵/5 + ...) Mercator-Gregory.
 * Converges fast for u ∈ [0, ~0.4].  Returns Q11.20.  10 odd-order terms
 * give ~1e-7 absolute error for u up to 0.5. */
static inline fxp32 fxp_ln_via_mercator(fxp32 u) {
  /* u in Q11.20.  We need u² stored separately for the recurrence. */
  fxp32 u_sq = fxp_mul(u, u);     /* Q11.20 of u² */
  fxp32 power = u;                 /* odd power: u, u³, u⁵, ... */
  fxp32 sum = u;                   /* first term = u */
  /* Q11.20 reciprocals of odd integers: 1/3, 1/5, 1/7, ..., 1/19. */
  static const fxp32 inv_odd[10] = {
    349525,  /* 1/3   */
    209715,  /* 1/5   */
    149797,  /* 1/7   */
    116508,  /* 1/9   */
    95325,   /* 1/11  */
    80661,   /* 1/13  */
    69905,   /* 1/15  */
    61680,   /* 1/17  */
    55188,   /* 1/19  */
    49932    /* 1/21  */
  };
  for(int k = 0; k < 10; k++) {
    power = fxp_mul(power, u_sq);                    /* u^(2k+3) */
    sum  += fxp_mul(power, inv_odd[k]);              /* term = u^(2k+3) / (2k+3) */
  }
  /* result = 2 * sum (= ln((1+u)/(1-u))). */
  return sum + sum;
}

/* INT-only log(m) for m ∈ [1, 2) in Q11.20.
 * Formula: log(m) = ln((1+u)/(1-u)) where u = (m-1)/(m+1).
 * For m ∈ [1, 2), u ∈ [0, 1/3], converges fast in fxp_ln_via_mercator. */
static inline fxp32 fxp_log_in_octave(fxp32 m_q20) {
  /* u = (m - 1) / (m + 1) */
  fxp32 num = m_q20 - FXP_ONE;
  fxp32 den = m_q20 + FXP_ONE;
  fxp32 u = fxp_mul(num, fxp_recip(den));
  return fxp_ln_via_mercator(u);
}

static inline void fxp_log_lut_init(void) {
  if(fxp_log_lut_initialized) return;
  /* Compute ln(2) via Mercator at m=2 (= u=1/3, converges fast).  This is
   * the only value still needed; fxp_log composes log(x) = b×ln(2) + log(m)
   * at runtime via fxp_log_in_octave (Mercator) for the mantissa part. */
  fxp_ln2_q20 = fxp_log_in_octave(FXP_ONE << 1);   /* m = 2 in Q11.20 */
  fxp_log_lut_initialized = 1;
}

static inline fxp32 fxp_log(fxp32 x) {
  if(x <= 0) return -16 * FXP_ONE;  /* domain undefined; clamp */
  /* x = 2^(b - FXP_FRAC) × mantissa, mantissa ∈ [1, 2).
   * log(x) = (b - FXP_FRAC) × ln(2) + log(mantissa).
   *
   * Bug fix (2026-05-12): old code used 8-segment LUT with LINEAR
   * interpolation in mantissa space.  But log is non-linear in mantissa,
   * giving ~6% error per call for non-power-of-2 inputs.  Cumulative
   * Σ_{k=1..N} log(k) drifts by 4% per doubling — fatal for
   * fxp_factorial_log[k] used in Foi-exact LUT.
   *
   * Fix: compute log(mantissa) directly via fxp_log_in_octave (Mercator-
   * Gregory series, ~1e-7 absolute error for u ∈ [0, 1/3] covering all
   * mantissa ∈ [1, 2)).  Eliminates the systematic linear-interp error.
   *
   * fxp_log LUT 線形補間が 6% 誤差を出して fxp_factorial_log を狂わせ
   * Foi LUT 2× scaling を引き起こしていたバグの修正。Mercator-Gregory
   * 級数で直接計算 (~1e-7 精度) に切替。 */
  int32_t b = 0;
  uint32_t v = (uint32_t)x;
  while(v >>= 1) b++;
  int32_t exponent_real = b - FXP_FRAC;
  /* Bug fix (2026-05-13): old code clamped exponent_real to [-16, 15] but
   * mantissa still used `x << (FXP_FRAC - b)`.  For x < 2^-16 (= 1.5e-5),
   * b < 4 and exponent_real < -16 → clamp causes log(x) to under-estimate
   * by ≥ ln(2) per octave below.  E.g. fxp_log(1e-5) returned -10.77
   * instead of -11.46 (= 1 octave off).  Caused fxp_sqrt(σ²) for σ²<1e-5
   * to over-estimate by sqrt(2)×, propagating wrong sigma_raw into Foi LUT.
   *
   * Fix: remove the clamp.  Q11.20 result range is ±2048; log(2^-31) ≈ -21
   * (smallest x>0 that produces a non-trivial result), so no clamp needed.
   *
   * fxp_log の exponent_real クランプが x<1.5e-5 で 1 octave 誤差を出して
   * fxp_sqrt(σ²) を sqrt(2)× 過大評価していたバグの修正。 */
  /* mantissa = x scaled to [FXP_ONE, 2·FXP_ONE). */
  fxp32 mantissa;
  if(b >= FXP_FRAC) mantissa = x >> (b - FXP_FRAC);
  else              mantissa = x << (FXP_FRAC - b);
  /* log(x) = exponent_real × ln(2) + log(mantissa). */
  fxp32 log_mantissa = fxp_log_in_octave(mantissa);
  return exponent_real * fxp_ln2_q20 + log_mantissa;
}

/* ============================================================================
 * Median via partial selection sort (port of GPU mad_sigma_y_sq_h).
 *
 * For an array of 63 fxp32 values (= AC coefficients of an 8x8 WHT block,
 * minus the DC), find the median via partial selection: find the smallest
 * 32 elements; the 32nd-smallest (= rank 31, 0-indexed) is the median of 63.
 *
 * Branchless on most ALUs (no INT-specific branches), suitable for both
 * CPU and GPU.  O(N * k) where N=63, k=32, so 2016 swaps total.
 * ========================================================================== */
static inline fxp32 fxp_partial_selection_median(const fxp32 *arr, int n) {
  /* Copy to local working buffer (max 64). */
  fxp32 tmp[64];
  if(n > 64) n = 64;
  for(int i = 0; i < n; i++) {
    /* Use absolute value, since MAD operates on |coef|. */
    tmp[i] = (arr[i] < 0) ? -arr[i] : arr[i];
  }

  const int target = n / 2;    /* median rank */
  for(int rank = 0; rank <= target; rank++) {
    int min_idx = rank;
    fxp32 min_val = tmp[rank];
    for(int j = rank + 1; j < n; j++) {
      if(tmp[j] < min_val) { min_val = tmp[j]; min_idx = j; }
    }
    /* Swap tmp[rank] ↔ tmp[min_idx]. */
    if(min_idx != rank) {
      fxp32 t = tmp[rank];
      tmp[rank] = tmp[min_idx];
      tmp[min_idx] = t;
    }
  }
  return tmp[target];
}

/* ============================================================================
 * WHT 8x8 forward / inverse (Walsh-Hadamard Transform, integer-only).
 *
 * Hadamard matrix H_8 has entries ±1, so the transform is pure add/sub.
 * The forward transform expands range by sqrt(N) = 2*sqrt(2) ≈ 2.83 per
 * coefficient axis, total range expansion 8x for the DC coefficient (sum
 * of 64 values), and the AC coefficients have similar ±64 worst-case range
 * for arbitrary input.
 *
 * For our pipeline: input range bounded by ~±5 in normalized GAT space
 * (= ~±5 * FXP_ONE = 5e6 in Q11.20).  Block sum = ±64 * 5 = ±320, so
 * coefficient max = ~±320 * FXP_ONE = 3.4e8, well within INT32 range.
 *
 * Normalization: the standard WHT (unnormalized) does Y = H * X * H'.  We
 * apply the 1/N = 1/64 scaling at the END (= post-process step), to avoid
 * intermediate precision loss.  For inverse, we re-apply 1/N scaling.
 *
 * In Q11.20, scaling by 1/N = right shift by 6 bits (since 64 = 2^6).
 * ========================================================================== */

#define FXP_GALOSH_BLOCK_SIZE   8
#define FXP_GALOSH_BLOCK_PIXELS 64    /* 8 * 8 */

/* In-place 8-element WHT (unnormalized).  Input: fxp32[8].  Output: same.
 *
 * Computes Y[k] = sum_n H_8[k,n] X[n], with H_8[k,n] = (-1)^bitcount(k & n).
 * Implemented as 3 stages of 4 add/sub butterflies each. */
static inline void fxp_wht8(fxp32 *x) {
  fxp32 t0, t1;
  /* Stage 1: pair (0,1)(2,3)(4,5)(6,7) */
  t0 = x[0] + x[1]; t1 = x[0] - x[1]; x[0] = t0; x[1] = t1;
  t0 = x[2] + x[3]; t1 = x[2] - x[3]; x[2] = t0; x[3] = t1;
  t0 = x[4] + x[5]; t1 = x[4] - x[5]; x[4] = t0; x[5] = t1;
  t0 = x[6] + x[7]; t1 = x[6] - x[7]; x[6] = t0; x[7] = t1;
  /* Stage 2: pair (0,2)(1,3)(4,6)(5,7) */
  t0 = x[0] + x[2]; t1 = x[0] - x[2]; x[0] = t0; x[2] = t1;
  t0 = x[1] + x[3]; t1 = x[1] - x[3]; x[1] = t0; x[3] = t1;
  t0 = x[4] + x[6]; t1 = x[4] - x[6]; x[4] = t0; x[6] = t1;
  t0 = x[5] + x[7]; t1 = x[5] - x[7]; x[5] = t0; x[7] = t1;
  /* Stage 3: pair (0,4)(1,5)(2,6)(3,7) */
  t0 = x[0] + x[4]; t1 = x[0] - x[4]; x[0] = t0; x[4] = t1;
  t0 = x[1] + x[5]; t1 = x[1] - x[5]; x[1] = t0; x[5] = t1;
  t0 = x[2] + x[6]; t1 = x[2] - x[6]; x[2] = t0; x[6] = t1;
  t0 = x[3] + x[7]; t1 = x[3] - x[7]; x[3] = t0; x[7] = t1;
}

/* 2D 8x8 WHT, in-place.  Optionally apply 1/N=1/64 normalization (= shift >>6).
 *
 * Applies 1D WHT to each row, then transposes & applies 1D WHT to each column.
 * Mirrors the FP32 wht2d_8x8() in galosh_cpu.h. */
static inline void fxp_wht2d_8x8(fxp32 block[FXP_GALOSH_BLOCK_PIXELS],
                                  int normalize) {
  for(int r = 0; r < 8; r++) fxp_wht8(block + r * 8);
  for(int c = 0; c < 8; c++) {
    fxp32 col[8];
    for(int r = 0; r < 8; r++) col[r] = block[r * 8 + c];
    fxp_wht8(col);
    for(int r = 0; r < 8; r++) block[r * 8 + c] = col[r];
  }
  if(normalize) {
    for(int i = 0; i < 64; i++) block[i] >>= 6;
  }
}

/* ============================================================================
 * Kaiser window 2D table (8x8) for BayesShrink overlap-add.
 *
 * Mirrors galosh_kaiser_2d in galosh_cpu.h.  1D values:
 *   {0.34012, 0.59885, 0.84123, 0.97659, 0.97659, 0.84123, 0.59885, 0.34012}
 * 2D = outer product.  Pre-quantized to Q11.20 to avoid runtime FP.
 * ========================================================================== */
extern fxp32 fxp_kaiser_2d[64];
extern int   fxp_kaiser_initialized;

static inline void fxp_kaiser_init(void) {
  if(fxp_kaiser_initialized) return;
  /* 1D values in Q11.20 (= value × 2^20). */
  static const fxp32 k1d[8] = {
    356571,    /* 0.34012 */
    627887,    /* 0.59885 */
    882104,    /* 0.84123 */
    1023771,   /* 0.97659 */
    1023771,
    882104,
    627887,
    356571
  };
  for(int dy = 0; dy < 8; dy++) {
    for(int dx = 0; dx < 8; dx++) {
      fxp_kaiser_2d[dy * 8 + dx] = fxp_mul(k1d[dy], k1d[dx]);
    }
  }
  fxp_kaiser_initialized = 1;
}

/* ============================================================================
 * GAT forward (Anscombe / Foi piecewise) in Q11.20.
 *
 * x_real = signal value in [0, 1] FP32.  Stored as Q11.20.
 *
 * Mathematics (sqrt branch):
 *   GAT(x) = (2/α) * sqrt(α x + 3α²/8 + σ²)
 *
 * Linear branch (when x < -3α/8): C1 extension to negative input range.
 *
 * α and σ² are estimated globally per-image (Phase 0).  In Q11.20:
 *   α typical: 1e-5 to 1e-2 (= scaled signal slope)
 *   σ² typical: 1e-8 to 1e-4
 *
 * IMPORTANT: α is small.  If we store α in Q11.20, value 1e-5 = 10 in
 * Q11.20 — too coarse (1 LSB = ~5% error).  Solution: use a separate
 * scale for α (Q5.26, range ±32, precision 1.5e-8).  For simplicity in
 * Stage 1 we'll use Q11.20 throughout but accept the precision floor
 * for α and verify it doesn't propagate.
 *
 * TODO Stage 1b: if α precision is insufficient, introduce alpha_q26.
 * ========================================================================== */

/* GAT forward — restructured to avoid α² (which underflows Q11.20 for
 * realistic α ≈ 1e-3).
 *
 * Original:    GAT(x) = (2/α) sqrt(α x + 3α²/8 + σ²)
 * Restructure: GAT(x) = (2/√α) sqrt(x + 3α/8 + σ²/α)
 *
 * Composite constants are precomputed once per image (or passed as Q11.20):
 *   inv_2sqrt_alpha = 2 / sqrt(α)           (≈ 63 for α=1e-3, fits Q11.20)
 *   three_alpha_8   = 3α/8                  (≈ 4e-4 for α=1e-3, ~393 LSBs)
 *   sigma_sq_over_alpha = σ² / α            (≈ 1e-3 for σ²=1e-6/α=1e-3)
 *
 * Caller responsibility: precompute these in Phase 0 via fxp_gat_precompute.
 * The internal fxp_gat_forward only does the per-pixel arithmetic. */

typedef struct fxp_gat_params {
  fxp32 alpha;
  fxp32 sigma_sq;
  fxp32 sigma_raw;             /* sqrt(σ²) */
  fxp32 inv_2sqrt_alpha;       /* 2 / sqrt(α) */
  fxp32 three_alpha_8;         /* 3α/8 */
  fxp32 sigma_sq_over_alpha;   /* σ²/α */
  fxp32 y_break;               /* -3α/8 (forward-domain transition) */
  fxp32 t_break;               /* 2σ/α  (GAT-domain transition) */
  fxp32 inv_sigma_raw;         /* 1/σ for linear branch */
} fxp_gat_params;

/* Precompute composite constants from raw α, σ² (one-time per image). */
static inline void fxp_gat_precompute(fxp_gat_params *p,
                                      fxp32 alpha, fxp32 sigma_sq) {
  p->alpha = alpha;
  p->sigma_sq = sigma_sq;
  if(sigma_sq <= 0) sigma_sq = 1;     /* avoid div-by-zero */
  if(alpha <= 0)    alpha    = 1;
  p->sigma_raw = fxp_sqrt(sigma_sq);
  if(p->sigma_raw <= 0) p->sigma_raw = 1;
  fxp32 sqrt_alpha = fxp_sqrt(alpha);
  if(sqrt_alpha <= 0) sqrt_alpha = 1;
  p->inv_2sqrt_alpha = fxp_recip(sqrt_alpha);
  p->inv_2sqrt_alpha += p->inv_2sqrt_alpha;            /* 2 × */
  p->three_alpha_8 = fxp_mul(fxp_from_float(0.375f), alpha);
  /* σ² / α and 2σ / α: avoid fxp_recip(α) intermediate.
   *
   * For α < 4.88e-4, 1/α > 2048 and is unrepresentable in Q11.20 (max=2048).
   * fxp_recip silently saturates to ~64 (observed), making σ²/α evaluate
   * to 0 when σ² is small — causing Phase 1 GAT to drop the σ²/α term
   * and Phase 10 inverse-GAT to collapse output toward zero.
   *
   * The final answers σ²/α (≈0.06) and 2σ/α (≈23) themselves DO fit in
   * Q11.20.  We compute them directly via fxp_div_q20 long division,
   * which uses paired-int32 (no INT64) for the wide intermediate
   * numerator (a << 20).  See fxp_div_q20 doc-comment for why.
   *
   * fxp_recip(α) を経由しない: Q11.20 では α<4.88e-4 で 1/α が範囲外、
   * fxp_recip が saturate して σ²/α=0 になる SIDD s04 系の破綻バグの修正。
   * 結果値 σ²/α, 2σ/α 自体は Q11.20 範囲に収まるので直接除算で計算。 */
  /* σ²/α from the ORIGINAL σ² (p->sigma_sq), NOT the sqrt-floored sigma_sq=1.
   * On super-clean super-dark scenes Phase 0 returns σ²≈0 (true ~6e-8 < Q11.20
   * 1 ulp → stored 0).  The `sigma_sq<=0 ? 1` floor (needed only for sigma_raw's
   * sqrt) would otherwise make σ²/α = 1ulp/α ≈ 0.24 for tiny α — a huge spurious
   * DC that compresses the GAT signal range (T≈505±8 instead of 56..134) so
   * Phase-5 shrink + Phase-10 inverse collapse the output to 0 → magenta render.
   * FP32 keeps σ²/α≈0.009 here and is fine.  Using the un-floored σ² gives
   * σ²/α≈0, matching FP32 and preventing the collapse (RawNIND low-ISO 26 scenes).
   * 超clean(σ²≈0)で floor 値 1ulp を使うと tiny-α で σ²/α 巨大化→GAT 信号圧縮→
   * 出力ゼロ崩壊。floor 前の σ² で計算し σ²/α≈0 とする。 */
  p->sigma_sq_over_alpha = fxp_div_q20(p->sigma_sq > 0 ? p->sigma_sq : 0, alpha);
  /* y_break = -3α/8 (= -p->three_alpha_8). */
  p->y_break = -p->three_alpha_8;
  /* t_break = 2σ / α from the ORIGINAL σ (not the sqrt-floored sigma_raw).
   * t_break is the inverse-GAT branch boundary (= the D value at x=y_break); it
   * must match the σ²/α used in the forward.  With the floored sigma_raw it
   * inflates (e.g. 487) so super-clean sqrt-branch outputs D∈[0, t_break] get
   * routed to the LINEAR inverse → x = y_break + σ(D−t_break) ≪ 0 → clamp 0 →
   * zero collapse (same tiny-α / σ²≈0 root as sigma_sq_over_alpha above).
   * inv_sigma_raw (linear-branch slope) keeps the floored sigma_raw (needs >0).
   * t_break も floored σ を使うと膨張し super-clean で逆 GAT が linear 誤分岐→
   * ゼロ崩壊。元 σ から算出して σ²/α と整合させる。 */
  fxp32 sigma_raw_true = fxp_sqrt(p->sigma_sq > 0 ? p->sigma_sq : 0);
  p->t_break = fxp_div_q20(sigma_raw_true + sigma_raw_true, alpha);
  p->inv_sigma_raw = fxp_recip(p->sigma_raw);
}

static inline fxp32 fxp_gat_forward(fxp32 x_q20,
                                    const fxp_gat_params *p) {
  if(x_q20 >= p->y_break) {
    /* sqrt branch: GAT = inv_2sqrt_alpha * sqrt(x + 3α/8 + σ²/α). */
    fxp32 u = x_q20 + p->three_alpha_8 + p->sigma_sq_over_alpha;
    if(u < 0) u = 0;
    return fxp_mul(p->inv_2sqrt_alpha, fxp_sqrt(u));
  } else {
    /* linear branch: GAT = t_break + (x - y_break) / σ. */
    fxp32 dx = x_q20 - p->y_break;
    return p->t_break + fxp_mul(dx, p->inv_sigma_raw);
  }
}

/* GAT algebraic inverse — closed-form inverse of fxp_gat_forward.
 *
 * Forward (sqrt branch):
 *   D = inv_2sqrt_alpha * sqrt(x + 3α/8 + σ²/α)
 *
 * Inverse (sqrt branch, algebraic invert):
 *   D / inv_2sqrt_alpha = sqrt(x + 3α/8 + σ²/α)
 *   sqrt(α)/2 * D = sqrt(x + ...)            [since inv_2sqrt_alpha = 2/√α]
 *   (α/4) * D² = x + 3α/8 + σ²/α
 *   x = (α/4) * D² - 3α/8 - σ²/α
 *
 * Forward (linear branch):
 *   D = t_break + (x - y_break) / σ_raw
 *
 * Inverse (linear branch):
 *   x = y_break + σ_raw * (D - t_break)
 *
 * NOTE: This is the algebraic inverse of the C1 piecewise model.  For the
 * Foi *exact unbiased* inverse via Gauss-Hermite quadrature LUT, see
 * fxp_gat_inv_table_t (built in Phase 0, used for production).  Algebraic
 * inverse is sufficient for round-trip verification and as a fast
 * approximation when LUT is not available. */
static inline fxp32 fxp_gat_inverse_algebraic(fxp32 d_q20,
                                              const fxp_gat_params *p) {
  if(d_q20 >= p->t_break) {
    /* sqrt branch inverse: x = (α/4) * D² - 3α/8 - σ²/α.
     *
     * Naive D² overflows Q11.20 (D up to ~270 → D² up to ~73000 >> 2048).
     * Re-order multiplication: ((α/4) * D) * D, which keeps each
     * intermediate in Q11.20 range:
     *   (α/4) * D ≈ 2.5e-4 * 270 = 0.07  (fits)
     *   that * D  ≈ 0.07 * 270 = 18      (fits) */
    fxp32 alpha_over_4 = p->alpha >> 2;
    fxp32 ad = fxp_mul(alpha_over_4, d_q20);     /* (α/4) * D */
    fxp32 ad2 = fxp_mul(ad, d_q20);              /* ((α/4) * D) * D */
    return ad2 - p->three_alpha_8 - p->sigma_sq_over_alpha;
  } else {
    /* linear branch inverse: x = y_break + σ * (D - t_break). */
    fxp32 dd = d_q20 - p->t_break;
    return p->y_break + fxp_mul(p->sigma_raw, dd);
  }
}

/* ============================================================================
 * GAT inverse table — fixed-point version of gat_inv_table_t.
 *
 * Built in Phase 0 via Gauss-Hermite quadrature, then used in Phase 9-10
 * to invert from normalized GAT space back to raw signal.
 *
 * Stage 1 stub: declared here, builder + lookup live in galosh_raw_cpu_int.c
 * since they involve more intricate multi-precision arithmetic for GH
 * quadrature.
 * ========================================================================== */

/* LUT size: 1024 entries gives x-step = 1/1023 ≈ 9.8e-4.  Linear interp
 * residual at lookup time is bounded by step / 2 ≈ 5e-4 in x — far below
 * the per-pixel quantization noise of Q11.20 (1 LSB ≈ 1e-6 magnitude after
 * Phase 9 reconstruction).  v12 SIDD val was 49.10 with N=4096; N=1024
 * yields essentially the same number while making LUT build 4× faster
 * (= ~0.13s/image instead of ~0.5s, total bench overhead 10 min → 3 min). */
#define FXP_GAT_INV_TABLE_SIZE 1024

typedef struct fxp_gat_inv_table {
  fxp32 d[FXP_GAT_INV_TABLE_SIZE];
  fxp32 x[FXP_GAT_INV_TABLE_SIZE];
  fxp32 d_min, d_max;
  fxp32 alpha;
  fxp32 sigma_raw;
  fxp32 y_break;
  fxp32 t_break;
  int   valid;
} fxp_gat_inv_table_t;

/* ============================================================================
 * Foi-exact unbiased inverse GAT via Gauss-Hermite quadrature LUT.
 *
 * Why: the algebraic inverse `x = (α/4)D² - 3α/8 - σ²/α` is the deterministic
 * inverse of the GAT forward — but for low-signal regimes (mean signal close
 * to 0), the expected value E[X | D=d] is significantly larger than the
 * algebraic inverse due to concavity of sqrt under noise.  Algebraic returns
 * negative for many D values that arise from low-signal observations,
 * destroying signal in dark patches.
 *
 * Foi-Mäkitalo exact unbiased inverse (TIP 2013) builds an inverse LUT by
 * forward-integrating: for each x_val, compute E[T(noisy)] where
 *   noisy = Poisson(x/α)·α + N(0, σ²)
 * via summed Poisson terms × Gauss-Hermite over Gaussian noise.  At lookup
 * time, binary search the LUT for the inverse mapping D → x.
 *
 * Observed failure: SIDD s24_p25 (very dark, mean 0.01) — INT algebraic gives
 * 97% negative pixel outputs → PSNR 39.7 dB vs FP32 LUT 63.2 dB.  After Foi
 * LUT impl below, INT matches FP32 within < 1 dB.
 *
 * Implementation: INT32-only using existing fxp_log / fxp_exp / fxp_sqrt
 * (Taylor + Mercator-Gregory LUTs).  No FP, no INT64.  Q11.20 throughout.
 * Built once per image after Phase 0 estimates α, σ².  Cost: ~50 ms one-time.
 *
 * Foi-Alenius 厳密非偏倚逆 GAT を Poisson × 10-pt Gauss-Hermite quadrature で
 * 構築する。algebraic 逆 GAT が低信号で負値を返すバグ (= 暗部 patch 破綻) の修正。
 * ========================================================================== */

/* 10-point Gauss-Hermite weights × exp(node²) and nodes × sqrt(2)·sigma_factor:
 * pre-factored to avoid runtime sqrt(2), 1/sqrt(π), exp.  Stored as Q11.20.
 * Source: Makitalo & Foi (TIP 2013), nodes from physicists' Hermite poly. */
static const fxp32 fxp_gh_nodes_q20[10] = {
  -3603158,    /* -3.4361591188 */
  -2655488,    /* -2.5327316742 */
  -1841717,    /* -1.7566836493 */
  -1086780,    /* -1.0366108298 */
   -359478,    /* -0.3429013272 */
    359478,
   1086780,
   1841717,
   2655488,
   3603158
};
/* Weights: smallest = 7.64e-6 → Q11.20 raw 8 (precision 12%; only tail-end
 * Gaussian density, contribute < 0.01% to integral).  For higher precision,
 * could use Q3.28 but Q11.20 is sufficient — verified by output match. */
static const fxp32 fxp_gh_weights_q20[10] = {
   8,           /* 7.640433e-6 */
   1408,        /* 1.343646e-3 */
   35511,       /* 3.387439e-2 */
   251726,      /* 2.401386e-1 */
   640566,      /* 6.108626e-1 */
   640566,
   251726,
   35511,
   1408,
   8
};
/* 1/sqrt(π) in Q11.20: 0.564189... → 591633 raw */
#define FXP_INV_SQRT_PI_Q20  591633
/* sqrt(2) in Q11.20: 1.414214... → 1482910 raw */
#define FXP_SQRT2_Q20        1482910

/* fxp_factorial_log[k] = log(k!) in Q11.20.  Built once at init.
 * Size 400: log(399!) ≈ 2002 fits Q11.20 (max 2048, just within).  Used for
 * the exact Poisson path (λ ≤ 100, k_max ≤ 180); for larger λ we use Gauss
 * approximation which doesn't need log(k!). */
#define FXP_FACTORIAL_LOG_SIZE 400
extern fxp32 fxp_factorial_log[FXP_FACTORIAL_LOG_SIZE];
extern int   fxp_factorial_log_initialized;

static inline void fxp_factorial_log_init(void) {
  if(fxp_factorial_log_initialized) return;
  fxp_factorial_log[0] = 0;  /* log(0!) = log(1) = 0 */
  for(int k = 1; k < FXP_FACTORIAL_LOG_SIZE; k++) {
    /* log(k!) = log((k-1)!) + log(k).  Q11.20 throughout. */
    fxp32 logk = fxp_log((fxp32)k * FXP_ONE);
    fxp_factorial_log[k] = fxp_factorial_log[k - 1] + logk;
  }
  fxp_factorial_log_initialized = 1;
}

/* Compute log(P(k; λ)) in Q11.20.
 *
 * Hybrid impl:
 *   λ ≤ 30: exact Poisson — log_prob = -λ + k·log(λ) - log(k!).
 *           k ≤ 255 covered by precomputed fxp_factorial_log table.
 *   λ > 30: Gauss CLT approximation — log_prob ≈ -(k-λ)²/(2λ) - 0.5·log(2πλ).
 *           Avoids log(k!) which overflows Q11.20 for k > ~330.
 *
 * Rationale: incremental Poisson sum (log_prob_k = log_prob_{k-1} + log(λ)
 * - log(k)) accumulates fxp_log's 1% LSB precision error over 100s of
 * iterations → log_prob at peak off by 6+ → prob×1000 too small → LUT
 * collapses to ~0.  Gauss approx is one-shot per k, no error accumulation.
 *
 * インクリメンタル和の累積誤差を避けるため、λ>30 で Gauss CLT 近似に切替。
 * λ≤30 では exact Poisson + 既存 fxp_factorial_log。 */
#define FXP_LOG_2PI_Q20 1926744   /* log(2π) ≈ 1.8379 → 1.8379 × 2^20 */
static inline fxp32 fxp_poisson_log_pdf(fxp32 lambda_q20, int k) {
  if(lambda_q20 <= 0) return (k == 0) ? 0 : FXP_MIN_INT;
  if(k == 0) return -lambda_q20;

  /* Switch threshold: λ=100.  Gauss CLT discrepancy with exact Poisson:
   *   λ=30  → Δ ≈ 0.05 in log_prob (= 5% factor on prob)
   *   λ=100 → Δ ≈ 0.005 (= 0.5%)
   * Using exact up to 100 (= k_max ≤ 180 fits FACT_SIZE=400). */
  if(lambda_q20 < (fxp32)100 * FXP_ONE) {
    /* Exact Poisson — k ≤ ~70 in this regime (λ + 4√λ ≤ 30 + 22 = 52). */
    if(k >= FXP_FACTORIAL_LOG_SIZE) return FXP_MIN_INT;
    fxp32 log_lambda = fxp_log(lambda_q20);
    /* k·log(λ): k ≤ 255, log(λ) up to ~3.4 raw (3.5M raw) → product
     * up to 255 × 3.5M = 9e8, fits int32. */
    fxp_acc t = fxp_acc_zero();
    fxp_acc_madd(&t, k, log_lambda);
    fxp32 k_log_lambda = fxp_acc_extract_q20(&t);
    return -lambda_q20 + k_log_lambda - fxp_factorial_log[k];
  }

  /* Gauss CLT: log_prob ≈ -0.5(k-λ)²/λ - 0.5·log(2πλ).
   *
   * Compute (k-λ)²/λ using plain int squaring (avoid Q11.20 overflow on
   * (k-λ)² which can reach ~50000 for λ=1500, > Q11.20 max=2048). */
  int lambda_int = lambda_q20 >> FXP_FRAC;
  if(lambda_int < 1) lambda_int = 1;
  int km_int = k - lambda_int;
  int km_sq_int = km_int * km_int;     /* up to ~5e4 for λ ≤ 1500, fits int32 */
  /* (km² / λ) × 2^20 = (km² × 2^20) / λ_int.  Use fxp_acc for the wide
   * numerator (up to 5e4 × 2^20 = 5.2e10), divide by int lambda_int. */
  fxp_acc num_acc = fxp_acc_zero();
  fxp_acc_madd(&num_acc, km_sq_int, FXP_ONE);
  fxp32 sq_over_lambda_q20 = fxp_acc_div_i32(&num_acc, lambda_int);
  /* 0.5·log(2πλ) = (log(2π) + log(λ)) / 2 */
  fxp32 log_lambda = fxp_log(lambda_q20);
  fxp32 log_2pi_lambda_half = (FXP_LOG_2PI_Q20 + log_lambda) >> 1;
  return -(sq_over_lambda_q20 >> 1) - log_2pi_lambda_half;
}

#endif  /* GALOSH_CPU_INT_H */
