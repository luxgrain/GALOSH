/* galosh_yuv420.h — GALOSH-420 shared front-end: planar YCbCr 4:2:0/4:2:2/4:0:0
 * container handling (siting / EOTF / matrix / range / depth).
 *
 * Normative spec: docs/yuv420_frontend_spec.md (draft v0.1, 2026-07-11).
 * Design decision (A/B, benchmark/scripts/ab_yuv420.py, 2026-07-11): chroma is
 * denoised at its NATIVE half-resolution lattice with a siting-phased
 * downsampled-Y guide — beats upsample-to-444-first by +0.3..0.5 dB Cb/Cr and
 * on LPIPS (62-64/80 SIDD scenes), ~4x cheaper chroma; gap grows with noise.
 *
 * EN: This header is pure front-end: container codes <-> float gamma-domain
 *     planes, siting-phased luma guide construction, and the NCL matrix.
 *     It contains NO denoising math — the core pipeline (galosh_cpu.h) is
 *     reused unchanged at the chroma lattice scale.  Shared by the CPU
 *     reference driver and (host-side) by the OpenCL / Vulkan drivers.
 * JP: このヘッダは純粋なフロントエンド（コンテナ符号<->float ガンマ域プレーン、
 *     siting 位相ガイド生成、NCL 行列）。デノイズ本体は一切含まない —
 *     コア（galosh_cpu.h）をクロマ格子スケールでそのまま再利用する。
 *     CPU リファレンスと OpenCL / Vulkan ドライバ（ホスト側）で共有。
 *
 * ---------------------------------------------------------------------------
 * Chroma siting — 4-layer coordinate framework (NEVER mix the layers):
 *   Luma pixel-corner coordinates; Y[0,0] covers [0,1]x[0,1].
 *   1. chroma sample-center position:
 *        center  = (1.0, 1.0)   JPEG/JFIF, MPEG-1        (H.273 type 1)
 *        left    = (0.5, 1.0)   MPEG-2/AVC/HEVC default  (H.273 type 0)
 *        topleft = (0.5, 0.5)   BT.2020/BT.2100          (H.273 type 2)
 *   2. equivalent 2x2 chroma-cell top-left: (0,0) / (-0.5,0) / (-0.5,-0.5)
 *   3. phase offset vs center-sited:        (0,0) / (-0.5,0) / (-0.5,-0.5)
 *   4. filter kernel: NEVER determined by siting — the kernels below are
 *      implementation choices evaluated AT the siting phase.
 *   The names describe sample-center locations, not cell anchors.
 *   Verified against FFmpeg pixfmt.h AVChromaLocation and Microsoft
 *   MFVideoChromaSubsampling ("1/2 pixel to the right/down if not cosited").
 * JP: siting は位置（位相）のみを規定しカーネルは決めない。名前は
 *     サンプル中心位置であってセルアンカーではない。
 *
 * Measured (2026-07-11, SIDD 80): siting choice itself does NOT affect
 * denoise quality (sRGB PSNR 35.77/35.77/35.74); only a guide-phase MISMATCH
 * costs (Cb -0.13 / Cr -0.19 dB, 18/20 scenes) — hence the MANDATORY
 * affine-field phase selftest below (galosh420_phase_selftest).
 * ---------------------------------------------------------------------------
 * (code: Apache-2.0)
 */
#ifndef GALOSH_YUV420_H
#define GALOSH_YUV420_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* ================================================================
 * Enums / parameters
 * ================================================================ */
typedef enum
{
  GALOSH420_PIX_444 = 0,   /* legacy sRGB float path (not planar) */
  GALOSH420_PIX_420 = 1,
  GALOSH420_PIX_422 = 2,   /* provisionally via horizontal up to 444 */
  GALOSH420_PIX_400 = 3    /* Y-only: chroma stage skipped */
} galosh420_pix_t;

typedef enum
{
  GALOSH420_SITING_CENTER  = 0,
  GALOSH420_SITING_LEFT    = 1,
  GALOSH420_SITING_TOPLEFT = 2
} galosh420_siting_t;

typedef enum
{
  GALOSH420_EOTF_SRGB   = 0,
  GALOSH420_EOTF_G22    = 1,
  GALOSH420_EOTF_G24    = 2,
  GALOSH420_EOTF_BT709  = 3,   /* inverse camera OETF (scene-linear) */
  GALOSH420_EOTF_HLG    = 4,   /* BT.2100 inverse OETF (scene-linear; the
                                * display OOTF is deliberately NOT applied —
                                * documented approximation, spec §eotf) */
  GALOSH420_EOTF_PQ     = 5,   /* ST.2084 EOTF, normalised 1.0 = 10000 nit */
  GALOSH420_EOTF_LINEAR = 6
} galosh420_eotf_t;

typedef enum
{
  GALOSH420_RANGE_FULL    = 0,
  GALOSH420_RANGE_LIMITED = 1
} galosh420_range_t;

/* H.273 NCL matrix coefficients.  Kg = 1 - Kr - Kb.
 * Primaries (e.g. Display P3) are NOT a matrix axis: GALOSH performs no
 * gamut conversion and is primaries-agnostic (tag passthrough). */
typedef struct { float kr, kb; } galosh420_matrix_t;

static const galosh420_matrix_t GALOSH420_MAT_BT601  = { 0.2990f, 0.1140f };
static const galosh420_matrix_t GALOSH420_MAT_BT709  = { 0.2126f, 0.0722f };
static const galosh420_matrix_t GALOSH420_MAT_BT2020 = { 0.2627f, 0.0593f };

/* ================================================================
 * EOTF pairs — gamma-domain V <-> (relative) linear L.
 *
 * EN: Out-of-range excursions (ycc2rgb of denoised/arbitrary chroma can
 *     leave [0,1]) are handled sign-symmetrically for pow-based curves:
 *     f(v) = sign(v)*f(|v|) (scRGB convention).  The sRGB pair keeps the
 *     exact piecewise form of the legacy driver so eotf=srgb reproduces
 *     the validated A/B chain.  PQ clamps |v| to 1 (super-white has no
 *     meaning above the 10000-nit code point).
 * JP: 範囲外は pow 系で符号対称拡張（scRGB 流儀）。sRGB はレガシー
 *     ドライバの区分式と同一 = A/B 検証済みチェーンを再現。
 * ================================================================ */
static inline float galosh420_eotf_inv_f(float v, const galosh420_eotf_t e)
{
  const float s = (v < 0.0f) ? -1.0f : 1.0f;
  const float a = fabsf(v);
  switch(e)
  {
    case GALOSH420_EOTF_SRGB:
      /* IEC 61966-2-1 piecewise — identical to legacy srgb_to_linear_f. */
      return (v <= 0.04045f) ? v / 12.92f
                             : powf((v + 0.055f) / 1.055f, 2.4f);
    case GALOSH420_EOTF_G22:   return s * powf(a, 2.2f);
    case GALOSH420_EOTF_G24:   return s * powf(a, 2.4f);
    case GALOSH420_EOTF_BT709: /* BT.709 inverse OETF */
      return (a < 0.081f) ? v / 4.5f
                          : s * powf((a + 0.099f) / 1.099f, 1.0f / 0.45f);
    case GALOSH420_EOTF_HLG:
    {
      /* BT.2100 HLG inverse OETF, scene-linear in [0,1]. */
      const float ha = 0.17883277f, hb = 0.28466892f, hc = 0.55991073f;
      const float l = (a <= 0.5f) ? (a * a) / 3.0f
                                  : (expf((a - hc) / ha) + hb) / 12.0f;
      return s * l;
    }
    case GALOSH420_EOTF_PQ:
    {
      /* SMPTE ST.2084 EOTF; 1.0 = 10000 nit.  Typical SDR-graded content
       * lands near 0.01 linear — harmless: the blind PG fit + GAT
       * normalisation are scale-adaptive. */
      const float m1 = 2610.0f / 16384.0f, m2 = 2523.0f / 4096.0f * 128.0f;
      const float c1 = 3424.0f / 4096.0f;
      const float c2 = 2413.0f / 4096.0f * 32.0f;
      const float c3 = 2392.0f / 4096.0f * 32.0f;
      const float ac = (a > 1.0f) ? 1.0f : a;
      const float vp = powf(ac, 1.0f / m2);
      const float num = vp - c1;
      const float l = (num <= 0.0f) ? 0.0f
                    : powf(num / (c2 - c3 * vp), 1.0f / m1);
      return s * l;
    }
    default: return v;   /* LINEAR */
  }
}

static inline float galosh420_eotf_fwd_f(float l, const galosh420_eotf_t e)
{
  const float s = (l < 0.0f) ? -1.0f : 1.0f;
  const float a = fabsf(l);
  switch(e)
  {
    case GALOSH420_EOTF_SRGB:
      /* identical to legacy linear_to_srgb_f incl. its [0,1] clip. */
      if(l <= 0.0f) return 0.0f;
      if(l >= 1.0f) return 1.0f;
      return (l <= 0.0031308f) ? 12.92f * l
                               : 1.055f * powf(l, 1.0f / 2.4f) - 0.055f;
    case GALOSH420_EOTF_G22:   return s * powf(a, 1.0f / 2.2f);
    case GALOSH420_EOTF_G24:   return s * powf(a, 1.0f / 2.4f);
    case GALOSH420_EOTF_BT709: /* BT.709 OETF */
      return (a < 0.018f) ? 4.5f * l
                          : s * (1.099f * powf(a, 0.45f) - 0.099f);
    case GALOSH420_EOTF_HLG:
    {
      const float ha = 0.17883277f, hb = 0.28466892f, hc = 0.55991073f;
      const float v = (a <= 1.0f / 12.0f) ? sqrtf(3.0f * a)
                                          : ha * logf(12.0f * a - hb) + hc;
      return s * v;
    }
    case GALOSH420_EOTF_PQ:
    {
      const float m1 = 2610.0f / 16384.0f, m2 = 2523.0f / 4096.0f * 128.0f;
      const float c1 = 3424.0f / 4096.0f;
      const float c2 = 2413.0f / 4096.0f * 32.0f;
      const float c3 = 2392.0f / 4096.0f * 32.0f;
      const float ac = (a > 1.0f) ? 1.0f : a;
      const float lp = powf(ac, m1);
      return s * powf((c1 + c2 * lp) / (1.0f + c3 * lp), m2);
    }
    default: return l;
  }
}

/* ================================================================
 * NCL matrix (H.273): gamma-domain R'G'B' <-> Y'CbCr.
 *   Y' = Kr R' + Kg G' + Kb B';  Cb = (B'-Y')/(2(1-Kb));  Cr = (R'-Y')/(2(1-Kr))
 * Same formulas as the A/B rig (rgb2ycc / ycc2rgb).
 * ================================================================ */
static inline void galosh420_ncl_fwd(const float r, const float g, const float b,
                                     const galosh420_matrix_t m,
                                     float *y, float *cb, float *cr)
{
  const float kg = 1.0f - m.kr - m.kb;
  const float yy = m.kr * r + kg * g + m.kb * b;
  *y  = yy;
  *cb = (b - yy) / (2.0f * (1.0f - m.kb));
  *cr = (r - yy) / (2.0f * (1.0f - m.kr));
}

static inline void galosh420_ncl_inv(const float y, const float cb, const float cr,
                                     const galosh420_matrix_t m,
                                     float *r, float *g, float *b)
{
  const float kg = 1.0f - m.kr - m.kb;
  const float rr = y + 2.0f * (1.0f - m.kr) * cr;
  const float bb = y + 2.0f * (1.0f - m.kb) * cb;
  *r = rr;
  *b = bb;
  *g = (y - m.kr * rr - m.kb * bb) / kg;
}

/* ================================================================
 * Range / depth: integer container codes <-> float gamma-domain values.
 *
 * EN: limited range uses the 8-bit constants x 2^(n-8) (BT.601/709/2020);
 *     full range divides by 2^n - 1 with the chroma zero at 2^(n-1)
 *     (JFIF convention).  Dequant does NOT clamp (sub-black / super-white
 *     excursions are legal and preserved); requant rounds-to-nearest and
 *     clamps to the full code range [0, 2^n - 1].
 * JP: dequant はクランプしない（super-white 等は合法・保存）。requant は
 *     四捨五入 + フル符号域クランプ。
 * ================================================================ */
static inline float galosh420_dequant_y(const float code, const int depth,
                                        const galosh420_range_t range)
{
  if(range == GALOSH420_RANGE_LIMITED)
  {
    const float s = (float)(1 << (depth - 8));
    return (code - 16.0f * s) / (219.0f * s);
  }
  return code / (float)((1 << depth) - 1);
}

static inline float galosh420_dequant_c(const float code, const int depth,
                                        const galosh420_range_t range)
{
  if(range == GALOSH420_RANGE_LIMITED)
  {
    const float s = (float)(1 << (depth - 8));
    return (code - 128.0f * s) / (224.0f * s);
  }
  return (code - (float)(1 << (depth - 1))) / (float)((1 << depth) - 1);
}

static inline int galosh420_requant_y(const float v, const int depth,
                                      const galosh420_range_t range)
{
  float code;
  if(range == GALOSH420_RANGE_LIMITED)
  {
    const float s = (float)(1 << (depth - 8));
    code = v * 219.0f * s + 16.0f * s;
  }
  else code = v * (float)((1 << depth) - 1);
  const int c = (int)lrintf(code);
  const int hi = (1 << depth) - 1;
  return (c < 0) ? 0 : (c > hi) ? hi : c;
}

static inline int galosh420_requant_c(const float v, const int depth,
                                      const galosh420_range_t range)
{
  float code;
  if(range == GALOSH420_RANGE_LIMITED)
  {
    const float s = (float)(1 << (depth - 8));
    code = v * 224.0f * s + 128.0f * s;
  }
  else code = v * (float)((1 << depth) - 1) + (float)(1 << (depth - 1));
  const int c = (int)lrintf(code);
  const int hi = (1 << depth) - 1;
  return (c < 0) ? 0 : (c > hi) ? hi : c;
}

/* ================================================================
 * Siting-phased luma guide: full-res Y' plane -> chroma-lattice guide.
 *
 * EN: Ports the A/B rig kernels verbatim (down420 in ab_yuv420.py) — the
 *     exact code validated by the phase tests and the siting x quality
 *     experiment.  Kernels are per-siting IMPLEMENTATION CHOICES evaluated
 *     at the siting phase (layer 4 of the framework):
 *       center : 2x2 box                        -> centers (2j+1, 2i+1)
 *       left   : horizontal 3-tap tent [.25,.5,.25] at even columns
 *                + vertical 2-tap box           -> centers (2j+0.5, 2i+1)
 *       topleft: 3-tap tent both axes at even rows/cols
 *                                               -> centers (2j+0.5, 2i+0.5)
 *     Box and tent are affine-exact, so galosh420_phase_selftest() proves
 *     the phase to float rounding.  Edge handling: clamp (edge-pad).
 * JP: A/B リグのカーネルを逐語移植（検証済み実装がそのまま基準）。
 *     box/tent はアフィン厳密 → selftest が位相を機械証明する。
 *
 * W, H even; out is (W/2) x (H/2).  In-place NOT allowed.
 * ================================================================ */
static inline void galosh420_down_luma(const float *restrict yp,
                                       const int W, const int H,
                                       float *restrict out,
                                       const galosh420_siting_t siting)
{
  const int W2 = W / 2, H2 = H / 2;
  if(siting == GALOSH420_SITING_CENTER)
  {
    for(int i = 0; i < H2; i++)
      for(int j = 0; j < W2; j++)
      {
        const float *r0 = yp + (size_t)(2 * i) * W + 2 * j;
        const float *r1 = r0 + W;
        out[(size_t)i * W2 + j] =
            0.25f * (r0[0] + r0[1] + r1[0] + r1[1]);
      }
    return;
  }
  /* left / topleft share the horizontal tent at even columns. */
  for(int i = 0; i < H2; i++)
  {
    const int y0 = 2 * i, y1 = 2 * i + 1;
    for(int j = 0; j < W2; j++)
    {
      const int x  = 2 * j;
      const int xm = (x > 0) ? x - 1 : 0;          /* edge clamp */
      const int xp = (x < W - 1) ? x + 1 : W - 1;
      if(siting == GALOSH420_SITING_LEFT)
      {
        const float *ra = yp + (size_t)y0 * W;
        const float *rb = yp + (size_t)y1 * W;
        const float ha = 0.25f * ra[xm] + 0.5f * ra[x] + 0.25f * ra[xp];
        const float hb = 0.25f * rb[xm] + 0.5f * rb[x] + 0.25f * rb[xp];
        out[(size_t)i * W2 + j] = 0.5f * (ha + hb);
      }
      else /* TOPLEFT: vertical tent at even rows over horizontal tents */
      {
        const int ym = (y0 > 0) ? y0 - 1 : 0;
        const float *rm = yp + (size_t)ym * W;
        const float *rc = yp + (size_t)y0 * W;
        const float *rp = yp + (size_t)y1 * W;   /* y0+1 = 2i+1 <= H-1 */
        const float hm = 0.25f * rm[xm] + 0.5f * rm[x] + 0.25f * rm[xp];
        const float hc = 0.25f * rc[xm] + 0.5f * rc[x] + 0.25f * rc[xp];
        const float hp = 0.25f * rp[xm] + 0.5f * rp[x] + 0.25f * rp[xp];
        out[(size_t)i * W2 + j] = 0.25f * hm + 0.5f * hc + 0.25f * hp;
      }
    }
  }
}

/* ================================================================
 * 4:2:2 helpers (provisional 444 treatment; video 422 is horizontally
 * co-sited by convention — JPEG-style centered 422 unsupported in v1).
 *   up  : even col = own sample (co-sited), odd col = midway 0.5/0.5
 *   down: horizontal 3-tap tent [.25,.5,.25] at even columns (edge clamp)
 * ================================================================ */
static inline void galosh420_up422_h(const float *restrict cin,
                                     const int W2, const int H,
                                     float *restrict out /* W2*2 x H */)
{
  const int W = W2 * 2;
  for(int y = 0; y < H; y++)
  {
    const float *r = cin + (size_t)y * W2;
    float *o = out + (size_t)y * W;
    for(int j = 0; j < W2; j++)
    {
      const float a = r[j];
      const float b = (j < W2 - 1) ? r[j + 1] : r[j];
      o[2 * j]     = a;
      o[2 * j + 1] = 0.5f * (a + b);
    }
  }
}

static inline void galosh420_down422_h(const float *restrict cin,
                                       const int W, const int H,
                                       float *restrict out /* W/2 x H */)
{
  const int W2 = W / 2;
  for(int y = 0; y < H; y++)
  {
    const float *r = cin + (size_t)y * W;
    float *o = out + (size_t)y * W2;
    for(int j = 0; j < W2; j++)
    {
      const int x  = 2 * j;
      const int xm = (x > 0) ? x - 1 : 0;
      const int xp = (x < W - 1) ? x + 1 : W - 1;
      o[j] = 0.25f * r[xm] + 0.5f * r[x] + 0.25f * r[xp];
    }
  }
}

/* ================================================================
 * MANDATORY affine-field phase selftest (spec §guide construction).
 *
 * EN: Feed the analytic affine field f(x,y) = ax + by + c (evaluated at
 *     luma pixel centers (k+0.5, i+0.5) in corner coords) through
 *     galosh420_down_luma; interior guide samples must equal
 *     f(chroma sample-center position) to float rounding (box/tent are
 *     affine-exact, so ANY residual = phase error).  Detection power is
 *     also asserted: a deliberately CENTER-phased guide evaluated against
 *     LEFT-sited positions must show mean |err| = 0.5*|a| (the 2026-07-11
 *     session measured 1e-16 / 0.00650 vs theory 0.00650 in the rig).
 * JP: アフィン場をガイド生成に通し、内部サンプルが解析値と一致することを
 *     機械検証（残差 = 位相誤差）。center ガイド vs left 位置 = 0.5|a| の
 *     検出力も同時に確認。
 *
 * Returns 0 on PASS, 1 on FAIL (diagnostics on stderr).
 * ================================================================ */
static int galosh420_phase_selftest(void)
{
  enum { W = 64, H = 48 };
  const int W2 = W / 2, H2 = H / 2;
  const float ax = 0.0130f, by = -0.0071f, cc = 0.4f;
  static float yp[(size_t)W * H], g[(size_t)(W / 2) * (H / 2)];
  for(int i = 0; i < H; i++)
    for(int k = 0; k < W; k++)
      yp[(size_t)i * W + k] = ax * (k + 0.5f) + by * (i + 0.5f) + cc;

  /* Layer-1 sample-center positions per siting (corner coords). */
  const float cx[3] = { 1.0f, 0.5f, 0.5f };   /* + 2j */
  const float cy[3] = { 1.0f, 1.0f, 0.5f };   /* + 2i */
  const char *names[3] = { "center", "left", "topleft" };
  int fail = 0;

  for(int s = 0; s < 3; s++)
  {
    galosh420_down_luma(yp, W, H, g, (galosh420_siting_t)s);
    float maxres = 0.0f;
    for(int i = 1; i < H2 - 1; i++)          /* interior: edge clamp breaks */
      for(int j = 1; j < W2 - 1; j++)        /* affinity on the border ring */
      {
        const float want = ax * (2 * j + cx[s]) + by * (2 * i + cy[s]) + cc;
        const float res = fabsf(g[(size_t)i * W2 + j] - want);
        if(res > maxres) maxres = res;
      }
    const int ok = (maxres < 1e-5f);
    fprintf(stderr, "[420-selftest] %-7s phase residual max = %.3g  %s\n",
            names[s], (double)maxres, ok ? "PASS" : "FAIL");
    if(!ok) fail = 1;
  }

  /* Detection power: center-phased guide vs LEFT positions = 0.5*|ax|. */
  galosh420_down_luma(yp, W, H, g, GALOSH420_SITING_CENTER);
  double acc = 0.0; int n = 0;
  for(int i = 1; i < H2 - 1; i++)
    for(int j = 1; j < W2 - 1; j++)
    {
      const float want = ax * (2 * j + 0.5f) + by * (2 * i + 1.0f) + cc;
      acc += fabsf(g[(size_t)i * W2 + j] - want);
      n++;
    }
  const double mean_err = acc / (double)n;
  const double theory = 0.5 * fabsf(ax);
  const int ok = fabs(mean_err - theory) < 1e-4;
  fprintf(stderr, "[420-selftest] mismatch detection: mean err %.6f vs "
                  "theory 0.5|ax| = %.6f  %s\n",
          mean_err, theory, ok ? "PASS" : "FAIL");
  if(!ok) fail = 1;
  return fail;
}

/* ================================================================
 * CLI string parsers (shared across drivers so flag vocabulary never
 * drifts between CPU / OpenCL / Vulkan).  Return 0 on success.
 * ================================================================ */
static inline int galosh420_parse_siting(const char *s, galosh420_siting_t *out)
{
  if(!strcmp(s, "center"))  { *out = GALOSH420_SITING_CENTER;  return 0; }
  if(!strcmp(s, "left"))    { *out = GALOSH420_SITING_LEFT;    return 0; }
  if(!strcmp(s, "topleft")) { *out = GALOSH420_SITING_TOPLEFT; return 0; }
  return 1;
}

static inline int galosh420_parse_eotf(const char *s, galosh420_eotf_t *out)
{
  if(!strcmp(s, "srgb"))   { *out = GALOSH420_EOTF_SRGB;   return 0; }
  if(!strcmp(s, "g22"))    { *out = GALOSH420_EOTF_G22;    return 0; }
  if(!strcmp(s, "g24"))    { *out = GALOSH420_EOTF_G24;    return 0; }
  if(!strcmp(s, "bt709"))  { *out = GALOSH420_EOTF_BT709;  return 0; }
  if(!strcmp(s, "hlg"))    { *out = GALOSH420_EOTF_HLG;    return 0; }
  if(!strcmp(s, "pq"))     { *out = GALOSH420_EOTF_PQ;     return 0; }
  if(!strcmp(s, "linear")) { *out = GALOSH420_EOTF_LINEAR; return 0; }
  return 1;
}

static inline int galosh420_parse_matrix(const char *s, galosh420_matrix_t *out)
{
  if(!strcmp(s, "bt601"))  { *out = GALOSH420_MAT_BT601;  return 0; }
  if(!strcmp(s, "bt709"))  { *out = GALOSH420_MAT_BT709;  return 0; }
  if(!strcmp(s, "bt2020")) { *out = GALOSH420_MAT_BT2020; return 0; }
  if(!strncmp(s, "custom:", 7))
  {
    float kr = 0.0f, kb = 0.0f;
    if(sscanf(s + 7, "%f,%f", &kr, &kb) == 2 &&
       kr > 0.0f && kb > 0.0f && kr + kb < 1.0f)
    { out->kr = kr; out->kb = kb; return 0; }
  }
  return 1;
}

static inline int galosh420_parse_range(const char *s, galosh420_range_t *out)
{
  if(!strcmp(s, "full"))    { *out = GALOSH420_RANGE_FULL;    return 0; }
  if(!strcmp(s, "limited")) { *out = GALOSH420_RANGE_LIMITED; return 0; }
  return 1;
}

#endif /* GALOSH_YUV420_H */
