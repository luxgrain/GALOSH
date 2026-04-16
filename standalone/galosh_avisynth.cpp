/*
 * galosh_avisynth.cpp
 * AviSynth+ plugin wrapper for yuv_galosh_core.c
 *
 * Exports:
 *   GALOSHDenoise(clip ref,
 *                 clip b1, clip f1,
 *                 clip "b2", clip "f2",
 *                 float "sy"    = 1.0,
 *                 float "sc"    = 1.0,
 *                 float "gamma" = 2.2,   (default; use 1.0 for linear RAW)
 *                 int   "stride_y" = 2,
 *                 int   "stride_c" = 2)
 *
 *   ref  : current noisy frame (YUV420 8-bit or 16-bit)
 *   b1   : MVCompensate backward  delta=1
 *   f1   : MVCompensate forward   delta=1
 *   b2   : MVCompensate backward  delta=2 (optional, enables tr=2)
 *   f2   : MVCompensate forward   delta=2 (optional, enables tr=2)
 *
 * Build (MSYS2 UCRT64):
 *   export PATH="/c/msys64/ucrt64/bin:$PATH"
 *   g++ -O3 -march=native -shared -o GALOSHDenoise.dll galosh_avisynth.cpp \
 *       -I. -fopenmp -lm -DGALOSH_AVS_PLUGIN \
 *       -Wl,--enable-auto-import
 *
 * Install:
 *   Copy GALOSHDenoise.dll to AviSynth+ plugins folder.
 */

#include "avisynth.h"
#include <cstring>

/* Required by AviSynth+ plugin interface */
const AVS_Linkage *AVS_linkage = nullptr;
#include <cstdlib>
#include <cmath>
#include <algorithm>

/* galosh_yuv_denoise is compiled separately from yuv_galosh_core.c (as C).
 * Include only the public API header here. */
#include "yuv_galosh_api.h"

/* ── helpers ─────────────────────────────────────────────────────── */
/* comp_size: vi.ComponentSize() — 1 for 8-bit, 2 for 16-bit           */

static void avs_luma_to_float(const PVideoFrame &frame,
                               float *out, int W, int H, int comp_size)
{
    const int   pitch = frame->GetPitch(PLANAR_Y);
    const BYTE *src   = frame->GetReadPtr(PLANAR_Y);

    if(comp_size == 1) {
        const float scale = 1.0f / 255.0f;
        for(int y = 0; y < H; y++) {
            const BYTE *row = src + (size_t)y * pitch;
            float      *dst = out + (size_t)y * W;
            for(int x = 0; x < W; x++) dst[x] = row[x] * scale;
        }
    } else {
        const float scale = 1.0f / 65535.0f;
        for(int y = 0; y < H; y++) {
            const uint16_t *row = reinterpret_cast<const uint16_t *>(src + (size_t)y * pitch);
            float          *dst = out + (size_t)y * W;
            for(int x = 0; x < W; x++) dst[x] = row[x] * scale;
        }
    }
}

/* Chroma: offset-binary (centre = 128 / 32768) → signed [-0.5, +0.5] */
static void avs_chroma_to_float(const PVideoFrame &frame, int plane,
                                 float *out, int cW, int cH, int comp_size)
{
    const int   pitch = frame->GetPitch(plane);
    const BYTE *src   = frame->GetReadPtr(plane);

    if(comp_size == 1) {
        const float scale = 1.0f / 255.0f;
        for(int y = 0; y < cH; y++) {
            const BYTE *row = src + (size_t)y * pitch;
            float      *dst = out + (size_t)y * cW;
            for(int x = 0; x < cW; x++) dst[x] = row[x] * scale - 0.5f;
        }
    } else {
        const float scale = 1.0f / 65535.0f;
        for(int y = 0; y < cH; y++) {
            const uint16_t *row = reinterpret_cast<const uint16_t *>(src + (size_t)y * pitch);
            float          *dst = out + (size_t)y * cW;
            for(int x = 0; x < cW; x++) dst[x] = row[x] * scale - 0.5f;
        }
    }
}

static void float_to_avs_luma(const float *in,
                               const PVideoFrame &frame,
                               int W, int H, int comp_size)
{
    const int  pitch = frame->GetPitch(PLANAR_Y);
    BYTE      *dst   = frame->GetWritePtr(PLANAR_Y);

    if(comp_size == 1) {
        for(int y = 0; y < H; y++) {
            BYTE        *row = dst + (size_t)y * pitch;
            const float *src = in  + (size_t)y * W;
            for(int x = 0; x < W; x++) {
                const float v = src[x] < 0.0f ? 0.0f : (src[x] > 1.0f ? 1.0f : src[x]);
                row[x] = static_cast<BYTE>(v * 255.0f + 0.5f);
            }
        }
    } else {
        for(int y = 0; y < H; y++) {
            uint16_t    *row = reinterpret_cast<uint16_t *>(dst + (size_t)y * pitch);
            const float *src = in  + (size_t)y * W;
            for(int x = 0; x < W; x++) {
                const float v = src[x] < 0.0f ? 0.0f : (src[x] > 1.0f ? 1.0f : src[x]);
                row[x] = static_cast<uint16_t>(v * 65535.0f + 0.5f);
            }
        }
    }
}

static void float_to_avs_chroma(const float *in,
                                 const PVideoFrame &frame, int plane,
                                 int cW, int cH, int comp_size)
{
    const int  pitch = frame->GetPitch(plane);
    BYTE      *dst   = frame->GetWritePtr(plane);

    if(comp_size == 1) {
        for(int y = 0; y < cH; y++) {
            BYTE        *row = dst + (size_t)y * pitch;
            const float *src = in  + (size_t)y * cW;
            for(int x = 0; x < cW; x++) {
                const float v = src[x] + 0.5f;
                const float c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                row[x] = static_cast<BYTE>(c * 255.0f + 0.5f);
            }
        }
    } else {
        for(int y = 0; y < cH; y++) {
            uint16_t    *row = reinterpret_cast<uint16_t *>(dst + (size_t)y * pitch);
            const float *src = in  + (size_t)y * cW;
            for(int x = 0; x < cW; x++) {
                const float v = src[x] + 0.5f;
                const float c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                row[x] = static_cast<uint16_t>(c * 65535.0f + 0.5f);
            }
        }
    }
}

/* ── Filter class ────────────────────────────────────────────────── */

class GALOSHDenoise : public GenericVideoFilter
{
public:
    /* Up to 4 MC clips: b1 f1 b2 f2 */
    static const int MAX_MC = 4;

    PClip     mc[MAX_MC];
    int       n_mc;          /* number of MC clips actually provided */
    float     sy, sc, gamma_curve;
    float     syf, scf;      /* sigma scale factor (0 or 1 = auto, >1 = multiply MAD estimate) */
    int       stride_y, stride_c;
    int       use_3dwht;     /* 0=temporal-mean+Wiener, 1=3D WHT BayesShrink */

    GALOSHDenoise(PClip _ref,
                  PClip _b1, PClip _f1,
                  PClip _b2, PClip _f2,
                  float _sy, float _sc, float _gamma,
                  float _syf, float _scf,
                  int _stride_y, int _stride_c,
                  int _use_3dwht,
                  IScriptEnvironment *env)
        : GenericVideoFilter(_ref),
          sy(_sy), sc(_sc), gamma_curve(_gamma),
          syf(_syf), scf(_scf),
          stride_y(_stride_y), stride_c(_stride_c),
          use_3dwht(_use_3dwht)
    {
        /* Validate format: must be YUV420 */
        if(!vi.IsYUV() || vi.NumComponents() < 3 ||
           vi.GetPlaneWidthSubsampling(PLANAR_U) != 1 ||
           vi.GetPlaneHeightSubsampling(PLANAR_U) != 1)
            env->ThrowError("GALOSHDenoise: input must be YUV420 (4:2:0)");

        mc[0] = _b1;  mc[1] = _f1;
        mc[2] = _b2;  mc[3] = _f2;

        n_mc = 0;
        if(_b1 && _f1) { n_mc = 2; }
        if(n_mc == 2 && _b2 && _f2) { n_mc = 4; }

        /* All MC clips must share format with ref */
        for(int i = 0; i < n_mc; i++) {
            if(!mc[i]) env->ThrowError("GALOSHDenoise: MC clip %d is null", i+1);
            const VideoInfo &mvi = mc[i]->GetVideoInfo();
            if(mvi.width != vi.width || mvi.height != vi.height)
                env->ThrowError("GALOSHDenoise: MC clip %d size mismatch", i+1);
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
        const int W  = vi.width;
        const int H  = vi.height;
        const int cW = W / 2;
        const int cH = H / 2;
        const size_t y_npix = (size_t)W  * H;
        const size_t c_npix = (size_t)cW * cH;

        /* --- fetch source frames --- */
        PVideoFrame ref_frame = child->GetFrame(n, env);

        /* MC frames (may be NULL if n_mc==0) */
        PVideoFrame mc_frame[MAX_MC] = {};
        for(int i = 0; i < n_mc; i++)
            mc_frame[i] = mc[i]->GetFrame(n, env);

        /* --- allocate float buffers --- */
        /* ref plane buffers */
        float *y_in = static_cast<float *>(_aligned_malloc(y_npix * sizeof(float), 64));
        float *u_in = static_cast<float *>(_aligned_malloc(c_npix * sizeof(float), 64));
        float *v_in = static_cast<float *>(_aligned_malloc(c_npix * sizeof(float), 64));
        /* output plane buffers */
        float *y_out = static_cast<float *>(_aligned_malloc(y_npix * sizeof(float), 64));
        float *u_out = static_cast<float *>(_aligned_malloc(c_npix * sizeof(float), 64));
        float *v_out = static_cast<float *>(_aligned_malloc(c_npix * sizeof(float), 64));

        if(!y_in||!u_in||!v_in||!y_out||!u_out||!v_out) {
            _aligned_free(y_in);  _aligned_free(u_in);  _aligned_free(v_in);
            _aligned_free(y_out); _aligned_free(u_out); _aligned_free(v_out);
            env->ThrowError("GALOSHDenoise: out of memory");
        }

        const int cs = vi.ComponentSize();  /* 1=8bit, 2=16bit */

        /* fill ref */
        avs_luma_to_float  (ref_frame,           y_in, W,  H,  cs);
        avs_chroma_to_float(ref_frame, PLANAR_U, u_in, cW, cH, cs);
        avs_chroma_to_float(ref_frame, PLANAR_V, v_in, cW, cH, cs);

        /* MC plane buffers */
        float *mc_y_buf[MAX_MC] = {};
        float *mc_u_buf[MAX_MC] = {};
        float *mc_v_buf[MAX_MC] = {};
        const float *mc_y_arr[MAX_MC] = {};
        const float *mc_u_arr[MAX_MC] = {};
        const float *mc_v_arr[MAX_MC] = {};

        for(int i = 0; i < n_mc; i++) {
            mc_y_buf[i] = static_cast<float *>(_aligned_malloc(y_npix * sizeof(float), 64));
            mc_u_buf[i] = static_cast<float *>(_aligned_malloc(c_npix * sizeof(float), 64));
            mc_v_buf[i] = static_cast<float *>(_aligned_malloc(c_npix * sizeof(float), 64));
            if(!mc_y_buf[i]||!mc_u_buf[i]||!mc_v_buf[i])
                env->ThrowError("GALOSHDenoise: MC alloc failed at clip %d", i+1);

            avs_luma_to_float  (mc_frame[i],           mc_y_buf[i], W,  H,  cs);
            avs_chroma_to_float(mc_frame[i], PLANAR_U, mc_u_buf[i], cW, cH, cs);
            avs_chroma_to_float(mc_frame[i], PLANAR_V, mc_v_buf[i], cW, cH, cs);

            mc_y_arr[i] = mc_y_buf[i];
            mc_u_arr[i] = mc_u_buf[i];
            mc_v_arr[i] = mc_v_buf[i];
        }

        /* --- run GALOSH --- */
        galosh_yuv_params_t params = {};
        params.sigma_y     = 1.0f;
        params.sigma_c     = 1.0f;
        params.stride_y    = 2;
        params.stride_c    = 2;
        params.tr          = 0;
        params.gamma_curve = 1.0f;
        params.yuv_format  = GALOSH_YUV_420;
        params.sigma_y       = sy;
        params.sigma_c       = sc;
        params.gamma_curve   = gamma_curve;
        params.sigma_y_scale = syf;
        params.sigma_c_scale = scf;
        params.stride_y      = stride_y;
        params.stride_c      = stride_c;
        params.tr            = n_mc / 2;
        params.use_3dwht     = use_3dwht;

        galosh_yuv_denoise(y_out, u_out, v_out,
                           y_in,  u_in,  v_in,
                           W, H,
                           n_mc > 0 ? mc_y_arr : nullptr,
                           n_mc > 0 ? mc_u_arr : nullptr,
                           n_mc > 0 ? mc_v_arr : nullptr,
                           n_mc, &params);

        /* --- write output frame --- */
        PVideoFrame dst = env->NewVideoFrameP(vi, &ref_frame);

        float_to_avs_luma  (y_out, dst,           W,  H,  cs);
        float_to_avs_chroma(u_out, dst, PLANAR_U, cW, cH, cs);
        float_to_avs_chroma(v_out, dst, PLANAR_V, cW, cH, cs);

        /* cleanup */
        _aligned_free(y_in);  _aligned_free(u_in);  _aligned_free(v_in);
        _aligned_free(y_out); _aligned_free(u_out); _aligned_free(v_out);
        for(int i = 0; i < n_mc; i++) {
            _aligned_free(mc_y_buf[i]);
            _aligned_free(mc_u_buf[i]);
            _aligned_free(mc_v_buf[i]);
        }

        return dst;
    }

    static AVSValue __cdecl Create(AVSValue args, void *, IScriptEnvironment *env)
    {
        /* args[0]=ref  [1]=b1  [2]=f1  [3]=b2  [4]=f2
           [5]=sy  [6]=sc  [7]=gamma  [8]=syf  [9]=scf
           [10]=stride_y  [11]=stride_c */
        PClip b1 = args[1].Defined() ? args[1].AsClip() : PClip();
        PClip f1 = args[2].Defined() ? args[2].AsClip() : PClip();
        PClip b2 = args[3].Defined() ? args[3].AsClip() : PClip();
        PClip f2 = args[4].Defined() ? args[4].AsClip() : PClip();

        if(args[1].Defined() != args[2].Defined())
            env->ThrowError("GALOSHDenoise: b1 and f1 must both be provided");
        if(args[3].Defined() != args[4].Defined())
            env->ThrowError("GALOSHDenoise: b2 and f2 must both be provided for tr=2");

        return new GALOSHDenoise(
            args[0].AsClip(),
            b1, f1, b2, f2,
            (float)args[5].AsFloat(1.0f),   /* sy */
            (float)args[6].AsFloat(1.0f),   /* sc */
            (float)args[7].AsFloat(2.2f),   /* gamma */
            (float)args[8].AsFloat(0.0f),   /* syf: 0/1=auto, >1=scale */
            (float)args[9].AsFloat(0.0f),   /* scf: 0/1=auto, >1=scale */
            args[10].AsInt(2),               /* stride_y */
            args[11].AsInt(2),               /* stride_c */
            args[12].AsBool(false) ? 1 : 0, /* use_3dwht */
            env
        );
    }
};

/* ── Plugin entry point ──────────────────────────────────────────── */

/* Diagnostic: GALOSHPing() -> int 1 (no args, tests basic AddFunction path) */
static AVSValue __cdecl GALOSHPing_Create(AVSValue, void *, IScriptEnvironment *)
{
    return AVSValue(1);
}

extern "C" __declspec(dllexport)
const char * __stdcall AvisynthPluginInit3(IScriptEnvironment *env,
                                            const AVS_Linkage * const vectors)
{
    AVS_linkage = vectors;

    /* Simple no-arg function to verify AddFunction works at all */
    env->AddFunction("GALOSHPing", "", GALOSHPing_Create, nullptr);

    env->AddFunction("GALOSHDenoise",
        /* ref [b1] [f1] [b2] [f2] [sy] [sc] [gamma] [syf] [scf] [stride_y] [stride_c] [use_3dwht] */
        "c[b1]c[f1]c[b2]c[f2]c[sy]f[sc]f[gamma]f[syf]f[scf]f[stride_y]i[stride_c]i[use_3dwht]b",
        GALOSHDenoise::Create, nullptr);

    return "GALOSHDenoise v1.0 - GALOSH YUV temporal denoiser";
}
