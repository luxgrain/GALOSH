/*
    This file is part of darktable,
    Copyright (C) 2026 darktable-rawforge contributors.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GALOSH-YUV — post-demosaic sRGB denoiser.

    Linear Y-GAT + 2-pass LOSH on luma + Y-guided LOESS chroma regression.
    Fully blind (Foi-Alenius-style MAD sigma on Y plane, quasi-linear GAT
    for signal-independent regime).  Operates on 4-channel RGBA pixels
    from darktable's post-demosaic pipeline; alpha is preserved.
*/
#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/imagebuf.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "develop/imageop_math.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Pull in GALOSH algorithm core + sRGB/YCbCr driver. */
#define GALOSH_DT_ENV
#include "galosh_core.h"
#include "galosh_yuv.h"

DT_MODULE_INTROSPECTION(1, dt_iop_galosh_yuv_params_t)

typedef struct dt_iop_galosh_yuv_params_t
{
  float strength_y;  // $MIN: 0.0 $MAX: 4.0 $DEFAULT: 1.0 $DESCRIPTION: "luma strength"
  float strength_c;  // $MIN: 0.0 $MAX: 4.0 $DEFAULT: 1.0 $DESCRIPTION: "chroma strength"
  float alpha;       // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "alpha (0 = blind)"
  float sigma_sq;    // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "sigma^2 (0 = blind)"
} dt_iop_galosh_yuv_params_t;

typedef struct dt_iop_galosh_yuv_gui_data_t
{
  GtkWidget *strength_y;
  GtkWidget *strength_c;
  GtkWidget *alpha;
  GtkWidget *sigma_sq;
} dt_iop_galosh_yuv_gui_data_t;

typedef struct dt_iop_galosh_yuv_data_t
{
  float strength_y, strength_c, alpha, sigma_sq;
} dt_iop_galosh_yuv_data_t;

typedef struct dt_iop_galosh_yuv_global_data_t
{
} dt_iop_galosh_yuv_global_data_t;


const char *name()
{
  return _("GALOSH-YUV");
}

const char *aliases()
{
  return _("galosh|yuv|denoise");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self,
    _("post-demosaic sRGB denoise via GAT + WHT-LOSH (luma) and Y-guided LOESS (chroma)"),
    _("corrective"),
    _("linear, RGB, scene-referred"),
    _("linear, RGB"),
    _("linear, RGB, scene-referred"));
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ALLOW_TILING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}


/* --- process ---
 * Interleave-unpack RGBA to planar RGB, call galosh_yuv_denoise_srgb,
 * then re-interleave.  Alpha channel is passed through unchanged. */
void process(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_galosh_yuv_data_t *const d = piece->data;

  const int W = roi_in->width;
  const int H = roi_in->height;
  const size_t npx = (size_t)W * (size_t)H;

  const float *const restrict in  = (const float *)ivoid;
  float       *const restrict out = (float       *)ovoid;

  if(d->strength_y <= 0.0f && d->strength_c <= 0.0f)
  {
    dt_iop_image_copy_by_size(ovoid, ivoid, W, H, piece->colors);
    return;
  }

  float *const restrict tri_in  = dt_alloc_align_float(npx * 3);
  float *const restrict tri_out = dt_alloc_align_float(npx * 3);
  if(!tri_in || !tri_out)
  {
    dt_free_align(tri_in);
    dt_free_align(tri_out);
    dt_iop_image_copy_by_size(ovoid, ivoid, W, H, piece->colors);
    return;
  }

  /* RGBA -> RGB (drop alpha for denoise). */
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
  {
    tri_in[3 * i + 0] = in[4 * i + 0];
    tri_in[3 * i + 1] = in[4 * i + 1];
    tri_in[3 * i + 2] = in[4 * i + 2];
  }

  galosh_yuv_denoise_srgb(tri_in, tri_out, W, H,
                          d->strength_y, d->strength_c,
                          d->alpha, d->sigma_sq);

  /* RGB -> RGBA (preserve alpha from input). */
  DT_OMP_FOR()
  for(size_t i = 0; i < npx; i++)
  {
    out[4 * i + 0] = tri_out[3 * i + 0];
    out[4 * i + 1] = tri_out[3 * i + 1];
    out[4 * i + 2] = tri_out[3 * i + 2];
    out[4 * i + 3] = in [4 * i + 3];
  }

  dt_free_align(tri_in);
  dt_free_align(tri_out);
}


void commit_params(dt_iop_module_t *self, dt_iop_params_t *params,
                   dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_galosh_yuv_params_t *p = (dt_iop_galosh_yuv_params_t *)params;
  dt_iop_galosh_yuv_data_t *d = piece->data;
  d->strength_y = p->strength_y;
  d->strength_c = p->strength_c;
  d->alpha      = p->alpha;
  d->sigma_sq   = p->sigma_sq;
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_galosh_yuv_data_t));
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}


void gui_init(dt_iop_module_t *self)
{
  dt_iop_galosh_yuv_gui_data_t *g = IOP_GUI_ALLOC(galosh_yuv);

  g->strength_y = dt_bauhaus_slider_from_params(self, "strength_y");
  dt_bauhaus_slider_set_digits(g->strength_y, 2);
  gtk_widget_set_tooltip_text(g->strength_y,
    _("luma denoise strength — controls the WHT-LOSH threshold on Y."));

  g->strength_c = dt_bauhaus_slider_from_params(self, "strength_c");
  dt_bauhaus_slider_set_digits(g->strength_c, 2);
  gtk_widget_set_tooltip_text(g->strength_c,
    _("chroma denoise strength — controls the LOESS slope prior on Cb/Cr."));

  g->alpha = dt_bauhaus_slider_from_params(self, "alpha");
  dt_bauhaus_slider_set_digits(g->alpha, 4);
  gtk_widget_set_tooltip_text(g->alpha,
    _("Poisson gain alpha (0 = auto, blind MAD estimation on Y)."));

  g->sigma_sq = dt_bauhaus_slider_from_params(self, "sigma_sq");
  dt_bauhaus_slider_set_digits(g->sigma_sq, 6);
  gtk_widget_set_tooltip_text(g->sigma_sq,
    _("Gaussian read-noise variance (0 = auto, blind MAD estimation on Y)."));
}
