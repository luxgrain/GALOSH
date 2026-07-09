#!/usr/bin/env bash
# build_vk.sh — compile the ACTIVE o32 shaders to SPIR-V + the Vulkan host.
# Usage: PATH=/c/msys64/ucrt64/bin:$PATH ./build_vk.sh
# [FP16 v1] --target-env=vulkan1.2: StorageBuffer16BitAccess / Float16
# SPIR-V capabilities are core in 1.1/1.2 (no per-vendor extension strings);
# host creates the device with apiVersion 1.2 + the two feature structs.
set -eu
cd "$(dirname "$0")"

ACTIVE="o32_ne_block_stats o32_ne_finalize o32_ne_dark_thresh_hist \
o32_ne_dark_thresh_finalize o32_ne_dark_lap_hist o32_ne_dark_finalize \
o32_build_inv_lut o32_lut_finalize o32_gat_forward_full o32_sigma_per_cfa \
o32_unified_sigma o32_normalize_apply o32_dark_ref_reduce_mwg \
o32_dark_ref_finalize_mwg o32_dark_resid_reduce_mwg o32_dark_resid_finalize_mwg \
o32_dark_sub_full o32_forward_l_stride1 o32_chroma_extract_halfres \
o32_pass12 o32_pass1_dump o32_lpixel_lh_den_fused o32_box_downsample_2x \
o32_box_downsample_2x_3p o32_loess_chroma_3p_tiled o32_crop_2d_topleft \
o32_k16_jbu_3p o32_pad_2d_edge o32_smoothstep_blend_3p o32_inverse_wht_dark_gat \
o32_pass12_wht4 o32_fastup_3p \
o32_box_downsample_2x_h16 o32_crop_2d_topleft_h16 o32_loess_chroma_3p_tiled_g16 \
o32_k16_jbu_3p_f16 o32_fastup_3p_f16"

for k in $ACTIVE; do
  glslc -O --target-env=vulkan1.2 "shaders/$k.comp" -o "shaders/$k.spv"
done
echo "shaders: $(echo $ACTIVE | wc -w) compiled"

gcc -O2 -std=c11 galosh_vk.c -o galosh_vk.exe -lvulkan-1 -lm
echo "galosh_vk.exe built"
