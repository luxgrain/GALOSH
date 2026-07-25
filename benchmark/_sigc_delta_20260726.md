# sigma_C radius + one-sided eps rerun delta (new - old), 2026-07-26

old = envelope-canonical v0.5.0 DLL (420 adaptive chroma radius semantically dead at R=2, constant ridge eps); new = sigma_C-driven R (T_C=0.020) + one-sided MAP-ridge eps (anchor 0.012, cap 4.0), defaults ON.  Mean over all cells present in BOTH runs.  420 lanes only (444 untouched by the change).

| track | method | dPSNR | dSSIM | dLPIPS | dDISTS | dNIQE | cells |
|---|---|---|---|---|---|---|---|
| awgn-420 | galosh-cpu-fit | +0.069 | +0.0183 | -0.0087 | -0.0058 | -0.055 | 40 |
| awgn-420 | galosh-cpu-hold | +0.071 | +0.0184 | -0.0087 | -0.0059 | -0.056 | 40 |
| awgn-420 | galosh-vk-fit | +0.070 | +0.0183 | -0.0088 | -0.0059 | -0.054 | 40 |
| awgn-420 | galosh-vk-hold | +0.056 | +0.0168 | -0.0099 | -0.0060 | -0.051 | 40 |
| pg-core-420 | galosh-cpu-fit | +0.057 | +0.0109 | -0.0038 | -0.0069 | +0.013 | 56 |
| pg-core-420 | galosh-cpu-hold | +0.064 | +0.0111 | -0.0039 | -0.0071 | +0.013 | 56 |
| pg-core-420 | galosh-vk-fit | +0.057 | +0.0110 | -0.0039 | -0.0069 | +0.013 | 56 |
| pg-core-420 | galosh-vk-hold | +0.019 | +0.0099 | -0.0027 | -0.0069 | +0.002 | 56 |
| pg-cmp-420 | galosh-cpu-fit | +0.021 | +0.0048 | -0.0012 | -0.0029 | +0.004 | 56 |
| pg-cmp-420 | galosh-cpu-hold | +0.018 | +0.0049 | -0.0011 | -0.0031 | +0.001 | 56 |
| pg-cmp-420 | galosh-vk-fit | +0.021 | +0.0048 | -0.0012 | -0.0029 | +0.004 | 56 |
| pg-cmp-420 | galosh-vk-hold | -0.004 | +0.0041 | -0.0003 | -0.0024 | +0.007 | 56 |
| crvd-420 | galosh-cpu-fit | +0.061 | +0.0023 | -0.0022 | -0.0019 | +0.022 | 55 |
| crvd-420 | galosh-cpu-hold | +0.061 | +0.0024 | -0.0022 | -0.0019 | +0.018 | 55 |
| crvd-420 | galosh-vk-fit | +0.061 | +0.0023 | -0.0022 | -0.0019 | +0.021 | 55 |
| crvd-420 | galosh-vk-hold | +0.111 | +0.0049 | -0.0062 | -0.0028 | +0.018 | 55 |

## per-level dPSNR (galosh-cpu-fit)

- **awgn-420**: s10: +0.00 / s20: +0.05 / s30: +0.09 / s40: +0.11 / s50: +0.09
- **pg-core-420**: ISO400: -0.00 / ISO800: +0.00 / ISO1600: +0.01 / ISO3200: +0.05 / ISO6400: +0.09 / ISO12800: +0.14 / ISO25600: +0.12
- **pg-cmp-420**: ISO400: +0.00 / ISO800: +0.00 / ISO1600: +0.00 / ISO3200: +0.00 / ISO6400: +0.01 / ISO12800: +0.04 / ISO25600: +0.09
- **crvd-420**: ISO1600: -0.00 / ISO3200: -0.00 / ISO6400: +0.01 / ISO12800: +0.04 / ISO25600: +0.26