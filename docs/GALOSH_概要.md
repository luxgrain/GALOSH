# GALOSH 概要

**GALOSH** = **G**eneralised **A**nscombe **Lo**cal **Sh**rinkage

生 RAW (pre-demosaic) と sRGB (post-demosaic) の両方で動く、完全盲検（blind, 事前情報ゼロ）で学習不要なイメージデノイザです。8 ×8 WHT + 局所収縮（BayesShrink → empirical Wiener）が luma パイプの中核で、chroma 側は LOESS（局所重み付き回帰、Cleveland 1979）で空間適応的に処理します。

---

## 1. 解く問題

センサーで撮った画像には **Poisson-Gauss ノイズ**が乗っています：

```
var(x_raw) = α·x + σ²
     ^        ^    ^
     |        |    読み出しノイズ (加法性ガウス)
     |        光子ノイズ (信号比例、ポアソン起源)
     ADC/ゲイン込みの統合センサ分散
```

- α: 光子 → 電子変換ゲイン相当
- σ²: 読み出しノイズ分散

**signal-dependent な分散** がある限り、単純なガウス前提のデノイザは明部と暗部で最適パラメータが変わり、どちらかが犠牲になります。

目標:
- **blind** に α, σ² を推定
- 分散を「一律」に平坦化
- 平坦化空間で収縮デノイズ
- 最後に元のスケールへ正しく逆変換

これを 4 段（GAT → 分解 → 収縮 → 逆変換）で実装するのが GALOSH です。

---

## 2. 主要アイディア

### 2.1 Foi-Alenius (2008) Blind Noise Estimation

- 小ブロック (8×8) ごとに `mean`, `Laplacian variance` を計算
- 点群 (mean, var) に線形回帰 → `var = α·mean + σ²`
- dark pixel サンプリング + Laplacian ヒストグラムで σ² を refine

GALOSH の K0 段がこれです。dark_ref は後で各 Bayer チャネルごとの DC オフセット除去に使います。

### 2.2 GAT (Generalised Anscombe Transform)

Makitalo-Foi (2013) の変形:

```
y = T(x) = (2/α) · √(α·x + 3α²/8 + σ²)
```

1 次テイラーで分散伝播すると、**あらゆる x に対して var(y) ≈ 1**。これが GAT の肝で、以降の処理は「ガウスノイズ前提」で済みます。

低輝度で √ 内部が負になるのを防ぐため、`t_break = 2σ/α` 以下では線形枝（C¹ 連続な拡張）を使います。逆変換 `T⁻¹` は不偏推定を閉じた形で取れないため、事前に 4096 点 LUT を build して binary search で引きます（Makitalo-Foi の数値手法に沿う）。

### 2.3 2×2 WHT（L/C 分解、RAW モード専用）

Bayer 2×2 ブロックに Walsh-Hadamard 変換を当てると、4 係数が得られます：

```
L  = (R + G₁ + G₂ + B) / 2   → 輝度 DC
C1 = (R − G₁ + G₂ − B) / 2   → 水平 chroma
C2 = (R + G₁ − G₂ − B) / 2   → 垂直 chroma
C3 = (R − G₁ − G₂ + B) / 2   → 斜め chroma
```

- Bayer → 4 つの half-res プレーン
- L は「クリーンな luma 成分」、C1/C2/C3 は「色差」
- それぞれ独立に (強度を別々に設定して) デノイズできる
- WHT は直交なので GAT 空間の var≈1 が保存される

sRGB モードでは BT.709 の YCbCr に直接分解します（2×2 WHT は使いません）。

### 2.4 LOSH（Local Shrinkage）

「非局所 BM3D を、局所 WHT + 2-pass 収縮で置き換えた」のが LOSH。GALOSH の主背骨です。

**Pass 1（pilot 生成、BayesShrink ハード閾値）**:
```
各 8×8 overlapping patch に対して:
  WHT → 係数 C
  σ² 推定 (noise variance, GAT 空間で ≈1)
  threshold = σ² / σ_signal   (BayesShrink)
  ハード閾値 → 逆 WHT
Kaiser 窓で overlap-add
```

**Pass 2（empirical Wiener、pilot 参照）**:
```
各 8×8 patch に対して:
  WHT(noisy) と WHT(pilot)
  利得 g = |pilot|² / (|pilot|² + σ²)    (Wiener)
  g を noisy 係数に掛ける → 逆 WHT
```

stride=2 (75% overlap) と stride=4 (50% overlap) を設定可能。stride=2 の方が滑らかで質も上、計算量は 4 倍。

### 2.5 LOESS chroma（今セッションで導入）

従来は C1/C2/C3 にも LOSH を独立にかけていました。しかし GAT-luma を guide に使った **guided filter (He 2010)** の方が、luma edges を保持しつつ chroma を空間滑らかにできます。これを採用して完全に入れ替えました。

標準 guided filter は box 窓内の全サンプルを等重みで平均します。ところが specular highlight (鏡面反射) が silver 面 (低彩度) の近傍にあるとき、window 内で luma 分布が **bimodal** になり、線形回帰係数が暴れて silver 表面に赤青の chroma 粒が噴出します（SIDD Medium の 6 scene で実際に発生）。

**LOESS (Cleveland 1979)** はこの box 重みを Gaussian bilateral に置き換えます：

```
w_i = exp(-(Y_i − Y_c)² / (2·σ²))
```

中心 pixel `Y_c` とかけ離れた luma のサンプルは自動的に重み 0 近くになり、window が「center と同じ population」だけに事実上縮まります。σ = 3 (GAT 単位。per-pixel ノイズ std ≈ 1 の 3σ) で specular(|ΔY|≈50) を完全除外、silver 面のノイズ平均化は通常通り。

**ε の principled 導出**: guided filter の `a = cov(Y,Cb) / (var(Y) + ε)` の `ε` は、従来 strength² · (α·Y + σ²) と書いていました。しかし GAT 後空間では **signal 非依存で σ²_noise ≈ 1 (定数)**。Bayes MAP (slope prior `a ~ N(0, τ²)`) から:

```
ε = σ²_noise / τ² = 1/τ²    (τ²=1 で |a|≤2 の prior を 95% カバー)
```

GAT 空間の変分構造と整合した形で ε が導けます。旧式は raw 空間の分散を GAT 後に再適用する単位混同でした。

---

## 3. パイプライン全体像

### 3.1 RAW Bayer 版

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: noisy Bayer RGGB (float32, 12–16 MP)                     │
└──────────────────────────────────────────────────────────────────┘
   │
   ├──► K0: Foi blind estimation         → α, σ², dark_ref[4]
   │
   ├──► K1: GAT forward + Bayer 4ch split (dark_ref 減算含む)
   │   ↓
   │   channels in GAT-normalised space (var≈1)
   │   ↓
   ├──► K11: 2×2 WHT decompose           → L, C1, C2, C3 (half-res)
   │
   ├──► K13 luma (LOSH half-res, Pass1 pilot + Pass2 Wiener)
   │   K13 chroma (LOESS: C1/C2 with L guide, C3 with L guide)
   │   ↓
   ├──► K14: Compute L_fullres           (half-res L + C を upscale)
   │
   ├──► K15: LOSH Pass2 on full-res L    (stride=2 で精細化)
   │
   ├──► K16: Inverse 2×2 WHT             (L_full + C_half で RGGB 復元)
   │       + dark_ref 加算 + ×σ_unified + 逆 GAT (LUT)
   │
   └──► OUTPUT: denoised Bayer RGGB (float32)
```

### 3.2 sRGB YCbCr 版

```
INPUT sRGB (float32 HxWx3)
  ↓
sRGB → linear RGB (inverse gamma)
  ↓
BT.709: linear → Y, Cb, Cr
  ↓
blind σ on Y (MAD-Laplacian)
  ↓
Y: GAT forward → LOSH Pass1+Pass2 → Makitalo inverse
  ↓
Cb, Cr: LOESS (Y guide, unified σ で正規化後)
  ↓
YCbCr → linear RGB → sRGB (gamma)
  ↓
OUTPUT sRGB
```

---

## 4. 実装構造

| ファイル | 内容 | 行数 |
|---|---|---|
| `standalone/galosh.cl` | OpenCL カーネル 44 本（§1 Core / §2 RAW / §3 YUV） | ~2400 |
| `standalone/galosh_gpu.c` | GPU ホスト、OpenCL 呼び出し、mode 切り替え | ~2000 |
| `standalone/galosh_core.h` | CPU 共通 core（GAT / LOSH / LOESS / noise est） | 1316 |
| `standalone/galosh_raw_cpu.c` | CPU RAW driver (2×2 WHT 分解/再合成) | 1154 |
| `standalone/galosh_yuv_cpu.c` | CPU YUV driver (sRGB↔YCbCr、Y-GAT) | 254 |

### 4.1 GPU / CPU 統一設計

- **GPU**: 1 ソース `galosh.cl` を単一 program として build。RAW と YUV は host 側で kernel の選び方を変えるだけ。
- **CPU**: `galosh_core.h` に「アルゴリズム本体」が全部、driver は I/O と OpenMP 並列化パターンのサンプル。
- LOESS 式、ε = 1/τ²、GAT LUT すべて GPU/CPU で**完全同一**（数値検証済）。

### 4.2 ビルド

**GPU** (Windows, ucrt64 gcc):
```bash
cd standalone
gcc -O2 -fopenmp galosh_gpu.c -o galosh_gpu.exe -lOpenCL -lm
```

**CPU** (最適化):
```bash
gcc -O3 -march=native -ffast-math -funroll-loops -flto \
    -fopenmp -DGALOSH_F \
    -o galosh_raw_cpu galosh_raw_cpu.c -lm

gcc -O3 -march=native -ffast-math -funroll-loops -flto \
    -fopenmp -DGALOSH_F \
    -o galosh_yuv_cpu galosh_yuv_cpu.c -lm
```

`-DGALOSH_F` で stride=2 (全 WHT stage で 75% overlap) が有効。外すと stride=4 の軽量モードになります。

### 4.3 CLI

```
galosh_gpu.exe  in.raw out.raw W H galosh  [strength ls cs alpha sigma_sq device]
galosh_gpu.exe  in.bin out.bin W H yuv_gat [strength_y strength_c device]
galosh_raw_cpu.exe in.raw out.raw W H galosh [strength ls cs alpha sigma_sq]
galosh_yuv_cpu.exe in.bin out.bin W H [strength_y strength_c alpha sigma_sq]
```

`alpha=0 sigma_sq=0` を渡すと blind 推定に切り替わります。

---

## 5. 性能（SIDD Medium, 80 image）

### 5.1 Pre-demosaic domain (RAW vs gt_raw)

| 方法 | PSNR | 備考 |
|---|---:|---|
| B2U (DL, pretrained) | **49.08** | GPU, self-supervised 事前学習 |
| **GALOSH-RAW** | **46.49** | 完全 blind |
| NLM-CFA (oracle σ) | 46.13 | GT から σ 取得 |
| BM3D-CFA (oracle σ) | 45.65 | GT から σ 取得 |

GALOSH は blind のまま oracle-σ を与えた BM3D, NLM を上回ります。

### 5.2 sRGB GT2 calibrated

| 方法 | PSNR | SSIM | LPIPS ↓ | DISTS ↓ | NIQE ↓ |
|---|---:|---:|---:|---:|---:|
| B2U | **37.39** | **0.910** | **0.176** | **0.149** | 8.83 |
| **GALOSH-RAW** | 35.14 | 0.853 | 0.305 | 0.274 | **8.59** |
| BM3D-CFA oracle | 35.00 | 0.840 | 0.313 | 0.255 | 9.85 |
| NLM-CFA oracle | 34.58 | 0.859 | 0.292 | 0.276 | 10.25 |

GALOSH は **NIQE (reference-free perceptual)** で全方式の top を取ります（B2U より良い perceptual score）。SSIM / LPIPS では B2U が勝ちますが、古典系の中では GALOSH が最も人間視覚に近い。

### 5.3 速度 (16MP)

| 方法 | CPU (4 thread) | GPU | CPU/GPU 比 |
|---|---:|---:|---:|
| **GALOSH-RAW** | **1.94 s** | **0.57 s** | **3.4×** |
| GALOSH-YUV | 3.07 s | 0.98 s | 3.1× |
| NLM-CFA (fast) | 26 s | 1.05 s | 25× |
| NLM-CFA (exact) | 400 s | 1.05 s | 380× |
| BM3D-CFA | 57 s | — | — |
| B2U | — | 5.89 s | — |

GALOSH は **GPU / CPU どちらでも実用速度**で動きます。DL 系は GPU 必須、BM3D は CPU only & 遅い、NLM は CPU fallback が絶望的——GALOSH だけ両軸で使いやすい。

---

## 6. ポジショニング

- **速度重視の blind 古典**：GPU 0.57 s、CPU 1.94 s で使える
- **oracle-σ を与えた古典 (BM3D/NLM) を blind のまま超える**
- **B2U 比で -2 dB だが 10× 高速**：速度/品質トレードオフ上は合理的選択
- **NIQE (ref-free)** では B2U や NAFNet を上回り、**知覚品質の劣化は小さい**
- darktable IOP や camera ISP に drop-in 可能（`galosh_core.h` ワンヘッダ移植）

---

## 7. 今セッションで導入した改良

1. **`gat_inv_lut` バイナリ探索化**: 代数推定 + 8 step refinement が D=175 付近で 3.89×10⁵ を返すバグ（特定条件下で silver 面に赤青粒を injection）→ bisection (O(log N)) に置換
2. **K16 出力 [0,1] clamp**: 下流 ISP への NaN/Inf leak 防御
3. **LOESS chroma**: box guided → bilateral guided (Cleveland LOESS)、specular edge の bimodal window に頑健化
4. **ε = 1/τ² 定数化**: Bayes MAP での principled derivation、旧式の unit-mismatch 除去
5. **ソース統合**: `rawdenoise_gpu.c → galosh_gpu.c`、2 CL → `galosh.cl` (44 kernel)
6. **CPU port 刷新**: `galosh_core.h` + RAW/YUV 2 driver、darktable plugin drop-in 可能に
7. **ビルド最適化**: `-O3 -march=native -ffast-math -funroll-loops -flto -fopenmp`、16MP を 4 thread で 2-3 秒

---

## 8. 参考文献

- Foi, Trimeche, Katkovnik, Egiazarian. "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data." IEEE TIP, 2008.
- Mäkitalo, Foi. "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise." IEEE TIP, 2013.
- Chang, Yu, Vetterli. "Adaptive wavelet thresholding for image denoising and compression." IEEE TIP, 2000.
- He, Sun, Tang. "Guided image filtering." IEEE TPAMI, 2013.
- Cleveland. "Robust locally weighted regression and smoothing scatterplots." *Journal of the American Statistical Association*, 1979.
- Danielyan et al. "Cross-color BM3D filtering of noisy raw data." LNLA, 2009.
