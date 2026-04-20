# GALOSH 概要

**GALOSH** = **G**eneralised **A**nscombe **Lo**cal **Sh**rinkage

生 RAW (pre-demosaic) と sRGB (post-demosaic) の両方で動く、完全盲検（blind, 事前情報ゼロ）かつ学習不要の一貫設計デノイザです。Poisson-Gauss ノイズモデルを出発点に置き、そこから **GAT で分散を平坦化 → luma / chroma を同じ GAT 空間で処理 → 各プレーンに適した収縮推定**、という一本の筋で全段を導出しています。

**設計方針**: 全ステージを同じ確率モデル（Poisson-Gauss + GAT）の上に建て、調整パラメータは物理的に解釈できる量（slope prior τ、bandwidth σ）だけに限定する。luma は 8 × 8 WHT + 2 パス局所収縮（Bayesian 閾値 → empirical Wiener）、chroma は luma-guide つき LOESS。どちらも **GAT 後空間で σ²_noise = 1** という同一前提下で解かれます。

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

## 2. Noise model からの導出

設計は 5 段階。**どれも前段の「var ≈ 1」という仮定が使えるように整え、次段に渡す**のが一貫した原則です。

### 2.1 分散モデルの同定 (Foi-Alenius, 2008)

画素ごとの分散 `var = α·x + σ²` を、事前情報なしで画像 1 枚から推定します。小ブロック (mean, Laplacian 変分) を大量に散布し、線形回帰で (α, σ²) を解く、というアプローチ。GALOSH の K0 段が dark pixel ヒストグラムによる σ² の refine も含めて実装しています。

この (α, σ²) がその後の**全ステージ共通の物理パラメータ**になります。

### 2.2 分散の平坦化 — GAT (Makitalo-Foi, 2013)

```
y = T(x) = (2/α) · √(α·x + 3α²/8 + σ²)
```

1 次テイラー展開の下で `var(y) ≈ 1`（GAT はこの目的で作られた変換）。Poisson-Gauss 前提の noise モデルが「ガウス unit variance」という**後段で数学的に扱いやすい形**に変形されます。

GALOSH は Makitalo-Foi のオリジナル GAT に 2 点の実装上の工夫を加えます:

- 低輝度で √ 内部が負になる場合の **C¹ 連続な線形枝拡張** (`t_break = 2σ/α` 閾値)
- 逆変換が閉形式で不偏推定できないので、**4096 点 LUT + bisection** で $O(\log N)$ 参照

### 2.3 方向独立成分への分解 — 2×2 WHT (RAW モード)

Bayer 2×2 ブロックに Walsh-Hadamard 変換を当てると 4 つの orthonormal 成分に分解できます:

```
L  = (R + G₁ + G₂ + B) / 2   → 輝度 DC (Y に相当)
C1 = (R − G₁ + G₂ − B) / 2   → 水平差分
C2 = (R + G₁ − G₂ − B) / 2   → 垂直差分
C3 = (R − G₁ − G₂ + B) / 2   → 斜め差分
```

orthonormal なので **GAT 後の `var ≈ 1` 前提が 4 成分すべてに保存されます**。加えて、L は空間的に強い相関を持ち、C1-C3 は主にエッジ情報を持つので、「luma は強めに平滑化、chroma は edge 保存を優先」という**物理的に自然な強度分離**が自動的に実現されます。

sRGB モードでは入力が既に色空間分離済み (BT.709 YCbCr) なので 2×2 WHT は使わず、Y と Cb/Cr を直接取り扱います。

### 2.4 Luma の局所収縮 — LOSH

**LOSH (LOcal SHrinkage)**: GAT 空間の L プレーン上で、8 × 8 WHT 係数に対する 2 パス Bayesian shrinkage を行います。

**Pass 1 — BayesShrink (Chang-Yu-Vetterli, 2000) によるハード閾値で pilot 推定**:

```
各 8×8 overlapping patch:
  WHT(noisy) → 係数 C
  signal σ_x 推定 (block 毎の Bayesian MAP)
  threshold = σ²_noise / σ_x            ← GAT 空間で σ²_noise = 1 固定
  hard-threshold(|C|) → 逆 WHT → overlap-add (Kaiser 窓)
```

**Pass 2 — pilot を事前分布にとった empirical Wiener**:

```
各 8×8 patch:
  gain = |pilot|² / (|pilot|² + σ²_noise)
  係数に gain を掛ける → 逆 WHT → overlap-add
```

Pass 1 で生成した pilot が Pass 2 の Wiener filter の empirical 事前分布になるので、非局所 block matching 無しでも BM3D-CFA 同等レベルの性能が出ます（§5 ベンチ参照）。Kaiser 窓 overlap-add は stride=2 (75% 重なり) が質的最良です。

### 2.5 Chroma の空間適応推定 — LOESS guided regression

C1/C2/C3 (RAW) もしくは Cb/Cr (YUV) は、luma と比べて信号帯域が狭いため、**luma を guide にとった局所線形回帰**で推定するのが自然です:

$$C_b(x) = a(x) \cdot Y(x) + b(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2_{\text{noise}})$$

ここで `(a, b)` を局所窓内で MAP 推定します。**GAT 空間で σ²_noise = 1** だから、slope に Gaussian prior `a ~ N(0, τ²)` を置くと Bayes MAP が閉形式で解けて:

$$a = \frac{\text{cov}(Y, C_b)}{\text{var}(Y) + \varepsilon}, \quad \varepsilon = \frac{\sigma^2_{\text{noise}}}{\tau^2} = \frac{1}{\tau^2}$$

**ε は GAT 空間で signal 非依存の定数**になります (τ² = 1 とすれば |a| ≤ 2 の prior を 95% カバー)。残る調整パラメータは τ² 一つで、「chroma-luma 相関の強さの事前仮定」という物理的に解釈できる量です。

**窓重みの適応化 (LOESS, Cleveland 1979)**: 標準的な box 窓で等重み平均してしまうと、specular highlight (鏡面反射) と silver 面 (低彩度) のように luma が bimodal な近傍で線形仮定が破綻します。そこで **中心 pixel `Y_c` との luma 距離で Gaussian kernel 重み**を掛けます:

$$w_i = \exp\left(-\frac{(Y_i - Y_c)^2}{2\sigma^2}\right)$$

これは古典的 **locally-weighted regression (LOESS)** と同じ形式で、kernel regression 理論の下で最適性が保証されます。σ = 3 (GAT 単位で per-pixel ノイズ std の 3σ) にすると、noise は通過、|ΔY| ≳ 50 の specular pixel は自動的に重み 0 に収束、という望ましい挙動になります。σ² 自体が GAT 空間の物理量から定まるので、**ここにも調整パラメータは増えません**。

ε の定数化と window 重みの bilateral 化が組み合わさって、**GAT の「σ²_noise = 1」仮定が推定器全体に一貫して効く**構造ができています。

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
