# GALOSH ノイズ除去ベンチ 全体マップ（2026-07-14 改訂3 — 全トラック全メソッド統一）

「ノイズの現実性」で 3+1 条件を並べる。左ほど非現実的、右ほど実写。
GALOSH は右へ行くほど相対順位が上がる（設計＝信号依存ノイズ向け）。

| # | 確定名 | dir | データ | ノイズ/劣化 | 圧縮 | レベル軸 | 状態 |
|---|---|---|---|---|---|---|---|
| ① | **AWGN** | results_set8_awgn | Set8 | フラット AWGN（信号非依存） | なし | σ {10..50} | ✅ |
| ② | **pgnoise core** | results_set8_pgnoise | Set8 | 信号依存 PG（unprocess→raw→再ISP）**スマホ trimmed 中央値カーブ** | なし | ISO {400..25600} ≈ σ{10..50}（level-matched!） | ✅ |
| ③ | **pgnoise cmp** | 同上 | Set8 | ②の noisy + H.264 CRF23 往復 | あり | ISO 7段 | ✅ |
| ④ | **CRVD** | results_crvd | CRVD | **実センサーノイズ**（IMX385 実測） | なし | ISO {1600..25600} | ✅ |

**ノイズカーブ（②③）**: SIDD per-image NLF 161行 → 機種毎正規化票 → スマホ
trimmed 中央値（GP/S6/N6; a=0.005763, b=6.34e-5 @ISO1600, a∝ISO^0.85 b∝ISO^1.5）。
旧「10機種中央値」は大型センサに引きずられ約9×甘かった（→ _ARCHIVE）。
実測等価: ISO400≈σ10 / 6400≈σ30 / 12800≈σ40 / 25600≈σ50。
provenance = results_set8_pgnoise/phone_median_curve.json + NOISE_CALIBRATION.md。

## 統一メソッドセット（全トラック共通、2026-07-14）

| 群 | メソッド | σ の扱い |
|---|---|---|
| GALOSH (4) | cpu-fit / cpu-hold / vk-fit / vk-hold（frameserver DLL） | **完全 blind** |
| BM3D 系 oracle (2) | bm3d1 / vbm3d | 測定σ（GT 使用 = 上限オラクル） |
| BM3D 系 **blind** (2) | **bm3d1b / vbm3db** | **MAD 推定σ**（Donoho, GT 不使用 = 実運用） |
| temporal (2) | knl d=1 / smdegrain tr=3 | 凍結 DAVIS 較正テーブル |
| reference (1) | hqdn3d | 無調整デフォルト |

420 = 全13、444 = GALOSH4 + bm3d1/bm3d1b/knl、CRVD は +galosh444。
**PNG は全手法＋GT＋Noisy をデフォルト保存**（--no-png はユーザ明示専用）。

## blind σ 軸の意味（本ベンチの核心）

MAD 推定は AWGN ではほぼ完璧（oracle±0.1dB）→ 合成PGで −1.5dB → 実CRVD で
−2dB超の過小推定。つまり「σ既知」前提はノイズが現実的になるほど成立しない。
blind GALOSH と blind BM3D の比較が実運用の比較。

## 成果物（各 dir 共通）

SUMMARY(.md) = 全指標表 / _metrics_*.json = シャード＋マージ / viewer.html =
条件×scene×ISO×全手法 pan/zoom ビューワー / png* = 全フレーム PNG。
旧結果は benchmark/_ARCHIVE/（旧trackb・旧全カメラ中央値 pgnoise・旧CRVD・yuv420_ab）。
