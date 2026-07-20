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

**ラベル方針 (2026-07-16)**: 「blind」という省略表記は廃止。σ の出自をそのまま書く。
MAD 推定σ を外付けした BM3D は blind ではなく「**σ=MAD 推定**」、フロア付きは
「**σ=MAD 推定 + 0.5 フロアガード**」。GALOSH は σ 入力自体が無い（推定内蔵）。

| 群 | メソッド | σ の扱い |
|---|---|---|
| GALOSH (4) | cpu-fit / cpu-hold / vk-fit / vk-hold（frameserver DLL） | **σ入力なし（ノイズ推定内蔵）** |
| BM3D 系 σ=実測 oracle (2) | bm3d1 / vbm3d | 測定σ（GT 使用 = 上限オラクル） |
| BM3D 系 **σ=MAD 推定** (2) | **bm3d1b / vbm3db** | **Donoho MAD 推定σ を外付け供給**（GT 不使用） |
| BM3D 系 **σ=MAD 推定+ガード** (2) | **bm3d1bg / vbm3dbg** | MAD 推定を per-plane **0.5 フロア**でクランプ |
| [T] temporal (2) | knl d=1 / smdegrain tr=3 | 凍結 DAVIS 較正テーブル |
| [T] reference (1) | hqdn3d | 無調整デフォルト |

420 = 全15、444 = GALOSH4 + bm3d1/bm3d1b/bm3d1bg/knl（galosh444 側注行は 2026-07-16
に _ARCHIVE へ退避済み）。
**PNG は全手法＋GT＋Noisy をデフォルト保存**（--no-png はユーザ明示専用）。
guarded twins はフロアが効くセル（σ=0 plane）のみ実走、他は σ入力同一＝出力同一
なので unguarded の値を複製（SUMMARY に明記）。

## σ供給軸の意味（本ベンチの核心）

MAD 推定は AWGN ではほぼ完璧（oracle±0.1dB）→ 合成PGで −1.5dB → 実CRVD で
−2dB超の過小推定。つまり「σ既知」前提はノイズが現実的になるほど成立しない。
σ入力なしの GALOSH と σ=MAD 推定の BM3D の比較が実運用の比較。
**σ=0 の罠 (2026-07-15)**: H.264 は 4:2:0 なので、デコード後 444 として扱うと
クロマは 420 アップサンプル＝最細スケール Haar エネルギーがゼロ → MAD が全ISOで
クロマσ=0 を返す（cmp-444 全56セル）。cmp-420 でも低ISO 11セルで発生。σ=0 を
食うと BM3D_CPU は能動的に壊れ（bm3d1b で −9dB、V-BM3D 連鎖は 4-7dB まで崩壊）、
0.5 フロアで完全回復。ナイーブな blind σ推定の実地の落とし穴として記録。

## 再現性ノート

- `_env`（DLL/EXE sha256 先頭16 + numpy/python/vs 版）: スクリプトには実装済みだが、
  **現行の結果 JSON には未記録**（2026-07-16 監査で全シャード・マージに `_env` 不在を確認 —
  実装後にシャード未再生成。バックフィルは未実施の TODO）。exe/DLL の出自は現状 mtime 頼み。
- **seed 開示**: PG ノイズ seed = `7000+ISO` で **同一 ISO の全 seq が共通**
  （乱数列は seq 非依存、実現値は信号依存分散経由でのみ変わる）。ISO 間は独立。

## 成果物（各 dir 共通）

SUMMARY(.md) = 全指標表 / _metrics_*.json = シャード＋マージ / viewer.html =
条件×scene×ISO×全手法 pan/zoom ビューワー / png* = 全フレーム PNG。
旧結果は benchmark/_ARCHIVE/（旧trackb・旧全カメラ中央値 pgnoise・旧CRVD・yuv420_ab）。

## 整理 2026-07-16（監査反映）

- **galosh420 (legacy exe) 幽霊行を削除**: 現行キャンペーンでは 0 セル（未実行）。旧データは
  _ARCHIVE/results_crvd のみ。report_crvd.py から legacy 名対応を除去し SUMMARY 再生成。
- **[T] マーカー導入**: 全 SUMMARY で時空間（複数フレーム）手法 = V-BM3D 系 / KNL /
  SMDegrain / hqdn3d に [T] を付与。無印 = 単フレーム空間処理。
- **SMDegrain 素通し問題を明文化**（監査での新発見・要注意）: 凍結 DAVIS thSAD が
  SAD ノイズ床を下回るセルで MDegrain が全ブロック棄却 → **出力が noisy とビット一致**。
  該当 = pg-core420 ISO12800 (8/8 seq)・pg-cmp420 ISO12800 (7/8)・CRVD ISO12800/25600
  (11/11 scene)・motorbike 全 ISO≥1600。当該セルは「較正転移の失敗」の測定値であり
  デノイズ性能ではない。各 SUMMARY 冒頭に注記済み。
- **KNL の CRVD クランプを明文化**: CRVD の実測 σY 3.2-12.4 は較正レンジ {10..50} の下限外
  → 全 55 セルが σ10 テーブル端（h=3.0 固定）。
- **CRVD 実測ノイズレベルを記録**: 現像後 σ_sRGB ≈ 4.5/6.2/8.6/12.1/17.6（ISO1600→25600、
  ≈AWGN σ5〜18）。スマホカーブの同 ISO ラベルより大幅に静か（raw 実測 (a,b) は
  results_set8_pgnoise/sensor_pg_library.json の CRVD_IMX385 に既存）。
- 未実施 TODO: bg 双子行の awgn/crvd への複製 / `_env` バックフィル /
  smdegrain・knl の現実 σ レンジ再較正（追加ベンチ候補）。

## ★メソッド統一 完了 2026-07-16★

**全 4 データセット(awgn / pg-core / pg-cmp / CRVD)× 2 レーン(420 / 444)で同一の
13 手法**(noisy + GALOSH×4 + BM3D 系 6 + knl + hqdn3d; smdegrain は drop)に統一。
- 新規実走: pg core-444 (+vbm3d/vbm3db/hqdn3d)・pg cmp-444 (+同+vbm3dbg 実走)・
  awgn-444 (+vbm3d/vbm3db/hqdn3d, フル尺)・**CRVD-444 レーン新設**(11 scene×5 ISO×10 手法、
  Set8-444 プロトコル、_env 付きシャード _metrics_crvd444_c1-3)。全キャンペーン FAILED 0。
- guard 双子コピー: 計 436 method-cells(フロア不作動をセル毎検証、全セル不作動)+
  PNG ハードリンク 12,600。copied_from マーカー付き。
- **CRVD-444 の旗艦結果**: GALOSH cpu-fit が σ-oracle BM3D を全 ISO で上回る
  (ISO25600: 29.90 vs 26.70 = +3.2dB)— 420 レーンと同型。
- **cmp-444 の σ=0 崩壊が最鮮明**: vbm3db は全 ISO で 4.4-4.6dB に崩壊
  (クロマ MAD=0 が全セル)、0.5 フロアガードで 30.7/27.2/22.2/17.3 に全快。
- viewer 3 種再生成(CRVD は 420/444 レーン切替 UI 追加)。

## 公平性ポリシー確定 + 完全アーカイブ 2026-07-17

- **BM3D 系の公平比較 = σ oracle(理論上限)vs σ MAD 推定+0.5 フロアガード(現実運用)**。
  ガードなし MAD 双子(bm3d1b/vbm3db)は「手法」ではなく「うっかり実装の事故」を測って
  いたため**完全アーカイブ**。崩壊の証拠数値は各 SUMMARY 注記に保全
  (vbm3db cmp-420 ISO400 = 12.01dB / cmp-444 全 ISO 4.4-4.6dB、ガードで全快)。
  注: ガードは σ=0 事故を防ぐだけで MAD の系統的過小推定(実ノイズ −1.5〜−2dB)は残る =
  それ自体が「推定の質込みの実運用差」として GALOSH との比較対象。
- **smdegrain も完全アーカイブ**(素通し欠陥に加え、この級のノイズ源には prefilter 前提の
  ツールを prefilter なしで走らせていた=実運用非代表)。
- 移動先 = `_ARCHIVE/dropped_methods_20260717/`(JSON 1,876 セル + PNG 1,035 dir)。
  synth 診断アーム(crvdsynth/crvdsynthphone)は分析記録のため未変更。
- 統一セットは **11 手法**(noisy + GALOSH×4 + bm3d1/bm3d1bg + vbm3d/vbm3dbg + knl + hqdn3d)
  × 4 データセット × 2 レーンに再定義。SUMMARY×3・viewer×3 再生成済み。

## 整理 2026-07-16 第2弾

- **galosh444 側注行を _ARCHIVE へ退避**（ユーザ判断）: JSON 55 セル（原本コピー保全）
  + PNG 55 dir → `_ARCHIVE/results_crvd_galosh444/`。ライブ JSON からは strip 済み。
- **残骸 dir を _ARCHIVE へ**: `results_set8_awgn/_calib_work`（616MB, AVS較正作業）
  → `_ARCHIVE/_calib_work_awgn`、`results_crvd/_work_*`×4 → `_ARCHIVE/results_crvd_workdirs/`。
- **ラベル全面改訂**: "blind"/"sigma-BLIND" 表記を全 SUMMARY 生成スクリプトから廃止 →
  「sigma: measured oracle / sigma: MAD estimate / sigma: MAD estimate + 0.5 floor guard /
  GALOSH = built-in noise est.」の省略なし表記。knl/smdegrain も knob の出自
  （frozen DAVIS table）をラベルに明記。
