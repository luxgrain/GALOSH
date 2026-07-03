# arXiv v1 re-centering memo（論文改稿の設計図, 2026-07-02）

現 draft `benchmark/raw_v2_results/galosh_raw_paper.tex`（RAW-only / ISP-first）は
たたき台。以下の確定 positioning へ軸替えして書き直す。

## 確定 positioning（user 2026-07-02）
**並列フレンドリな training-free classical デノイズの再設計 × 高速 × 高性能 × マルチドメイン**

> BM3D/NLM 系の学習不要クラシカルデノイズを、ブロックマッチングや探索に頼らず、
> 局所的・規則的・並列処理向きの構造で再設計した。RAW Bayer と YUV/RGB の両方で動く。

- ISP / 固定小数点 / streaming は **feasibility / mapping / designed-for** のみ
  （「実装済」「実機動作」「frame-buffer-free 完成」等は禁止）。
- タイトルから **ISP と 4K60 を外す**。
- DL は倒さない（正直な非対称提示）。Pareto 系用語・"SOTA"・"beats DL" 禁止。

## タイトル候補（ISP/4K60 抜き）
1. `GALOSH: Blind, Training-Free Image Denoising for Raw Bayer and YUV
   by Parallel-Friendly Local Shrinkage`
2. `GALOSH: Redesigning Training-Free Denoising Without Block Matching —
   a Blind, Parallel-Friendly Method for Raw and YUV Images`
3. `GALOSH: Fast Blind Denoising of Raw Bayer and YUV Images
   with Search-Free Local Shrinkage`

## Abstract 骨子
1. 課題：classical（BM3D/NLM）= 訓練不要だが σ 既知前提＋データ依存探索で並列/固定
   レイテンシに不向き。DL = 高品質だが要学習・ドメイン依存・重い。
2. 提案：GALOSH = blind PG 推定 + GAT + 局所 WHT shrinkage + luma-guided chroma
   LOESS。**探索なし・データ非依存・局所・並列規則構造**。RAW Bayer と YUV/RGB の
   両ドメインに同一 core。
3. 結果（正直な非対称）：
   - 両ドメインで classical を明確に上回る（RAW: LPIPS 0.203 vs BM3D-CFA 0.27;
     sRGB: oracle-σ を与えた CBM3D/NLM でも GALOSH 未満 = **σ でなくアルゴリズムの優位**）。
   - trained DL に対し：RAW では Blind2Unblind に肉薄、sRGB 高 ISO では SIDD 学習
     DL が上（と明記）。ただし GALOSH は training-free・全ドメイン一貫。
   - 速度：GPU で DL の 9–100×（RAW 1080p 18ms / sRGB 1024² 55ms）、CPU でも動く
     唯一の強手法クラス（32t 205ms ≈ Color-NLM）。
4. 固定小数点/streaming への **mapping は feasibility として**一段落
   （INT16 CPU↔GPU 一致・op-count model は measured fact として言及可）。

## 章立て
1. **Introduction** — classical vs DL のトレードオフ、search-free 再設計という主張、
   貢献 3 点（blind/training-free 両ドメイン法、classical 超えの実証＋正直な DL 比較、
   並列・固定小数点への適合性）。
2. **Background** — **Foi (GAT) / Chang (BayesShrink) / Cleveland (LOESS) /
   Kopf (joint bilateral)** ← 特許明細と整合、現 draft に欠落。＋BM3D/NLM/DL。
3. **Method** — 3.1 GALOSH core（共通：GAT・2-pass WHT shrinkage・Y-guided LOESS）
   / 3.2 GALOSH-RAW（CFA WHT・dark ref・K16 upsample・chroma clamp）
   / 3.3 GALOSH-YUV/RGB（sRGB→linear→BT.709、chroma full-res）。
4. **Experiments** — 4.1 RAW（SIDD Medium 80 full-frame + RawNIND 1493、
   BM3D-CFA/NLM-CFA/VST ablation/B2U）/ 4.2 sRGB（SIDD 80 + RawNIND 1493 全 blind、
   CBM3D/Color-NLM/guided/NAFNet/SCUNet/Restormer、**per-ISO 表＋NAFNet 暗部発散
   脚注**、σ-source 実験=参考）/ 4.3 Speed（CPU 同士・GPU 同士で公平比較、
   CBM3D 単スレ注記）/ 4.4 Ablation。
   **baseline がドメインごとに違うのは当然**（RAW は CFA 対応手法、sRGB は色 sRGB
   手法）と protocol で一文明記。
5. **Parallel & fixed-point mapping（feasibility）** — 現 §INT16 を縮約・降格。
   measured fact（INT16 CPU↔GPU bit-exact、op-count 3.4k MAC/px、line-buffer 見積）
   は書いてよいが、「ISP 実装済/real-time on ISP」とは書かない。
6. **Discussion** — 正直な限界：sRGB 高 ISO で SIDD 学習 DL に劣る／PSNR では B2U
   未到達／X15 corrector は future work 一段落。
7. **Conclusion**。

## 図表計画
- Fig: RAW 定性 2 枚（既存 qualitative_{rawnind,sidd_medium}.png）
- Fig: **YUV 定性（新規作成必要）** — SIDD sRGB + RawNIND sRGB crop、論文手法全部
- Tab: RAW SIDD / RawNIND（既存 galosh_raw_tables.tex 流用）
- Tab: sRGB SIDD / RawNIND（bench 済、**per-ISO 内訳 or 注記付き**）
- Tab: Speed 統合（CPU 32t 横並び + GPU 横並び、スレッド数明記）

## 禁止/許可 claim（再掲・確定）
- ❌ ISP 実装済 / frame-buffer-free 完成 / 4K60 / SOTA / beats DL / Pareto 語
- ✅ classical を両ドメインで上回る（oracle-σ でも）/ DL の 9–100× 高速 /
  training-free・fully blind / CPU+GPU 両対応 / fixed-point に自然に写像する設計
- 特許脚注は現行の generic 形を維持（INT16 を特許手法と示唆しない — 特許の組込
  claim は FP16 pre-scaling）。

## ロジ
著者 Yoshiro Sato + repo github.com/luxgrain/GALOSH（公開時に実名↔handle が紐づく
点は user 了承済み前提で最終確認）。コード Apache-2.0 / 論文 CC BY 4.0。
arXiv → SPL → IPOL。投稿前に ChatGPT 査読。
