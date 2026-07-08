#!/usr/bin/env python3
"""GALOSH-RAW distributable wrapper — DNG in, denoised DNG out.

DNG を読み、GALOSH-RAW（完全ブラインド、学習不要）でデノイズし、
「元ファイル名_GALOSH_l<L>_c<C>.dng」を元ファイルと同じフォルダに書き出す。
メタデータ（EXIF / GPS / MakerNotes / XMP / DNG 色タグ）は ExifTool の
-TagsFromFile でオリジナルから完全コピーする。

Pipeline / パイプライン:
  1. rawpy (LibRaw) reads the Bayer mosaic + sensor metadata.
  2. Per-channel black/white normalization to [0,1] float32.
  3. galosh_raw_cpu.exe (FP32 reference) or galosh_raw_gpu.exe (--gpu, o32)
     denoises the mosaic fully blind (per-image Poisson-Gaussian fit).
  4. A minimal valid DNG skeleton is written with tifffile (uncompressed CFA,
     16-bit) carrying the essential rendering tags from the sensor metadata.
  5. ExifTool overlays ALL tags from the original file (-all:all -unsafe),
     so color matrices, camera calibration, lens opcodes, EXIF, GPS and
     MakerNotes match the original exactly. Structural strip/geometry tags
     are protected by ExifTool and cannot be clobbered.

Usage / 使い方:
  galosh_raw.exe photo.dng                     # defaults l=1 c=1
  galosh_raw.exe -l 0.8 -c 1.2 a.dng b.dng     # explicit strengths
  (drag & drop DNG files onto the exe = defaults)

Only 2x2 Bayer CFA DNGs are supported (X-Trans and monochrome are rejected).
2x2 ベイヤー CFA の DNG のみ対応（X-Trans・モノクロは明示エラー）。
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Locate bundled binaries / 同梱バイナリの探索
# frozen (PyInstaller): <exe_dir>/bin ; source: repo standalone/ + PATH
# ----------------------------------------------------------------------
def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _find_tool(name: str) -> Path:
    base = _base_dir()
    cands = [base / "bin" / name, base / name]
    if not getattr(sys, "frozen", False):
        # source checkout: ../../standalone (repo layout)
        cands.append(Path(__file__).resolve().parents[2] / "standalone" / name)
    for c in cands:
        if c.is_file():
            return c
    import shutil
    hit = shutil.which(name)
    if hit:
        return Path(hit)
    raise SystemExit(f"[GALOSH] required tool not found: {name} "
                     f"(looked in {', '.join(str(c) for c in cands)})")


def _fmt(v: float) -> str:
    """1.0 -> '1', 0.85 -> '0.85' (filename-friendly)."""
    s = f"{v:g}"
    return s.replace("-", "m")  # negative never expected; stay filename-safe


# ----------------------------------------------------------------------
# DNG skeleton writing / DNG 骨格の生成
# Essential tags only — everything else is overlaid from the original
# by ExifTool afterwards. 必須タグのみ; 残りは ExifTool が上書きする。
# ----------------------------------------------------------------------
TIFF_ASCII, TIFF_SHORT, TIFF_LONG, TIFF_RATIONAL = 2, 3, 4, 5
TIFF_BYTE, TIFF_SRATIONAL = 1, 10


def _rat(values, denom=10000):
    out = []
    for v in values:
        out.extend([int(round(float(v) * denom)), denom])
    return out


def write_dng_skeleton(path: Path, data_u16: np.ndarray, cfa_pattern: bytes,
                       black4: list, white: int, color_matrix, as_shot,
                       camera_model: str):
    import tifffile
    extratags = [
        (50706, TIFF_BYTE, 4, (1, 4, 0, 0), True),                 # DNGVersion 1.4
        (50707, TIFF_BYTE, 4, (1, 1, 0, 0), True),                 # DNGBackwardVersion
        (50708, TIFF_ASCII, None, camera_model, True),             # UniqueCameraModel
        (33421, TIFF_SHORT, 2, (2, 2), True),                      # CFARepeatPatternDim
        (33422, TIFF_BYTE, 4, tuple(cfa_pattern), True),           # CFAPattern
        (50713, TIFF_SHORT, 2, (2, 2), True),                      # BlackLevelRepeatDim
        (50714, TIFF_RATIONAL, 4, _rat(black4, 1), True),          # BlackLevel (quad)
        (50717, TIFF_LONG, 1, int(white), True),                   # WhiteLevel
        (50778, TIFF_SHORT, 1, 21, True),                          # CalibrationIlluminant1 = D65
    ]
    # NOTE tifffile counts RATIONALs, not ints (each value = 2 packed ints).
    if color_matrix is not None:
        extratags.append((50721, TIFF_SRATIONAL, 9,
                          tuple(_rat(color_matrix.flatten())), True))  # ColorMatrix1
    if as_shot is not None:
        extratags.append((50728, TIFF_RATIONAL, 3,
                          tuple(_rat(as_shot)), True))                 # AsShotNeutral
    tifffile.imwrite(
        str(path), data_u16,
        photometric=32803,          # CFA
        compression=None,
        planarconfig=1,
        rowsperstrip=data_u16.shape[0],
        extratags=extratags,
    )


# ----------------------------------------------------------------------
# Per-file processing / 1 ファイルの処理
# ----------------------------------------------------------------------
def _mode_suffix(l_str: float, c_str: float, wht: int, upsample: str) -> str:
    """Filename suffix encodes non-default modes so outputs never collide.
    非デフォルトのモードはファイル名に刻む（衝突防止＋出所自明）。"""
    parts = [f"l{_fmt(l_str)}", f"c{_fmt(c_str)}"]
    if wht == 4:
        parts.append("wht4")
    if upsample == "fast":
        parts.append("upfast")
    return "_GALOSH_" + "_".join(parts)


def process_dng(src: Path, l_str: float, c_str: float, use_gpu: bool,
                exe_cpu: Path, exe_gpu, exiftool: Path,
                wht: int = 8, upsample: str = "jinc") -> Path:
    import rawpy

    raw = rawpy.imread(str(src))
    img = raw.raw_image  # full sensor incl. masked borders / マスク領域込み
    if img.ndim != 2 or raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
        raise SystemExit(f"[GALOSH] {src.name}: not a 2x2 Bayer CFA raw "
                         "(X-Trans / monochrome / linear DNG unsupported)")

    H, W = img.shape
    # CFAPattern: raw_pattern indexes into color_desc (e.g. b'RGBG');
    # DNG CFA colors: 0=R 1=G 2=B. Gr/Gb both map to 1.
    desc = raw.color_desc.decode("ascii")
    letter_to_cfa = {"R": 0, "G": 1, "B": 2}
    try:
        cfa = bytes(letter_to_cfa[desc[raw.raw_pattern[r, c]]]
                    for r in range(2) for c in range(2))
    except KeyError:
        raise SystemExit(f"[GALOSH] {src.name}: unsupported CFA colors {desc!r}")

    # Per-channel normalization to [0,1] / チャネル別 black-white 正規化
    blacks = np.asarray(raw.black_level_per_channel, dtype=np.float32)
    white = float(raw.white_level)
    colors = raw.raw_colors  # HxW channel index 0..3
    black_map = blacks[colors]
    scale_map = np.maximum(white - black_map, 1.0)
    f32 = ((img.astype(np.float32) - black_map) / scale_map).clip(0.0, 1.0)

    # GALOSH quads are aligned at (0,0); process the even-sized region.
    # 奇数サイズは偶数領域のみ処理（残り 1 行/列は原本のまま）。
    We, He = W & ~1, H & ~1

    with tempfile.TemporaryDirectory(prefix="galosh_") as td:
        tin = Path(td) / "in.bin"
        tout = Path(td) / "out.bin"
        f32[:He, :We].tofile(tin)

        mode_flags = []
        if wht == 4:
            mode_flags.append("--wht=4")
        if upsample == "fast":
            mode_flags.append("--upsample=fast")
        if use_gpu and exe_gpu is not None and mode_flags:
            # The OpenCL GPU exe predates the V2.0 fast modes; fall back.
            # OpenCL 版は V2.0 fast モード未対応 → CPU にフォールバック。
            print("[GALOSH]   note: --wht/--upsample not supported by the GPU "
                  "pipeline yet; using CPU.")
            use_gpu = False
        if use_gpu and exe_gpu is not None:
            cmd = [str(exe_gpu), str(tin), str(tout), str(We), str(He),
                   "1.0", f"{l_str}", f"{c_str}", "0", "0", "0"]
            cwd = str(exe_gpu.parent)  # galosh.cl lives next to the exe
        else:
            cmd = [str(exe_cpu), str(tin), str(tout), str(We), str(He),
                   "galosh", "1.0", f"{l_str}", f"{c_str}", "0", "0",
                   *mode_flags]
            cwd = str(exe_cpu.parent)
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
        if r.returncode != 0 or not tout.is_file():
            sys.stderr.write(r.stderr or "")
            raise SystemExit(f"[GALOSH] denoiser failed on {src.name} "
                             f"(exit {r.returncode})")
        dn = np.fromfile(tout, dtype=np.float32).reshape(He, We)

    # De-normalize back to sensor code values / センサー値へ逆正規化
    out_u16 = img.copy()
    dn_codes = dn * scale_map[:He, :We] + black_map[:He, :We]
    out_u16[:He, :We] = np.clip(np.rint(dn_codes), 0, 65535).astype(np.uint16)

    # ColorMatrix1 from LibRaw (XYZ->cam, 3x3); zeros row = absent.
    cm = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    if not np.any(cm):
        cm = None
    wb = np.asarray(raw.camera_whitebalance, dtype=np.float64)
    as_shot = None
    if wb[0] > 0 and wb[1] > 0 and wb[2] > 0:
        as_shot = [wb[1] / wb[0], 1.0, wb[1] / wb[2]]

    dst = src.with_name(src.stem + _mode_suffix(l_str, c_str, wht, upsample)
                        + ".dng")
    write_dng_skeleton(dst, out_u16, cfa, [float(b) for b in blacks[:4]],
                       int(white), cm, as_shot, "GALOSH denoised")
    raw.close()

    # Full metadata overlay from the original / 全メタデータをオリジナルから複写
    # -all:all -unsafe copies EXIF/GPS/XMP/MakerNotes; the DNG calibration
    # tags are protected from bulk copy and must be named explicitly.
    # NOTE deliberately NOT copied: LinearizationTable (rawpy output is
    # already linearized), BlackLevel/WhiteLevel (ours match the data),
    # strip/geometry tags (protected). NoiseProfile describes noise that
    # no longer exists; ExifTool drops it, which is semantically correct.
    calib = ["-ColorMatrix1", "-ColorMatrix2", "-CalibrationIlluminant1",
             "-CalibrationIlluminant2", "-ForwardMatrix1", "-ForwardMatrix2",
             "-CameraCalibration1", "-CameraCalibration2", "-AnalogBalance",
             "-AsShotNeutral", "-BaselineExposure", "-BaselineNoise",
             "-BaselineSharpness", "-LinearResponseLimit", "-BayerGreenSplit",
             "-AntiAliasStrength", "-DefaultCropOrigin", "-DefaultCropSize",
             "-DefaultScale", "-ActiveArea", "-OpcodeList1", "-OpcodeList2",
             "-OpcodeList3", "-DNGPrivateData", "-UniqueCameraModel",
             "-CameraSerialNumber", "-CameraCalibrationSignature",
             "-ProfileCalibrationSignature"]
    r = subprocess.run(
        [str(exiftool), "-TagsFromFile", str(src), "-all:all", "-unsafe",
         "-icc_profile", *calib, "-F",
         "-Software=GALOSH (arXiv:2607.03768)"
         + _mode_suffix(l_str, c_str, wht, upsample).replace("_GALOSH_", " ")
           .replace("_", " "),
         "-overwrite_original", str(dst)],
        capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        print(f"[GALOSH] WARNING: metadata copy failed for {dst.name}; "
              "the DNG itself is valid.")
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="galosh_raw",
        description="GALOSH-RAW blind denoiser: DNG in -> denoised DNG "
                    "next to the original, metadata fully copied.")
    ap.add_argument("files", nargs="+", help="input .dng files")
    ap.add_argument("-l", "--luma", type=float, default=1.0,
                    help="luminance strength s_L (default 1.0)")
    ap.add_argument("-c", "--chroma", type=float, default=1.0,
                    help="chrominance strength s_C (default 1.0)")
    ap.add_argument("--gpu", action="store_true",
                    help="use the OpenCL GPU pipeline (o32) if available")
    ap.add_argument("--wht", type=int, choices=(8, 4), default=8,
                    help="luma WHT block: 8 = canonical (default), "
                         "4 = fast (coarser grain)")
    ap.add_argument("--upsample", choices=("jinc", "fast"), default="jinc",
                    help="chroma upsample: jinc = canonical K16 (default), "
                         "fast = guided bilinear (ring-free, slightly softer)")
    args = ap.parse_args(argv)

    exe_cpu = _find_tool("galosh_raw_cpu.exe")
    exe_gpu = None
    if args.gpu:
        try:
            exe_gpu = _find_tool("galosh_raw_gpu.exe")
        except SystemExit:
            print("[GALOSH] GPU exe not found — falling back to CPU.")
    exiftool = _find_tool("exiftool.exe")

    ok = 0
    for f in args.files:
        src = Path(f)
        if not src.is_file():
            print(f"[GALOSH] skip (not found): {f}")
            continue
        print(f"[GALOSH] {src.name}  (l={_fmt(args.luma)} c={_fmt(args.chroma)}"
              f"{' gpu' if exe_gpu else ''}) ...", flush=True)
        try:
            dst = process_dng(src, args.luma, args.chroma, args.gpu,
                              exe_cpu, exe_gpu, exiftool,
                              wht=args.wht, upsample=args.upsample)
            print(f"[GALOSH]   -> {dst}")
            ok += 1
        except SystemExit as e:
            print(str(e))
    print(f"[GALOSH] done: {ok}/{len(args.files)} file(s).")
    if os.name == "nt" and getattr(sys, "frozen", False) and ok < len(args.files):
        # drag & drop: keep the console readable on errors
        # ドラッグ&ドロップ時にエラーを読めるよう一時停止
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
    return 0 if ok == len(args.files) else 1


if __name__ == "__main__":
    sys.exit(main())
