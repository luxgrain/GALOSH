#!/usr/bin/env python3
"""GALOSH-YUV distributable wrapper — sRGB PNG in, denoised PNG out.

PNG を読み、GALOSH-YUV（完全ブラインド、学習不要、sRGB→linear→BT.709
YCbCr で L/C 分離デノイズ）で処理し、「元名_GALOSH_l<L>_c<C>.png」を
元ファイルと同じフォルダへ書き出す。メタデータ（eXIf / XMP / tEXt /
ICC プロファイル）は ExifTool でオリジナルから完全コピーする。

Usage / 使い方:
  galosh_yuv.exe photo.png                     # defaults l=1 c=1
  galosh_yuv.exe -l 0.8 -c 1.2 a.png b.png     # explicit strengths
  (drag & drop PNG files onto the exe = defaults)

Notes / 補足:
  - RGB / RGBA / grayscale PNG を受け付ける（alpha は無加工で保持、
    grayscale は RGB として処理し grayscale へ戻す）。
  - 16-bit PNG は 8-bit 精度で処理される（Pillow の制約; 警告を表示）。
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _find_tool(name: str) -> Path:
    base = _base_dir()
    cands = [base / "bin" / name, base / name]
    if not getattr(sys, "frozen", False):
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
    return f"{v:g}".replace("-", "m")


def _png_bit_depth(path: Path) -> int:
    """Read bit depth from the IHDR chunk / IHDR からビット深度を読む."""
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return 8  # not a PNG signature; PIL will decide
        f.read(8)  # IHDR length+type
        ihdr = f.read(13)
        return struct.unpack(">B", ihdr[8:9])[0]


def process_png(src: Path, l_str: float, c_str: float,
                exe_cpu: Path, exiftool: Path) -> Path:
    from PIL import Image

    if _png_bit_depth(src) > 8:
        print(f"[GALOSH]   note: {src.name} is >8-bit; processed at "
              "8-bit precision (Pillow limitation).")

    im = Image.open(src)
    icc = im.info.get("icc_profile")
    orig_mode = im.mode
    alpha = None
    if orig_mode == "RGBA":
        alpha = im.getchannel("A")
        rgb = im.convert("RGB")
    elif orig_mode == "RGB":
        rgb = im
    else:  # L, P, LA, I;16 ... → RGB で処理して元モードへ戻す
        rgb = im.convert("RGB")

    arr = np.asarray(rgb, dtype=np.float32) / 255.0  # HxWx3 sRGB [0,1]
    H, W, _ = arr.shape

    with tempfile.TemporaryDirectory(prefix="galosh_") as td:
        tin = Path(td) / "in.bin"
        tout = Path(td) / "out.bin"
        arr.tofile(tin)
        cmd = [str(exe_cpu), str(tin), str(tout), str(W), str(H),
               f"{l_str}", f"{c_str}", "0", "0"]
        r = subprocess.run(cmd, cwd=str(exe_cpu.parent),
                           capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
        if r.returncode != 0 or not tout.is_file():
            sys.stderr.write(r.stderr or "")
            raise SystemExit(f"[GALOSH] denoiser failed on {src.name} "
                             f"(exit {r.returncode})")
        dn = np.fromfile(tout, dtype=np.float32).reshape(H, W, 3)

    out8 = np.clip(np.rint(dn * 255.0), 0, 255).astype(np.uint8)
    out_im = Image.fromarray(out8, "RGB")
    if orig_mode == "RGBA" and alpha is not None:
        out_im = out_im.convert("RGBA")
        out_im.putalpha(alpha)
    elif orig_mode in ("L", "I;16", "LA"):
        out_im = out_im.convert("L")

    dst = src.with_name(f"{src.stem}_GALOSH_l{_fmt(l_str)}_c{_fmt(c_str)}.png")
    save_kw = {}
    if icc:
        save_kw["icc_profile"] = icc  # ICC は PIL 側でも直接保持
    out_im.save(dst, "PNG", **save_kw)

    # eXIf / XMP / tEXt をオリジナルから複写 / copy metadata chunks
    r = subprocess.run(
        [str(exiftool), "-TagsFromFile", str(src), "-all:all", "-unsafe",
         "-icc_profile", "-F",
         f"-Software=GALOSH (arXiv:2607.03768) l={_fmt(l_str)} c={_fmt(c_str)}",
         "-overwrite_original", str(dst)],
        capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        print(f"[GALOSH] WARNING: metadata copy failed for {dst.name}; "
              "the PNG itself is valid.")
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="galosh_yuv",
        description="GALOSH-YUV blind denoiser: sRGB PNG in -> denoised PNG "
                    "next to the original, metadata fully copied.")
    ap.add_argument("files", nargs="+", help="input .png files")
    ap.add_argument("-l", "--luma", type=float, default=1.0,
                    help="luminance strength s_L (default 1.0)")
    ap.add_argument("-c", "--chroma", type=float, default=1.0,
                    help="chrominance strength s_C (default 1.0)")
    args = ap.parse_args(argv)

    exe_cpu = _find_tool("galosh_yuv_cpu.exe")
    exiftool = _find_tool("exiftool.exe")

    ok = 0
    for f in args.files:
        src = Path(f)
        if not src.is_file():
            print(f"[GALOSH] skip (not found): {f}")
            continue
        print(f"[GALOSH] {src.name}  (l={_fmt(args.luma)} c={_fmt(args.chroma)}) ...",
              flush=True)
        try:
            dst = process_png(src, args.luma, args.chroma, exe_cpu, exiftool)
            print(f"[GALOSH]   -> {dst}")
            ok += 1
        except SystemExit as e:
            print(str(e))
    print(f"[GALOSH] done: {ok}/{len(args.files)} file(s).")
    if os.name == "nt" and getattr(sys, "frozen", False) and ok < len(args.files):
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
    return 0 if ok == len(args.files) else 1


if __name__ == "__main__":
    sys.exit(main())
