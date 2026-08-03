#!/usr/bin/env bash
# install_linux.sh — one-command GALOSH setup for Linux (no Python knowledge
# needed). Builds the CLI denoisers, creates a private Python environment
# inside this folder (nothing is installed system-wide), and generates two
# launchers:
#
#   ./galosh-dng photo.dng|.cr3|.nef|.arw...   raw photo -> denoised DNG
#   ./galosh-png image.png                     sRGB PNG  -> denoised PNG
#
# Requirements: gcc and python3 with the venv module.
#   Debian/Ubuntu:  sudo apt install gcc python3-venv
#   Arch/Manjaro:   sudo pacman -S gcc python
#   Fedora:         sudo dnf install gcc python3
# Optional: exiftool for full metadata copy (EXIF/GPS/MakerNotes) —
#   apt: libimage-exiftool-perl / pacman: perl-image-exiftool. Without it
#   the outputs still render correctly; metadata copy is skipped.
set -eu
cd "$(dirname "$0")"

echo "== GALOSH Linux setup =="

command -v gcc >/dev/null 2>&1 || {
  echo "ERROR: gcc not found."
  echo "  Debian/Ubuntu: sudo apt install gcc"
  echo "  Arch/Manjaro:  sudo pacman -S gcc"
  exit 1
}
command -v python3 >/dev/null 2>&1 || {
  echo "ERROR: python3 not found."
  echo "  Debian/Ubuntu: sudo apt install python3 python3-venv"
  echo "  Arch/Manjaro:  sudo pacman -S python"
  exit 1
}

echo "-- [1/3] building the CLI denoisers (gcc -O3) ..."
(cd standalone && bash build.sh raw >/dev/null && bash build.sh yuv >/dev/null)
echo "   built: standalone/galosh_raw_cpu.exe, standalone/galosh_yuv_cpu.exe"
echo "   (the .exe suffix is just the build script's naming — these are"
echo "    normal Linux binaries)"

echo "-- [2/3] creating a private Python environment in .venv/ ..."
if ! python3 -m venv .venv 2>/dev/null; then
  echo "ERROR: python3 -m venv failed."
  echo "  Debian/Ubuntu: sudo apt install python3-venv   then re-run."
  exit 1
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet rawpy numpy tifffile pillow
echo "   installed: rawpy numpy tifffile pillow (inside .venv/ only)"

echo "-- [3/3] generating launchers ..."
cat > galosh-dng <<'SH'
#!/usr/bin/env bash
exec "$(dirname "$0")/.venv/bin/python" "$(dirname "$0")/tools/dist/galosh_dng.py" "$@"
SH
cat > galosh-png <<'SH'
#!/usr/bin/env bash
exec "$(dirname "$0")/.venv/bin/python" "$(dirname "$0")/tools/dist/galosh_png.py" "$@"
SH
chmod +x galosh-dng galosh-png

if command -v exiftool >/dev/null 2>&1; then
  echo "   exiftool: found ($(command -v exiftool)) — full metadata copy enabled"
else
  echo "   exiftool: not found — outputs still render fine; metadata copy is"
  echo "             skipped (optional: apt install libimage-exiftool-perl)"
fi

echo ""
echo "== done. Usage =="
echo "  ./galosh-dng /path/to/photo.CR3            # -> photo_GALOSH_l1_c1.dng"
echo "  ./galosh-png /path/to/image.png            # -> image_GALOSH_l1_c1.png"
echo "  ./galosh-dng -l 0.7 -c 0.7 photo.dng       # gentler strengths"
echo "Outputs are written NEXT TO the input; originals are never modified."
echo "More: docs/QUICKSTART_LINUX.md"
