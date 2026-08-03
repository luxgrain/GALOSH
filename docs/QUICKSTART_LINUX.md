# GALOSH on Linux — quick start (no Python knowledge needed)

GALOSH is a blind, training-free denoiser: you give it a photo, it figures
out the noise by itself and writes a denoised copy **next to the original**.
The original file is never modified.

## Setup (once)

```sh
git clone https://github.com/luxgrain/GALOSH.git
cd GALOSH
./install_linux.sh
```

That's the whole installation. The script

- compiles the two denoiser programs with `gcc` (about 10 seconds),
- creates a **private** Python folder `.venv/` inside `GALOSH/` for the
  raw-file reader (`rawpy`) — nothing is installed system-wide, and you
  never need to "activate" anything,
- generates two launchers, `./galosh-dng` and `./galosh-png`.

If the script stops with an error it tells you the exact package to
install (`gcc`, `python3-venv`, …) for your distribution.

Optional: install `exiftool` (Debian/Ubuntu `libimage-exiftool-perl`,
Arch `perl-image-exiftool`) if you want the full EXIF/GPS/MakerNotes
block copied onto the results. Without it everything still works — the
outputs carry the color calibration they need to render correctly — and
you can copy tags yourself later (e.g. with `exiv2`).

## Denoise a raw photo (DNG, CR3, CR2, NEF, ARW, …)

```sh
./galosh-dng /path/to/photo.CR3
# -> /path/to/photo_GALOSH_l1_c1.dng
```

Anything LibRaw can read with a regular Bayer sensor works; the output is
always a DNG. No noise level to enter, no profiles: the noise model is
estimated from the photo itself.

## Denoise an sRGB image (PNG)

```sh
./galosh-png /path/to/image.png
# -> /path/to/image_GALOSH_l1_c1.png
```

## Strength

The two knobs (defaults are `1.0`, the calibrated strength):

```sh
./galosh-dng -l 0.7 -c 0.7 photo.dng     # gentler luma + color
./galosh-png -l 0.7 photo.png            # keep a little more grain
```

`-l` (luma) controls grain/brightness noise, `-c` (chroma) controls color
blotches. For lightly noisy material `-l 0.7` is a good perceptual sweet
spot; keep `1.0` for high-ISO shots. The chosen values are embedded in the
output filename, so experiments never overwrite each other.

Skip the metadata copy on purpose (e.g. you prefer your own exiv2 flow):

```sh
./galosh-dng --no-exiftool photo.CR3
```

## Trouble?

- `Read N floats, expected M` from the low-level CLI: you ran the bare
  `standalone/galosh_*.exe` on an image file. Those programs read raw
  float32 planes, not image containers — use the launchers above, or see
  `standalone/README.md` for the CLI contract.
- Anything else: [open an issue](https://github.com/luxgrain/GALOSH/issues)
  with the console output.

## Under the hood / going further

The launchers wrap `tools/dist/galosh_dng.py` / `galosh_png.py`, which
call the compiled reference denoisers in `standalone/`. Direct CLI use,
GPU (OpenCL/Vulkan) builds, planar YCbCr modes and every option are
documented in `standalone/README.md`. For denoising *video* inside
VapourSynth/AviSynth+, see
[GALOSH-frameserver](https://github.com/luxgrain/GALOSH-frameserver).
