import sys, re, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

keywords = [
    'galosh', 'GALOSH', 'rawdenoise', 'rawforge', 'forge',
    'denoise', 'denoising', 'DENOISE', 'denoiser',
    'bm3d', 'BM3D', 'WHT', 'wht', 'GAT', 'anscombe', 'Anscombe',
    'poisson', 'Poisson', 'shot noise', 'photon',
    'private', 'secret',
    'luxgrain',
    'integrate.*denois', 'denois.*integrat',
    'future.*work', 'planned',
]

files = [
    r'C:\Users\luxgrain\darktable-rawforge\darktable\src\iop\demosaicing\ari.c',
    r'C:\Users\luxgrain\darktable-rawforge\darktable\src\iop\demosaicing\menon.c',
    r'C:\Users\luxgrain\darktable-rawforge\darktable\src\iop\demosaic.c',
]

for fpath in files:
    fname = os.path.basename(fpath)
    with open(fpath, encoding='utf-8') as f:
        lines = f.readlines()
    hits = []
    for i, line in enumerate(lines, 1):
        ll = line.lower()
        for kw in keywords:
            if re.search(kw.lower(), ll):
                hits.append((i, kw, line.rstrip()))
                break
    if hits:
        print(f'\n=== {fname}: {len(hits)} hits ===')
        for ln, kw, text in hits:
            print(f'  line {ln} [{kw}]: {text[:120]}')
    else:
        print(f'{fname}: CLEAN')
