import numpy as np
import subprocess, os, tempfile

EXE  = r'C:\Users\luxgrain\denoise_eval\standalone\yuv_galosh.exe'
BASH = r'C:\msys64\usr\bin\bash.exe'

H, W = 64, 64
Y  = (np.random.rand(H, W) * 0.5 + 0.2).astype(np.float32)
Cb = np.zeros((H, W), dtype=np.float32)
Cr = np.zeros((H, W), dtype=np.float32)

tmpdir = tempfile.mkdtemp()
in_path  = os.path.join(tmpdir, 'in.yuv')
out_path = os.path.join(tmpdir, 'out.yuv')
with open(in_path, 'wb') as f:
    Y.tofile(f); Cb.tofile(f); Cr.tofile(f)

def win2bash(p):
    return p.replace('C:', '/c').replace('\\', '/')

cmd = '{} {} {} {} {} 444 1.0 1.0 2 2'.format(
    win2bash(EXE), win2bash(in_path), win2bash(out_path), W, H)
env = dict(os.environ)
env['PATH'] = 'C:/msys64/ucrt64/bin;C:/msys64/usr/bin;' + env.get('PATH', '')
r = subprocess.run([BASH, '-lc', cmd], capture_output=True,
                   text=True, encoding='utf-8', errors='replace', env=env)
print('rc:', r.returncode)
print('stdout:', r.stdout[:300])
print('stderr:', r.stderr[:300])
print('out exists:', os.path.exists(out_path))
if os.path.exists(out_path):
    d = np.fromfile(out_path, dtype=np.float32)
    print('output floats:', len(d), 'expected:', H * W * 3)
    print('Y range: [{:.4f}, {:.4f}]'.format(d[:H*W].min(), d[:H*W].max()))
