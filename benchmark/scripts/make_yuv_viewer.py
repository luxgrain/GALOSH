"""Build a SYNCHRONIZED image-comparison viewer for the YUV sRGB benches
(wheel=zoom, drag=pan, all panes move together — same mechanism as the RAW viewer.html).

Per dataset/scene: a full-resolution display JPG for every method + noisy + GT, plus the
per-scene metrics. Output: benchmark/viewer_srgb/{viewer.html, imgs/}.
Display JPG is the native crop (SIDD 1024², RawNIND 512²); zoom on it reveals detail.
"""
import json
from pathlib import Path
from PIL import Image

GAL = Path(__file__).resolve().parents[2]
OUTV = GAL / "benchmark" / "viewer_srgb"
(OUTV / "imgs").mkdir(parents=True, exist_ok=True)

ORDER = [("noisy", "Noisy (input)"), ("galosh_yuv_gpu_fp32", "GALOSH-YUV GPU (training-free)"),
         ("cbm3d", "CBM3D"), ("color_nlm", "Color-NLM"), ("guided", "Guided"),
         ("nafnet", "NAFNet-SIDD (DL)"), ("scunet", "SCUNet-real (DL)"),
         ("restormer", "Restormer-SIDD (DL)"), ("gt", "Ground truth")]
DATASETS = [("sidd", "SIDD-Medium sRGB", GAL / "benchmark" / "results_srgb_sidd_1024"),
            ("rawnind", "RawNIND-rendered sRGB", GAL / "benchmark" / "results_srgb_rawnind")]
RESDIR = {"sidd": "../results_srgb_sidd_1024", "rawnind": "../results_srgb_rawnind"}
MAXW = 1024


def to_jpg(src_png, dst_jpg):
    im = Image.open(src_png).convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)), Image.LANCZOS)
    im.save(dst_jpg, quality=88)
    return im.width, im.height


datasets = []
for key, name, root in DATASETS:
    if not (root / "_metrics.json").exists():
        print(f"skip {key}"); continue
    rows = json.load(open(root / "_metrics.json"))
    by_tag = {}
    for r in rows:
        if r.get("ok") is False: continue
        by_tag.setdefault(r["tag"], {})[r["method"]] = r
    scenes = []
    tags = sorted(by_tag)
    if key == "rawnind" and len(tags) > 220:   # viewer: representative subset (table uses full 1493)
        step = len(tags) / 220
        tags = [tags[int(i * step)] for i in range(220)]
    for i, tag in enumerate(tags):
        w = h = 0; imgs = []
        for mdir, label in ORDER:
            png = root / mdir / f"{tag}.png"
            if not png.exists(): continue
            jd = OUTV / "imgs" / key / mdir; jd.mkdir(parents=True, exist_ok=True)
            w, h = to_jpg(png, jd / f"{tag}.jpg")
            mr = by_tag[tag].get("_noisy" if mdir == "noisy" else mdir, {})
            met = ""
            if "psnr" in mr:
                met = (f"PSNR {mr['psnr']:.2f} · SSIM {mr['ssim']:.3f} · "
                       f"LPIPS {mr['lpips']:.3f} · DISTS {mr['dists']:.3f}")
            imgs.append({"k": mdir, "label": label,
                         "img": f"imgs/{key}/{mdir}/{tag}.jpg",
                         "full": f"{RESDIR[key]}/{mdir}/{tag}.png", "met": met})
        scenes.append({"scene": tag, "w": w, "h": h, "imgs": imgs})
        if (i + 1) % 25 == 0: print(f"  {key}: {i+1} scenes")
    datasets.append({"name": name, "key": key, "scenes": scenes})
    print(f"{key}: {len(scenes)} scenes")

DATA = json.dumps({"datasets": datasets})

HTML = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>GALOSH-YUV sRGB — synchronized comparison</title><style>
 :root{--bg:#0e0f13;--card:#171a21;--line:#2a2f3a;--fg:#e6e9ef;--mut:#9aa3b2;--galosh:#1f6feb}
 *{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:10px 16px;border-bottom:1px solid var(--line)}
 h1{font-size:16px;margin:0 0 7px}
 .row{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
 button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer;font-size:12.5px}
 button.on{background:var(--galosh);border-color:var(--galosh)}
 select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px}
 .hint{color:var(--mut);font-size:12px;margin-left:8px}
 #grid{display:grid;gap:8px;padding:10px 16px;grid-template-columns:repeat(var(--cols,4),1fr)}
 .tile{background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden;position:relative}
 .tile.galosh{border-color:var(--galosh)}
 .vp{position:relative;width:100%;overflow:hidden;cursor:grab;background:#000}
 .vp.drag{cursor:grabbing}
 .vp img{position:absolute;left:0;top:0;transform-origin:0 0;will-change:transform;image-rendering:pixelated;-webkit-user-drag:none;user-select:none}
 .cap{position:absolute;left:0;right:0;bottom:0;padding:4px 8px;font-size:12px;background:linear-gradient(transparent,rgba(0,0,0,.82));color:#fff;pointer-events:none}
 .cap .m{color:#cdd;font-variant-numeric:tabular-nums}
 .tile.galosh .cap{color:#9cc2ff;font-weight:600}
 .open{position:absolute;top:5px;right:6px;font-size:11px;color:#cdd;background:rgba(0,0,0,.5);padding:2px 6px;border-radius:5px;text-decoration:none;z-index:3}
</style></head><body>
<header><h1>GALOSH-YUV sRGB — 同期画像比較 (all blind)</h1>
 <div class="row"><span id="tabs"></span><select id="scene"></select>
   <button id="reset">reset view</button>
   <label>cols <input id="cols" type="range" min="2" max="5" value="4" style="vertical-align:middle"></label>
   <span class="hint">wheel = zoom · drag = pan (全ペイン連動) · click "full‑res" = PNG</span>
 </div></header>
<div id="grid"></div>
<script>
const D=__DATA__;
const tabs=document.getElementById('tabs'),sel=document.getElementById('scene'),grid=document.getElementById('grid');
let di=0,si=0,view={s:1,x:0,y:0},natW=1,natH=1,imgs=[];
D.datasets.forEach((d,i)=>{const b=document.createElement('button');b.textContent=`${d.name} (${d.scenes.length})`;
  b.className=i===0?'on':'';b.onclick=()=>{di=i;si=0;[...tabs.children].forEach((x,j)=>x.className=j===i?'on':'');fillScenes();load()};tabs.appendChild(b);});
function fillScenes(){sel.innerHTML='';D.datasets[di].scenes.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.scene;sel.appendChild(o);});}
sel.onchange=()=>{si=+sel.value;load()};
document.getElementById('reset').onclick=()=>{fitView();apply()};
document.getElementById('cols').oninput=e=>grid.style.setProperty('--cols',e.target.value);
grid.style.setProperty('--cols',4);
function load(){const sc=D.datasets[di].scenes[si];natW=sc.w;natH=sc.h;sel.value=si;
  grid.innerHTML='';imgs=[];
  for(const im of sc.imgs){const g=im.k.startsWith('galosh')?' galosh':'';
    const t=document.createElement('div');t.className='tile'+g;
    t.innerHTML=`<a class="open" href="${im.full}" target="_blank">full‑res ↗</a><div class="vp"><img src="${im.img}"></div><div class="cap">${im.label}<br><span class=m>${im.met||''}</span></div>`;
    grid.appendChild(t);
    const vp=t.querySelector('.vp'),img=t.querySelector('img');
    imgs.push({vp,img});attach(vp,img);
    img.addEventListener('load',()=>{fitView();apply();},{once:true});
  }
  requestAnimationFrame(()=>requestAnimationFrame(()=>{fitView();apply();}));
}
function fitView(){const vpw=imgs[0]?imgs[0].vp.clientWidth:natW;
  imgs.forEach(o=>o.vp.style.height=(vpw*natH/natW)+'px');view={s:1,x:0,y:0};}
function apply(){const vpw=imgs[0]?imgs[0].vp.clientWidth:natW;const base=vpw/natW;
  imgs.forEach(o=>{o.img.style.transform=`translate(${view.x}px,${view.y}px) scale(${base*view.s})`;
    o.img.style.width=natW+'px';o.img.style.height=natH+'px';});}
function attach(vp,img){
  vp.addEventListener('wheel',e=>{e.preventDefault();
    const r=vp.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const f=e.deltaY<0?1.18:1/1.18;const ns=Math.min(40,Math.max(1,view.s*f));const k=ns/view.s;
    view.x=mx-(mx-view.x)*k;view.y=my-(my-view.y)*k;view.s=ns;
    if(view.s<=1.0001){view.s=1;view.x=0;view.y=0;}apply();},{passive:false});
  let dr=null;
  vp.addEventListener('mousedown',e=>{dr={x:e.clientX,y:e.clientY,ox:view.x,oy:view.y};vp.classList.add('drag');});
  window.addEventListener('mousemove',e=>{if(!dr)return;view.x=dr.ox+(e.clientX-dr.x);view.y=dr.oy+(e.clientY-dr.y);apply();});
  window.addEventListener('mouseup',()=>{if(dr){dr=null;document.querySelectorAll('.vp').forEach(v=>v.classList.remove('drag'));}});
}
window.addEventListener('resize',()=>{fitView();apply();});
fillScenes();load();
</script></body></html>"""
(OUTV / "viewer.html").write_text(HTML.replace("__DATA__", DATA), encoding="utf-8")
# remove the old non-synced thumbs/crops dirs if present
import shutil
for d in ("thumbs", "crops"):
    p = OUTV / d
    if p.exists(): shutil.rmtree(p)
print("wrote", OUTV / "viewer.html")
