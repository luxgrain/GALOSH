#!/usr/bin/env python3
"""CRVD real-noise viewer: scene x ISO, noisy -> GALOSH/SMDegrain -> clean,
pan/zoom-synced with per-panel metrics.  Reads results_crvd/_metrics_crvd.json
and the png/ tree (noisy, gt_ref, galosh420, smdegrain).  Regenerate after
crvd_png_gen.py: python make_crvd_viewer.py  ->  results_crvd/viewer.html
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "benchmark" / "results_crvd"
ISOS = ["ISO1600", "ISO3200", "ISO6400", "ISO12800", "ISO25600"]
METHODS = ["gt_ref", "noisy", "galosh-cpu-fit", "galosh-cpu-hold",
           "galosh-vk-fit", "galosh-vk-hold", "galosh444", "bm3d1", "bm3d1b",
           "vbm3d", "vbm3db", "knl", "smdegrain", "hqdn3d"]
LABEL = {"gt_ref": "clean (reference)", "noisy": "noisy (real sensor)",
         "galosh-cpu-fit": "GALOSH cpu-fit (blind)",
         "galosh-cpu-hold": "GALOSH cpu-hold (blind)",
         "galosh-vk-fit": "GALOSH vk-fit (blind)",
         "galosh-vk-hold": "GALOSH vk-hold (blind)",
         "galosh444": "GALOSH 444 (blind)",
         "bm3d1": "BM3D (sigma-oracle)", "bm3d1b": "BM3D (sigma-blind MAD)",
         "vbm3d": "V-BM3D (sigma-oracle temporal)",
         "vbm3db": "V-BM3D (sigma-blind MAD temporal)",
         "knl": "KNL d=1 (temporal)", "smdegrain": "SMDegrain (temporal)",
         "hqdn3d": "hqdn3d (untuned default)"}


def main():
    m = json.loads((OUT / "_metrics_crvd.json").read_text())
    scenes = sorted([s for s in m if s.startswith("scene")],
                    key=lambda s: int(s.replace("scene", "")))
    # frame counts + available ISOs per scene, metrics for label
    data = {"scenes": {}, "metrics": {}, "methods": METHODS,
            "labels": LABEL}
    png = OUT / "png"
    for s in scenes:
        isos = [i for i in ISOS if i in m[s]]
        if not isos:
            continue
        # frame count from the noisy png dir (fallback 7)
        d = png / s / isos[0] / "noisy"
        n = len(list(d.glob("*.png"))) if d.exists() else 7
        data["scenes"][s] = {"isos": isos, "n": max(n, 1)}
        data["metrics"][s] = {}
        for iso in isos:
            data["metrics"][s][iso] = {}
            for meth in METHODS:
                e = m[s][iso].get(meth)
                if e and meth != "gt_ref":
                    data["metrics"][s][iso][meth] = {
                        "psnr": round(e["psnr"], 2), "lpips": round(e["lpips"], 3)}
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    (OUT / "viewer.html").write_text(html, encoding="utf-8")
    print("saved:", OUT / "viewer.html", "| scenes:", len(data["scenes"]))


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>GALOSH CRVD viewer</title><style>
 :root{color-scheme:dark;}*{box-sizing:border-box;}
 body{margin:0;background:#111;color:#ddd;font:13px/1.4 system-ui,sans-serif;}
 #bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:6px 10px;
   background:#1c1c1c;position:sticky;top:0;z-index:5;border-bottom:1px solid #333;}
 #bar label{color:#999;} select,button{background:#2a2a2a;color:#ddd;
   border:1px solid #444;border-radius:4px;padding:2px 6px;}button{cursor:pointer;}
 button:hover{background:#3a3a3a;} #grid{display:grid;gap:2px;height:calc(100vh - 46px);}
 .panel{position:relative;overflow:hidden;background:#000;min-height:0;}
 .phead{position:absolute;top:0;left:0;right:0;z-index:3;display:flex;gap:8px;
   align-items:center;padding:3px 6px;background:rgba(20,20,20,.75);}
 .pmet{color:#8fc7ff;font-size:12px;white-space:nowrap;}
 .pimg{position:absolute;inset:0;} .pimg img{position:absolute;left:0;top:0;
   transform-origin:0 0;image-rendering:pixelated;user-select:none;-webkit-user-drag:none;}
 #help{color:#777;}
</style></head><body>
<div id="bar">
 <label>scene</label><select id="scene"></select>
 <label>ISO</label><select id="iso"></select>
 <label>layout</label><select id="layout"><option value="2">2-up</option><option value="4" selected>2&times;2</option></select>
 <button id="prev">&#9664;</button><button id="play">&#9654;</button><button id="next">&#9654;&#9654;</button>
 <input type="range" id="frame" min="0" value="0" style="width:180px"><span id="fn"></span>
 <button id="rz">1:1</button>
 <span id="help">real SONY IMX385 noise &middot; wheel=zoom drag=pan dblclick=fit</span>
</div><div id="grid"></div><script>
const DATA=/*__DATA__*/;const $=id=>document.getElementById(id);
const st={scene:null,iso:null,frame:0,layout:4,meth:["noisy","galosh-cpu-fit","bm3d1b","gt_ref"],
  playing:false,timer:null,z:{s:1,tx:0,ty:0},fitted:false};
function nF(){return DATA.scenes[st.scene].n;}
function pad(i){return String(i).padStart(2,"0");}
function url(m,f){return `png/${st.scene}/${st.iso}/${m}/${pad(f)}.png`;}
function metTxt(m){if(m==="gt_ref")return "clean reference";
  const e=DATA.metrics[st.scene][st.iso][m];
  return e?`PSNR ${e.psnr.toFixed(2)}  LPIPS ${e.lpips.toFixed(3)}`:"";}
function buildGrid(){const g=$("grid");
  g.style.gridTemplateColumns=st.layout===4?"1fr 1fr":"1fr 1fr";
  g.style.gridTemplateRows=st.layout===4?"1fr 1fr":"1fr";g.innerHTML="";
  const np=st.layout===4?4:2;
  for(let i=0;i<np;i++){const p=document.createElement("div");p.className="panel";
    const hd=document.createElement("div");hd.className="phead";
    const sel=document.createElement("select");
    DATA.methods.forEach(mm=>{const o=document.createElement("option");o.textContent=mm;sel.appendChild(o);});
    sel.value=st.meth[i];sel.onchange=()=>{st.meth[i]=sel.value;draw();};hd.appendChild(sel);
    const met=document.createElement("span");met.className="pmet";hd.appendChild(met);
    const wrap=document.createElement("div");wrap.className="pimg";
    const img=document.createElement("img");wrap.appendChild(img);
    p.appendChild(hd);p.appendChild(wrap);attachNav(wrap);g.appendChild(p);}
  st.fitted=false;}
function draw(){document.querySelectorAll("#grid .panel").forEach((p,i)=>{
  const m=st.meth[i];const img=p.querySelector("img");img.src=url(m,st.frame);
  if(!st.fitted)img.onload=()=>fit(img);
  p.querySelector(".pmet").textContent=(LABEL(m))+"  "+metTxt(m);applyZ(img);});
  $("frame").max=nF()-1;$("frame").value=st.frame;$("fn").textContent=`${st.frame+1}/${nF()}`;}
function LABEL(m){return (DATA.labels&&DATA.labels[m])||m;}
function fit(img){const w=img.parentElement.clientWidth,h=img.parentElement.clientHeight;
  if(!img.naturalWidth)return;const s=Math.min(w/img.naturalWidth,h/img.naturalHeight);
  st.z={s:s,tx:(w-img.naturalWidth*s)/2,ty:(h-img.naturalHeight*s)/2};st.fitted=true;applyAll();}
function applyZ(img){img.style.transform=`translate(${st.z.tx}px,${st.z.ty}px) scale(${st.z.s})`;}
function applyAll(){document.querySelectorAll("#grid img").forEach(applyZ);}
function attachNav(el){el.addEventListener("wheel",e=>{e.preventDefault();
  const r=el.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const f=e.deltaY<0?1.25:0.8;st.z.tx=mx-(mx-st.z.tx)*f;st.z.ty=my-(my-st.z.ty)*f;st.z.s*=f;applyAll();},{passive:false});
  let drag=null;el.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY};el.setPointerCapture(e.pointerId);});
  el.addEventListener("pointermove",e=>{if(!drag)return;st.z.tx+=e.clientX-drag.x;st.z.ty+=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};applyAll();});
  el.addEventListener("pointerup",()=>drag=null);el.addEventListener("dblclick",()=>fit(el.querySelector("img")));}
function step(d){st.frame=(st.frame+d+nF())%nF();draw();}
function setPlay(o){st.playing=o;$("play").innerHTML=o?"&#10074;&#10074;":"&#9654;";clearInterval(st.timer);if(o)st.timer=setInterval(()=>step(1),120);}
function fillIso(){const s=$("iso");s.innerHTML="";DATA.scenes[st.scene].isos.forEach(i=>{const o=document.createElement("option");o.textContent=i;s.appendChild(o);});
  if(!DATA.scenes[st.scene].isos.includes(st.iso))st.iso=DATA.scenes[st.scene].isos[0];s.value=st.iso;}
$("scene").onchange=e=>{st.scene=e.target.value;fillIso();st.frame=0;st.fitted=false;draw();};
$("iso").onchange=e=>{st.iso=e.target.value;st.frame=Math.min(st.frame,nF()-1);draw();};
$("layout").onchange=e=>{st.layout=+e.target.value;buildGrid();draw();};
$("prev").onclick=()=>step(-1);$("next").onclick=()=>step(1);$("play").onclick=()=>setPlay(!st.playing);
$("frame").oninput=e=>{st.frame=+e.target.value;draw();};$("rz").onclick=()=>{st.z={s:1,tx:0,ty:0};applyAll();};
document.addEventListener("keydown",e=>{if(e.target.tagName==="SELECT")return;
  if(e.key==="ArrowLeft"){step(-1);e.preventDefault();}if(e.key==="ArrowRight"){step(1);e.preventDefault();}});
Object.keys(DATA.scenes).forEach(s=>{const o=document.createElement("option");o.textContent=s;$("scene").appendChild(o);});
st.scene=Object.keys(DATA.scenes)[0];$("scene").value=st.scene;fillIso();buildGrid();draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
