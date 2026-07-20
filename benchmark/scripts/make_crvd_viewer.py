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
# [2026-07-16 unification] two lanes (420 production / 444 protocol);
# galosh444 side-check archived; smdegrain kept only where its PNGs exist
# (420, as a saved artifact of the dropped method).
GAL = ["galosh-cpu-fit", "galosh-cpu-hold", "galosh-vk-fit", "galosh-vk-hold"]
# [2026-07-17] unguarded MAD twins + smdegrain fully archived
BASE = ["bm3d1", "bm3d1bg", "vbm3d", "vbm3dbg", "knl"]
M420 = ["gt_ref", "noisy"] + GAL + BASE + ["hqdn3d"]
M444 = ["gt_ref", "noisy"] + GAL + BASE + ["hqdn3d"]
# synth arms (2026-07-17 generator-fidelity): no bg twins / no smdegrain
MSYN = ["gt_ref", "noisy"] + GAL + ["bm3d1", "bm3d1b", "vbm3d", "vbm3db",
                                    "knl", "hqdn3d"]
LANES = [("420", "_metrics_crvd.json", "png", M420),
         ("444", "_metrics_crvd444.json", "png444", M444),
         ("420 synth imx385", "_metrics_crvdsynth.json", "png_synth", MSYN),
         ("420 synth phone-match", "_metrics_crvdsynthphone.json",
          "png_synth_phone", MSYN)]
LABEL = {"gt_ref": "clean (reference)", "noisy": "noisy (real sensor)",
         "galosh-cpu-fit": "GALOSH cpu-fit",
         "galosh-cpu-hold": "GALOSH cpu-hold",
         "galosh-vk-fit": "GALOSH vk-fit",
         "galosh-vk-hold": "GALOSH vk-hold",
         "bm3d1": "BM3D (sigma: measured oracle)",
         "bm3d1b": "BM3D (sigma: MAD estimate)",
         "bm3d1bg": "BM3D (sigma: MAD estimate + 0.5 floor guard)",
         "vbm3d": "[T] V-BM3D (sigma: measured oracle)",
         "vbm3db": "[T] V-BM3D (sigma: MAD estimate)",
         "vbm3dbg": "[T] V-BM3D (sigma: MAD estimate + 0.5 floor guard)",
         "knl": "[T] KNL d=1", "smdegrain": "[T] SMDegrain (dropped 07-16)",
         "hqdn3d": "[T] hqdn3d (untuned default)"}


def main():
    data = {"lanes": {}, "labels": LABEL}
    for lane, mfile, sub, methods in LANES:
        mp = OUT / mfile
        png = OUT / sub
        if not mp.exists() or not png.exists():
            continue
        m = json.loads(mp.read_text())
        scenes = sorted([s for s in m if s.startswith("scene")],
                        key=lambda s: int(s.replace("scene", "")))
        ld = {"pngdir": sub, "scenes": {}, "metrics": {}, "methods": methods}
        for s in scenes:
            isos = [i for i in ISOS if i in m[s]]
            if not isos:
                continue
            d = png / s / isos[0] / "noisy"
            n = len(list(d.glob("*.png"))) if d.exists() else 7
            ld["scenes"][s] = {"isos": isos, "n": max(n, 1)}
            ld["metrics"][s] = {}
            for iso in isos:
                ld["metrics"][s][iso] = {}
                for meth in methods:
                    e = m[s][iso].get(meth)
                    if e and meth != "gt_ref":
                        ld["metrics"][s][iso][meth] = {
                            "psnr": round(e["psnr"], 2),
                            "lpips": round(e["lpips"], 3)}
        data["lanes"][lane] = ld
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    (OUT / "viewer.html").write_text(html, encoding="utf-8")
    print("saved:", OUT / "viewer.html",
          "| lanes:", ", ".join(f"{k}={len(v['scenes'])}sc"
                                for k, v in data["lanes"].items()))


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
 <label>lane</label><select id="lane"></select>
 <label>scene</label><select id="scene"></select>
 <label>ISO</label><select id="iso"></select>
 <label>layout</label><select id="layout"><option value="2">2-up</option><option value="4" selected>2&times;2</option></select>
 <button id="prev">&#9664;</button><button id="play">&#9654;</button><button id="next">&#9654;&#9654;</button>
 <input type="range" id="frame" min="0" value="0" style="width:180px"><span id="fn"></span>
 <button id="rz">1:1</button>
 <span id="help">real SONY IMX385 noise &middot; wheel=zoom drag=pan dblclick=fit</span>
</div><div id="grid"></div><script>
const DATA=/*__DATA__*/;const $=id=>document.getElementById(id);
const st={lane:Object.keys(DATA.lanes)[0],scene:null,iso:null,frame:0,layout:4,
  meth:["noisy","galosh-cpu-fit","bm3d1b","gt_ref"],
  playing:false,timer:null,z:{s:1,tx:0,ty:0},fitted:false};
function C(){return DATA.lanes[st.lane];}
function nF(){return C().scenes[st.scene].n;}
function pad(i){return String(i).padStart(2,"0");}
function url(m,f){return `${C().pngdir}/${st.scene}/${st.iso}/${m}/${pad(f)}.png`;}
function metTxt(m){if(m==="gt_ref")return "clean reference";
  const e=C().metrics[st.scene][st.iso][m];
  return e?`PSNR ${e.psnr.toFixed(2)}  LPIPS ${e.lpips.toFixed(3)}`:"";}
function buildGrid(){const g=$("grid");
  g.style.gridTemplateColumns=st.layout===4?"1fr 1fr":"1fr 1fr";
  g.style.gridTemplateRows=st.layout===4?"1fr 1fr":"1fr";g.innerHTML="";
  const np=st.layout===4?4:2;
  for(let i=0;i<np;i++){const p=document.createElement("div");p.className="panel";
    const hd=document.createElement("div");hd.className="phead";
    const sel=document.createElement("select");
    C().methods.forEach(mm=>{const o=document.createElement("option");o.textContent=(DATA.labels&&DATA.labels[mm])||mm;o.value=mm;sel.appendChild(o);});
    if(!C().methods.includes(st.meth[i]))st.meth[i]=C().methods[Math.min(i,C().methods.length-1)];
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
function fillIso(){const s=$("iso");s.innerHTML="";C().scenes[st.scene].isos.forEach(i=>{const o=document.createElement("option");o.textContent=i;s.appendChild(o);});
  if(!C().scenes[st.scene].isos.includes(st.iso))st.iso=C().scenes[st.scene].isos[0];s.value=st.iso;}
function fillScenes(){const sc=$("scene");sc.innerHTML="";
  Object.keys(C().scenes).forEach(s=>{const o=document.createElement("option");o.textContent=s;sc.appendChild(o);});
  if(!C().scenes[st.scene])st.scene=Object.keys(C().scenes)[0];sc.value=st.scene;}
$("scene").onchange=e=>{st.scene=e.target.value;fillIso();st.frame=0;st.fitted=false;draw();};
$("iso").onchange=e=>{st.iso=e.target.value;st.frame=Math.min(st.frame,nF()-1);draw();};
$("layout").onchange=e=>{st.layout=+e.target.value;buildGrid();draw();};
$("prev").onclick=()=>step(-1);$("next").onclick=()=>step(1);$("play").onclick=()=>setPlay(!st.playing);
$("frame").oninput=e=>{st.frame=+e.target.value;draw();};$("rz").onclick=()=>{st.z={s:1,tx:0,ty:0};applyAll();};
document.addEventListener("keydown",e=>{if(e.target.tagName==="SELECT")return;
  if(e.key==="ArrowLeft"){step(-1);e.preventDefault();}if(e.key==="ArrowRight"){step(1);e.preventDefault();}});
Object.keys(DATA.lanes).forEach(l=>{const o=document.createElement("option");o.textContent=l;$("lane").appendChild(o);});
$("lane").value=st.lane;
$("lane").onchange=e=>{st.lane=e.target.value;fillScenes();fillIso();st.frame=0;st.fitted=false;buildGrid();draw();};
st.scene=Object.keys(C().scenes)[0];fillScenes();fillIso();buildGrid();draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
