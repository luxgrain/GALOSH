#!/usr/bin/env python3
"""PG-noise track viewer: condition x track x scene x ISO, method panels pan/zoom-
synced with per-panel metrics.  Conditions = core (median-sensor PG noise) and
cmp (+H.264 CRF23); tracks = 420 (limited) and 444 (full).  Reads the merged
report_set8_pgnoise metrics (_metrics_pg_{mode}[ _cmp].json) for labels and the png
trees (png420[/_cmp23], png[/_cmp23]) written by a --no-png-off PNG slice.

Regenerate:  python report_set8_pgnoise.py  (merges metrics)
             python bench_set8_pgnoise.py ... (PNG slice, no --no-png)
             python make_set8_pgnoise_viewer.py  ->  results_set8_pgnoise/viewer.html
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "benchmark" / "results_set8_pgnoise"
ISOS = ["ISO400", "ISO800", "ISO1600", "ISO3200", "ISO6400", "ISO12800",
        "ISO25600"]
# (condition-key, png subdir, merged-metrics file, method list)
# expose EVERY saved method (persist-everything: all 15 for 420 / 10 for 444)
GAL = ["galosh-cpu-fit", "galosh-cpu-hold", "galosh-vk-fit", "galosh-vk-hold"]
# [2026-07-17] unguarded MAD twins + smdegrain fully archived; both lanes
# share the fair set: oracle + guarded-estimate + knl + hqdn3d
M420 = ["gt_ref", "noisy"] + GAL + ["bm3d1", "bm3d1bg", "vbm3d",
                                    "vbm3dbg", "knl", "hqdn3d"]
M444 = list(M420)
CONDS = [
    ("core-420", "png420",        "_metrics_pg_420.json",     M420),
    ("cmp-420",  "png420_cmp23",  "_metrics_pg_420_cmp.json", M420),
    ("core-444", "png",           "_metrics_pg_444.json",     M444),
    ("cmp-444",  "png_cmp23",     "_metrics_pg_444_cmp.json", M444),
]
LABEL = {"gt_ref": "clean (ref)", "noisy": "noisy (developed)",
         "galosh-cpu-fit": "GALOSH cpu-fit", "galosh-cpu-hold": "GALOSH cpu-hold",
         "galosh-vk-fit": "GALOSH vk-fit", "galosh-vk-hold": "GALOSH vk-hold",
         "bm3d1": "BM3D (sigma: measured oracle)",
         "bm3d1b": "BM3D (sigma: MAD estimate)",
         "bm3d1bg": "BM3D (sigma: MAD estimate + 0.5 floor guard)",
         "vbm3d": "[T] V-BM3D (sigma: measured oracle)",
         "vbm3db": "[T] V-BM3D (sigma: MAD estimate)",
         "vbm3dbg": "[T] V-BM3D (sigma: MAD estimate + 0.5 floor guard)",
         "knl": "[T] KNL d=1", "smdegrain": "[T] SMDegrain (dropped 07-16)",
         "hqdn3d": "[T] hqdn3d (untuned default)"}


def main():
    data = {"conditions": {}, "labels": LABEL}
    for ckey, sub, mfile, methods in CONDS:
        mp = OUT / mfile
        png = OUT / sub
        if not mp.exists() or not png.exists():
            continue
        m = json.loads(mp.read_text())
        scenes = {}
        metrics = {}
        for s in sorted(k for k in m if "n_frames" in m[k]):
            isos = [i for i in ISOS if i in m[s]]
            avail = []
            for iso in isos:
                d = png / s / iso / "noisy"
                n = len(list(d.glob("*.png"))) if d.exists() else 0
                if n == 0:
                    continue
                avail.append(iso)
                metrics.setdefault(s, {})[iso] = {}
                for meth in methods:
                    e = m[s][iso].get(meth)
                    if e and meth != "gt_ref":
                        metrics[s][iso][meth] = {
                            "psnr": round(e.get("psnr", 0), 2),
                            "lpips": round(e.get("lpips", 0), 3),
                            "cr": round(e["cr_psnr"], 2) if "cr_psnr" in e
                            else None}
                # frame count from noisy dir
                metrics[s][iso]["_n"] = len(
                    list((png / s / iso / "noisy").glob("*.png")))
            if avail:
                scenes[s] = {"isos": avail}
        if scenes:
            data["conditions"][ckey] = {
                "methods": methods, "scenes": scenes, "metrics": metrics,
                "png": sub}
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    (OUT / "viewer.html").write_text(html, encoding="utf-8")
    nc = len(data["conditions"])
    print("saved:", OUT / "viewer.html", "| conditions:", nc)


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>GALOSH PG-noise track viewer</title><style>
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
 <label>condition</label><select id="cond"></select>
 <label>scene</label><select id="scene"></select>
 <label>ISO</label><select id="iso"></select>
 <label>layout</label><select id="layout"><option value="2">2-up</option><option value="4" selected>2&times;2</option><option value="6">2&times;3</option></select>
 <button id="prev">&#9664;</button><button id="play">&#9654;</button><button id="next">&#9654;&#9654;</button>
 <input type="range" id="frame" min="0" value="0" style="width:180px"><span id="fn"></span>
 <button id="rz">1:1</button>
 <span id="help">median-sensor PG noise &middot; wheel=zoom drag=pan dblclick=fit</span>
</div><div id="grid"></div><script>
const DATA=/*__DATA__*/;const $=id=>document.getElementById(id);
const st={cond:null,scene:null,iso:null,frame:0,layout:4,meth:[],
  playing:false,timer:null,z:{s:1,tx:0,ty:0},fitted:false};
function C(){return DATA.conditions[st.cond];}
function nF(){const mm=C().metrics[st.scene][st.iso];return (mm&&mm._n)||1;}
function pad(i){return String(i).padStart(4,"0");}
function url(m,f){return `${C().png}/${st.scene}/${st.iso}/${m}/${pad(f)}.png`;}
function metTxt(m){if(m==="gt_ref")return "clean reference";
  const e=C().metrics[st.scene][st.iso][m];
  if(!e)return "";let t=`PSNR ${e.psnr.toFixed(2)}  LPIPS ${e.lpips.toFixed(3)}`;
  if(e.cr!=null)t+=`  Cr ${e.cr.toFixed(2)}`;return t;}
function defMeth(){const M=C().methods;
  const pick=["noisy","galosh-cpu-fit","bm3d1","gt_ref","galosh-vk-fit","smdegrain"];
  const o=pick.filter(x=>M.includes(x));while(o.length<6)o.push(M[0]);return o;}
function buildGrid(){const g=$("grid");
  g.style.gridTemplateColumns=st.layout===6?"1fr 1fr 1fr":"1fr 1fr";
  g.style.gridTemplateRows=st.layout===2?"1fr":(st.layout===6?"1fr 1fr":"1fr 1fr");g.innerHTML="";
  const np=st.layout;
  for(let i=0;i<np;i++){const p=document.createElement("div");p.className="panel";
    const hd=document.createElement("div");hd.className="phead";
    const sel=document.createElement("select");
    C().methods.forEach(mm=>{const o=document.createElement("option");o.textContent=DATA.labels[mm]||mm;o.value=mm;sel.appendChild(o);});
    sel.value=st.meth[i];sel.onchange=()=>{st.meth[i]=sel.value;draw();};hd.appendChild(sel);
    const met=document.createElement("span");met.className="pmet";hd.appendChild(met);
    const wrap=document.createElement("div");wrap.className="pimg";
    const img=document.createElement("img");wrap.appendChild(img);
    p.appendChild(hd);p.appendChild(wrap);attachNav(wrap);g.appendChild(p);}
  st.fitted=false;}
function draw(){document.querySelectorAll("#grid .panel").forEach((p,i)=>{
  const m=st.meth[i];const img=p.querySelector("img");img.src=url(m,st.frame);
  if(!st.fitted)img.onload=()=>fit(img);
  p.querySelector(".pmet").textContent=metTxt(m);applyZ(img);});
  $("frame").max=nF()-1;$("frame").value=st.frame;$("fn").textContent=`${st.frame+1}/${nF()}`;}
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
function fillSel(id,items,cur){const s=$(id);s.innerHTML="";items.forEach(i=>{const o=document.createElement("option");o.textContent=i;s.appendChild(o);});
  s.value=items.includes(cur)?cur:items[0];return s.value;}
function fillScene(){const sc=Object.keys(C().scenes);st.scene=fillSel("scene",sc,st.scene);}
function fillIso(){const is=C().scenes[st.scene].isos;st.iso=fillSel("iso",is,st.iso);}
function reset(full){if(full){fillScene();}fillIso();st.meth=defMeth();st.frame=0;st.fitted=false;buildGrid();draw();}
$("cond").onchange=e=>{st.cond=e.target.value;reset(true);};
$("scene").onchange=e=>{st.scene=e.target.value;fillIso();st.frame=0;st.fitted=false;draw();};
$("iso").onchange=e=>{st.iso=e.target.value;st.frame=Math.min(st.frame,nF()-1);draw();};
$("layout").onchange=e=>{st.layout=+e.target.value;st.meth=defMeth();buildGrid();draw();};
$("prev").onclick=()=>step(-1);$("next").onclick=()=>step(1);$("play").onclick=()=>setPlay(!st.playing);
$("frame").oninput=e=>{st.frame=+e.target.value;draw();};$("rz").onclick=()=>{st.z={s:1,tx:0,ty:0};applyAll();};
document.addEventListener("keydown",e=>{if(e.target.tagName==="SELECT")return;
  if(e.key==="ArrowLeft"){step(-1);e.preventDefault();}if(e.key==="ArrowRight"){step(1);e.preventDefault();}});
Object.keys(DATA.conditions).forEach(c=>{const o=document.createElement("option");o.textContent=c;$("cond").appendChild(o);});
st.cond=Object.keys(DATA.conditions)[0];$("cond").value=st.cond;reset(true);
</script></body></html>
"""

if __name__ == "__main__":
    main()
