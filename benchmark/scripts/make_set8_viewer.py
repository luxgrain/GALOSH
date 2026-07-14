#!/usr/bin/env python3
"""Generate the Set8 visual-comparison viewer (single self-contained HTML).

Reads _metrics_{444,420}.json + _metrics_420_baselines.json and the saved
PNG trees, embeds everything the page needs (frame counts, GT file lists,
per-seq/sigma metrics) and writes results_set8_awgn/viewer.html.

Open locally (file://) — images are referenced relatively (png/, png420/);
the 444-track GT frames point at the dataset via file:///E:/... URIs.

Panels are pan/zoom-synced; 1-up mode is a hold-Space A/B flipper.
Regenerate any time the benches are re-run: python make_set8_viewer.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "results_set8_awgn"
SET8 = Path("E:/img_dataset/Set8/test_sequences")
DERF = ["tractor", "touchdown", "park_joy", "sunflower"]
GOPRO = ["hypersmooth", "motorbike", "rafting", "snowboard"]
DERF_CAP = 85
SIGMAS = [10, 20, 30, 40, 50]
# [2026-07-14] unified galosh- naming; galosh-fix/-old (420-fix A/B) archived.
# PNG-backed methods only (GALOSH pngs exist for cpu-fit + vk-hold).
M444 = ["gt", "noisy", "galosh-cpu-fit", "galosh-vk-hold", "bm3d1", "bm3d1b",
        "knl"]
M420 = ["gt_ref", "noisy", "galosh-cpu-fit", "galosh-vk-hold", "bm3d1",
        "bm3d1b", "vbm3d", "vbm3db", "knl", "smdegrain", "hqdn3d"]
# cross-track entries: same seq/sigma/frame viewed from the OTHER track
X444 = ["444:gt", "444:noisy", "444:galosh-cpu-fit", "444:bm3d1", "444:knl"]
X420 = ["420:gt_ref", "420:noisy", "420:galosh-cpu-fit", "420:bm3d1",
        "420:knl"]


def gt_files(seq):
    d = SET8 / ("gopro_540p" if seq in GOPRO else "") / seq
    fs = sorted(d.glob("*.png"))
    if seq in DERF:
        fs = fs[:DERF_CAP]
    return ["file:///" + str(f).replace("\\", "/") for f in fs]


def pick_metrics(ent, method):
    e = ent.get(method)
    if not e:
        return None
    return {k: round(e[k], 4) for k in ("psnr", "ssim", "lpips",
                                        "dists", "niqe") if k in e}


def main():
    m444 = json.loads((OUT / "_metrics_444.json").read_text())
    m420 = json.loads((OUT / "_metrics_420.json").read_text())
    mbl = json.loads((OUT / "_metrics_420_baselines.json").read_text())
    bl4 = OUT / "_metrics_444_baselines.json"
    mbl4 = json.loads(bl4.read_text()) if bl4.exists() else {}

    def fold(base, path):        # add-on shards (blind twins) into base
        if path.exists():
            for s, cells in json.loads(path.read_text()).items():
                for k, v in cells.items():
                    if k != "n_frames":
                        base.setdefault(s, {}).setdefault(k, {}).update(v)
        return base
    mbl = fold(mbl, OUT / "_metrics_420_baselines_blind.json")
    mbl4 = fold(mbl4, OUT / "_metrics_444_baselines_blind.json")
    seqs = DERF + GOPRO

    data = {"tracks": {}}
    t = {"methods": M444 + X420, "seqs": {}, "metrics": {}}
    for s in seqs:
        t["seqs"][s] = {"n": m444[s]["n_frames"], "gt": gt_files(s)}
        t["metrics"][s] = {}
        for sig in SIGMAS:
            ent = dict(mbl4.get(s, {}).get(f"s{sig}", {}))
            ent.update(m444[s][f"s{sig}"])
            t["metrics"][s][str(sig)] = {
                m: pick_metrics(ent, m) for m in M444 if m != "gt"}
    data["tracks"]["444"] = t

    t = {"methods": M420 + X444, "seqs": {}, "metrics": {}}
    for s in seqs:
        t["seqs"][s] = {"n": m420[s]["n_frames"]}
        t["metrics"][s] = {}
        for sig in SIGMAS:
            ent = dict(mbl[s][f"s{sig}"])
            ent.update(m420[s][f"s{sig}"])   # galosh + noisy win on clash
            t["metrics"][s][str(sig)] = {
                m: pick_metrics(ent, m) for m in M420 if m != "gt_ref"}
    data["tracks"]["420"] = t

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    (OUT / "viewer.html").write_text(html, encoding="utf-8")
    print("saved:", OUT / "viewer.html")


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>GALOSH Set8 viewer</title>
<style>
 :root { color-scheme: dark; }
 * { box-sizing: border-box; }
 body { margin:0; background:#111; color:#ddd;
        font:13px/1.4 system-ui, sans-serif; }
 #bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center;
        padding:6px 10px; background:#1c1c1c; position:sticky; top:0;
        z-index:5; border-bottom:1px solid #333; }
 #bar label { color:#999; margin-right:2px; }
 select, button, input[type=range] { background:#2a2a2a; color:#ddd;
        border:1px solid #444; border-radius:4px; padding:2px 6px; }
 button { cursor:pointer; } button:hover { background:#3a3a3a; }
 #frameNum { min-width:70px; display:inline-block; }
 #grid { display:grid; gap:2px; height:calc(100vh - 46px); }
 .panel { position:relative; overflow:hidden; background:#000;
          min-height:0; }
 .phead { position:absolute; top:0; left:0; right:0; z-index:3;
          display:flex; gap:8px; align-items:center; padding:3px 6px;
          background:rgba(20,20,20,.75); }
 .pmet { color:#8fc7ff; font-size:12px; white-space:nowrap;
         overflow:hidden; text-overflow:ellipsis; }
 .pimg { position:absolute; inset:0; }
 .pimg img { position:absolute; left:0; top:0; transform-origin:0 0;
             image-rendering:pixelated; user-select:none;
             -webkit-user-drag:none; }
 .flipnote { position:absolute; bottom:6px; left:8px; z-index:3;
             color:#ffd479; font-size:12px;
             background:rgba(20,20,20,.6); padding:1px 6px;
             border-radius:4px; }
 #help { color:#777; }
</style></head><body>
<div id="bar">
 <label>track</label><select id="track"><option>444</option><option selected>420</option></select>
 <label>seq</label><select id="seq"></select>
 <label>&sigma;</label><select id="sigma"><option>10</option><option selected>20</option><option>30</option><option>40</option><option>50</option></select>
 <label>layout</label><select id="layout"><option value="1">1-up A/B</option><option value="2" selected>2-up</option><option value="4">2&times;2</option></select>
 <button id="prev">&#9664;</button>
 <button id="play">&#9654;</button>
 <button id="next">&#9654;&#9654;</button>
 <select id="fps"><option>5</option><option selected>10</option><option>25</option></select>
 <input type="range" id="frame" min="0" value="0" style="width:220px">
 <span id="frameNum"></span>
 <button id="rzoom">1:1</button>
 <span id="help">wheel=zoom drag=pan dblclick=fit &larr;&rarr;=frame Space=B(1-up)</span>
</div>
<div id="grid"></div>
<script>
const DATA = /*__DATA__*/;
const $ = id => document.getElementById(id);
const st = { track:"420", seq:"tractor", sigma:"20", frame:0, layout:2,
             meth:["galosh-cpu-fit","bm3d1b","noisy","gt_ref"], flipB:"gt_ref",
             flip:false, playing:false, timer:null,
             z:{s:1, tx:0, ty:0}, fitted:false };
function tk(){ return DATA.tracks[st.track]; }
function nFrames(){ return tk().seqs[st.seq].n; }
function pad(i){ return String(i).padStart(4,"0"); }
function split(m){          // "444:cpu-fit" -> [track, method]
  const i = m.indexOf(":");
  return i < 0 ? [st.track, m] : [m.slice(0, i), m.slice(i + 1)];
}
function url(m, f){
  const [tr, mm] = split(m);
  const s4 = "s" + st.sigma;
  if(tr === "444")
    return mm === "gt" ? DATA.tracks["444"].seqs[st.seq].gt[f]
                       : `png/${st.seq}/${s4}/${mm}/${pad(f)}.png`;
  return mm === "gt_ref" ? `png420/${st.seq}/gt_ref/${pad(f)}.png`
                         : `png420/${st.seq}/${s4}/${mm}/${pad(f)}.png`;
}
function metTxt(m){
  const [tr, mm] = split(m);
  if(mm === "gt" || mm === "gt_ref") return "reference (clean)";
  const e = DATA.tracks[tr].metrics[st.seq][st.sigma][mm];
  return e ? `PSNR ${e.psnr.toFixed(2)}  LPIPS ${e.lpips.toFixed(3)}`
           : "(no metrics)";
}
function panels(){ return st.layout === 1 ? 1 : st.layout; }
function buildGrid(){
  const g = $("grid");
  g.style.gridTemplateColumns = st.layout === 4 ? "1fr 1fr" : (st.layout === 2 ? "1fr 1fr" : "1fr");
  g.style.gridTemplateRows = st.layout === 4 ? "1fr 1fr" : "1fr";
  g.innerHTML = "";
  for(let i = 0; i < panels(); i++){
    const p = document.createElement("div"); p.className = "panel";
    const hd = document.createElement("div"); hd.className = "phead";
    const sel = document.createElement("select");
    tk().methods.forEach(m => {
      const o = document.createElement("option"); o.textContent = m;
      sel.appendChild(o);
    });
    sel.value = st.meth[i];
    sel.onchange = () => { st.meth[i] = sel.value; draw(); };
    hd.appendChild(sel);
    if(st.layout === 1){
      const selB = document.createElement("select"); selB.id = "selB";
      tk().methods.forEach(m => {
        const o = document.createElement("option"); o.textContent = m;
        selB.appendChild(o);
      });
      selB.value = st.flipB;
      selB.onchange = () => { st.flipB = selB.value; draw(); };
      const lab = document.createElement("span");
      lab.textContent = "⇄"; lab.style.color = "#777";
      hd.appendChild(lab); hd.appendChild(selB);
    }
    const met = document.createElement("span"); met.className = "pmet";
    hd.appendChild(met);
    const wrap = document.createElement("div"); wrap.className = "pimg";
    const img = document.createElement("img");
    wrap.appendChild(img); p.appendChild(hd); p.appendChild(wrap);
    if(st.layout === 1){
      const fn = document.createElement("div"); fn.className = "flipnote";
      p.appendChild(fn);
    }
    attachNav(wrap); g.appendChild(p);
  }
  st.fitted = false;
}
function methodShown(i){
  return (st.layout === 1 && st.flip) ? st.flipB : st.meth[i];
}
function draw(){
  document.querySelectorAll("#grid .panel").forEach((p, i) => {
    const m = methodShown(i);
    const img = p.querySelector("img");
    img.src = url(m, st.frame);
    if(!st.fitted) img.onload = () => fit(img);
    p.querySelector(".pmet").textContent = metTxt(m);
    const fn = p.querySelector(".flipnote");
    if(fn) fn.textContent = (st.flip ? "B: " + st.flipB : "A: " + st.meth[i])
                            + "  (hold Space for B)";
    applyZ(img);
  });
  $("frame").max = nFrames() - 1;
  $("frame").value = st.frame;
  $("frameNum").textContent = `${st.frame + 1}/${nFrames()}`;
  preload();
}
function preload(){
  [st.frame + 1, st.frame - 1].forEach(f => {
    if(f < 0 || f >= nFrames()) return;
    for(let i = 0; i < panels(); i++) (new Image()).src = url(methodShown(i), f);
    if(st.layout === 1) (new Image()).src = url(st.flipB, f);
  });
}
function fit(img){
  const w = img.parentElement.clientWidth, h = img.parentElement.clientHeight;
  if(!img.naturalWidth) return;
  const s = Math.min(w / img.naturalWidth, h / img.naturalHeight);
  st.z = { s:s, tx:(w - img.naturalWidth * s) / 2,
           ty:(h - img.naturalHeight * s) / 2 };
  st.fitted = true; applyAll();
}
function applyZ(img){
  img.style.transform = `translate(${st.z.tx}px,${st.z.ty}px) scale(${st.z.s})`;
}
function applyAll(){ document.querySelectorAll("#grid img").forEach(applyZ); }
function attachNav(el){
  el.addEventListener("wheel", e => {
    e.preventDefault();
    const r = el.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = e.deltaY < 0 ? 1.25 : 0.8, s2 = st.z.s * f;
    st.z.tx = mx - (mx - st.z.tx) * f;
    st.z.ty = my - (my - st.z.ty) * f;
    st.z.s = s2; applyAll();
  }, { passive:false });
  let drag = null;
  el.addEventListener("pointerdown", e => {
    drag = { x:e.clientX, y:e.clientY }; el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", e => {
    if(!drag) return;
    st.z.tx += e.clientX - drag.x; st.z.ty += e.clientY - drag.y;
    drag = { x:e.clientX, y:e.clientY }; applyAll();
  });
  el.addEventListener("pointerup", () => drag = null);
  el.addEventListener("dblclick", () => fit(el.querySelector("img")));
}
function step(d){ st.frame = (st.frame + d + nFrames()) % nFrames(); draw(); }
function setPlay(on){
  st.playing = on;
  $("play").innerHTML = on ? "&#10074;&#10074;" : "&#9654;";
  clearInterval(st.timer);
  if(on) st.timer = setInterval(() => step(1), 1000 / (+$("fps").value));
}
function rebuild(){
  const ms = tk().methods;
  st.meth = st.meth.map(m => ms.includes(m) ? m : ms[Math.min(2, ms.length-1)]);
  if(!ms.includes(st.flipB)) st.flipB = ms[0];
  st.frame = Math.min(st.frame, nFrames() - 1);
  buildGrid(); draw();
}
$("track").onchange = e => {
  st.track = e.target.value;
  st.meth = st.track === "444" ? ["galosh-cpu-fit","gt","noisy","bm3d1b"]
                               : ["galosh-cpu-fit","bm3d1b","noisy","gt_ref"];
  st.flipB = st.track === "444" ? "gt" : "gt_ref";
  rebuild();
};
$("seq").onchange = e => { st.seq = e.target.value; st.fitted = false; rebuild(); };
$("sigma").onchange = e => { st.sigma = e.target.value; draw(); };
$("layout").onchange = e => { st.layout = +e.target.value; buildGrid(); draw(); };
$("frame").oninput = e => { st.frame = +e.target.value; draw(); };
$("prev").onclick = () => step(-1);
$("next").onclick = () => step(1);
$("play").onclick = () => setPlay(!st.playing);
$("fps").onchange = () => { if(st.playing) setPlay(true); };
$("rzoom").onclick = () => {
  st.z = { s:1, tx:0, ty:0 }; applyAll();
};
document.addEventListener("keydown", e => {
  if(e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if(e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
  if(e.key === "ArrowRight") { step(1); e.preventDefault(); }
  if(e.key === " " && st.layout === 1 && !st.flip)
  { st.flip = true; draw(); e.preventDefault(); }
});
document.addEventListener("keyup", e => {
  if(e.key === " " && st.flip) { st.flip = false; draw(); }
});
Object.keys(DATA.tracks["420"].seqs).forEach(s => {
  const o = document.createElement("option"); o.textContent = s;
  $("seq").appendChild(o);
});
$("seq").value = st.seq;
buildGrid(); draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
