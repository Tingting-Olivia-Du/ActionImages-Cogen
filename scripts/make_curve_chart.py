"""Metric-versus-training-step curves for every modality x every conditioning mode.

    python scripts/make_curve_chart.py --out reports/curves.html

FORM: small multiples, one panel per modality, lines within a panel = conditioning mode.
Four modalities is NOT four series on one axis -- absrel (lower better, 0.05-1.3), macro-mIoU
(higher better, 0.32-0.59), cosine (higher better, 0.89-0.95) and metres (lower better,
0.018-0.148) share no scale and no direction. One shared y would be a dual-axis chart wearing a
disguise; each panel carries its own.

COLOUR is by CONDITIONING MODE and is identical across panels -- the entity is the mode, so a
reader tracking `fifi` follows one hue everywhere. Slots 1/4/5/6 of the reference palette
(blue/yellow/magenta/green): the default first-four FAILS the all-pairs gate small multiples
require (orange-yellow normal-vision dE 13.7 < 15 floor); this subset passes every check in both
modes -- light worst CVD dE 13.0, normal-vision 19.6; dark 6.9 / 19.3.

Two validator WARNs drive two design obligations, and both are honoured below:
  * light-mode contrast for yellow (2.11:1) and magenta (2.62:1) is under 3:1 -> direct labels
    on every line AND a table view;
  * dark-mode CVD lands in the 6-8 floor band -> secondary encoding, so each mode also gets its
    own marker shape. Hue is never the only thing telling two lines apart.
"""
import argparse, glob, json, os, collections

MODES = ["iiii", "fiii", "fifi", "policy"]
SHAPE = {"iiii": "circle", "fiii": "square", "fifi": "triangle", "policy": "diamond"}
PANELS = [
    ("action",       "pos_err_m",  "6-DoF 位置误差 (m)",  "lower",
     "把生成的动作热图三角化回位姿，与 GT 轨迹比。取代了会被任何 RGB 图饱和的 r_peak。"),
    ("depth",        "absrel",     "AbsRel",              "lower",
     "解码深度对 GT 的相对误差，只在有效像素上算。"),
    ("segmentation", "macro_miou", "macro mIoU",          "higher",
     "按角色的 IoU，丢掉 GT 中不存在的类再取平均。"),
    ("normal",       "mean_cos",   "平均余弦",             "higher",
     "解码法线与从 depth+intrinsics 重算的 GT 法线的逐像素余弦。"),
]

def collect(root="reports/modality_mode_grid"):
    series = collections.defaultdict(dict)     # (modality, mode) -> {step: value}
    for f in sorted(glob.glob(os.path.join(root, "arm6_*", "metrics.json"))):
        tag = os.path.basename(os.path.dirname(f))
        if "_ep_" in tag:
            continue                            # per-episode probes are a different question
        try:
            step = int(tag.split("_")[1])
        except (IndexError, ValueError):
            continue
        for key, rec in json.load(open(f))["results"].items():
            _tpl, mode = key.split("|")
            tgt = [e for e in rec["segments"] if e["modality"] != "video" and e["view"] == 0]
            if not tgt:
                continue
            e = tgt[0]
            v, metric = e.get("value"), e.get("metric")
            if v is None:
                continue
            # step 1500 scored `action` with r_peak, a different quantity on a different scale.
            # Plotting it beside pos_err_m would draw a line between two unrelated numbers.
            if e["modality"] == "action" and metric != "pos_err_m":
                continue
            series[(e["modality"], mode)][step] = v
    return series

TPL_ORDER = ["video+action", "video+segmentation", "video+depth", "video+normal", "video"]
TPL_SHAPE = {"video+action": "circle", "video+segmentation": "square",
             "video+depth": "triangle", "video+normal": "diamond", "video": "circle"}


def collect_heldout(path="reports/heldout_val.json"):
    """Per-template held-out diffusion loss vs step, from heldout_val.py.

    This is a DIFFERENT quantity from the four panels above: it is the training objective
    (flow-matching MSE) evaluated on variation1/2, not a decoded-output metric. It answers
    "is the model overfitting", which no decoded metric can -- absrel/mIoU/cosine are all
    measured on one training episode. Kept in its own section with its own legend because
    its hue keys the TEMPLATE, not the conditioning mode; sharing the small-multiple legend
    would make one colour mean two things on one page.
    """
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    res = d.get("results", {})
    if not res:
        return None
    per = collections.defaultdict(dict)
    overall = {}
    for step_s, rec in res.items():
        step = int(step_s)
        overall[step] = rec["mean"]
        for tpl, v in rec.get("per_template", {}).items():
            per[tpl][step] = v["mean"]
    steps = sorted(overall)
    series = []
    for tpl in TPL_ORDER:
        if tpl in per and per[tpl]:
            series.append({"tpl": tpl, "shape": TPL_SHAPE[tpl],
                           "points": [[st, per[tpl][st]] for st in sorted(per[tpl])]})
    return {"steps": steps, "series": series,
            "overall": [[st, overall[st]] for st in steps],
            "n": d.get("n"), "variations": d.get("variations"),
            "sem": {str(st): res[str(st)]["sem"] for st in steps}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/modality_mode_grid")
    ap.add_argument("--out", default="reports/curves.html")
    ap.add_argument("--heldout", default="reports/heldout_val.json")
    a = ap.parse_args()

    series = collect(a.root)
    steps = sorted({s for d in series.values() for s in d})
    data = {"steps": steps, "panels": []}
    for mod, metric, ylabel, better, blurb in PANELS:
        rows = []
        for mode in MODES:
            pts = series.get((mod, mode))
            if not pts:
                continue
            rows.append({"mode": mode, "shape": SHAPE[mode],
                         "points": [[s, pts[s]] for s in sorted(pts)]})
        if rows:
            data["panels"].append({"modality": mod, "metric": metric, "ylabel": ylabel,
                                   "better": better, "blurb": blurb, "series": rows})
    data["heldout"] = collect_heldout(a.heldout)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open(a.out, "w") as f:
        f.write(html)
    n = sum(len(s["points"]) for p in data["panels"] for s in p["series"])
    ho = data["heldout"]
    print(f"wrote {a.out}  ({len(data['panels'])} panels, {n} points, steps {steps})"
          + (f"  + held-out {len(ho['steps'])} steps x {len(ho['series'])} templates" if ho else "  (no held-out yet)"))

TEMPLATE = r"""<title>arm6 多模态生成能力 vs 训练步数</title>
<style>
.viz-root{color-scheme:light;
 --surface-1:#fcfcfb;--surface-2:#f3f3f1;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#84837c;
 --grid:#e4e4e0;--axis:#c9c8c2;
 --s-iiii:#2a78d6;--s-fiii:#eda100;--s-fifi:#e87ba4;--s-policy:#008300;--h-action:#eb6834;--h-segmentation:#1baf7a;--h-depth:#4a3aa7;--h-normal:#e34948;--ref:#84837c;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--surface-2:#232322;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8e8d84;
 --grid:#2e2e2c;--axis:#454440;
 --s-iiii:#3987e5;--s-fiii:#c98500;--s-fifi:#d55181;--s-policy:#008300;--h-action:#d95926;--h-segmentation:#199e70;--h-depth:#9085e9;--h-normal:#e66767;--ref:#8e8d84;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--surface-2:#232322;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8e8d84;
 --grid:#2e2e2c;--axis:#454440;
 --s-iiii:#3987e5;--s-fiii:#c98500;--s-fifi:#d55181;--s-policy:#008300;--h-action:#d95926;--h-segmentation:#199e70;--h-depth:#9085e9;--h-normal:#e66767;--ref:#8e8d84;}
body{margin:0;background:var(--surface-1);color:var(--text-primary);
 font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.6;padding:0 20px 80px}
.wrap{max-width:1080px;margin:0 auto}
header{padding:48px 0 20px;border-bottom:1px solid var(--grid);margin-bottom:28px}
h1{font-size:clamp(24px,3.4vw,34px);line-height:1.15;margin:0 0 10px;letter-spacing:-.015em;text-wrap:balance}
.sub{color:var(--text-secondary);max-width:66ch;margin:0;font-size:16px}
.meta{display:flex;flex-wrap:wrap;gap:6px 22px;margin-top:16px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
 font-size:12px;color:var(--text-muted)}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;margin:22px 0 6px;align-items:center}
.lg{display:inline-flex;align-items:center;gap:7px;font-size:13.5px;color:var(--text-secondary)}
.lg svg{display:block}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:26px;margin-top:10px}
figure{margin:0;background:var(--surface-2);border:1px solid var(--grid);border-radius:6px;padding:16px 14px 10px;overflow-x:auto}
figcaption{font-size:12.5px;color:var(--text-muted);margin-top:8px;max-width:60ch}
.ptitle{font-size:15px;font-weight:650;margin:0 0 2px}
.punit{font-size:12px;color:var(--text-muted);font-family:ui-monospace,Menlo,monospace;margin:0 0 8px}
svg.chart{display:block;max-width:100%;height:auto;overflow:visible}
.tick{font-size:10.5px;fill:var(--text-muted);font-family:ui-monospace,Menlo,monospace}
.dlabel{font-size:11px;font-weight:650;font-family:ui-monospace,Menlo,monospace}
.hit{cursor:crosshair}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--surface-1);
 border:1px solid var(--axis);border-radius:5px;padding:7px 10px;font-size:12.5px;
 font-family:ui-monospace,Menlo,monospace;box-shadow:0 3px 12px rgba(0,0,0,.16);z-index:9;white-space:pre}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums;min-width:560px}
th,td{padding:7px 11px;border-bottom:1px solid var(--grid);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);font-weight:500;white-space:nowrap}
.tablewrap{overflow-x:auto;border:1px solid var(--grid);border-radius:6px;background:var(--surface-2);margin-top:14px}
details{margin-top:30px}summary{cursor:pointer;font-weight:600;font-size:15px;padding:6px 0}
h2{font-size:20px;margin:36px 0 6px;letter-spacing:-.01em}
.note{background:var(--surface-2);border:1px solid var(--grid);border-left:3px solid var(--s-iiii);
 border-radius:5px;padding:13px 16px;margin:16px 0;font-size:14.5px}
code{font-family:ui-monospace,Menlo,monospace;font-size:.88em;background:var(--surface-2);padding:1px 5px;border-radius:3px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="viz-root"><div class="wrap">
<header>
<h1>arm6：四种模态 × 四种条件化模式，随训练步数的变化</h1>
<p class="sub">同一个 checkpoint、同一条指令、同一个 seed，只改 prompt tag 和条件化方式。
每个面板一个模态，因为四个指标的量纲和方向都不同——共用一根 y 轴就是伪装过的双轴图。
颜色跟随<strong>模式</strong>，四个面板里同一模式永远同色。</p>
<div class="meta">
<span>arm6 · video+action@0.4 depth@0.2 seg@0.2 normal@0.2</span>
<span>scene_roles · 512px · frame_interval 3</span>
<span>open_drawer/variation0/episode0 · view 0 · seed 42</span>
</div>
</header>
<div class="legend" id="legend"></div>
<div class="grid2" id="panels"></div>
<div id="tip"></div>
<section id="hosec" hidden>
<h2>held-out 验证损失（variation1 + variation2）</h2>
<p style="color:var(--text-secondary);max-width:68ch;margin:0 0 4px">
上面四个面板量的是<strong>解码输出的质量</strong>，而且只在一个训练 episode 上。这一节量的是
<strong>训练目标本身</strong>（flow-matching MSE）在训练从未见过的 variation 上的值——它是唯一能回答
「有没有过拟合」的量。此节<strong>颜色键的是模板</strong>，与上面的模式配色不相交。</p>
<div class="legend" id="holegend"></div>
<figure id="hofig"></figure>
</section>
<h2>怎么读</h2>
<div class="note">
<p style="margin:0 0 8px"><strong>模式是训练时真实抽到的条件化方式</strong>，不是事后想出来的消融：</p>
<p style="margin:0;font-family:ui-monospace,Menlo,monospace;font-size:13px">
含 &lt;action&gt; 的模板　iiii 0.81 ｜ fiii 0.045 ｜ fifi 0.045 ｜ policy 0.10<br>
感知模板 (M2)　　　　iiii 0.90 ｜ fiii 0.05 ｜ fifi 0.05
</p>
</div>
<ul style="font-size:14.5px;color:var(--text-secondary);max-width:70ch">
<li><code>iiii</code> 每段只给第一帧——闭环 rollout 用的就是这个</li>
<li><code>fiii</code> 第 0 段整段给出，其余只给首帧</li>
<li><code>fifi</code> 锚点（video）两个视角都整段给出</li>
<li><code>policy</code> 视觉段压成单帧、动作段保持全长（仅 rlbench，10%）</li>
</ul>
<details open><summary>数据表（黄/品红在亮色下对比度低于 3:1，此表是必需的相应措施）</summary>
<div class="tablewrap"><table id="tbl"></table></div></details>
</div></div>
<script>
const D=__DATA__;
const MODES=["iiii","fiii","fifi","policy"];
const CSSVAR={iiii:"--s-iiii",fiii:"--s-fiii",fifi:"--s-fifi",policy:"--s-policy"};
const col=m=>getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(CSSVAR[m]).trim();
function marker(shape,x,y,c,r=4.5){
 if(shape==="circle")return `<circle cx="${x}" cy="${y}" r="${r}" fill="${c}" stroke="var(--surface-2)" stroke-width="2"/>`;
 if(shape==="square")return `<rect x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}" fill="${c}" stroke="var(--surface-2)" stroke-width="2"/>`;
 if(shape==="triangle")return `<polygon points="${x},${y-r-1} ${x+r+1},${y+r} ${x-r-1},${y+r}" fill="${c}" stroke="var(--surface-2)" stroke-width="2"/>`;
 return `<polygon points="${x},${y-r-1} ${x+r+1},${y} ${x},${y+r+1} ${x-r-1},${y}" fill="${c}" stroke="var(--surface-2)" stroke-width="2"/>`;}
const usedModes=[...new Set(D.panels.flatMap(p=>p.series.map(s=>s.mode)))].sort((a,b)=>MODES.indexOf(a)-MODES.indexOf(b));
document.getElementById("legend").innerHTML=usedModes.map(m=>{
 const c=col(m),sh=D.panels.flatMap(p=>p.series).find(s=>s.mode===m).shape;
 return `<span class="lg"><svg width="26" height="14"><line x1="1" y1="7" x2="25" y2="7" stroke="${c}" stroke-width="2"/>${marker(sh,13,7,c,4)}</svg>${m}</span>`;}).join("");
const tip=document.getElementById("tip");
function draw(p){
 const W=470,H=250,ml=58,mr=64,mt=12,mb=34;
 const xs=D.steps,x0=Math.min(...xs),x1=Math.max(...xs);
 const vals=p.series.flatMap(s=>s.points.map(q=>q[1]));
 let lo=Math.min(...vals),hi=Math.max(...vals);const pad=(hi-lo)*0.14||0.01;lo-=pad;hi+=pad;
 if(p.better==="higher"&&lo<0)lo=0;
 const X=v=>ml+(v-x0)/((x1-x0)||1)*(W-ml-mr), Y=v=>H-mb-(v-lo)/((hi-lo)||1)*(H-mt-mb);
 let g="";
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=Y(v);
  g+=`<line x1="${ml}" y1="${y}" x2="${W-mr}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`;
  g+=`<text class="tick" x="${ml-8}" y="${y+3.5}" text-anchor="end">${v.toFixed(v<0.2?3:2)}</text>`;}
 xs.forEach(s=>{g+=`<text class="tick" x="${X(s)}" y="${H-mb+16}" text-anchor="middle">${s}</text>`;});
 g+=`<line x1="${ml}" y1="${H-mb}" x2="${W-mr}" y2="${H-mb}" stroke="var(--axis)" stroke-width="1"/>`;
 p.series.forEach(s=>{const c=col(s.mode);
  const pts=s.points.map(q=>[X(q[0]),Y(q[1])]);
  g+=`<polyline points="${pts.map(q=>q.join(",")).join(" ")}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"/>`;
  pts.forEach((q,i)=>{g+=marker(s.shape,q[0],q[1],c);});
  const last=pts[pts.length-1];
  g+=`<text class="dlabel" x="${last[0]+9}" y="${last[1]+4}" fill="${c}">${s.mode}</text>`;});
 p.series.forEach(s=>s.points.forEach(q=>{
  g+=`<circle class="hit" cx="${X(q[0])}" cy="${Y(q[1])}" r="11" fill="transparent"
      data-t="${s.mode} · step ${q[0]}\n${p.ylabel} ${q[1].toFixed(4)}"/>`;}));
 return `<figure><p class="ptitle">&lt;${p.modality==="segmentation"?"scene-seg":p.modality}&gt;</p>
 <p class="punit">${p.ylabel} · ${p.better==="lower"?"越低越好":"越高越好"}</p>
 <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${p.modality} ${p.ylabel} 随训练步数变化，按条件化模式分线">${g}</svg>
 <figcaption>${p.blurb}</figcaption></figure>`;}
document.getElementById("panels").innerHTML=D.panels.map(draw).join("");
document.querySelectorAll(".hit").forEach(el=>{
 el.addEventListener("mouseenter",e=>{tip.textContent=el.dataset.t;tip.style.opacity=1;});
 el.addEventListener("mousemove",e=>{tip.style.left=(e.clientX+14)+"px";tip.style.top=(e.clientY-10)+"px";});
 el.addEventListener("mouseleave",()=>tip.style.opacity=0);});
(function(){
 const H=D.heldout; if(!H||!H.series.length) return;
 document.getElementById("hosec").hidden=false;
 const KEY=t=>"--h-"+t.replace("video+","");
 const hcol=t=>getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(KEY(t)).trim();
 // `video` 是 10% action-dropout 的副产物(n=9)，不是设计出来的模板：入表不入图。
 const plot=H.series.filter(s=>s.tpl!=="video");
 document.getElementById("holegend").innerHTML=plot.map(s=>{
  const c=hcol(s.tpl);
  return `<span class="lg"><svg width="26" height="14"><line x1="1" y1="7" x2="25" y2="7" stroke="${c}" stroke-width="2"/>${marker(s.shape,13,7,c,4)}</svg>${s.tpl}</span>`;})
  .join("")+`<span class="lg"><svg width="26" height="14"><line x1="1" y1="7" x2="25" y2="7" stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 3"/></svg>训练 loss（同区间均值 0.0641）</span>`;
 const W=980,Hh=330,ml=64,mr=150,mt=14,mb=40;
 const xs=H.steps,x0=Math.min(...xs),x1=Math.max(...xs);
 const vals=plot.flatMap(s=>s.points.map(q=>q[1])).concat([0.0641]);
 let lo=Math.min(...vals),hi=Math.max(...vals);const pad=(hi-lo)*0.16;lo-=pad;hi+=pad;
 const X=v=>ml+(v-x0)/((x1-x0)||1)*(W-ml-mr), Y=v=>Hh-mb-(v-lo)/((hi-lo)||1)*(Hh-mt-mb);
 let g="";
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=Y(v);
  g+=`<line x1="${ml}" y1="${y}" x2="${W-mr}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`;
  g+=`<text class="tick" x="${ml-8}" y="${y+3.5}" text-anchor="end">${v.toFixed(3)}</text>`;}
 xs.forEach(st=>{g+=`<text class="tick" x="${X(st)}" y="${Hh-mb+16}" text-anchor="middle">${st}</text>`;});
 g+=`<line x1="${ml}" y1="${Hh-mb}" x2="${W-mr}" y2="${Hh-mb}" stroke="var(--axis)" stroke-width="1"/>`;
 g+=`<line x1="${ml}" y1="${Y(0.0641)}" x2="${W-mr}" y2="${Y(0.0641)}" stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 3"/>`;
 plot.forEach(s=>{const c=hcol(s.tpl),pts=s.points.map(q=>[X(q[0]),Y(q[1])]);
  g+=`<polyline points="${pts.map(q=>q.join(",")).join(" ")}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"/>`;
  pts.forEach(q=>{g+=marker(s.shape,q[0],q[1],c);});
  const last=pts[pts.length-1];
  g+=`<text class="dlabel" x="${last[0]+10}" y="${last[1]+4}" fill="${c}">${s.tpl}</text>`;});
 plot.forEach(s=>s.points.forEach(q=>{
  g+=`<circle class="hit" cx="${X(q[0])}" cy="${Y(q[1])}" r="11" fill="transparent"
      data-t="${s.tpl} · step ${q[0]}\nheld-out loss ${q[1].toFixed(5)}"/>`;}));
 document.getElementById("hofig").innerHTML=
  `<p class="ptitle">held-out flow-matching loss</p><p class="punit">越低越好 · n=${H.n} 个冻结样本 · variations ${H.variations}</p>
   <svg class="chart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="held-out 损失随训练步数变化，按模板分线">${g}</svg>
   <figcaption>样本集跨 checkpoint 冻结（同样的 episode/窗口/模板/时间步/噪声），所以点与点之间的差是模型的差。
   虚线是同区间的训练 loss 均值——两者重合即没有泛化间隙。<code>video</code>（10% action-dropout 的副产物，n=9）只在下表。</figcaption>`;
 document.querySelectorAll("#hofig .hit").forEach(el=>{
  el.addEventListener("mouseenter",()=>{tip.textContent=el.dataset.t;tip.style.opacity=1;});
  el.addEventListener("mousemove",e=>{tip.style.left=(e.clientX+14)+"px";tip.style.top=(e.clientY-10)+"px";});
  el.addEventListener("mouseleave",()=>tip.style.opacity=0);});
})();
let th=`<thead><tr><th>模态</th><th>指标</th><th>模式</th>${D.steps.map(s=>`<th>${s}</th>`).join("")}</tr></thead><tbody>`;
D.panels.forEach(p=>p.series.forEach(s=>{const m=Object.fromEntries(s.points);
 th+=`<tr><td>${p.modality}</td><td>${p.ylabel}</td><td>${s.mode}</td>`+
  D.steps.map(st=>`<td>${m[st]!==undefined?m[st].toFixed(4):"—"}</td>`).join("")+`</tr>`;}));
if(D.heldout){const H=D.heldout;
 H.series.forEach(s=>{const m=Object.fromEntries(s.points);
  th+=`<tr><td>held-out</td><td>flow-matching loss</td><td>${s.tpl}</td>`+
   D.steps.map(st=>`<td>${m[st]!==undefined?m[st].toFixed(4):"—"}</td>`).join("")+`</tr>`;});
 const o=Object.fromEntries(H.overall);
 th+=`<tr><td>held-out</td><td>flow-matching loss</td><td><strong>全部</strong></td>`+
  D.steps.map(st=>`<td>${o[st]!==undefined?"<strong>"+o[st].toFixed(4)+"</strong>":"—"}</td>`).join("")+`</tr>`;}
document.getElementById("tbl").innerHTML=th+"</tbody>";
</script>
"""

if __name__ == "__main__":
    main()
