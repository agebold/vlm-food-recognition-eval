"""
Publication-quality D3.js charts.

Rules:
  - ZERO low-opacity data elements. Transparency = gray = bad.
    If something needs to be "muted", use a real muted color (#CBD5E1 etc).
  - X/Y axes start at round numbers. Percentages start at 0%.
  - Data fill >= 90% opacity. Lines >= 3px for main series.
  - Labels on the data, not in a separate legend box.

Run: python3 gen_d3_plots.py
"""

import json, pathlib, sys, collections, statistics

sys.path.insert(0, ".")
from metric import sim

THRESH = 0.4
PLOT_DIR = pathlib.Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

MODELS = {
    "Kimi K2.5":       "results/bedrock_kimi25_overhead.json",
    "Claude Opus 4.8": "results/claude_opus_overhead_full.json",
    "Llama4 Maverick": "results/bedrock_llama4_maverick_full.json",
    "Llama4 Scout":    "results/bedrock_llama4_scout_full.json",
    "Nova Pro":        "results/bedrock_nova_pro_full.json",
    "Nova Lite":       "results/bedrock_nova_lite_full.json",
}

# All colors pass WCAG AA (4.5:1) on white. No pastels.
COLOR = {
    "Kimi K2.5":       "#1D4ED8",
    "Claude Opus 4.8": "#7C3AED",
    "Llama4 Maverick": "#047857",
    "Llama4 Scout":    "#B45309",
    "Nova Pro":        "#B91C1C",
    "Nova Lite":       "#0369A1",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:#F0F4F8;
  padding:48px 44px;
  color:#111827;
}
svg{
  display:block;
  background:#fff;
  border-radius:14px;
  box-shadow:0 2px 6px rgba(0,0,0,.07),0 12px 32px rgba(0,0,0,.07);
}
text{font-family:'Inter',system-ui,sans-serif}
.ttl{font:800 18px/1.3 'Inter',sans-serif;fill:#111827}
.sub{font:500 13px/1.5 'Inter',sans-serif;fill:#1F2937}
.axl{font:600 11px/1   'Inter',sans-serif;fill:#374151}
"""

def page(title, js):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>{CSS}</style>
</head>
<body><script>\n{js}\n</script></body>
</html>"""

# ── data loading ──────────────────────────────────────────────────────────────
def dish_stats(samples):
    out = []
    for s in samples:
        gt, pred = s["ground_truth"], s["predicted"]
        correct = [p for p in pred if max((sim(p,g) for g in gt), default=0) >= THRESH]
        wrong   = [p for p in pred if max((sim(p,g) for g in gt), default=0) < THRESH]
        missed  = [g for g in gt  if max((sim(g,p) for p in pred), default=0) < THRESH]
        out.append({
            "n_gt":len(gt), "n_pred":len(pred), "n_correct":len(correct),
            "pred_acc": len(correct)/len(pred) if pred else 0.0,
            "missed":missed, "halluc":wrong,
        })
    return out

raw = {}
for name, path in MODELS.items():
    samps = json.loads(pathlib.Path(path).read_text())["samples"]
    raw[name] = {"samples": samps, "stats": dish_stats(samps)}

names = list(MODELS.keys())


# ════════════════════════════════════════════════════════════════════════
# FIG 1 — Horizontal bar chart, prediction accuracy
#   x from 0 → 90%.  Solid bars, full color, percentage at bar end.
# ════════════════════════════════════════════════════════════════════════
fig1 = sorted([{
    "model": n, "color": COLOR[n],
    "acc": round(100*statistics.mean(d["pred_acc"] for d in raw[n]["stats"]), 2),
} for n in names], key=lambda r: r["acc"])   # lowest at top, highest at bottom

js1 = f"""
const data = {json.dumps(fig1)};

const W=560, rowH=64, H=data.length*rowH;
const m={{t:96,r:80,b:56,l:164}};

const svg = d3.select("body").append("svg")
  .attr("width",W+m.l+m.r).attr("height",H+m.t+m.b);

svg.append("text").attr("class","ttl").attr("x",m.l).attr("y",36)
  .text("When a model names an ingredient, how often is it correct?");
svg.append("text").attr("class","sub").attr("x",m.l).attr("y",60)
  .text("Prediction accuracy = correct guesses ÷ total guesses · averaged across 507 dishes · sorted ascending");

const g = svg.append("g").attr("transform",`translate(${{m.l}},${{m.t}})`);

const x = d3.scaleLinear().domain([0,90]).range([0,W]);
const y = d3.scaleBand().domain(data.map(d=>d.model)).range([0,H]).padding(0.35);

// Light vertical grid only (20%, 40%, 60%, 80%)
[20,40,60,80].forEach(v => {{
  g.append("line")
   .attr("x1",x(v)).attr("x2",x(v)).attr("y1",0).attr("y2",H)
   .attr("stroke","#E5E7EB").attr("stroke-width",1);
}});

// Bars — full color, no opacity tricks
g.selectAll(".bar").data(data).join("rect")
  .attr("x",0).attr("y",d=>y(d.model))
  .attr("width",d=>x(d.acc))
  .attr("height",y.bandwidth())
  .attr("fill",d=>d.color)
  .attr("rx",4);

// Percentage label at bar end — bold, matching color
g.selectAll(".pct").data(data).join("text")
  .attr("x",d=>x(d.acc)+10)
  .attr("y",d=>y(d.model)+y.bandwidth()/2)
  .attr("dy","0.35em")
  .attr("font-size",14).attr("font-weight","800")
  .attr("fill",d=>d.color)
  .text(d=>d.acc.toFixed(1)+"%");

// Y axis — model names, colored, no tick marks, no axis line
const yAx = g.append("g").call(d3.axisLeft(y).tickSize(0).tickPadding(12));
yAx.select(".domain").remove();
yAx.selectAll("text")
  .attr("font-size",13).attr("font-weight","700")
  .each(function(d) {{
    const row = data.find(r => r.model === d);
    d3.select(this).attr("fill", row ? row.color : "#374151");
  }});

// X axis — clean, 0 to 90%
const xAx = g.append("g").attr("transform",`translate(0,${{H}})`).call(
  d3.axisBottom(x).tickValues([0,20,40,60,80]).tickFormat(d=>d+"%").tickSize(5));
xAx.select(".domain").attr("stroke","#9CA3AF");
xAx.selectAll("text").attr("font-size",11).attr("fill","#111827");
xAx.selectAll(".tick line").attr("stroke","#9CA3AF");

g.append("text").attr("class","axl")
  .attr("x",W/2).attr("y",H+46).attr("text-anchor","middle")
  .text("Prediction accuracy (%)");

// Annotation: winner
const best=data[data.length-1];
g.append("text")
  .attr("x",x(best.acc)+10)
  .attr("y",y(best.model)+y.bandwidth()/2-22)
  .attr("font-size",11).attr("font-weight","600")
  .attr("fill",best.color).attr("font-style","italic")
  .text("highest accuracy");
"""

pathlib.Path("plots/fig01_accuracy.html").write_text(page("Fig 1 – Accuracy", js1))
print("✓ fig01_accuracy.html")


# ════════════════════════════════════════════════════════════════════════
# FIG 2 — 2×3 small multiples, correct/dish histograms
# ════════════════════════════════════════════════════════════════════════
fig2 = [{
    "model":n, "color":COLOR[n],
    "vals":[d["n_correct"] for d in raw[n]["stats"]],
    "med": round(statistics.median(d["n_correct"] for d in raw[n]["stats"]),1),
    "mean":round(statistics.mean  (d["n_correct"] for d in raw[n]["stats"]),2),
} for n in names]

js2 = f"""
const panels = {json.dumps(fig2)};
const COLS=3, CW=260, CH=200, GAP=28;
const pm={{t:44,r:20,b:46,l:48}};
const TW=COLS*(CW+pm.l+pm.r)+(COLS-1)*GAP;
const ROWS=Math.ceil(panels.length/COLS);
const TH=84+ROWS*(CH+pm.t+pm.b)+(ROWS-1)*GAP;

const svg=d3.select("body").append("svg").attr("width",TW).attr("height",TH);

svg.append("text").attr("class","ttl").attr("x",32).attr("y",36)
  .text("Correctly identified ingredients per dish");
svg.append("text").attr("class","sub").attr("x",32).attr("y",58)
  .text("Distribution across 507 dishes · black dashed = median · gray dashed = mean");

panels.forEach((m,i)=>{{
  const col=i%COLS, row=Math.floor(i/COLS);
  const ox=col*(CW+pm.l+pm.r+GAP)+pm.l;
  const oy=76+row*(CH+pm.t+pm.b+GAP)+pm.t;
  const g=svg.append("g").attr("transform",`translate(${{ox}},${{oy}})`);

  const maxV=d3.max(m.vals)||1;
  const bins=d3.bin().domain([0,maxV+1]).thresholds(d3.range(0,maxV+2))(m.vals);
  const x=d3.scaleLinear().domain([0,maxV+0.5]).range([0,CW]);
  const y=d3.scaleLinear().domain([0,d3.max(bins,b=>b.length)*1.1]).range([CH,0]);

  // Very light grid
  y.ticks(4).forEach(v=>{{
    g.append("line")
     .attr("x1",0).attr("x2",CW).attr("y1",y(v)).attr("y2",y(v))
     .attr("stroke","#F3F4F6").attr("stroke-width",1);
  }});

  // Bars — full color
  g.selectAll("rect").data(bins).join("rect")
    .attr("x",d=>x(d.x0)+1).attr("width",d=>Math.max(0,x(d.x1)-x(d.x0)-2))
    .attr("y",d=>y(d.length)).attr("height",d=>CH-y(d.length))
    .attr("fill",m.color).attr("rx",3);

  // Median line — dark, visible
  g.append("line")
   .attr("x1",x(m.med)).attr("x2",x(m.med)).attr("y1",0).attr("y2",CH)
   .attr("stroke","#111827").attr("stroke-width",2.5).attr("stroke-dasharray","6,3");

  // Mean — clearly distinct color
  g.append("line")
   .attr("x1",x(m.mean)).attr("x2",x(m.mean)).attr("y1",0).attr("y2",CH)
   .attr("stroke","#D97706").attr("stroke-width",2).attr("stroke-dasharray","2,4");

  // Stats label
  g.append("text").attr("x",CW).attr("y",12).attr("text-anchor","end")
   .attr("font-size",10).attr("font-weight","600").attr("fill","#111827")
   .text(`median ${{m.med}}  ·  mean ${{m.mean}}`);

  // Panel title — model color, bold
  g.append("text").attr("x",0).attr("y",-20)
   .attr("font-size",12).attr("font-weight","800").attr("fill",m.color)
   .text(m.model);

  // X axis
  const xAx=g.append("g").attr("transform",`translate(0,${{CH}})`).call(
    d3.axisBottom(x).ticks(maxV+1).tickFormat(d3.format("d")).tickSize(4));
  xAx.select(".domain").attr("stroke","#9CA3AF");
  xAx.selectAll("text").attr("font-size",10).attr("font-weight","600").attr("fill","#111827");
  xAx.selectAll(".tick line").attr("stroke","#9CA3AF");

  if(col===0){{
    const yAx=g.append("g").call(d3.axisLeft(y).ticks(4).tickSize(4));
    yAx.select(".domain").attr("stroke","#9CA3AF");
    yAx.selectAll("text").attr("font-size",10).attr("font-weight","600").attr("fill","#111827");
    yAx.selectAll(".tick line").attr("stroke","#9CA3AF");
    g.append("text").attr("class","axl").attr("transform","rotate(-90)")
     .attr("x",-CH/2).attr("y",-40).attr("text-anchor","middle").text("dishes");
  }}
  if(row===ROWS-1)
    g.append("text").attr("class","axl").attr("x",CW/2).attr("y",CH+38)
     .attr("text-anchor","middle").text("correct / dish");
}});
"""

pathlib.Path("plots/fig02_correct_dist.html").write_text(page("Fig 2 – Correct per Dish", js2))
print("✓ fig02_correct_dist.html")


# ════════════════════════════════════════════════════════════════════════
# FIG 3 — Missed ingredients horizontal bar
# ════════════════════════════════════════════════════════════════════════
all_missed = collections.Counter()
for n in names:
    for d in raw[n]["stats"]:
        for ing in d["missed"]:
            all_missed[ing] += 1
fig3 = [{"ing":ing,"count":cnt} for ing,cnt in all_missed.most_common(25)]
INVIS = {"olive oil","salt","pepper","garlic","mustard","vinegar","lemon juice",
         "onions","shallots","thyme","basil","rosemary","ginger","cumin","paprika",
         "oregano","flour","white wine","cinnamon"}

js3 = f"""
const data  = {json.dumps(fig3)};
const INVIS = new Set({json.dumps(list(INVIS))});
const W=520, rowH=28, H=data.length*rowH;
const m={{t:110,r:72,b:52,l:148}};

const svg=d3.select("body").append("svg")
  .attr("width",W+m.l+m.r).attr("height",H+m.t+m.b);

svg.append("text").attr("class","ttl").attr("x",m.l).attr("y",34)
  .text("Top missed ingredients are physically invisible in photos");
svg.append("text").attr("class","sub").attr("x",m.l).attr("y",58)
  .text("Ground-truth ingredient present, but no model named it · 6 models × 507 dishes");

// Legend — use actual solid colors
[["#B91C1C","Invisible ingredient (dissolved oil, salt, spice)"],
 ["#0369A1","Visible ingredient — model just missed it"]].forEach(([col,lbl],i)=>{{
  svg.append("rect").attr("x",m.l+i*290).attr("y",76).attr("width",14).attr("height",14)
   .attr("fill",col).attr("rx",3);
  svg.append("text").attr("x",m.l+i*290+20).attr("y",89)
   .attr("font-size",11).attr("fill","#374151").text(lbl);
}});

const g=svg.append("g").attr("transform",`translate(${{m.l}},${{m.t}})`);
const x=d3.scaleLinear().domain([0,d3.max(data,d=>d.count)*1.05]).range([0,W]).nice();
const y=d3.scaleBand().domain(data.map(d=>d.ing)).range([0,H]).padding(0.2);

x.ticks(5).forEach(v=>{{
  g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",0).attr("y2",H)
   .attr("stroke","#F3F4F6").attr("stroke-width",1);
}});

// Bars — full solid colors, no opacity
g.selectAll("rect").data(data).join("rect")
  .attr("x",0).attr("y",d=>y(d.ing))
  .attr("width",d=>x(d.count)).attr("height",y.bandwidth())
  .attr("fill",d=>INVIS.has(d.ing)?"#B91C1C":"#0369A1")
  .attr("rx",3);

// Count labels
g.selectAll(".cnt").data(data).join("text")
  .attr("x",d=>x(d.count)+6).attr("y",d=>y(d.ing)+y.bandwidth()/2).attr("dy","0.35em")
  .attr("font-size",10).attr("font-weight","600").attr("fill","#374151")
  .text(d=>d.count.toLocaleString());

// Y axis
const yAx=g.append("g").call(d3.axisLeft(y).tickSize(0).tickPadding(10));
yAx.select(".domain").remove();
yAx.selectAll("text")
  .attr("font-size",10.5)
  .attr("font-weight","600")
  .each(function(d){{
    d3.select(this).attr("fill", INVIS.has(d) ? "#B91C1C" : "#111827");
  }});

// X axis
const xAx=g.append("g").attr("transform",`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(5));
xAx.select(".domain").attr("stroke","#9CA3AF");
xAx.selectAll("text").attr("font-size",10).attr("font-weight","600").attr("fill","#111827");
xAx.selectAll(".tick line").attr("stroke","#9CA3AF");
g.append("text").attr("class","axl").attr("x",W/2).attr("y",H+42).attr("text-anchor","middle")
  .text("total misses");

// Callout annotation
g.append("text")
  .attr("x",x(data[0].count)+6).attr("y",y(data[0].ing)-7)
  .attr("font-size",10).attr("font-style","italic").attr("font-weight","600").attr("fill","#B91C1C")
  .text("877× — liquid absorbed into food, no visual trace");
"""

pathlib.Path("plots/fig03_missed.html").write_text(page("Fig 3 – Missed Ingredients", js3))
print("✓ fig03_missed.html")


# ════════════════════════════════════════════════════════════════════════
# FIG 4 — Hallucinations 2×3 grid
# ════════════════════════════════════════════════════════════════════════
fig4 = []
for n in names:
    ctr = collections.Counter(ing for d in raw[n]["stats"] for ing in d["halluc"])
    fig4.append({"model":n,"color":COLOR[n],
                 "items":[{"ing":i,"count":c} for i,c in ctr.most_common(10)]})

js4 = f"""
const panels = {json.dumps(fig4)};
const COLS=3, CW=270, CH=220, GAP=24;
const pm={{t:42,r:16,b:36,l:122}};
const TW=COLS*(CW+pm.l+pm.r)+(COLS-1)*GAP;
const ROWS=Math.ceil(panels.length/COLS);
const TH=80+ROWS*(CH+pm.t+pm.b)+(ROWS-1)*GAP;

const svg=d3.select("body").append("svg").attr("width",TW).attr("height",TH);
svg.append("text").attr("class","ttl").attr("x",32).attr("y",34)
  .text("What each model hallucinates — predicted but not in the dish");
svg.append("text").attr("class","sub").attr("x",32).attr("y",56)
  .text("Top 10 ingredients predicted when ground truth says they were absent");

panels.forEach((m,i)=>{{
  const col=i%COLS, row=Math.floor(i/COLS);
  const ox=col*(CW+pm.l+pm.r+GAP)+pm.l;
  const oy=70+row*(CH+pm.t+pm.b+GAP)+pm.t;
  const g=svg.append("g").attr("transform",`translate(${{ox}},${{oy}})`);

  const maxC=d3.max(m.items,d=>d.count)||1;
  const x=d3.scaleLinear().domain([0,maxC*1.1]).range([0,CW]).nice();
  const y=d3.scaleBand().domain(m.items.map(d=>d.ing)).range([0,CH]).padding(0.28);

  x.ticks(4).forEach(v=>{{
    g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",0).attr("y2",CH)
     .attr("stroke","#F3F4F6").attr("stroke-width",1);
  }});

  // Bars — full model color
  g.selectAll("rect").data(m.items).join("rect")
    .attr("x",0).attr("y",d=>y(d.ing))
    .attr("width",d=>x(d.count)).attr("height",y.bandwidth())
    .attr("fill",m.color).attr("rx",3);

  // Count labels
  g.selectAll(".lbl").data(m.items).join("text")
    .attr("x",d=>x(d.count)+5).attr("y",d=>y(d.ing)+y.bandwidth()/2).attr("dy","0.35em")
    .attr("font-size",10).attr("font-weight","600").attr("fill","#374151")
    .text(d=>d.count);

  // Panel title
  g.append("text").attr("x",0).attr("y",-20)
   .attr("font-size",12).attr("font-weight","800").attr("fill",m.color).text(m.model);

  const yAx=g.append("g").call(d3.axisLeft(y).tickSize(0).tickPadding(8));
  yAx.select(".domain").remove();
  yAx.selectAll("text").attr("font-size",9.5).attr("fill","#374151").attr("font-weight","500");

  const xAx=g.append("g").attr("transform",`translate(0,${{CH}})`).call(d3.axisBottom(x).ticks(4));
  xAx.select(".domain").attr("stroke","#9CA3AF");
  xAx.selectAll("text").attr("font-size",9).attr("font-weight","600").attr("fill","#111827");
  xAx.selectAll(".tick line").attr("stroke","#9CA3AF");
}});
"""

pathlib.Path("plots/fig04_hallucinated.html").write_text(page("Fig 4 – Hallucinations", js4))
print("✓ fig04_hallucinated.html")


# ════════════════════════════════════════════════════════════════════════
# FIG 5 — SLOPEGRAPH  F1 by complexity
#   All 6 models in their own strong colors. NO gray band.
#   Key 3 (Kimi / Claude / Maverick): 4px lines, r=10 circles.
#   Other 3: 2.5px lines, r=7 circles.
#   F1 labels at every point for all models, bold.
# ════════════════════════════════════════════════════════════════════════
BUCKETS = [(1,3,"Simple"),(4,8,"Moderate"),(9,99,"Complex")]
BUCK_SUB = ["1–3 ingredients","4–8 ingredients","9+ ingredients"]
HIGHLIGHT = ["Kimi K2.5","Claude Opus 4.8","Llama4 Maverick"]

all_f1 = {}
for n in names:
    pts = []
    for lo,hi,label in BUCKETS:
        grp = [s["scores"]["f1"] for s in raw[n]["samples"]
               if lo <= len(s["ground_truth"]) <= hi]
        pts.append({"b":label,"f1":round(statistics.mean(grp),4) if grp else 0,"n":len(grp)})
    all_f1[n] = pts

fig5 = [{"model":n,"color":COLOR[n],"pts":all_f1[n],"hi": n in HIGHLIGHT} for n in names]
ns_per_bucket = [all_f1["Kimi K2.5"][i]["n"] for i in range(3)]

# Data-driven y domain — spread the chart, no cramping
_all_f1_vals = [pt["f1"] for n in names for pt in all_f1[n]]
_y_lo = round(min(_all_f1_vals) - 0.05, 2)
_y_hi = round(max(_all_f1_vals) + 0.05, 2)

js5 = f"""
const series  = {json.dumps(fig5)};
const bsub    = {json.dumps(BUCK_SUB)};
const ns      = {json.dumps(ns_per_bucket)};
const buckets = ["Simple","Moderate","Complex"];

const W=520, H=400;
const m={{t:104,r:234,b:80,l:64}};

const svg=d3.select("body").append("svg")
  .attr("width",W+m.l+m.r).attr("height",H+m.t+m.b);

svg.append("text").attr("class","ttl").attr("x",m.l).attr("y",36)
  .text("F1 score collapses on complex dishes — all 6 models");
svg.append("text").attr("class","sub").attr("x",m.l).attr("y",60)
  .text("Average F1 (PMC13092701) by dish complexity · Kimi K2.5 degrades least");
svg.append("text").attr("class","sub").attr("x",m.l).attr("y",80).attr("font-style","italic")
  .text("Bold lines = Kimi / Claude / Maverick · Thin lines = Scout / Nova Pro / Nova Lite");

const g=svg.append("g").attr("transform",`translate(${{m.l}},${{m.t}})`);

const x=d3.scalePoint().domain(buckets).range([0,W]).padding(0.3);
const y=d3.scaleLinear().domain([{json.dumps(_y_lo)},{json.dumps(_y_hi)}]).range([H,0]);

// Hairline grid
y.ticks(5).forEach(v=>{{
  g.append("line")
   .attr("x1",0).attr("x2",W).attr("y1",y(v)).attr("y2",y(v))
   .attr("stroke","#E5E7EB").attr("stroke-width",0.8);
  g.append("text").attr("x",-10).attr("y",y(v)).attr("dy","0.35em")
   .attr("text-anchor","end").attr("font-size",11).attr("font-weight","700").attr("fill","#111827")
   .text(v.toFixed(2));
}});

// Draw non-highlighted models first (bottom layer), then highlighted on top
const layers=[series.filter(s=>!s.hi), series.filter(s=>s.hi)];
const lineGen=d3.line().x(d=>x(d.b)).y(d=>y(d.f1)).curve(d3.curveMonotoneX);

layers.forEach(layer=>{{
  // Lines
  layer.forEach(s=>{{
    g.append("path").datum(s.pts)
     .attr("d",lineGen).attr("fill","none")
     .attr("stroke",s.color)
     .attr("stroke-width",s.hi?4:2.5)
     .attr("stroke-linejoin","round").attr("stroke-linecap","round");
  }});

  // Circles + F1 labels
  layer.forEach(s=>{{
    const r=s.hi?10:7;
    s.pts.forEach(pt=>{{
      g.append("circle").attr("cx",x(pt.b)).attr("cy",y(pt.f1)).attr("r",r+3).attr("fill","#fff");
      g.append("circle").attr("cx",x(pt.b)).attr("cy",y(pt.f1)).attr("r",r)
       .attr("fill",s.color).attr("stroke","#fff").attr("stroke-width",2.5);
      const labelOffset=s.hi?-20:-16;
      g.append("text").attr("x",x(pt.b)).attr("y",y(pt.f1)+labelOffset)
       .attr("text-anchor","middle")
       .attr("font-size",s.hi?12:10).attr("font-weight","800")
       .attr("fill",s.color).text(pt.f1.toFixed(2));
    }});
  }});
}});

// Right-edge labels — sorted by complex F1, anti-overlap
const ends=series.map(s=>({{model:s.model,color:s.color,f1:s.pts[2].f1,hi:s.hi}}))
  .sort((a,b)=>b.f1-a.f1);
const placed=[];
ends.forEach(ep=>{{
  let ly=y(ep.f1);
  placed.forEach(py=>{{ if(Math.abs(py-ly)<18) ly=py+(ly>py?18:-18); }});
  placed.push(ly);
  g.append("line").attr("x1",W+6).attr("x2",W+16)
   .attr("y1",y(ep.f1)).attr("y2",ly)
   .attr("stroke",ep.color).attr("stroke-width",ep.hi?2:1.2);
  g.append("text").attr("x",W+20).attr("y",ly).attr("dy","0.35em")
   .attr("font-size",ep.hi?12:11).attr("font-weight",ep.hi?"700":"600")
   .attr("fill",ep.color).text(ep.model+" "+ep.f1.toFixed(3));
}});

// X axis — custom labels, large & bold
buckets.forEach((b,i)=>{{
  const cx=x(b);
  g.append("line").attr("x1",cx).attr("x2",cx).attr("y1",H).attr("y2",H+8)
   .attr("stroke","#374151").attr("stroke-width",1.5);
  g.append("text").attr("x",cx).attr("y",H+24).attr("text-anchor","middle")
   .attr("font-size",13).attr("font-weight","700").attr("fill","#111827").text(b);
  g.append("text").attr("x",cx).attr("y",H+42).attr("text-anchor","middle")
   .attr("font-size",11).attr("font-weight","600").attr("fill","#1F2937").text(bsub[i]);
  g.append("text").attr("x",cx).attr("y",H+60).attr("text-anchor","middle")
   .attr("font-size",10).attr("font-weight","600").attr("fill","#374151").text("n="+ns[i]+" dishes");
}});

// Y axis label
g.append("text").attr("transform","rotate(-90)")
  .attr("x",-H/2).attr("y",-52).attr("text-anchor","middle")
  .attr("font-size",11).attr("font-weight","600").attr("fill","#374151")
  .text("Average F1 score");

// Annotation: drop for Kimi
const kPts=series.find(s=>s.model==="Kimi K2.5").pts;
const drop=(kPts[0].f1-kPts[2].f1).toFixed(2);
g.append("text").attr("x",x("Complex")+16).attr("y",y(kPts[2].f1)+32)
  .attr("font-size",10.5).attr("font-weight","700").attr("font-style","italic")
  .attr("fill","#1D4ED8").text("Kimi: −"+drop+" from simple→complex");
"""

pathlib.Path("plots/fig05_complexity.html").write_text(page("Fig 5 – F1 by Complexity", js5))
print("✓ fig05_complexity.html")


# ════════════════════════════════════════════════════════════════════════
# FIG 6 — STACKED BAR: accuracy buckets per model
#   Immediately readable: "What fraction of 507 dishes did each model
#   nail / do ok / struggle on?"
#
#   4 buckets (real colors, no opacity tricks):
#     Perfect  = 100%          → #059669 green
#     Good     = 75–99%        → #1D4ED8 blue
#     Fair     = 50–74%        → #D97706 amber
#     Poor     = 0–49%         → #DC2626 red
#
#   One horizontal stacked bar per model, sorted by "Perfect" descending.
#   % label inside each segment if wide enough.
# ════════════════════════════════════════════════════════════════════════
BUCKETS6 = [
    ("Perfect (100%)",  100, 100, "#059669"),
    ("Good (75–99%)",    75,  99, "#1D4ED8"),
    ("Fair (50–74%)",    50,  74, "#D97706"),
    ("Poor  (<50%)",      0,  49, "#DC2626"),
]

fig6 = []
for n in names:
    vals = [d["pred_acc"]*100 for d in raw[n]["stats"]]
    total = len(vals)
    segs = []
    for label, lo, hi, col in BUCKETS6:
        count = sum(1 for v in vals if lo <= v <= hi)
        segs.append({"label": label, "count": count,
                     "pct": round(100*count/total, 1), "color": col})
    fig6.append({"model": n, "segs": segs,
                 "perfect_pct": segs[0]["pct"]})

fig6.sort(key=lambda r: r["perfect_pct"], reverse=True)

js6 = f"""
const data = {json.dumps(fig6)};
const BUCKETS = {json.dumps([{"label":b[0],"color":b[3]} for b in BUCKETS6])};

const W=600, rowH=68, H=data.length*rowH;
const m={{t:130,r:44,b:56,l:162}};

const svg=d3.select("body").append("svg")
  .attr("width",W+m.l+m.r).attr("height",H+m.t+m.b);

// Title
svg.append("text").attr("class","ttl").attr("x",m.l).attr("y",34)
  .text("How many dishes did each model get right?");
svg.append("text").attr("class","sub").attr("x",m.l).attr("y",58)
  .text("507 dishes per model · each bar = 100% of dishes · sorted by perfect-accuracy dishes");

// Legend — horizontal, clear colored squares
BUCKETS.forEach((b,i)=>{{
  const lx=m.l+i*152;
  svg.append("rect").attr("x",lx).attr("y",78).attr("width",16).attr("height",16)
   .attr("fill",b.color).attr("rx",3);
  svg.append("text").attr("x",lx+22).attr("y",90)
   .attr("font-size",11).attr("font-weight","500").attr("fill","#374151")
   .text(b.label);
}});

const g=svg.append("g").attr("transform",`translate(${{m.l}},${{m.t}})`);

const x=d3.scaleLinear().domain([0,100]).range([0,W]);
const y=d3.scaleBand().domain(data.map(d=>d.model)).range([0,H]).padding(0.28);

// For each model: stacked segments
data.forEach(row=>{{
  let cumPct=0;
  row.segs.forEach(seg=>{{
    const bx=x(cumPct), bw=x(seg.pct)-x(0);
    const by=y(row.model), bh=y.bandwidth();

    // Bar segment — solid color, no opacity
    g.append("rect")
     .attr("x",bx).attr("y",by).attr("width",bw).attr("height",bh)
     .attr("fill",seg.color).attr("rx",seg===row.segs[0]?4:0);

    // Percentage label inside bar if wide enough
    if(seg.pct>=5){{
      g.append("text")
       .attr("x",bx+bw/2).attr("y",by+bh/2).attr("dy","0.35em")
       .attr("text-anchor","middle")
       .attr("font-size",seg.pct>=12?13:10).attr("font-weight","800")
       .attr("fill","#fff")
       .text(seg.pct.toFixed(0)+"%");
    }}
    cumPct+=seg.pct;
  }});

  // Model name — left, bold, dark
  g.append("text").attr("x",-12).attr("y",y(row.model)+y.bandwidth()/2).attr("dy","0.35em")
   .attr("text-anchor","end").attr("font-size",12).attr("font-weight","700")
   .attr("fill","#1F2937").text(row.model);
}});

// X axis — % of dishes
const xAx=g.append("g").attr("transform",`translate(0,${{H}})`).call(
  d3.axisBottom(x).tickValues([0,25,50,75,100]).tickFormat(d=>d+"%").tickSize(5));
xAx.select(".domain").attr("stroke","#9CA3AF");
xAx.selectAll("text").attr("font-size",11).attr("font-weight","700").attr("fill","#111827");
xAx.selectAll(".tick line").attr("stroke","#9CA3AF");
g.append("text").attr("class","axl").attr("x",W/2).attr("y",H+46)
  .attr("text-anchor","middle").text("Share of dishes (%)");

// Subtle vertical guides
[25,50,75].forEach(v=>{{
  g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",0).attr("y2",H)
   .attr("stroke","#fff").attr("stroke-width",1.5);
}});
"""

pathlib.Path("plots/fig06_buckets.html").write_text(page("Fig 6 – Accuracy Buckets", js6))
print("✓ fig06_buckets.html")

print("\nAll 6 charts written to eval/plots/")
