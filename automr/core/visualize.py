"""
AutoMR Visualization Script
============================
Run this script from your AutoMR-Framework folder:
    python visualize.py

It reads  automr_results_detailed.csv  and generates  dashboard.html
Open dashboard.html in any browser — no internet required.
"""

import csv
import json
import os
from collections import Counter, defaultdict

# ── 1. Locate the CSV ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

CSV_FILE = os.path.join(BASE_DIR, "automr_results_detailed.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "dashboard.html")

if not os.path.exists(CSV_FILE):
    print(f"[ERROR] '{CSV_FILE}' not found.")
    print("Make sure you run this script from the AutoMR-Framework folder.")
    exit(1)

print(f"[✓] Found {CSV_FILE}")

# ── 2. Read & parse CSV ───────────────────────────────────────────────────────
with open(CSV_FILE, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

total        = len(rows)
pass_count   = sum(1 for r in rows if r["status"] == "PASS")
fail_count   = total - pass_count
unique_samples = len(set(r["frame_id"] for r in rows))

print(f"[✓] Loaded {total:,} rows  |  {pass_count:,} PASS  |  {fail_count:,} FAIL")

# ── 3. Aggregate statistics ───────────────────────────────────────────────────
mr_status   = defaultdict(Counter)          # mr → {PASS, FAIL}
mr_param    = defaultdict(lambda: defaultdict(lambda: {"pass": 0, "total": 0}))
mr_diffs    = defaultdict(list)

for r in rows:
    mr   = r["mr"]
    param = r["parameter"]
    mr_status[mr][r["status"]] += 1
    mr_param[mr][param]["total"] += 1
    if r["status"] == "PASS":
        mr_param[mr][param]["pass"] += 1
    mr_diffs[mr].append(float(r["difference"]))

# Build overview per MR
mr_overview = {}
for mr, counts in mr_status.items():
    t = counts["PASS"] + counts["FAIL"]
    mr_overview[mr] = {
        "pass":      counts["PASS"],
        "fail":      counts["FAIL"],
        "pass_rate": round(counts["PASS"] / t * 100, 1),
        "avg_diff":  round(sum(mr_diffs[mr]) / len(mr_diffs[mr]), 4),
    }

# Build param trend per MR
param_trends = {}
for mr in mr_param:
    param_trends[mr] = [
        {
            "param":     float(p),
            "pass_rate": round(d["pass"] / d["total"] * 100, 1),
        }
        for p, d in sorted(mr_param[mr].items(), key=lambda x: float(x[0]))
    ]

# Pack everything the dashboard needs
dashboard_data = {
    "summary": {
        "total":          total,
        "pass":           pass_count,
        "fail":           fail_count,
        "pass_rate":      round(pass_count / total * 100, 1),
        "unique_samples": unique_samples,
        "mr_types":       len(mr_status),
    },
    "mr_overview":   mr_overview,
    "param_trends":  param_trends,
}

# ── 4. Write HTML dashboard (all-in-one, no internet needed) ──────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AutoMR — Test Results Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root{{
    --bg:#0b0f1a; --panel:#111827; --border:#1f2d45;
    --accent:#38bdf8; --accent2:#818cf8; --accent3:#34d399; --accent4:#fb923c;
    --text:#e2e8f0; --muted:#64748b; --pass:#34d399; --fail:#f87171;
    --font:'Segoe UI',system-ui,sans-serif;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh;padding:2rem}}
  h1{{font-size:1.8rem;font-weight:700;letter-spacing:-.5px;margin-bottom:.25rem}}
  h1 span{{color:var(--accent)}}
  .subtitle{{color:var(--muted);font-size:.9rem;margin-bottom:2rem}}
  .grid-4{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1rem;margin-bottom:2rem}}
  .card{{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:1.4rem}}
  .card .label{{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.4rem}}
  .card .value{{font-size:2rem;font-weight:700}}
  .card .sub{{font-size:.8rem;color:var(--muted);margin-top:.3rem}}
  .pass-color{{color:var(--pass)}}
  .fail-color{{color:var(--fail)}}
  .accent-color{{color:var(--accent)}}
  .grid-2{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:1.2rem;margin-bottom:2rem}}
  .chart-card{{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:1.5rem}}
  .chart-card h2{{font-size:.95rem;font-weight:600;margin-bottom:1rem;color:var(--text)}}
  canvas{{max-height:260px}}
  .mr-table{{width:100%;border-collapse:collapse;font-size:.875rem}}
  .mr-table th{{text-align:left;padding:.6rem .8rem;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);font-size:.75rem;text-transform:uppercase;letter-spacing:.06em}}
  .mr-table td{{padding:.65rem .8rem;border-bottom:1px solid var(--border)}}
  .mr-table tr:last-child td{{border:none}}
  .badge{{display:inline-block;padding:.2rem .6rem;border-radius:99px;font-size:.75rem;font-weight:600}}
  .badge.pass{{background:rgba(52,211,153,.15);color:var(--pass)}}
  .badge.warn{{background:rgba(251,146,60,.15);color:var(--accent4)}}
  .badge.fail{{background:rgba(248,113,113,.15);color:var(--fail)}}
  .bar-wrap{{background:var(--border);border-radius:99px;height:6px;width:100%;margin-top:.3rem}}
  .bar-fill{{height:6px;border-radius:99px}}
  footer{{text-align:center;color:var(--muted);font-size:.8rem;margin-top:2rem}}
</style>
</head>
<body>
<h1>Auto<span>MR</span> - Metamorphic Testing Dashboard</h1>
<p class="subtitle">Results from <code>{CSV_FILE}</code> &nbsp;·&nbsp; Generated automatically by visualize.py</p>

<!-- ── KPI cards ── -->
<div class="grid-4" id="kpiCards"></div>

<!-- ── Charts row ── -->
<div class="grid-2">
  <div class="chart-card">
    <h2>📊 Pass / Fail by Metamorphic Relation</h2>
    <canvas id="barChart"></canvas>
  </div>
  <div class="chart-card">
    <h2>🥧 Overall Pass vs Fail</h2>
    <canvas id="doughnut"></canvas>
  </div>
</div>

<!-- ── Param trend charts ── -->
<div class="grid-2" id="trendCharts"></div>

<!-- ── MR table ── -->
<div class="chart-card" style="margin-bottom:2rem">
  <h2>📋 Metamorphic Relation Summary</h2>
  <table class="mr-table" id="mrTable"></table>
</div>

<footer>AutoMR Visualization &nbsp;·&nbsp; {total:,} total tests &nbsp;·&nbsp; {unique_samples} samples</footer>

<script>
const DATA = {json.dumps(dashboard_data, separators=(',',':'))};

// ── helpers ──────────────────────────────────────────────────────────────────
const mr_colors = ['#38bdf8','#818cf8','#34d399','#fb923c'];
const mr_short  = mr => mr.replace('Relation','');

function badge(rate){{
  if(rate>=75) return `<span class="badge pass">${{rate}}%</span>`;
  if(rate>=50) return `<span class="badge warn">${{rate}}%</span>`;
  return `<span class="badge fail">${{rate}}%</span>`;
}}

// ── KPI cards ────────────────────────────────────────────────────────────────
const s = DATA.summary;
const kpis = [
  {{label:'Total Tests',    value:s.total.toLocaleString(),        sub:'across all MRs',         cls:'accent-color'}},
  {{label:'Tests Passed',   value:s.pass.toLocaleString(),         sub:`${{s.pass_rate}}% pass rate`,cls:'pass-color'}},
  {{label:'Tests Failed',   value:s.fail.toLocaleString(),         sub:'MR violations found',    cls:'fail-color'}},
  {{label:'Input Samples',  value:s.unique_samples.toLocaleString(),sub:'unique inputs tested',  cls:'accent-color'}},
  {{label:'MR Types',       value:s.mr_types,                       sub:'metamorphic relations',  cls:'accent-color'}},
  {{label:'Pass Rate',      value:s.pass_rate+'%',                 sub:'overall consistency',    cls: s.pass_rate>=60?'pass-color':'fail-color'}},
];
document.getElementById('kpiCards').innerHTML = kpis.map(k=>
  `<div class="card"><div class="label">${{k.label}}</div>
   <div class="value ${{k.cls}}">${{k.value}}</div>
   <div class="sub">${{k.sub}}</div></div>`
).join('');

// ── Bar chart ─────────────────────────────────────────────────────────────────
const mrKeys = Object.keys(DATA.mr_overview);
new Chart(document.getElementById('barChart'),{{
  type:'bar',
  data:{{
    labels: mrKeys.map(mr_short),
    datasets:[
      {{label:'PASS', data:mrKeys.map(k=>DATA.mr_overview[k].pass), backgroundColor:'rgba(52,211,153,.75)', borderRadius:6}},
      {{label:'FAIL', data:mrKeys.map(k=>DATA.mr_overview[k].fail), backgroundColor:'rgba(248,113,113,.75)', borderRadius:6}},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:true,
    plugins:{{legend:{{labels:{{color:'#e2e8f0',boxRadius:4}}}}}},
    scales:{{
      x:{{stacked:true, ticks:{{color:'#94a3b8'}}, grid:{{color:'#1f2d45'}}}},
      y:{{stacked:true, ticks:{{color:'#94a3b8'}}, grid:{{color:'#1f2d45'}}}}
    }}
  }}
}});

// ── Doughnut ──────────────────────────────────────────────────────────────────
new Chart(document.getElementById('doughnut'),{{
  type:'doughnut',
  data:{{
    labels:['PASS','FAIL'],
    datasets:[{{data:[s.pass,s.fail], backgroundColor:['rgba(52,211,153,.8)','rgba(248,113,113,.8)'], borderWidth:0, hoverOffset:8}}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:true,
    cutout:'65%',
    plugins:{{
      legend:{{position:'bottom',labels:{{color:'#e2e8f0',padding:20,boxRadius:4}}}},
      tooltip:{{callbacks:{{label:ctx=>`${{ctx.label}}: ${{ctx.parsed.toLocaleString()}} (${{(ctx.parsed/s.total*100).toFixed(1)}}%)`}}}}
    }}
  }}
}});

// ── Param trend line charts ───────────────────────────────────────────────────
const trendWrap = document.getElementById('trendCharts');
Object.keys(DATA.param_trends).forEach((mr,i)=>{{
  const trend = DATA.param_trends[mr];
  const color = mr_colors[i % mr_colors.length];
  const id = 'trend_'+i;
  const div = document.createElement('div');
  div.className='chart-card';
  div.innerHTML=`<h2>📈 ${{mr_short(mr)}} — Pass Rate by Parameter</h2><canvas id="${{id}}"></canvas>`;
  trendWrap.appendChild(div);
  new Chart(document.getElementById(id),{{
    type:'line',
    data:{{
      labels:trend.map(d=>d.param),
      datasets:[{{
        label:'Pass Rate %',
        data:trend.map(d=>d.pass_rate),
        borderColor:color,
        backgroundColor:color+'28',
        tension:.35, fill:true, pointRadius:5,
        pointBackgroundColor:color, borderWidth:2.5
      }}]
    }},
    options:{{
      responsive:true, maintainAspectRatio:true,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>`Pass Rate: ${{ctx.parsed.y}}%`}}}}
      }},
      scales:{{
        x:{{title:{{display:true,text:'Parameter Value',color:'#64748b'}}, ticks:{{color:'#94a3b8'}}, grid:{{color:'#1f2d45'}}}},
        y:{{min:0,max:100, title:{{display:true,text:'Pass %',color:'#64748b'}}, ticks:{{color:'#94a3b8',callback:v=>v+'%'}}, grid:{{color:'#1f2d45'}}}}
      }}
    }}
  }});
}});

// ── MR summary table ──────────────────────────────────────────────────────────
const tbl = document.getElementById('mrTable');
tbl.innerHTML=`<tr>
  <th>Metamorphic Relation</th><th>Pass</th><th>Fail</th>
  <th>Pass Rate</th><th>Avg Δ Output</th><th>Robustness</th>
</tr>`;
mrKeys.forEach(mr=>{{
  const d = DATA.mr_overview[mr];
  const barW = d.pass_rate;
  const barColor = barW>=75?'var(--pass)':barW>=50?'var(--accent4)':'var(--fail)';
  tbl.innerHTML+=`<tr>
    <td><strong>${{mr}}</strong></td>
    <td class="pass-color">${{d.pass.toLocaleString()}}</td>
    <td class="fail-color">${{d.fail.toLocaleString()}}</td>
    <td>
      ${{badge(d.pass_rate)}}
      <div class="bar-wrap"><div class="bar-fill" style="width:${{barW}}%;background:${{barColor}}"></div></div>
    </td>
    <td style="color:var(--muted)">${{d.avg_diff}}</td>
    <td>${{d.pass_rate>=75?'🟢 Good':d.pass_rate>=50?'🟡 Moderate':'🔴 Weak'}}</td>
  </tr>`;
}});
</script>
</body>
</html>"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[✓] Dashboard written → {OUTPUT_FILE}")
print(f"    Open it in your browser: file://{os.path.abspath(OUTPUT_FILE)}")
print()
print("── Quick Summary ──────────────────────────────────────")
for mr, d in mr_overview.items():
    rob = "Good" if d["pass_rate"] >= 75 else "Moderate" if d["pass_rate"] >= 50 else "Weak"
    print(f"  {mr:<22}  {d['pass_rate']:5.1f}% pass  [{rob}]")
print(f"\n  Overall: {dashboard_data['summary']['pass_rate']}% pass rate across {total:,} tests")