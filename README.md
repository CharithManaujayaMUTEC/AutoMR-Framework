<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoMR — Metamorphic Testing Framework</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
--bg: #0d0d0f;
--bg-2: #131316;
--bg-3: #1a1a1f;
--bg-4: #202026;
--border: rgba(255,255,255,0.07);
--border-md: rgba(255,255,255,0.12);
--text-1: #f0efe8;
--text-2: #9998a0;
--text-3: #5e5d66;
--accent: #7c6fcd;
--accent-2: #a89fe0;
--accent-dim: rgba(124,111,205,0.12);
--accent-dim-2: rgba(124,111,205,0.06);
--green: #4caf82;
--green-dim: rgba(76,175,130,0.1);
--amber: #d4924a;
--amber-dim: rgba(212,146,74,0.1);
--blue: #5b9bd5;
--blue-dim: rgba(91,155,213,0.1);
--red: #c0736a;
--red-dim: rgba(192,115,106,0.1);
--radius-sm: 5px;
--radius-md: 8px;
--radius-lg: 12px;
--font-sans: 'DM Sans', sans-serif;
--font-mono: 'DM Mono', monospace;
}

html { font-size: 15px; }

body {
background: var(--bg);
color: var(--text-1);
font-family: var(--font-sans);
font-weight: 300;
line-height: 1.7;
-webkit-font-smoothing: antialiased;
min-height: 100vh;
}

.page {
max-width: 820px;
margin: 0 auto;
padding: 4rem 2.5rem 6rem;
}

/_ ── HERO ─────────────────────────────────────────── _/
.hero {
padding-bottom: 3.5rem;
border-bottom: 1px solid var(--border);
margin-bottom: 3rem;
}

.hero-eyebrow {
display: inline-flex;
align-items: center;
gap: 8px;
font-family: var(--font-mono);
font-size: 11px;
letter-spacing: 0.14em;
text-transform: uppercase;
color: var(--accent-2);
margin-bottom: 1.5rem;
}

.hero-eyebrow::before {
content: '';
display: inline-block;
width: 18px;
height: 1px;
background: var(--accent-2);
opacity: 0.6;
}

.hero-title {
font-family: var(--font-mono);
font-size: clamp(2.8rem, 6vw, 4.2rem);
font-weight: 300;
letter-spacing: -0.02em;
line-height: 1;
color: var(--text-1);
margin-bottom: 1.25rem;
}

.hero-title .accent { color: var(--accent-2); }

.hero-desc {
font-size: 1.05rem;
color: var(--text-2);
max-width: 580px;
line-height: 1.75;
margin-bottom: 2rem;
font-weight: 300;
}

.badges {
display: flex;
flex-wrap: wrap;
gap: 8px;
}

.badge {
display: inline-flex;
align-items: center;
gap: 6px;
padding: 4px 11px;
border-radius: 4px;
font-family: var(--font-mono);
font-size: 11px;
font-weight: 400;
letter-spacing: 0.02em;
border: 1px solid;
}

.badge-purple { background: var(--accent-dim-2); color: var(--accent-2); border-color: rgba(124,111,205,0.25); }
.badge-green { background: var(--green-dim); color: var(--green); border-color: rgba(76,175,130,0.25); }
.badge-blue { background: var(--blue-dim); color: var(--blue); border-color: rgba(91,155,213,0.25); }
.badge-amber { background: var(--amber-dim); color: var(--amber); border-color: rgba(212,146,74,0.25); }

.badge-dot {
width: 5px; height: 5px;
border-radius: 50%;
background: currentColor;
opacity: 0.7;
}

/_ ── SECTIONS ─────────────────────────────────────── _/
.section { margin-bottom: 2.75rem; }

.section-header {
display: flex;
align-items: center;
gap: 12px;
margin-bottom: 1.25rem;
}

.section-label {
font-family: var(--font-mono);
font-size: 10.5px;
letter-spacing: 0.14em;
text-transform: uppercase;
color: var(--text-3);
white-space: nowrap;
}

.section-line {
flex: 1;
height: 1px;
background: var(--border);
}

.section-title {
font-size: 1.1rem;
font-weight: 400;
color: var(--text-1);
margin-bottom: 1rem;
letter-spacing: -0.01em;
}

/_ ── CARDS GRID ───────────────────────────────────── _/
.card-grid {
display: grid;
grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
gap: 1px;
background: var(--border);
border: 1px solid var(--border);
border-radius: var(--radius-lg);
overflow: hidden;
}

.metric-card {
background: var(--bg-2);
padding: 1.25rem 1.5rem;
}

.metric-label {
font-family: var(--font-mono);
font-size: 10px;
letter-spacing: 0.12em;
text-transform: uppercase;
color: var(--text-3);
margin-bottom: 0.6rem;
}

.metric-title {
font-size: 1rem;
font-weight: 400;
color: var(--text-1);
margin-bottom: 0.3rem;
line-height: 1.3;
}

.metric-sub {
font-size: 0.8rem;
color: var(--text-2);
line-height: 1.5;
}

/_ ── FEATURE PILLS ────────────────────────────────── _/
.pill-group {
display: flex;
flex-wrap: wrap;
gap: 8px;
}

.pill {
font-family: var(--font-mono);
font-size: 12px;
padding: 5px 13px;
border-radius: 4px;
border: 1px solid var(--border-md);
color: var(--text-2);
background: var(--bg-3);
letter-spacing: 0.01em;
transition: border-color 0.15s, color 0.15s;
}

.pill:hover { border-color: rgba(124,111,205,0.35); color: var(--text-1); }

/_ ── CODE BLOCKS ──────────────────────────────────── _/
.code-wrap {
background: var(--bg-2);
border: 1px solid var(--border);
border-radius: var(--radius-lg);
overflow: hidden;
margin-bottom: 1rem;
}

.code-header {
display: flex;
align-items: center;
justify-content: space-between;
padding: 10px 16px;
border-bottom: 1px solid var(--border);
background: var(--bg-3);
}

.code-lang {
font-family: var(--font-mono);
font-size: 10.5px;
color: var(--text-3);
letter-spacing: 0.1em;
text-transform: uppercase;
}

.code-dots {
display: flex;
gap: 5px;
}

.code-dot {
width: 8px; height: 8px;
border-radius: 50%;
background: var(--border-md);
}

pre {
padding: 1.25rem 1.5rem;
font-family: var(--font-mono);
font-size: 12.5px;
line-height: 1.85;
overflow-x: auto;
color: var(--text-2);
}

.kw { color: var(--accent-2); }
.fn { color: var(--green); }
.str { color: var(--amber); }
.cm { color: var(--text-3); font-style: italic; }
.num { color: var(--blue); }
.path { color: #7eb8a4; }

/_ ── FLOW STEPS ───────────────────────────────────── _/
.flow {
background: var(--bg-2);
border: 1px solid var(--border);
border-radius: var(--radius-lg);
overflow: hidden;
}

.flow-step {
display: flex;
gap: 1.25rem;
align-items: flex-start;
padding: 1rem 1.5rem;
border-bottom: 1px solid var(--border);
transition: background 0.15s;
}

.flow-step:last-child { border-bottom: none; }
.flow-step:hover { background: var(--bg-3); }

.step-num {
width: 22px;
height: 22px;
border-radius: 50%;
background: var(--accent-dim);
border: 1px solid rgba(124,111,205,0.2);
color: var(--accent-2);
font-family: var(--font-mono);
font-size: 10px;
font-weight: 500;
display: flex;
align-items: center;
justify-content: center;
flex-shrink: 0;
margin-top: 2px;
}

.step-body { flex: 1; }
.step-title { font-size: 0.9rem; color: var(--text-1); font-weight: 400; margin-bottom: 2px; }
.step-desc { font-size: 0.82rem; color: var(--text-2); }
.step-desc code { font-family: var(--font-mono); font-size: 11px; color: var(--green); background: var(--green-dim); padding: 1px 6px; border-radius: 3px; }

/_ ── OUTPUT FILES ─────────────────────────────────── _/
.output-list { display: flex; flex-direction: column; gap: 6px; }

.output-item {
display: flex;
align-items: center;
gap: 12px;
padding: 10px 14px;
background: var(--bg-2);
border: 1px solid var(--border);
border-radius: var(--radius-md);
transition: border-color 0.15s;
}

.output-item:hover { border-color: var(--border-md); }

.output-indicator {
width: 6px;
height: 6px;
border-radius: 50%;
flex-shrink: 0;
}

.output-name {
font-family: var(--font-mono);
font-size: 12px;
color: var(--green);
flex: 1;
}

.output-desc {
font-size: 12px;
color: var(--text-3);
text-align: right;
}

/_ ── MR TABLE ─────────────────────────────────────── _/
.mr-table-wrap {
background: var(--bg-2);
border: 1px solid var(--border);
border-radius: var(--radius-lg);
overflow: hidden;
}

table {
width: 100%;
border-collapse: collapse;
font-size: 13px;
}

thead tr {
background: var(--bg-3);
border-bottom: 1px solid var(--border);
}

th {
padding: 10px 16px;
text-align: left;
font-family: var(--font-mono);
font-size: 10px;
letter-spacing: 0.12em;
text-transform: uppercase;
color: var(--text-3);
font-weight: 400;
}

td {
padding: 11px 16px;
border-bottom: 1px solid var(--border);
color: var(--text-2);
vertical-align: middle;
}

tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg-3); }

td:first-child {
font-family: var(--font-mono);
font-size: 12px;
color: var(--accent-2);
white-space: nowrap;
}

.mr-tag {
display: inline-block;
font-family: var(--font-mono);
font-size: 10px;
padding: 2px 8px;
border-radius: 3px;
border: 1px solid;
letter-spacing: 0.05em;
text-transform: uppercase;
}

.tag-inv { background: var(--blue-dim); color: var(--blue); border-color: rgba(91,155,213,0.25); }
.tag-rob { background: var(--amber-dim); color: var(--amber); border-color: rgba(212,146,74,0.25); }
.tag-mon { background: var(--green-dim); color: var(--green); border-color: rgba(76,175,130,0.25); }

/_ ── ARCH GRID ────────────────────────────────────── _/
.arch-grid {
display: grid;
grid-template-columns: repeat(4, 1fr);
gap: 1px;
background: var(--border);
border: 1px solid var(--border);
border-radius: var(--radius-lg);
overflow: hidden;
}

.arch-card {
background: var(--bg-2);
padding: 1.25rem;
}

.arch-label {
font-family: var(--font-mono);
font-size: 10px;
color: var(--text-3);
text-transform: uppercase;
letter-spacing: 0.1em;
margin-bottom: 6px;
}

.arch-name {
font-size: 0.95rem;
font-weight: 400;
color: var(--text-1);
margin-bottom: 3px;
}

.arch-role {
font-size: 0.78rem;
color: var(--text-3);
}

/_ ── COLUMNS TABLE (output columns) ──────────────── _/
.col-table-wrap {
background: var(--bg-2);
border: 1px solid var(--border);
border-radius: var(--radius-lg);
overflow: hidden;
}

/_ ── DESIGN PRINCIPLES ────────────────────────────── _/
.principle-grid {
display: grid;
grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
gap: 1px;
background: var(--border);
border: 1px solid var(--border);
border-radius: var(--radius-lg);
overflow: hidden;
}

.principle-card {
background: var(--bg-2);
padding: 1.25rem 1.5rem;
}

.principle-mark {
display: inline-block;
font-family: var(--font-mono);
font-size: 10px;
color: var(--green);
background: var(--green-dim);
border: 1px solid rgba(76,175,130,0.2);
border-radius: 3px;
padding: 1px 7px;
margin-bottom: 10px;
letter-spacing: 0.1em;
text-transform: uppercase;
}

.principle-title {
font-size: 0.95rem;
font-weight: 400;
color: var(--text-1);
margin-bottom: 5px;
}

.principle-desc {
font-size: 0.8rem;
color: var(--text-2);
line-height: 1.6;
}

.principle-desc code {
font-family: var(--font-mono);
font-size: 11px;
color: var(--green);
background: var(--green-dim);
padding: 1px 5px;
border-radius: 3px;
}

/_ ── AUTHORS ──────────────────────────────────────── _/
.authors {
display: flex;
gap: 12px;
flex-wrap: wrap;
margin-bottom: 1rem;
}

.author-chip {
display: flex;
align-items: center;
gap: 10px;
background: var(--bg-2);
border: 1px solid var(--border);
border-radius: 50px;
padding: 6px 16px 6px 6px;
transition: border-color 0.15s;
}

.author-chip:hover { border-color: var(--border-md); }

.author-av {
width: 30px; height: 30px;
border-radius: 50%;
display: flex;
align-items: center;
justify-content: center;
font-family: var(--font-mono);
font-size: 10px;
font-weight: 500;
flex-shrink: 0;
}

.av-purple { background: var(--accent-dim); color: var(--accent-2); }
.av-teal { background: var(--green-dim); color: var(--green); }

.author-name {
font-size: 13px;
font-weight: 400;
color: var(--text-1);
}

.author-handle {
font-family: var(--font-mono);
font-size: 10.5px;
color: var(--text-3);
}

/_ ── FOOTER ───────────────────────────────────────── _/
.footer {
margin-top: 3.5rem;
padding-top: 2rem;
border-top: 1px solid var(--border);
display: flex;
align-items: center;
justify-content: space-between;
flex-wrap: gap;
gap: 12px;
}

.footer-project {
font-size: 12px;
color: var(--text-3);
max-width: 420px;
line-height: 1.6;
}

.footer-license {
display: inline-flex;
align-items: center;
gap: 7px;
padding: 5px 14px;
background: var(--bg-3);
border: 1px solid var(--border);
border-radius: 4px;
font-family: var(--font-mono);
font-size: 11px;
color: var(--text-2);
white-space: nowrap;
}

/_ ── PROJECT STRUCTURE ────────────────────────────── _/
.tree { padding: 1.25rem 1.5rem; }
.tree-line { font-family: var(--font-mono); font-size: 12.5px; line-height: 1.9; color: var(--text-2); }
.tree-dir { color: var(--accent-2); }
.tree-file-green { color: var(--green); }
.tree-file-amber { color: var(--amber); }
.tree-sym { color: var(--text-3); }

/_ ── DIVIDER ──────────────────────────────────────── _/
.divider {
height: 1px;
background: var(--border);
margin: 2.5rem 0;
}

/_ ── INLINE CODE ──────────────────────────────────── _/
code {
font-family: var(--font-mono);
font-size: 12px;
background: var(--bg-3);
border: 1px solid var(--border);
padding: 1px 6px;
border-radius: 3px;
color: var(--text-2);
}

@media (max-width: 560px) {
.page { padding: 2.5rem 1.25rem 4rem; }
.arch-grid { grid-template-columns: repeat(2, 1fr); }
.hero-title { font-size: 2.4rem; }
}
</style>

</head>
<body>
<div class="page">

  <!-- ── HERO ── -->
  <div class="hero">
    <div class="hero-eyebrow">Metamorphic Testing Framework</div>
    <h1 class="hero-title">Auto<span class="accent">MR</span></h1>
    <p class="hero-desc">
      A model-agnostic, input-agnostic, and output-agnostic framework for evaluating
      regressional autonomous driving AI/ML models without requiring ground-truth labels.
      Verifies expected behaviors under controlled input transformations via metamorphic relations.
    </p>
    <div class="badges">
      <span class="badge badge-blue"><span class="badge-dot"></span>Python 3.8+</span>
      <span class="badge badge-green"><span class="badge-dot"></span>MIT License</span>
      <span class="badge badge-purple"><span class="badge-dot"></span>Status: Active</span>
      <span class="badge badge-amber"><span class="badge-dot"></span>Final Year Project</span>
    </div>
  </div>

  <!-- ── OBJECTIVE ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Objective</span>
      <div class="section-line"></div>
    </div>
    <div class="card-grid">
      <div class="metric-card">
        <div class="metric-label">Problem 01</div>
        <div class="metric-title">No labeled data</div>
        <div class="metric-sub">Test ML models without requiring ground-truth labels</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Problem 02</div>
        <div class="metric-title">Real-world perturbations</div>
        <div class="metric-sub">Measure model robustness under realistic conditions</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Problem 03</div>
        <div class="metric-title">Failure detection</div>
        <div class="metric-sub">Identify when and how models begin to fail</div>
      </div>
    </div>
  </div>

  <!-- ── KEY FEATURES ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Key Features</span>
      <div class="section-line"></div>
    </div>
    <div class="pill-group">
      <span class="pill">Model-agnostic</span>
      <span class="pill">Input-agnostic</span>
      <span class="pill">Output-agnostic</span>
      <span class="pill">Built-in MR pipeline</span>
      <span class="pill">Parametric testing</span>
      <span class="pill">Automated analysis</span>
      <span class="pill">Auto CSV export</span>
      <span class="pill">Progress tracking</span>
      <span class="pill">TensorFlow</span>
      <span class="pill">PyTorch</span>
      <span class="pill">sklearn</span>
      <span class="pill">Custom models</span>
    </div>
  </div>

  <!-- ── INSTALLATION ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Installation</span>
      <div class="section-line"></div>
    </div>
    <div class="code-wrap">
      <div class="code-header">
        <span class="code-lang">bash</span>
        <div class="code-dots"><div class="code-dot"></div><div class="code-dot"></div><div class="code-dot"></div></div>
      </div>
      <pre><span class="cm"># clone the repository</span>
git clone https://github.com/<span class="str">CharithManaujayaMUTEC</span>/AutoMR-Framework.git
cd AutoMR-Framework

<span class="cm"># create and activate virtual environment</span>
python -m venv venv
venv\Scripts\activate

<span class="cm"># install dependencies</span>
pip install -r requirements.txt</pre>
</div>

  </div>

  <!-- ── QUICK START ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Quick Start</span>
      <div class="section-line"></div>
    </div>
    <div class="code-wrap">
      <div class="code-header">
        <span class="code-lang">python</span>
        <div class="code-dots"><div class="code-dot"></div><div class="code-dot"></div><div class="code-dot"></div></div>
      </div>
      <pre><span class="kw">from</span> automr.api <span class="kw">import</span> AutoMR

automr = AutoMR(model)

df, results = automr.<span class="fn">run_full_test</span>(
dataset,
max_samples=<span class="num">2000</span>,
samples_per_mr=<span class="num">5</span>,
show_progress=<span class="kw">True</span>
)</pre>
</div>

  </div>

  <!-- ── EXECUTION FLOW ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Execution Flow</span>
      <div class="section-line"></div>
    </div>
    <div class="flow">
      <div class="flow-step">
        <div class="step-num">01</div>
        <div class="step-body">
          <div class="step-title">Load dataset</div>
          <div class="step-desc">User-defined input source — any format</div>
        </div>
      </div>
      <div class="flow-step">
        <div class="step-num">02</div>
        <div class="step-body">
          <div class="step-title">Load model</div>
          <div class="step-desc">Any model exposing a <code>predict(x)</code> interface</div>
        </div>
      </div>
      <div class="flow-step">
        <div class="step-num">03</div>
        <div class="step-body">
          <div class="step-title">AutoMR executes</div>
          <div class="step-desc">Applies transformations, generates predictions, validates metamorphic relations</div>
        </div>
      </div>
      <div class="flow-step">
        <div class="step-num">04</div>
        <div class="step-body">
          <div class="step-title">Analysis</div>
          <div class="step-desc">Computes failure rate, severity, and worst-case failures</div>
        </div>
      </div>
      <div class="flow-step">
        <div class="step-num">05</div>
        <div class="step-body">
          <div class="step-title">Export</div>
          <div class="step-desc">Results saved automatically to <code>/results</code></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── OUTPUT FILES ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Output Files</span>
      <div class="section-line"></div>
    </div>
    <div class="output-list">
      <div class="output-item">
        <div class="output-indicator" style="background: var(--accent);"></div>
        <span class="output-name">automr_results.csv</span>
        <span class="output-desc">Full per-sample test log</span>
      </div>
      <div class="output-item">
        <div class="output-indicator" style="background: var(--amber);"></div>
        <span class="output-name">failure_summary.csv</span>
        <span class="output-desc">Failure rate per MR</span>
      </div>
      <div class="output-item">
        <div class="output-indicator" style="background: var(--blue);"></div>
        <span class="output-name">severity_summary.csv</span>
        <span class="output-desc">Average deviation per MR</span>
      </div>
      <div class="output-item">
        <div class="output-indicator" style="background: var(--red);"></div>
        <span class="output-name">worst_cases.csv</span>
        <span class="output-desc">Highest deviation samples</span>
      </div>
      <div class="output-item">
        <div class="output-indicator" style="background: var(--green);"></div>
        <span class="output-name">failure_regions.txt</span>
        <span class="output-desc">Parametric failure boundaries</span>
      </div>
    </div>
  </div>

  <!-- ── OUTPUT COLUMNS ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Output Columns</span>
      <div class="section-line"></div>
    </div>
    <div class="col-table-wrap">
      <table>
        <thead>
          <tr>
            <th>Column</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>mr</td><td>Metamorphic relation identifier</td></tr>
          <tr><td>param</td><td>Transformation parameter value</td></tr>
          <tr><td>original</td><td>Original model prediction</td></tr>
          <tr><td>transformed</td><td>Prediction after transformation</td></tr>
          <tr><td>difference</td><td>Raw output difference</td></tr>
          <tr><td>percent_change</td><td>Percentage change between outputs</td></tr>
          <tr><td>status</td><td>PASS / FAIL verdict</td></tr>
          <tr><td>expected_behavior</td><td>Expected MR rule</td></tr>
          <tr><td>actual_behavior</td><td>Consistent / Violation</td></tr>
          <tr><td>sample_id</td><td>Input sample index</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── METAMORPHIC RELATIONS ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Metamorphic Relations</span>
      <div class="section-line"></div>
    </div>
    <div class="mr-table-wrap">
      <table>
        <thead>
          <tr>
            <th>Relation</th>
            <th>Description</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>BrightnessRelation</td>
            <td>Output invariant to lighting changes</td>
            <td><span class="mr-tag tag-inv">invariance</span></td>
          </tr>
          <tr>
            <td>RotationRelation</td>
            <td>Stable under small rotations</td>
            <td><span class="mr-tag tag-inv">invariance</span></td>
          </tr>
          <tr>
            <td>TranslationRelation</td>
            <td>Stable under image shifts</td>
            <td><span class="mr-tag tag-inv">invariance</span></td>
          </tr>
          <tr>
            <td>NoiseRelation</td>
            <td>Robust to random noise</td>
            <td><span class="mr-tag tag-rob">robustness</span></td>
          </tr>
          <tr>
            <td>FogRelation</td>
            <td>Robust to visibility degradation</td>
            <td><span class="mr-tag tag-rob">robustness</span></td>
          </tr>
          <tr>
            <td>TemporalSmoothness</td>
            <td>Consistency across frames</td>
            <td><span class="mr-tag tag-mon">monotonic</span></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── TRANSFORMATIONS ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Transformations</span>
      <div class="section-line"></div>
    </div>
    <div class="pill-group">
      <span class="pill">Brightness</span>
      <span class="pill">Rotation</span>
      <span class="pill">Translation</span>
      <span class="pill">Noise</span>
      <span class="pill">Fog</span>
      <span class="pill">Rain</span>
      <span class="pill">Blur</span>
    </div>
  </div>

  <!-- ── DESIGN PRINCIPLES ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Design Principles</span>
      <div class="section-line"></div>
    </div>
    <div class="principle-grid">
      <div class="principle-card">
        <div class="principle-mark">Model Agnostic</div>
        <div class="principle-title">Any model</div>
        <div class="principle-desc">Works with any model implementing <code>predict(x)</code> — TensorFlow, PyTorch, sklearn, or custom.</div>
      </div>
      <div class="principle-card">
        <div class="principle-mark">Input Agnostic</div>
        <div class="principle-title">Any input</div>
        <div class="principle-desc">Supports images, sequences, tabular data, and other input types.</div>
      </div>
      <div class="principle-card">
        <div class="principle-mark">Modular</div>
        <div class="principle-title">Clean separation</div>
        <div class="principle-desc">Model, Transform, Relation, and Analyzer are fully independent components.</div>
      </div>
    </div>
  </div>

  <!-- ── MODULAR ARCHITECTURE ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Architecture</span>
      <div class="section-line"></div>
    </div>
    <div class="arch-grid">
      <div class="arch-card">
        <div class="arch-label">Layer 01</div>
        <div class="arch-name">Model</div>
        <div class="arch-role">Prediction</div>
      </div>
      <div class="arch-card">
        <div class="arch-label">Layer 02</div>
        <div class="arch-name">Transform</div>
        <div class="arch-role">Input modification</div>
      </div>
      <div class="arch-card">
        <div class="arch-label">Layer 03</div>
        <div class="arch-name">Relation</div>
        <div class="arch-role">Expected behavior</div>
      </div>
      <div class="arch-card">
        <div class="arch-label">Layer 04</div>
        <div class="arch-name">Analyzer</div>
        <div class="arch-role">Failure analysis</div>
      </div>
    </div>
  </div>

  <!-- ── PROJECT STRUCTURE ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Project Structure</span>
      <div class="section-line"></div>
    </div>
    <div class="code-wrap">
      <div class="code-header">
        <span class="code-lang">tree</span>
        <div class="code-dots"><div class="code-dot"></div><div class="code-dot"></div><div class="code-dot"></div></div>
      </div>
      <div class="tree">
        <div class="tree-line"><span class="tree-dir">AutoMR-Framework/</span></div>
        <div class="tree-line"><span class="tree-sym">├── </span><span class="tree-dir">automr/</span></div>
        <div class="tree-line"><span class="tree-sym">│   ├── </span><span class="tree-file-green">api.py</span></div>
        <div class="tree-line"><span class="tree-sym">│   ├── </span><span class="tree-file-green">comparator.py</span></div>
        <div class="tree-line"><span class="tree-sym">│   ├── </span><span class="tree-dir">core/</span></div>
        <div class="tree-line"><span class="tree-sym">│   │   ├── </span><span class="tree-file-green">range_tester.py</span></div>
        <div class="tree-line"><span class="tree-sym">│   │   └── </span><span class="tree-file-green">failure_analysis.py</span></div>
        <div class="tree-line"><span class="tree-sym">│   ├── </span><span class="tree-dir">relations/</span></div>
        <div class="tree-line"><span class="tree-sym">│   ├── </span><span class="tree-dir">transforms/</span></div>
        <div class="tree-line"><span class="tree-sym">│   └── </span><span class="tree-dir">analysis/</span></div>
        <div class="tree-line"><span class="tree-sym">├── </span><span class="tree-file-amber">run_test_example.py</span></div>
        <div class="tree-line"><span class="tree-sym">└── </span><span class="tree-file-amber">requirements.txt</span></div>
      </div>
    </div>
  </div>

  <div class="divider"></div>

  <!-- ── LIMITATIONS ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Limitations</span>
      <div class="section-line"></div>
    </div>
    <div class="pill-group">
      <span class="pill">Image-focused transforms only</span>
      <span class="pill">Comparator tuning required per task</span>
      <span class="pill">Performance depends on model speed</span>
    </div>
  </div>

  <!-- ── FUTURE WORK ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Future Work</span>
      <div class="section-line"></div>
    </div>
    <div class="pill-group">
      <span class="pill">NLP extensions</span>
      <span class="pill">Tabular extensions</span>
      <span class="pill">Classification comparators</span>
      <span class="pill">Streamlit dashboard</span>
      <span class="pill">Cross-model MR testing</span>
      <span class="pill">Automated visualizations</span>
    </div>
  </div>

  <div class="divider"></div>

  <!-- ── AUTHORS ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-label">Authors</span>
      <div class="section-line"></div>
    </div>
    <div class="authors">
      <div class="author-chip">
        <div class="author-av av-purple">CM</div>
        <div>
          <div class="author-name">CharithManaujayaMUTEC</div>
          <div class="author-handle">github.com/CharithManaujayaMUTEC</div>
        </div>
      </div>
      <div class="author-chip">
        <div class="author-av av-teal">RP</div>
        <div>
          <div class="author-name">RaveeshaPeiris</div>
          <div class="author-handle">github.com/RaveeshaPeiris</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── FOOTER ── -->
  <div class="footer">
    <div class="footer-project">
      Final Year Project — Metamorphic Testing Framework for Regressional Based Autonomous Driving AI/ML Models
    </div>
    <div class="footer-license">MIT License</div>
  </div>

</div>
</body>
</html>
