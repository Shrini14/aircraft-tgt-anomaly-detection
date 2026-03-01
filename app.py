import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import datetime
import pipeline_script as pipeline

# ========================
# Page Config
# ========================
st.set_page_config(
    page_title="TGT Anomaly Monitor | MRO Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================
# Custom CSS — Light Aviation / MRO Operations Theme
# ========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ─── Root Variables ─── */
:root {
    --bg-page:        #f0f4f8;
    --bg-surface:     #ffffff;
    --bg-card:        #ffffff;
    --bg-card-alt:    #f7fafc;
    --bg-header:      #0b1f3a;

    --navy:           #0b1f3a;
    --navy-mid:       #1a3a5c;
    --steel:          #2c6ea6;
    --steel-light:    #4a90c4;
    --sky:            #e8f3fc;

    --accent-blue:    #1565c0;
    --accent-amber:   #e67e00;
    --accent-red:     #c0392b;
    --accent-green:   #1a7f5a;

    --text-primary:   #0d1b2a;
    --text-secondary: #3a5068;
    --text-muted:     #7a92a8;
    --text-on-dark:   #e8f0f8;

    --border:         #d0dce8;
    --border-strong:  #b0c4d8;
    --shadow-sm:      0 1px 4px rgba(11,31,58,0.08);
    --shadow-md:      0 4px 16px rgba(11,31,58,0.1);
}

/* ─── Base ─── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(180deg, #e8eef5 0%, #f0f4f8 120px), var(--bg-page) !important;
}

/* ─── Top Nav ─── */
.top-nav {
    background: var(--bg-header);
    margin: -1rem -1rem 0 -1rem;
    padding: 0.6rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid #2c6ea6;
}

.nav-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.nav-divider { width:1px; height:18px; background:rgba(255,255,255,0.2); display:inline-block; margin: 0 0.8rem; vertical-align:middle; }

.nav-module {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    vertical-align: middle;
}

.nav-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    padding: 0.18rem 0.65rem;
    border-radius: 20px;
    text-transform: uppercase;
    display: inline-block;
    margin-left: 0.6rem;
}

.nav-pill.blue  { background: rgba(70,130,195,0.25); color:#90c4e8; border:1px solid rgba(70,130,195,0.4); }
.nav-pill.green { background: rgba(26,127,90,0.25);  color:#5ecfa0; border:1px solid rgba(26,127,90,0.4); }

/* ─── Page Header ─── */
.page-header {
    padding: 2rem 0 1.4rem 0;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}

.page-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
}

.page-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
    letter-spacing: 0.02em;
}

.page-meta {
    text-align: right;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    line-height: 1.8;
}

/* ─── Intro Block ─── */
.intro-block {
    background: var(--sky);
    border: 1px solid #c5dff2;
    border-left: 3px solid var(--steel);
    border-radius: 0 6px 6px 0;
    padding: 0.85rem 1.2rem;
    margin-bottom: 1.4rem;
    font-size: 0.86rem;
    color: var(--text-secondary);
    line-height: 1.7;
}

/* ─── Section Headers ─── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
}

.section-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--steel);
    background: var(--sky);
    border: 1px solid #c5dff2;
    border-radius: 3px;
    padding: 0.1rem 0.45rem;
    letter-spacing: 0.1em;
}

.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--navy);
    text-transform: uppercase;
}

.section-line { flex:1; height:1px; background:var(--border); }

/* ─── Metric Cards ─── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem 1.6rem 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s, transform 0.2s;
}

.metric-card:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }

.metric-card::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: var(--steel);
    border-radius: 0 0 8px 8px;
}

.metric-card.amber::after { background: var(--accent-amber); }
.metric-card.red::after   { background: var(--accent-red); }
.metric-card.green::after { background: var(--accent-green); }

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--navy);
    line-height: 1;
}

.metric-card.amber .metric-value { color: var(--accent-amber); }
.metric-card.red   .metric-value { color: var(--accent-red); }
.metric-card.green .metric-value { color: var(--accent-green); }

.metric-sublabel { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.35rem; }

.metric-icon { position:absolute; top:1rem; right:1.2rem; font-size:1.4rem; opacity:0.1; color:var(--navy); }

/* ─── Trend Pills ─── */
.trend-up  { display:inline-block; font-family:'DM Mono',monospace; font-size:0.62rem; color:var(--accent-red);   background:#fdecea; border-radius:12px; padding:0.1rem 0.5rem; }
.trend-ok  { display:inline-block; font-family:'DM Mono',monospace; font-size:0.62rem; color:var(--accent-green); background:#e6f4ef; border-radius:12px; padding:0.1rem 0.5rem; }

/* ─── Alert Banner ─── */
.alert-banner {
    background: #fff8e6;
    border: 1px solid #f0c060;
    border-left: 4px solid var(--accent-amber);
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.2rem;
    font-size: 0.84rem;
    color: #7a4800;
}

.alert-banner.red   { background:#fdecea; border-color:#e57373; border-left-color:var(--accent-red);   color:#7f1e1e; }
.alert-banner.green { background:#e8f5ee; border-color:#81c995; border-left-color:var(--accent-green); color:#1a4a35; }

/* ─── Health Bar ─── */
.health-bar-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.6rem;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1.2rem;
}

.health-bar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    display: flex;
    justify-content: space-between;
}

.health-bar-track { height:8px; background:#e8eef4; border-radius:4px; overflow:hidden; }
.health-bar-fill  { height:100%; border-radius:4px; background:linear-gradient(90deg,#1a7f5a,#2c9e72); }
.health-bar-fill.warn { background:linear-gradient(90deg,#e67e00,#f0a030); }
.health-bar-fill.crit { background:linear-gradient(90deg,#c0392b,#e05040); }

/* ─── Buttons ─── */
.stButton > button {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: var(--navy) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.7rem 2rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.18s ease !important;
}

.stButton > button:hover {
    background: var(--navy-mid) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}

/* ─── Selectbox ─── */
.stSelectbox label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
}

.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 5px !important;
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
}

/* ─── LLM Summary Cards ─── */
.summary-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    position: relative;
}

.summary-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    border-radius: 8px 0 0 8px;
    background: var(--steel);
}

.summary-card.amber::before { background: var(--accent-amber); }
.summary-card.red::before   { background: var(--accent-red); }
.summary-card.green::before { background: var(--accent-green); }

.summary-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.summary-value { font-size: 0.9rem; color: var(--text-primary); line-height: 1.65; }

/* ─── Misc ─── */
hr, .stDivider { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
.stInfo    { background:var(--sky) !important; border:1px solid #c5dff2 !important; border-radius:6px !important; color:var(--text-secondary) !important; font-size:0.84rem !important; }
.stSuccess { background:#e8f5ee !important; border:1px solid #81c995 !important; border-radius:6px !important; color:#1a4a35 !important; font-size:0.84rem !important; }
.stPyplot  { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:8px !important; overflow:hidden !important; box-shadow:var(--shadow-sm) !important; }
.stDataFrame { border:1px solid var(--border) !important; border-radius:6px !important; overflow:hidden !important; box-shadow:var(--shadow-sm) !important; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg-page); }
::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:3px; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 3rem !important; max-width: 1200px; }
[data-testid="column"] { padding: 0 0.5rem !important; }

</style>
""", unsafe_allow_html=True)


# ========================
# TOP NAV BAR
# ========================
st.markdown("""
<div class="top-nav">
    <div>
        <span class="nav-logo">✈ Air Intelligence</span>
        <span class="nav-divider"></span>
        <span class="nav-module">Engine Health Monitoring · TGT Anomaly Detection</span>
    </div>
    <div>
        <span class="nav-pill blue">REV 4.2.1</span>
        <span class="nav-pill green">● SYSTEM ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ========================
# PAGE HEADER
# ========================
now = datetime.datetime.now()
st.markdown(f"""
<div class="page-header">
    <div>
        <div class="page-title">TGT Anomaly Monitor</div>
        <div class="page-subtitle">Turbine Gas Temperature · Fleet-Wide Residual Analysis · MRO Decision Support</div>
    </div>
    <div class="page-meta">
        <div>DATE &nbsp;&nbsp;<span style="color:var(--text-secondary)">{now.strftime('%d %b %Y')}</span></div>
        <div>LOCAL &nbsp;<span style="color:var(--text-secondary)">{now.strftime('%H:%M')} UTC</span></div>
        <div>USER &nbsp;&nbsp;<span style="color:var(--text-secondary)">MRO OPS</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="intro-block">
    This system identifies engines operating at elevated Turbine Gas Temperature (TGT) margins using
    regression-based residual analysis, fleet-level z-score normalization, and persistence-validated
    anomaly flagging. Results support MRO maintenance scheduling and airworthiness decision making.
</div>
""", unsafe_allow_html=True)


# ========================
# Initialize Session State
# ========================
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False


# ========================
# Run Pipeline
# ========================
col_btn, col_pad = st.columns([1, 4])
with col_btn:
    run_clicked = st.button("▶  Run Anomaly Detection", use_container_width=True)

if run_clicked:
    with st.spinner("Running anomaly detection pipeline..."):
        model = pipeline.load_model()
        df = pipeline.load_data()
        pipeline.validate_data(df)

        df = pipeline.predict_tgt(model, df)
        df = pipeline.compute_residuals(df)

        engine_stats = pipeline.compute_engine_stats(df)
        flagged = pipeline.identify_anomalies(engine_stats)

    st.session_state.pipeline_ran = True
    st.session_state.df = df
    st.session_state.engine_stats = engine_stats
    st.session_state.flagged = flagged

    st.success("✓  Pipeline complete — all modules executed successfully.")


# ========================
# RESULTS
# ========================
if st.session_state.pipeline_ran:

    df           = st.session_state.df
    engine_stats = st.session_state.engine_stats
    flagged      = st.session_state.flagged

    total_engines = engine_stats.shape[0]
    flagged_count = flagged.shape[0]
    healthy_count = total_engines - flagged_count
    health_pct    = round((healthy_count / total_engines) * 100, 1) if total_engines > 0 else 100
    zscore_max    = round(engine_stats["z_score"].max(), 2)

    # ── Smart Alert Banner ──
    if flagged_count == 0:
        st.markdown("""
        <div class="alert-banner green">
            <strong>✔ ALL CLEAR</strong> — No engines exceed anomaly thresholds. Fleet operating within normal TGT margins.
        </div>
        """, unsafe_allow_html=True)
    elif flagged_count >= 3 or zscore_max > 3:
        st.markdown(f"""
        <div class="alert-banner red">
            <strong>⚠ HIGH PRIORITY ALERT</strong> — {flagged_count} engine(s) flagged with elevated TGT residuals. Immediate MRO review recommended.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-banner">
            <strong>▲ ATTENTION</strong> — {flagged_count} engine(s) flagged for elevated TGT. Schedule inspection at next available opportunity.
        </div>
        """, unsafe_allow_html=True)

    # ── 01. Fleet Overview Metrics ──
    st.markdown("""
    <div class="section-header">
        <span class="section-num">01</span>
        <span class="section-title">Fleet Intelligence Overview</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">◉</div>
            <div class="metric-label">Engines Monitored</div>
            <div class="metric-value">{total_engines}</div>
            <div class="metric-sublabel">Active in fleet scan</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        hc = "green" if healthy_count == total_engines else ""
        st.markdown(f"""
        <div class="metric-card {hc}">
            <div class="metric-icon">✓</div>
            <div class="metric-label">Healthy Engines</div>
            <div class="metric-value">{healthy_count}</div>
            <div class="metric-sublabel"><span class="trend-ok">Within tolerance</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        fc = "red" if flagged_count > 0 else "green"
        trend = '<span class="trend-up">Requires action</span>' if flagged_count > 0 else '<span class="trend-ok">None flagged</span>'
        st.markdown(f"""
        <div class="metric-card {fc}">
            <div class="metric-icon">⚠</div>
            <div class="metric-label">Flagged Anomalies</div>
            <div class="metric-value">{flagged_count}</div>
            <div class="metric-sublabel">{trend}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        zc = "red" if zscore_max > 3 else "amber" if zscore_max > 2 else ""
        st.markdown(f"""
        <div class="metric-card {zc}">
            <div class="metric-icon">σ</div>
            <div class="metric-label">Peak Z-Score</div>
            <div class="metric-value">{zscore_max}</div>
            <div class="metric-sublabel">Fleet max deviation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fleet Health Index Bar ──
    bar_class    = "crit" if health_pct < 80 else "warn" if health_pct < 95 else ""
    health_label = "CRITICAL" if health_pct < 80 else "CAUTION" if health_pct < 95 else "NOMINAL"
    st.markdown(f"""
    <div class="health-bar-container">
        <div class="health-bar-label">
            <span>Fleet Health Index</span>
            <span style="color:var(--text-primary);font-weight:600;">{health_pct}% &nbsp;—&nbsp; {health_label}</span>
        </div>
        <div class="health-bar-track">
            <div class="health-bar-fill {bar_class}" style="width:{health_pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 02. Z-Score Distribution Chart ──
    st.markdown("""
    <div class="section-header">
        <span class="section-num">02</span>
        <span class="section-title">Fleet Z-Score Distribution</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    fig_dist, ax_dist = plt.subplots(figsize=(12, 3))
    fig_dist.patch.set_facecolor("#ffffff")
    ax_dist.set_facecolor("#f7fafc")

    engines_sorted = engine_stats.sort_values("z_score", ascending=False).reset_index(drop=True)
    bar_colors = ["#c0392b" if z > 3 else "#e67e00" if z > 2 else "#2c6ea6" for z in engines_sorted["z_score"]]

    ax_dist.bar(range(len(engines_sorted)), engines_sorted["z_score"],
                color=bar_colors, width=0.7, edgecolor="none")
    ax_dist.axhline(2, color="#e67e00", linewidth=1, linestyle="--", alpha=0.7)
    ax_dist.axhline(3, color="#c0392b", linewidth=1, linestyle="--", alpha=0.7)

    ax_dist.set_xlabel("Engine Index (sorted by Z-Score)", fontsize=8, color="#7a92a8", labelpad=6)
    ax_dist.set_ylabel("Z-Score", fontsize=8, color="#7a92a8", labelpad=6)
    ax_dist.set_title("TGT Residual Z-Score — All Engines",
                      fontsize=9, color="#3a5068", fontweight="bold", loc="left", pad=10)
    ax_dist.grid(True, axis="y", color="#e0e8f0", linewidth=0.6, linestyle="-")
    ax_dist.set_axisbelow(True)
    ax_dist.tick_params(colors="#7a92a8", labelsize=7.5, length=2)

    for spine in ax_dist.spines.values():
        spine.set_edgecolor("#d0dce8")

    legend_patches = [
        mpatches.Patch(color="#c0392b", label="Critical  Z > 3"),
        mpatches.Patch(color="#e67e00", label="Caution   Z > 2"),
        mpatches.Patch(color="#2c6ea6", label="Normal"),
    ]
    ax_dist.legend(handles=legend_patches, fontsize=7.5, frameon=True,
                   facecolor="white", edgecolor="#d0dce8", loc="upper right")

    plt.tight_layout()
    st.pyplot(fig_dist)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 03. Flagged Engines Table ──
    st.markdown("""
    <div class="section-header">
        <span class="section-num">03</span>
        <span class="section-title">Anomaly Registry — Flagged Engines</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    if flagged.empty:
        st.info("◎  No anomalies detected — all engines are within operational TGT tolerance.")
    else:
        st.dataframe(
            flagged.sort_values("z_score", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 04. Engine Drilldown ──
    st.markdown("""
    <div class="section-header">
        <span class="section-num">04</span>
        <span class="section-title">Engine Residual Analysis — Drilldown</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_engine = st.selectbox(
            "Select Engine Unit",
            sorted(engine_stats["engine no"].unique()),
            key="engine_select"
        )

    # Engine quick-stat badges
    eng_rows = engine_stats[engine_stats["engine no"] == selected_engine]
    if len(eng_rows) > 0:
        eng_row = eng_rows.iloc[0]
        with col_info:
            is_flagged = selected_engine in flagged["engine no"].values if "engine no" in flagged.columns else False
            flag_badge = (
                '<span style="background:#fdecea;color:#c0392b;padding:0.15rem 0.65rem;border-radius:12px;'
                'font-size:0.7rem;font-family:DM Mono,monospace;border:1px solid #e57373;">⚠ FLAGGED</span>'
                if is_flagged else
                '<span style="background:#e8f5ee;color:#1a7f5a;padding:0.15rem 0.65rem;border-radius:12px;'
                'font-size:0.7rem;font-family:DM Mono,monospace;border:1px solid #81c995;">✓ NORMAL</span>'
            )
            eng_z    = round(float(eng_row.get("z_score", 0)), 3)    if "z_score"       in eng_row.index else "N/A"
            eng_mean = round(float(eng_row.get("mean_residual", 0)), 2) if "mean_residual" in eng_row.index else "N/A"
            st.markdown(f"""
            <div style="display:flex;gap:1.5rem;align-items:center;padding:0.6rem 0;flex-wrap:wrap;">
                {flag_badge}
                <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--text-muted);">
                    Z-SCORE &nbsp;<strong style="color:var(--text-primary)">{eng_z}</strong>
                </span>
                <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--text-muted);">
                    MEAN RESIDUAL &nbsp;<strong style="color:var(--text-primary)">{eng_mean} °C</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)

    with st.spinner(f"⚙️  Rendering residual analysis for Engine {selected_engine}..."):

        df_engine = df[df["engine no"] == selected_engine].sort_values("datetime").copy()
        df_engine["rolling_residual"] = df_engine["residual"].rolling(50).mean()

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f7fafc")

        ax.grid(True, color="#e0e8f0", linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)
        ax.axhline(0, color="#b0c4d8", linewidth=1, linestyle="-")

        residual_std = df_engine["residual"].std()

        ax.fill_between(df_engine["datetime"], 2 * residual_std, -2 * residual_std,
                        alpha=0.06, color="#2c6ea6", label="±2σ Band")

        ax.fill_between(df_engine["datetime"], df_engine["residual"], 0,
                        where=(df_engine["residual"] > 0), alpha=0.12, color="#c0392b", interpolate=True)
        ax.fill_between(df_engine["datetime"], df_engine["residual"], 0,
                        where=(df_engine["residual"] <= 0), alpha=0.08, color="#2c6ea6", interpolate=True)

        ax.plot(df_engine["datetime"], df_engine["residual"],
                color="#b0c4d8", linewidth=0.5, alpha=0.7, label="Raw Residual")
        ax.plot(df_engine["datetime"], df_engine["rolling_residual"],
                color="#1565c0", linewidth=2, label="50-Cycle Rolling Mean")

        ax.axhline( 2 * residual_std, color="#c0392b", linewidth=0.8, linestyle=":", alpha=0.7, label="+2σ Threshold")
        ax.axhline(-2 * residual_std, color="#c0392b", linewidth=0.8, linestyle=":", alpha=0.7)

        ax.set_title(f"TGT Residual Trend  ·  Engine {selected_engine}",
                     fontsize=9, color="#3a5068", fontweight="bold", loc="left", pad=10)
        ax.set_xlabel("Datetime", fontsize=8, color="#7a92a8", labelpad=6)
        ax.set_ylabel("Residual — Actual minus Predicted TGT (°C)", fontsize=8, color="#7a92a8", labelpad=6)
        ax.tick_params(colors="#7a92a8", labelsize=7.5, length=2)

        for spine in ax.spines.values():
            spine.set_edgecolor("#d0dce8")

        ax.legend(fontsize=7.5, frameon=True, facecolor="white", edgecolor="#d0dce8", loc="upper left")

        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 05. LLM Engineering Assessment ──
    if not flagged.empty:

        st.markdown("""
        <div class="section-header">
            <span class="section-num">05</span>
            <span class="section-title">AI Engineering Assessment</span>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🧠  AI is analysing flagged engines and generating engineering assessment..."):
            summary = pipeline.generate_llm_summary(flagged)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"""
            <div class="summary-card red">
                <div class="summary-label">⬡  Critical Engines</div>
                <div class="summary-value">{summary["critical_engines"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class="summary-card amber">
                <div class="summary-label">◈  Risk Assessment</div>
                <div class="summary-value">{summary["risk_assessment"]}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">◉  Engineering Summary</div>
            <div class="summary-value">{summary["engineering_summary"]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="summary-card green">
            <div class="summary-label">▶  Recommended Action</div>
            <div class="summary-value">{summary["recommended_action"]}</div>
        </div>
        """, unsafe_allow_html=True)


# ========================
# FOOTER
# ========================
st.markdown("""
<div style="margin-top:4rem;padding:1.2rem 0;border-top:1px solid var(--border);
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
    <div style="display:flex;align-items:center;gap:1.2rem;">
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:0.95rem;font-weight:700;
              color:var(--navy);letter-spacing:0.08em;">AIR INTELLIGENCE</span>
        <span style="font-family:'DM Mono',monospace;font-size:0.6rem;color:var(--text-muted);
              letter-spacing:0.1em;">TGT ANOMALY MONITOR · ENGINE HEALTH SYSTEM</span>
    </div>
    <span style="font-family:'DM Mono',monospace;font-size:0.6rem;color:var(--text-muted);
          letter-spacing:0.1em;">REGRESSION-BASED RESIDUAL ANALYSIS · FLEET INTELLIGENCE v4.2</span>
</div>
""", unsafe_allow_html=True)