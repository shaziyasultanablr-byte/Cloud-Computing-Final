"""Visual presentation and interactive Plotly chart layer for VibeGuard Cloud.

This module never changes preprocessing, checkpoint weights, dataset splits,
labels, score calibration, or the underlying ML evaluation. It contains only
Streamlit presentation helpers, interactive Plotly figures, and deterministic
Matplotlib charts.
"""
from __future__ import annotations

from html import escape
from typing import Any, Mapping
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

MIDNIGHT = "#030407"
COBALT = "#1558C9"
LIME = "#D4FF24"
INK = "#080E19"
PANEL = "#0B1220"
PANEL_RAISED = "#101B2A"
BORDER = "#243248"
WHITE = "#F6F8FF"
MUTED = "#9DAAC2"
RED = "#FF746C"


def mount_theme() -> None:
    """Apply a consistent, accessible dark dashboard theme to Streamlit."""
    st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --vg-black:#030407; --vg-blue:#1558C9; --vg-lime:#D4FF24; --vg-border:#243248; --vg-ink:#0B1220; --vg-muted:#9DAAC2; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background:#030407 !important; color:#F6F8FF !important; }
html, body, p, label, button { font-family:'DM Sans',sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family:'Material Symbols Rounded' !important; font-weight:normal !important; font-style:normal !important; letter-spacing:normal !important; text-transform:none !important; }
[data-testid="stHeader"] { background:transparent !important; }
[data-testid="stAppViewBlockContainer"] { max-width:1400px !important; padding:1.25rem 2.0rem 4rem !important; }
[data-testid="stSidebar"] { background:#080C15 !important; border-right:1px solid #1C2B43 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:1.25rem 1.1rem 1.75rem; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color:#CAD3E4 !important; }
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#F6F8FF !important; }
[data-testid="stSidebarNav"] { background:transparent !important; }
h1,h2,h3 { font-family:'Space Grotesk','DM Sans',sans-serif !important; color:#F6F8FF !important; letter-spacing:-.03em; }
h2 { font-size:clamp(1.25rem, 1.8vw, 1.65rem) !important; }
h3 { font-size:1.12rem !important; }
a { color:#B4CBFF !important; }

/* Navigation Radio Bar (Interactive Section Switcher) */
div[role="radiogroup"] { gap:0.5rem !important; background:#080E19; padding:6px; border-radius:12px; border:1px solid #243248; }
div[role="radiogroup"] > label { background:#0B1220 !important; border:1px solid #243248 !important; border-radius:9px !important; padding:8px 16px !important; margin:0 !important; transition:all .15s ease; }
div[role="radiogroup"] > label:hover { border-color:#438AF0 !important; }
div[role="radiogroup"] > label[data-checked="true"], div[role="radiogroup"] > label:has(input:checked) { background:#D4FF24 !important; border-color:#D4FF24 !important; }
div[role="radiogroup"] > label[data-checked="true"] p, div[role="radiogroup"] > label:has(input:checked) p { color:#030407 !important; font-weight:800 !important; }

/* Primary & Secondary Action Buttons */
[data-testid="stBaseButton-primary"], button[kind="primary"] { background:#D4FF24 !important; color:#030407 !important; border:1px solid #D4FF24 !important; font-weight:800 !important; border-radius:10px !important; }
[data-testid="stBaseButton-primary"] p, button[kind="primary"] p { color:#030407 !important; font-weight:800 !important; }
[data-testid="stBaseButton-secondary"], button[kind="secondary"], [data-testid="stDownloadButton"] button { background:#0E1A2B !important; color:#F6F8FF !important; border:1px solid #2E4768 !important; font-weight:700 !important; border-radius:10px !important; }
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover { border-color:#D4FF24 !important; background:#13233B !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.45rem; border-bottom:1px solid #253248; padding-bottom:8px; overflow:auto; }
[data-testid="stTabs"] button[role="tab"] { color:#A4B1C7 !important; background:#0B1220 !important; background-color:#0B1220 !important; border:1px solid #27354A !important; border-radius:9px !important; padding:9px 14px !important; font-size:.82rem !important; font-weight:700 !important; white-space:nowrap; }
[data-testid="stTabs"] button[role="tab"] p { color:#A4B1C7 !important; font-size:.82rem !important; font-weight:700 !important; margin:0 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#030407 !important; background:#D4FF24 !important; background-color:#D4FF24 !important; border-color:#D4FF24 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color:#030407 !important; font-weight:800 !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:none !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }
[data-testid="stFileUploader"] { background:#0B1220; padding:10px; border-radius:12px; border:1px solid #243248; }
[data-testid="stFileUploaderDropzone"] { background:#0D1726 !important; border:1px dashed #49628A !important; border-radius:10px !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#0D1726 !important; border:1px solid #2D4260 !important; border-radius:10px !important; color:#F6F8FF !important; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #26364D !important; }
[data-testid="stExpander"] { background:#0A111E; border:1px solid #25364B; border-radius:12px; }
[data-testid="stAlert"] { border-radius:12px; border:1px solid #2B4561; }
[data-testid="stMarkdownContainer"] p { line-height:1.58; }
[data-testid="stMetric"] { background:#101B2A; padding:14px 16px; border-radius:12px; border:1px solid #243248; }
[data-testid="stMetricValue"] { color:#F6F8FF !important; }
[data-testid="stMetricLabel"] { color:#A4B2C9 !important; }
[data-testid="stToolbar"] { background:transparent !important; }

/* Compact Professional Landing Header (No fake buttons) */
.vg-compact-header { background:linear-gradient(125deg,#0C1729 0%,#060A12 70%); border:1px solid #263A54; border-radius:14px; padding:1.05rem 1.35rem; margin-bottom:.65rem; display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:1rem; }
.vg-compact-left { min-width:260px; flex:1 1 460px; }
.vg-eyebrow { font-size:.66rem; font-weight:800; letter-spacing:.18em; color:#D4FF24; text-transform:uppercase; margin-bottom:.25rem; }
.vg-compact-left h1 { font-size:clamp(1.45rem,2.3vw,2.05rem) !important; line-height:1.12; margin:0 0 .25rem; font-weight:700; color:#F6F8FF; }
.vg-compact-left h1 span { color:#D4FF24; }
.vg-compact-left p { color:#B0C0D8; font-size:.86rem; margin:0; max-width:780px; }
.vg-spec-grid { display:grid; grid-template-columns:repeat(2,minmax(130px,1fr)); gap:8px; flex:0 1 360px; }
.vg-spec-item { background:#0A1220; border-left:2px solid #1558C9; padding:6px 10px; border-radius:0 6px 6px 0; }
.vg-spec-item .k { font-size:.6rem; color:#8FA4C2; text-transform:uppercase; letter-spacing:.1em; font-weight:700; }
.vg-spec-item .v { font-family:'Space Grotesk',sans-serif; font-size:.8rem; color:#F6F8FF; font-weight:700; }

/* Workflow Stepper */
.vg-stepper { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:.4rem 0 .85rem; }
.vg-step { background:#09101C; border:1px solid #1F2E43; border-radius:10px; padding:8px 12px; display:flex; align-items:center; gap:9px; }
.vg-step.done { border-color:#2D5999; background:#0C182B; }
.vg-step.active { border-color:#D4FF24; background:#121C14; }
.vg-step .num { width:22px; height:22px; border-radius:6px; background:#162438; color:#9DAAC2; font-family:'Space Grotesk',sans-serif; font-size:.72rem; font-weight:800; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.vg-step.done .num { background:#1558C9; color:#F6F8FF; }
.vg-step.active .num { background:#D4FF24; color:#030407; }
.vg-step .txt { font-size:.76rem; font-weight:700; color:#C4D2E8; line-height:1.2; }
.vg-step.active .txt { color:#D4FF24; }

/* Active Input Banner */
.vg-input-banner { background:#0B1526; border:1px solid #284264; border-left:4px solid #D4FF24; border-radius:10px; padding:10px 14px; margin:.45rem 0 .85rem; display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; font-size:.82rem; color:#D0DDF0; }
.vg-input-banner strong { color:#F6F8FF; font-family:'Space Grotesk',sans-serif; }
.vg-input-banner .mode-tag { font-size:.68rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#D4FF24; background:#142117; border:1px solid #486123; padding:3px 8px; border-radius:6px; }

.vg-title { margin:1.05rem 0 .55rem; }
.vg-title p { margin:0 0 .15rem; font-size:.65rem; font-weight:800; color:#D4FF24; letter-spacing:.18em; text-transform:uppercase; }
.vg-title h2 { font-size:clamp(1.22rem,1.75vw,1.6rem); margin:0 0 .2rem; }
.vg-title div { font-size:.84rem; color:#9AB0CC; max-width:940px; }
.vg-card-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:.55rem 0 .85rem; }
.vg-card { border:1px solid #263A52; background:linear-gradient(160deg,#111F31,#0B1220); border-radius:13px; padding:14px 16px; min-width:0; }
.vg-card.featured { background:linear-gradient(145deg,#1558C9,#0B2855); border-color:#438AF0; }
.vg-card .label { color:#B5C6DF; font-size:.66rem; letter-spacing:.09em; text-transform:uppercase; font-weight:750; margin-bottom:6px; }
.vg-card .value { font-family:'Space Grotesk',sans-serif; font-size:clamp(1.15rem,1.65vw,1.5rem); color:#F6F8FF; font-weight:700; line-height:1.15; overflow-wrap:anywhere; }
.vg-card.featured .value { color:#D4FF24; }
.vg-card .micro { color:#9DB2D0; font-size:.74rem; margin-top:6px; }

.vg-sidebar-brand { display:flex; gap:10px; align-items:center; margin:.1rem 0 1.1rem; }
.vg-monogram { background:#D4FF24;color:#030407; font-family:'Space Grotesk'; font-weight:900; font-size:.95rem; letter-spacing:-.06em; padding:8px 10px; border-radius:9px; }
.vg-sidebar-brand strong { font-family:'Space Grotesk'; font-size:1.0rem; letter-spacing:-.03em; }
.vg-sidebar-brand small { display:block; color:#9AB0CC; font-size:.65rem; letter-spacing:.14em; font-weight:800; }
.vg-sidebar-box { background:#0E1A2B; border:1px solid #2C4562; border-radius:11px; padding:11px 13px; margin-bottom:10px; }
.vg-sidebar-box .head { font-size:.64rem; color:#9BB2D1; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
.vg-sidebar-box .big { margin:5px 0 4px; font-size:.88rem; color:#F3F7FF; font-weight:750; }
.vg-sidebar-box .small { color:#A3B5CE; line-height:1.48; font-size:.74rem; overflow-wrap:anywhere; }
.vg-live { display:inline-block;width:8px;height:8px;border-radius:50%;background:#D4FF24;box-shadow:0 0 10px #D4FF24;margin-right:6px; }
.vg-note { border-left:3px solid #1558C9; background:#0C1728; border-radius:0 10px 10px 0; padding:9px 14px; font-size:.8rem; line-height:1.52; color:#B8C9DF; margin:.35rem 0 .85rem; }
.vg-note b { color:#E2EDFF; }
.vg-result { border:1px solid #365575; background:linear-gradient(120deg,#10253F,#0A111D); border-radius:14px; padding:16px 20px; margin:.65rem 0; }
.vg-result .kicker { color:#D4FF24; font-size:.66rem; letter-spacing:.16em; font-weight:800; text-transform:uppercase; }
.vg-result .main { font-family:'Space Grotesk',sans-serif; font-size:clamp(1.55rem,2.5vw,2.25rem); font-weight:700; color:#F6F8FF; line-height:1.15; margin:.35rem 0 .4rem; overflow-wrap:break-word; }
.vg-result .caption { color:#B2C4DD; font-size:.81rem; }
.vg-result.review { border-color:#A65C50; background:linear-gradient(120deg,#341E23,#0A111D); }

/* Educational Fault Cards */
.vg-edu-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin:.6rem 0 1rem; }
.vg-edu-card { background:#0B1322; border:1px solid #263A54; border-radius:13px; padding:15px 17px; }
.vg-edu-card.active { border-color:#D4FF24; background:linear-gradient(145deg,#122238,#0B1322); box-shadow:0 0 0 1px rgba(212,255,36,.25); }
.vg-edu-card .tag { font-size:.64rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; color:#83B6FF; margin-bottom:4px; }
.vg-edu-card.active .tag { color:#D4FF24; }
.vg-edu-card h4 { font-family:'Space Grotesk',sans-serif; font-size:1.05rem; color:#F6F8FF; margin:0 0 8px; }
.vg-edu-card p { font-size:.8rem; color:#B6C6DC; line-height:1.52; margin:0 0 7px; }
.vg-edu-card .cwru-note { font-size:.75rem; color:#98AECB; background:#080F1B; border-left:2px solid #1558C9; padding:6px 10px; border-radius:0 6px 6px 0; margin-top:6px; }

.vg-arch-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:12px 0; }
.vg-arch-step { border-radius:12px; border:1px solid #28415E; background:#0D1727; padding:15px; }
.vg-arch-step .n { color:#D4FF24;font-size:.65rem;font-weight:800;letter-spacing:.13em; }
.vg-arch-step h3 { font-size:1.05rem !important; margin:.4rem 0; }
.vg-arch-step p { color:#AFC0D8;font-size:.78rem;line-height:1.5; margin:0; }
.vg-host-grid { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:.85rem 0; }
.vg-host { border:1px solid #31557E;background:linear-gradient(125deg,#0E284F,#0B1424);border-radius:13px;padding:17px; }
.vg-host.public { border-color:#7C9436;background:linear-gradient(125deg,#23320D,#101710); }
.vg-host .tag { color:#D4FF24;font-size:.65rem;letter-spacing:.14em;font-weight:850; }
.vg-host strong { display:block;margin:.4rem 0;font-size:1.05rem; }
.vg-host p { font-size:.8rem;color:#C2D1E5;line-height:1.55;margin:0; }
.vg-prov-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; margin:.35rem 0 .75rem; }
.vg-prov-item { border:1px solid #24364E; background:#0D1727; border-radius:10px; padding:10px 12px; min-width:0; }
.vg-prov-item .k { color:#98AECB; font-size:.63rem; letter-spacing:.11em; text-transform:uppercase; font-weight:800; margin-bottom:3px; }
.vg-prov-item .v { font-family:'Space Grotesk',sans-serif; color:#F6F8FF; font-size:.9rem; font-weight:700; overflow-wrap:anywhere; }
.vg-prov-item .v.ok { color:#D4FF24; }
.vg-prov-item .v.warn { color:#FF746C; }
@media(max-width:900px) {
  [data-testid="stAppViewBlockContainer"] { padding:.9rem .75rem 2.5rem !important; }
  .vg-compact-header { padding:.9rem 1rem; }
  .vg-card-grid, .vg-arch-grid, .vg-stepper, .vg-edu-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media(max-width:540px) {
  .vg-card-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
  .vg-card { padding:11px; }
  .vg-arch-grid, .vg-host-grid, .vg-prov-grid, .vg-edu-grid, .vg-spec-grid { grid-template-columns:1fr; }
  .vg-stepper { grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px; }
}
</style>''', unsafe_allow_html=True)


def hero() -> None:
    """Render a compact, non-misleading header banner so diagnostic controls stay above the fold."""
    st.markdown('''<header class="vg-compact-header">
      <div class="vg-compact-left">
        <div class="vg-eyebrow">VIBEGUARD CLOUD · MOTOR BEARING SIGNAL INTELLIGENCE</div>
        <h1>Load-Shift-Aware <span>Bearing Fault Diagnosis</span></h1>
        <p>Interactive 1D CNN &amp; Random Forest diagnostic workstation for CWRU drive-end accelerometer signals (trained on 0 &amp; 1 HP, validated on 2 HP, locked test on 3 HP).</p>
      </div>
      <div class="vg-spec-grid" aria-label="System Specifications">
        <div class="vg-spec-item"><div class="k">Architecture</div><div class="v">1D TinyFaultCNN (15,572 params)</div></div>
        <div class="vg-spec-item"><div class="k">Sampling / Window</div><div class="v">12 kHz · 2,048 samples</div></div>
        <div class="vg-spec-item"><div class="k">Locked 3 HP Test</div><div class="v">4 Recordings · 378 Windows</div></div>
        <div class="vg-spec-item"><div class="k">Prototype Status</div><div class="v">Lab Benchmark (Uncalibrated)</div></div>
      </div>
    </header>''', unsafe_allow_html=True)


def workflow_stepper(has_diagnosis: bool, details_open: bool = False) -> None:
    """Display the 4-step diagnostic workflow progression."""
    s1 = "done"
    s2 = "done" if has_diagnosis else "active"
    s3 = "done" if (has_diagnosis and details_open) else ("active" if has_diagnosis else "")
    s4 = "active" if (has_diagnosis and details_open) else ("done" if has_diagnosis else "")
    st.markdown(
        f'''<div class="vg-stepper" role="region" aria-label="Diagnosis Workflow Steps">
          <div class="vg-step {s1}"><span class="num">1</span><span class="txt">Select Recording or Upload File</span></div>
          <div class="vg-step {s2}"><span class="num">2</span><span class="txt">Run Signal Diagnosis</span></div>
          <div class="vg-step {s3}"><span class="num">3</span><span class="txt">Inspect Interactive Plots &amp; Fault Guide</span></div>
          <div class="vg-step {s4}"><span class="num">4</span><span class="txt">Export PDF / CSV Report</span></div>
        </div>''',
        unsafe_allow_html=True,
    )


def active_input_banner(source_name: str, input_mode: str, extra_info: str = "") -> None:
    """Show clearly which input source is currently selected and analyzed."""
    extra_html = f" · <span>{escape(extra_info)}</span>" if extra_info else ""
    st.markdown(
        f'''<div class="vg-input-banner">
          <div>Active Signal Input: <strong>{escape(source_name)}</strong>{extra_html}</div>
          <span class="mode-tag">{escape(input_mode)}</span>
        </div>''',
        unsafe_allow_html=True,
    )


def section(eyebrow: str, title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="vg-title"><p>{escape(eyebrow)}</p><h2>{escape(title)}</h2><div>{escape(description)}</div></div>',
        unsafe_allow_html=True,
    )


def scope_note() -> None:
    st.markdown('''<div class="vg-note"><b>RESEARCH PROTOTYPE NOTICE</b> — Evaluated on CWRU laboratory test-rig recordings (0.007-inch seeded EDM faults), not field-validated industrial machinery. Class scores are <b>uncalibrated softmax values</b>, not posterior probabilities. Does not estimate Remaining Useful Life (RUL) or replace certified engineering inspection.</div>''', unsafe_allow_html=True)


def sidebar_status(model_sha: str | None, run_id: str, epoch: Any, cnn_f1: float | None, rf_f1: float | None, s3_connected: bool, s3_message: str, confidence_threshold: float, agreement_threshold: float) -> None:
    st.markdown('''<div class="vg-sidebar-brand"><span class="vg-monogram">VG</span><div><strong>VibeGuard Cloud</strong><small>SIGNAL INTELLIGENCE</small></div></div>''', unsafe_allow_html=True)
    if model_sha:
        ckpt = '<span class="vg-live"></span> Checkpoint verified'
        meta = f'outputs/model.pt · SHA-256 {escape(model_sha)}…'
    else:
        ckpt = 'Checkpoint missing'
        meta = 'Train or restore outputs/model.pt before inference'
    st.markdown(f'<div class="vg-sidebar-box"><div class="head">MODEL STATUS</div><div class="big">{ckpt}</div><div class="small">{meta}</div></div>', unsafe_allow_html=True)
    cloud_title = 'AWS EC2 + Private S3' if s3_connected else 'Public Web Demo (No AWS Keys)'
    cloud_desc = s3_message if s3_connected else 'Hosted on Streamlit Community Cloud without AWS credentials or private S3 access. Uses SHA-256 verified CWRU recordings.'
    st.markdown(f'<div class="vg-sidebar-box"><div class="head">HOSTING ENVIRONMENT</div><div class="big">{escape(cloud_title)}</div><div class="small">{escape(cloud_desc)}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vg-sidebar-box"><div class="head">FROZEN 3 HP BENCHMARK</div><div class="big">CNN {cnn_f1:.3f} / RF {rf_f1:.3f} Macro-F1</div><div class="small">Locked 3 HP test · Epoch {escape(str(epoch))} · Run {escape(str(run_id))}</div></div>' if cnn_f1 is not None and rf_f1 is not None else '<div class="vg-sidebar-box"><div class="head">FROZEN 3 HP BENCHMARK</div><div class="small">Evaluation artifacts unavailable</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vg-sidebar-box"><div class="head">REVIEW GUARDRAILS</div><div class="big">Score ≥ {confidence_threshold:.2f} · Agree ≥ {agreement_threshold:.0%}</div><div class="small">Pre-specified research heuristics; below-threshold signals trigger a manual-review alert.</div></div>', unsafe_allow_html=True)


def metrics_row(items: list[tuple[str, str, str]], featured: int | None = None) -> None:
    content = []
    for idx, (label, value, micro) in enumerate(items):
        klass = 'vg-card featured' if featured == idx else 'vg-card'
        content.append(f'<div class="{klass}"><div class="label">{escape(str(label))}</div><div class="value">{escape(str(value))}</div><div class="micro">{escape(str(micro))}</div></div>')
    st.markdown('<div class="vg-card-grid">' + ''.join(content) + '</div>', unsafe_allow_html=True)


def result_hero(predicted_display: str, source_file: str, manual_review: bool, note: str = "") -> None:
    label = 'MANUAL REVIEW ADVISED' if manual_review else 'SIGNAL CLASSIFICATION COMPLETE'
    klass = 'vg-result review' if manual_review else 'vg-result'
    st.markdown(f'<div class="{klass}"><div class="kicker">{label}</div><div class="main">{escape(predicted_display)}</div><div class="caption">Analyzed Recording: {escape(source_file)} · {escape(note)}</div></div>', unsafe_allow_html=True)


def provenance_card(source_file: str, channel_key: str, sampling_hz: int, window_size: int, num_windows: int, manual_review: bool) -> None:
    review_text = "Flagged for review" if manual_review else "Not flagged"
    review_cls = "v warn" if manual_review else "v ok"
    khz = f"{sampling_hz / 1000:g} kHz"
    st.markdown(
        f'''<div class="vg-prov-grid">
          <div class="vg-prov-item"><div class="k">RECORDING</div><div class="v">{escape(str(source_file))}</div></div>
          <div class="vg-prov-item"><div class="k">CHANNEL</div><div class="v">{escape(str(channel_key))}</div></div>
          <div class="vg-prov-item"><div class="k">SAMPLING</div><div class="v">{escape(khz)}</div></div>
          <div class="vg-prov-item"><div class="k">PREPROCESSING</div><div class="v">Z-score · Clamped [-10, 10]</div></div>
          <div class="vg-prov-item"><div class="k">WINDOW SIZE</div><div class="v">{window_size:,} ({num_windows} segs)</div></div>
          <div class="vg-prov-item"><div class="k">REVIEW STATUS</div><div class="{review_cls}">{review_text}</div></div>
        </div>''',
        unsafe_allow_html=True,
    )


def eval_limitation_note() -> None:
    st.markdown(
        '''<div class="vg-note"><b>BENCHMARK LIMITATION</b> — The locked 3 HP test set comprises only <b>four related held-out recordings</b> from the same CWRU laboratory test rig (one recording per class: 200 normal, 60 inner, 59 ball, 59 outer windows). A 100% test macro-F1 demonstrates cross-load consistency across 0/1 HP → 3 HP on seeded 0.007-inch EDM faults, <b>not</b> real-world industrial reliability across different motors or progressive wear.</div>''',
        unsafe_allow_html=True,
    )


def educational_fault_guide(active_class_key: str | None = None) -> None:
    """Render educational engineering explanations of all 4 bearing conditions vs. CWRU scope."""
    cards = [
        (
            "normal",
            "CLASS 00 · BASELINE CONDITION",
            "Normal Bearing (Healthy Baseline)",
            "Smooth rolling contact between balls and raceways without localized surface spalls or pits. Vibration energy is dominated by fundamental shaft rotational speed (1× RPM) and low-amplitude structural background noise.",
            "Time-domain waveform exhibits low crest factor and near-Gaussian amplitude distribution without periodic high-frequency impulse trains.",
            "CWRU 3 HP Benchmark (100.mat): 200 windows at ~1,730 RPM; classified as Normal Baseline with 0.9869 mean uncalibrated CNN score (100% window agreement).",
        ),
        (
            "inner",
            "CLASS 01 · ROTATING RACEWAY DEFECT",
            "Inner Race Fault (BPFI Signature)",
            "A localized defect on the inner raceway rotates with the shaft, entering and leaving the radial load zone once per revolution. Rolling elements striking the fault generate high-frequency impulses at the Ball Pass Frequency of the Inner Race (BPFI), amplitude-modulated at 1× shaft speed.",
            "Pronounced periodic impulse bursts in the time domain with sidebands spaced at shaft rotational frequency around high-frequency structural resonances (2–4.5 kHz).",
            "CWRU 3 HP Benchmark (108.mat, 0.007 in EDM notch): 60 windows; 1D CNN achieves 0.9953 mean score (100% agreement), whereas the Random Forest baseline drops to 0.7273 due to load-induced spectral shift.",
        ),
        (
            "ball",
            "CLASS 02 · ROLLING ELEMENT DEFECT",
            "Ball Bearing Fault (BSF Signature)",
            "A spall or pit on a rolling element strikes both the inner and outer raceways at twice the Ball Spin Frequency (2× BSF) as the ball rotates with the cage (FTF). Because the ball's spin axis orientation can precess, impulse amplitudes often fluctuate across consecutive impacts.",
            "Broadband high-frequency resonance excitation modulated at cage frequency, typically exhibiting lower impulse-to-noise ratio than stationary outer race defects.",
            "CWRU 3 HP Benchmark (121.mat, 0.007 in EDM fault): 59 windows; classified as Ball Fault with 0.9890 mean uncalibrated CNN score and 0.9870 RF vote share.",
        ),
        (
            "outer",
            "CLASS 03 · STATIONARY RACEWAY DEFECT",
            "Outer Race Fault @ 6:00 (BPFO Signature)",
            "A defect on the stationary outer raceway positioned directly in the primary radial load zone (6 o'clock orthogonal load position) experiences consistent impact force every time a rolling element passes at the Ball Pass Frequency of the Outer Race (BPFO).",
            "Highly repeatable, unmodulated periodic impulse spikes in the time waveform and strong upper-band resonant peaks in the FFT spectrum.",
            "CWRU 3 HP Benchmark (133.mat, 0.007 in EDM notch at 6:00): 59 windows; classified as Outer Race Fault with 0.9851 mean uncalibrated CNN score and 0.9981 RF vote share.",
        ),
    ]
    html_parts = []
    for key, tag, title, mech, sig, cwru in cards:
        is_active = key == active_class_key
        cls_name = "vg-edu-card active" if is_active else "vg-edu-card"
        badge = " · [DETECTED IN CURRENT SIGNAL]" if is_active else ""
        html_parts.append(
            f'''<div class="{cls_name}">
              <div class="tag">{escape(tag)}{escape(badge)}</div>
              <h4>{escape(title)}</h4>
              <p><b>Mechanical Meaning:</b> {escape(mech)}</p>
              <p><b>Vibration Signature:</b> {escape(sig)}</p>
              <div class="cwru-note"><b>CWRU Empirical Scope:</b> {escape(cwru)}</div>
            </div>'''
        )
    st.markdown('<div class="vg-edu-grid">' + "".join(html_parts) + '</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Interactive Plotly Charts (Web UI)
# ---------------------------------------------------------------------------

def _apply_plotly_theme(fig: go.Figure, height: int = 320, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=MIDNIGHT,
        plot_bgcolor=PANEL,
        height=height,
        margin=dict(l=52, r=24, t=46 if title else 24, b=44),
        font=dict(family="DM Sans, sans-serif", color=WHITE, size=12),
        title=dict(
            text=title,
            font=dict(family="Space Grotesk, sans-serif", size=14, color=WHITE),
            x=0.01,
            xanchor="left",
        ) if title else None,
        hoverlabel=dict(bgcolor="#101B2A", bordercolor=COBALT, font=dict(color=WHITE, size=12)),
    )
    fig.update_xaxes(gridcolor="#1F3047", zerolinecolor="#243248", tickfont=dict(color=MUTED, size=11), title_font=dict(color=MUTED, size=11))
    fig.update_yaxes(gridcolor="#1F3047", zerolinecolor="#243248", tickfont=dict(color=MUTED, size=11), title_font=dict(color=MUTED, size=11))
    return fig


def interactive_score_chart(
    scores: Mapping[str, float],
    window_counts: Mapping[str, int],
    display_names: Mapping[str, str],
    labels: list[str],
) -> go.Figure:
    """Interactive horizontal bar chart for 4-class uncalibrated softmax scores."""
    vals = [float(scores[k]) for k in labels]
    counts = [int(window_counts.get(k, 0)) for k in labels]
    total_w = max(1, sum(counts))
    shares = [(c / total_w) * 100.0 for c in counts]
    best_idx = int(np.argmax(vals))
    colors_bar = [LIME if i == best_idx else COBALT for i in range(len(labels))]
    short_names = [display_names[k] for k in labels]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=short_names,
            orientation="h",
            marker=dict(color=colors_bar, line=dict(color=BORDER, width=1)),
            text=[f"{v:.4f} ({c}/{total_w} wins)" for v, c in zip(vals, counts)],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(family="Space Grotesk, sans-serif", size=12, color=WHITE),
            customdata=np.column_stack([counts, shares]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Mean Uncalibrated Softmax Score: <b>%{x:.6f}</b><br>"
                "Winning Windows: <b>%{customdata[0]}</b> (%{customdata[1]:.1f}%)<extra></extra>"
            ),
        )
    )
    _apply_plotly_theme(fig, height=275, title="Four-Class Mean Uncalibrated Softmax Scores")
    fig.update_layout(margin=dict(l=52, r=70, t=46, b=44))
    fig.update_xaxes(range=[0, 1.55], title_text="Mean Uncalibrated Softmax Score (0.0 – 1.0)")
    fig.update_yaxes(autorange="reversed", title_text="")
    return fig


def interactive_signal_charts(
    raw_signal: np.ndarray,
    window_norm: np.ndarray,
    window_index: int = 0,
    sampling_hz: int = 12000,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Return 3 interactive Plotly figures: Raw Waveform, Normalized Window, and FFT Spectrum."""
    n_raw = min(len(raw_signal), 4096)
    t_sec = np.arange(n_raw, dtype=np.float64) / float(sampling_hz)

    fig_raw = go.Figure(
        go.Scatter(
            x=t_sec,
            y=raw_signal[:n_raw],
            mode="lines",
            name="Drive-End Acceleration",
            line=dict(color=COBALT, width=1.2),
            hovertemplate="Time: <b>%{x:.4f} s</b><br>Acceleration: <b>%{y:.4f} g</b><extra></extra>",
        )
    )
    _apply_plotly_theme(fig_raw, height=300, title=f"01 / Raw Drive-End Vibration Waveform (First {n_raw:,} samples @ {sampling_hz/1000:g} kHz)")
    fig_raw.update_xaxes(title_text="Time (seconds)")
    fig_raw.update_yaxes(title_text="Acceleration (g)")

    sample_idx = np.arange(len(window_norm))
    fig_norm = go.Figure(
        go.Scatter(
            x=sample_idx,
            y=window_norm,
            mode="lines",
            name=f"Window #{window_index}",
            line=dict(color=LIME, width=1.15),
            hovertemplate="Sample Index: <b>%{x}</b><br>Normalized Z-Score: <b>%{y:.4f} σ</b><extra></extra>",
        )
    )
    _apply_plotly_theme(fig_norm, height=300, title=f"02 / Normalized Window #{window_index} (2,048 samples, Z-score)")
    fig_norm.update_xaxes(title_text="Window Sample Index (0 – 2,047)")
    fig_norm.update_yaxes(title_text="Normalized Amplitude (Z-Score)")

    freqs_khz = np.fft.rfftfreq(len(window_norm), d=1.0 / sampling_hz) / 1000.0
    fft_mag = np.abs(np.fft.rfft(window_norm)) / len(window_norm)
    peak_idx = int(np.argmax(fft_mag))
    peak_freq = float(freqs_khz[peak_idx])
    peak_val = float(fft_mag[peak_idx])

    fig_fft = go.Figure(
        go.Scatter(
            x=freqs_khz,
            y=fft_mag,
            mode="lines",
            name="FFT Magnitude",
            line=dict(color="#83B6FF", width=1.35),
            fill="tozeroy",
            fillcolor="rgba(21, 88, 201, 0.22)",
            hovertemplate="Frequency: <b>%{x:.3f} kHz</b><br>FFT Magnitude: <b>%{y:.5f}</b><extra></extra>",
        )
    )
    fig_fft.add_annotation(
        x=peak_freq,
        y=peak_val,
        text=f"Peak: {peak_freq:.2f} kHz ({peak_val:.3f})",
        showarrow=True,
        arrowhead=2,
        arrowcolor=LIME,
        font=dict(color=LIME, size=11, family="Space Grotesk, sans-serif"),
        bgcolor="#0B1220",
        bordercolor=LIME,
    )
    _apply_plotly_theme(fig_fft, height=300, title=f"03 / FFT Spectrum (Window #{window_index}, 0–{sampling_hz/2000:g} kHz)")
    fig_fft.update_xaxes(title_text="Frequency (kHz)")
    fig_fft.update_yaxes(title_text="Normalized FFT Magnitude")

    return fig_raw, fig_norm, fig_fft


def interactive_eval_comparison_chart(rec_rows: list[dict[str, Any]], rf_by_id: Mapping[int, dict[str, Any]]) -> go.Figure:
    """Interactive grouped bar chart comparing 1D CNN vs Random Forest top scores across the 4 held-out 3 HP recordings."""
    labels_x = [f"{r['filename']} ({r['true_label'].title()})" for r in rec_rows]
    cnn_scores = [float(r["top_uncalibrated_score"]) for r in rec_rows]
    cnn_margins = [float(r["second_best_margin"]) for r in rec_rows]
    rf_scores = [float(rf_by_id.get(r["file_id"], {}).get("top_uncalibrated_score", 0.0)) for r in rec_rows]
    rf_margins = [float(rf_by_id.get(r["file_id"], {}).get("second_best_margin", 0.0)) for r in rec_rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels_x,
            y=cnn_scores,
            name="1D TinyFaultCNN (Top Uncalibrated Score)",
            marker_color=LIME,
            text=[f"{v:.4f}" for v in cnn_scores],
            textposition="outside",
            customdata=cnn_margins,
            hovertemplate="<b>%{x}</b><br>1D CNN Top Score: <b>%{y:.4f}</b><br>Margin over 2nd Class: <b>%{customdata:.4f}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels_x,
            y=rf_scores,
            name="Random Forest Baseline (Top Vote Share)",
            marker_color=COBALT,
            text=[f"{v:.4f}" for v in rf_scores],
            textposition="outside",
            customdata=rf_margins,
            hovertemplate="<b>%{x}</b><br>Random Forest Top Score: <b>%{y:.4f}</b><br>Margin over 2nd Class: <b>%{customdata:.4f}</b><extra></extra>",
        )
    )
    fig.add_hline(
        y=0.70,
        line_dash="dash",
        line_color=RED,
        annotation_text="Manual Review Threshold (0.70)",
        annotation_position="bottom right",
        annotation_font=dict(color=RED, size=11),
    )
    _apply_plotly_theme(fig, height=360, title="Recording-Level Confidence Comparison on Locked 3 HP Test Split")
    fig.update_layout(
        barmode="group",
        margin=dict(l=52, r=24, t=78, b=44),
        title=dict(y=0.95),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_yaxes(range=[0, 1.16], title_text="Top Uncalibrated Score / Vote Share")
    fig.update_xaxes(title_text="Held-Out 3 HP Test Recording (True Condition)")
    return fig


def interactive_confusion_charts(
    cnn_matrix: list[list[int]],
    rf_matrix: list[list[int]],
    labels: list[str],
) -> go.Figure:
    """Side-by-side interactive Plotly confusion matrices for 1D CNN and Random Forest."""
    disp_labels = [s.title() for s in labels]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("1D TinyFaultCNN · Locked 3 HP Test (378 Windows)", "Random Forest Baseline · Locked 3 HP Test (378 Windows)"),
        horizontal_spacing=0.14,
    )
    colorscale = [[0.0, PANEL], [0.35, COBALT], [1.0, LIME]]

    for col_idx, mat in enumerate([cnn_matrix, rf_matrix], start=1):
        arr = np.asarray(mat, dtype=int)
        row_sums = np.maximum(1, arr.sum(axis=1, keepdims=True))
        pct = (arr / row_sums) * 100.0
        text_grid = [[f"{arr[i, j]}<br>({pct[i, j]:.0f}%)" for j in range(len(labels))] for i in range(len(labels))]
        fig.add_trace(
            go.Heatmap(
                z=arr,
                x=disp_labels,
                y=disp_labels,
                text=text_grid,
                texttemplate="%{text}",
                textfont=dict(family="Space Grotesk, sans-serif", size=12),
                colorscale=colorscale,
                showscale=(col_idx == 2),
                hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Windows: <b>%{z}</b><extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(title_text="Predicted Class", row=1, col=col_idx)
        fig.update_yaxes(title_text="Actual Class", autorange="reversed", row=1, col=col_idx)

    _apply_plotly_theme(fig, height=380, title="")
    return fig


def interactive_training_charts(history: dict[str, Any]) -> go.Figure:
    """Dual-panel interactive Plotly training/validation loss and macro-F1 curves."""
    rows = history.get("epochs", [])
    if not rows:
        raise ValueError("Training history has no epochs")
    ep = [int(r["epoch"]) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    tr_f1 = [float(r["train_macro_f1"]) for r in rows]
    va_f1 = [float(r["val_macro_f1"]) for r in rows]
    chosen = history.get("selected_epoch")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cross-Entropy Loss (0 & 1 HP Train vs. 2 HP Val)", "Macro-F1 Score (0 & 1 HP Train vs. 2 HP Val)"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(x=ep, y=tr_loss, mode="lines+markers", name="Train Loss (0 & 1 HP)", line=dict(color=COBALT, width=2.5), hovertemplate="Epoch %{x}<br>Train Loss: <b>%{y:.5f}</b><extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=ep, y=va_loss, mode="lines+markers", name="Validation Loss (2 HP)", line=dict(color=LIME, width=2.5), hovertemplate="Epoch %{x}<br>Val Loss: <b>%{y:.5f}</b><extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=ep, y=tr_f1, mode="lines+markers", name="Train Macro-F1 (0 & 1 HP)", line=dict(color=COBALT, width=2.5, dash="dot"), hovertemplate="Epoch %{x}<br>Train Macro-F1: <b>%{y:.4f}</b><extra></extra>"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=ep, y=va_f1, mode="lines+markers", name="Validation Macro-F1 (2 HP)", line=dict(color=LIME, width=2.5), hovertemplate="Epoch %{x}<br>Val Macro-F1: <b>%{y:.4f}</b><extra></extra>"),
        row=1,
        col=2,
    )
    if chosen:
        fig.add_vline(x=int(chosen), line_dash="dash", line_color="#9DAAC2", row=1, col=1)
        fig.add_vline(x=int(chosen), line_dash="dash", line_color="#9DAAC2", row=1, col=2)

    _apply_plotly_theme(fig, height=370, title="")
    fig.update_xaxes(title_text="Training Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Training Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Cross-Entropy Loss", row=1, col=1)
    fig.update_yaxes(title_text="Macro-F1", range=[0.0, 1.06], row=1, col=2)
    fig.update_layout(
        margin=dict(l=52, r=24, t=70, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.14, xanchor="left", x=0.0),
    )
    return fig


# ---------------------------------------------------------------------------
# Deterministic Matplotlib Figures (Backward Compatibility & PDF Export)
# ---------------------------------------------------------------------------

def _figure_axes(n: int, size: tuple[float, float] = (13.2, 3.8)):
    fig, axes = plt.subplots(1, n, figsize=size, dpi=125, facecolor=MIDNIGHT)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=7)
        ax.grid(color='#29405E', linestyle=':', linewidth=.75, alpha=.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(WHITE)
    return fig, axes


def score_plot(scores: Mapping[str, float], display_names: Mapping[str, str], labels: list[str]) -> plt.Figure:
    """Readable, horizontally oriented 4-class score comparison."""
    fig, axes = _figure_axes(1, (9, 3.3))
    ax = axes[0]
    values = [float(scores[k]) for k in labels]
    selected = int(np.argmax(values))
    colors = [LIME if i == selected else COBALT for i in range(len(labels))]
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=colors, height=.54)
    ax.set_yticks(ypos, [display_names[k].split(' (')[0] for k in labels])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, .25, .5, .75, 1.0], ['0', '.25', '.50', '.75', '1.0'])
    for i, v in enumerate(values):
        ax.text(min(v + .018, .995), i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold', color=WHITE if i != selected else LIME)
    ax.set_xlabel('Mean uncalibrated softmax score', fontsize=9, labelpad=12)
    fig.subplots_adjust(left=.23, right=.98, top=.91, bottom=.21)
    return fig


def signal_plot(raw_signal: np.ndarray, first_window_norm: np.ndarray, sampling_hz: int = 12000) -> plt.Figure:
    """Three dark signal panels with consistent colors, axes and annotations."""
    fig, axes = _figure_axes(3, (14.2, 3.9))
    n = min(len(raw_signal), 4096)
    t = np.arange(n) / float(sampling_hz)
    axes[0].plot(t, raw_signal[:n], color=COBALT, lw=.8)
    axes[0].set_title('01  /  RAW VIBRATION', loc='left', fontsize=11, fontweight='bold', color=WHITE, pad=15)
    axes[0].set_xlabel('Time (seconds)', fontsize=9)
    axes[0].set_ylabel('Acceleration (g)', fontsize=9)
    axes[1].plot(first_window_norm, color=LIME, lw=.75)
    axes[1].set_title('02  /  NORMALIZED WINDOW', loc='left', fontsize=11, fontweight='bold', color=WHITE, pad=15)
    axes[1].set_xlabel('Sample index', fontsize=9)
    axes[1].set_ylabel('Z-score', fontsize=9)
    freqs = np.fft.rfftfreq(len(first_window_norm), d=1.0 / sampling_hz)
    mag = np.abs(np.fft.rfft(first_window_norm)) / len(first_window_norm)
    axes[2].plot(freqs / 1000, mag, color='#83B6FF', lw=1.1)
    axes[2].fill_between(freqs / 1000, mag, alpha=.18, color=COBALT)
    axes[2].set_title('03  /  FREQUENCY SPECTRUM', loc='left', fontsize=11, fontweight='bold', color=WHITE, pad=15)
    axes[2].set_xlabel('Frequency (kHz)', fontsize=9)
    axes[2].set_ylabel('FFT magnitude', fontsize=9)
    fig.subplots_adjust(left=.055, right=.986, wspace=.29, top=.83, bottom=.21)
    return fig


def confusion_plot(cnn_matrix: list[list[int]], rf_matrix: list[list[int]], labels: list[str]) -> plt.Figure:
    fig, axes = _figure_axes(2, (12, 4.4))
    cmap = LinearSegmentedColormap.from_list('vibeguard', [PANEL, COBALT, LIME])
    for ax, mat, title in zip(axes, [cnn_matrix, rf_matrix], ['1D CNN · TEST', 'RANDOM FOREST · TEST']):
        a = np.array(mat, dtype=int)
        vmax = max(1, int(a.max()))
        ax.imshow(a, aspect='equal', vmin=0, vmax=vmax, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=17)
        ax.grid(False)
        ax.set_xticks(range(len(labels)), [s.title() for s in labels])
        ax.set_yticks(range(len(labels)), [s.title() for s in labels])
        ax.set_xlabel('Predicted class', fontsize=9, labelpad=10)
        ax.set_ylabel('Actual class', fontsize=9, labelpad=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = int(a[i, j])
                ax.text(j, i, str(val), ha='center', va='center', fontsize=11, fontweight='bold', color=MIDNIGHT if val > vmax * .65 else WHITE)
    fig.subplots_adjust(left=.065, right=.97, wspace=.28, top=.85, bottom=.17)
    return fig


def training_plot(history: dict[str, Any]) -> plt.Figure:
    rows = history.get('epochs', [])
    if not rows:
        raise ValueError('Training history has no epochs')
    ep = [int(r['epoch']) for r in rows]
    fig, axes = _figure_axes(2, (12, 3.9))
    axes[0].plot(ep, [float(r['train_loss']) for r in rows], label='Train loss', color=COBALT, lw=2.5)
    axes[0].plot(ep, [float(r['val_loss']) for r in rows], label='Validation loss', color=LIME, lw=2.5)
    axes[0].set_title('LOSS / LEARNING CURVE', loc='left', fontsize=11, fontweight='bold', color=WHITE, pad=16)
    axes[0].set_xlabel('Epoch', fontsize=9)
    axes[1].plot(ep, [float(r['train_macro_f1']) for r in rows], label='Train macro-F1', color=COBALT, lw=2.5)
    axes[1].plot(ep, [float(r['val_macro_f1']) for r in rows], label='Validation macro-F1', color=LIME, lw=2.5)
    axes[1].set_title('MODEL QUALITY / MACRO-F1', loc='left', fontsize=11, fontweight='bold', color=WHITE, pad=16)
    axes[1].set_xlabel('Epoch', fontsize=9)
    axes[1].set_ylim(0, 1.07)
    chosen = history.get('selected_epoch')
    if chosen:
        for ax in axes:
            ax.axvline(int(chosen), color='#95A9C7', lw=1.2, ls='--', alpha=.75)
    for ax in axes:
        ax.legend(loc='best', facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)
    fig.subplots_adjust(left=.055, right=.98, wspace=.22, top=.84, bottom=.18)
    return fig


def architecture() -> None:
    st.markdown('''<div class="vg-arch-grid">
      <div class="vg-arch-step"><div class="n">01 / SOURCE</div><h3>CWRU dataset</h3><p>16 verified MATLAB files · 12 kHz drive-end vibration · SHA-256 checksums.</p></div>
      <div class="vg-arch-step"><div class="n">02 / PROCESS</div><h3>Signal windows</h3><p>2,048 samples · disjoint motor-load splits · window-wise z-score normalization.</p></div>
      <div class="vg-arch-step"><div class="n">03 / INFER</div><h3>Deep learning</h3><p>1D TinyFaultCNN (15,572 params) and Random Forest baseline · four bearing classes.</p></div>
      <div class="vg-arch-step"><div class="n">04 / DELIVER</div><h3>Interactive results</h3><p>Predicted condition · interactive Plotly forensics · downloadable PDF/CSV reports.</p></div>
    </div>
    <div class="vg-host-grid">
      <div class="vg-host"><div class="tag">ENVIRONMENT A / AWS CLOUD INFRASTRUCTURE</div><strong>AWS EC2 + Private S3 + IAM Role</strong><p>Secured production-style cloud deployment using an instance IAM role (VibeGuardEC2Role) and a private S3 bucket (Block Public Access = ON). Accessed via encrypted SSH tunnel; EC2 remains stopped when idle for cost protection.</p></div>
      <div class="vg-host public"><div class="tag">ENVIRONMENT B / CREDENTIAL-FREE PUBLIC DEMO</div><strong>Streamlit Community Cloud</strong><p>Publicly accessible web interface running the exact same frozen TinyFaultCNN checkpoint and official SHA-256-verified CWRU recordings, operating with zero AWS credentials and no connection to private S3.</p></div>
    </div>''', unsafe_allow_html=True)
