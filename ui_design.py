"""Visual presentation layer for VibeGuard Cloud.

This module never changes preprocessing, checkpoint weights, dataset splits,
labels, score calibration, or the underlying ML evaluation. It contains only
Streamlit presentation helpers and deterministic matplotlib charts.

Install: copy into the repository root, then import in app.py.
"""
from __future__ import annotations

from html import escape
from typing import Mapping, Any
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
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
html,body, p, label, button { font-family:'DM Sans',sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family:'Material Symbols Rounded' !important; font-weight:normal !important; font-style:normal !important; letter-spacing:normal !important; text-transform:none !important; }
[data-testid="stHeader"] { background:transparent !important; }
[data-testid="stAppViewBlockContainer"] { max-width:1380px !important; padding:2.2rem 2.5rem 6rem !important; }
[data-testid="stSidebar"] { background:#080C15 !important; border-right:1px solid #1C2B43 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:1.75rem 1.2rem 2rem; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color:#CAD3E4 !important; }
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#F6F8FF !important; }
[data-testid="stSidebarNav"] { background:transparent !important; }
h1,h2,h3 { font-family:'Space Grotesk','DM Sans',sans-serif !important; color:#F6F8FF !important; letter-spacing:-.035em; }
h2 { font-size:clamp(1.4rem, 2.1vw, 2rem) !important; }
h3 { font-size:1.23rem !important; }
a { color:#B4CBFF !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.5rem; border-bottom:1px solid #253248; padding-bottom:10px; overflow:auto; }
[data-testid="stTabs"] button[role="tab"] { color:#A4B1C7 !important; background:#0B1220 !important; background-color:#0B1220 !important; border:1px solid #27354A !important; border-radius:10px !important; padding:12px 17px !important; font-size:.83rem !important; font-weight:750 !important; white-space:nowrap; }
[data-testid="stTabs"] button[role="tab"] p { color:#A4B1C7 !important; font-size:.83rem !important; font-weight:750 !important; margin:0 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#030407 !important; background:#D4FF24 !important; background-color:#D4FF24 !important; border-color:#D4FF24 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color:#030407 !important; font-weight:800 !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:none !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }
[data-testid="stFileUploader"] { background:#0B1220; padding:12px; border-radius:14px; border:1px solid #243248; }
[data-testid="stFileUploaderDropzone"] { background:#0D1726 !important; border:1px dashed #49628A !important; border-radius:12px !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#0D1726 !important; border:1px solid #2D4260 !important; border-radius:12px !important; color:#F6F8FF !important; }
[data-testid="stDataFrame"] { border-radius:13px; overflow:hidden; border:1px solid #26364D !important; }
[data-testid="stExpander"] { background:#0A111E; border:1px solid #25364B; border-radius:12px; }
[data-testid="stAlert"] { border-radius:12px; border:1px solid #2B4561; }
[data-testid="stMarkdownContainer"] p { line-height:1.65; }
[data-testid="stMetric"] { background:#101B2A; padding:18px 20px; border-radius:14px; border:1px solid #243248; }
[data-testid="stMetricValue"] { color:#F6F8FF !important; }
[data-testid="stMetricLabel"] { color:#A4B2C9 !important; }
[data-testid="stImage"] img { border-radius:13px; }
[data-testid="stToolbar"] { background:transparent !important; }
.vg-hero { position:relative; overflow:hidden; background:radial-gradient(circle at 82% 15%,rgba(21,88,201,.40),transparent 38%),linear-gradient(130deg,#0B1424,#030407 68%); border:1px solid #2A3C56; border-radius:20px; padding:2.2rem 2.6rem; margin-bottom:1.4rem; }
.vg-hero:after {content:'';position:absolute;right:-55px;top:18px;width:240px;height:240px;border:1px solid rgba(212,255,36,.22);border-radius:50%;box-shadow:0 0 0 46px rgba(21,88,201,.10),0 0 0 96px rgba(21,88,201,.05);pointer-events:none;}
.vg-eyebrow { font-size:.68rem; font-weight:850; letter-spacing:.2em; color:#D4FF24; text-transform:uppercase; margin-bottom:.75rem; }
.vg-hero h1 { line-height:1.04; font-size:clamp(2.2rem,4.1vw,4.4rem) !important; max-width:900px; margin:.1rem 0 .75rem; font-weight:700; }
.vg-hero h1 span { color:#D4FF24; }
.vg-hero p { max-width:760px; color:#BAC6D8; font-size:.96rem; margin:.6rem 0 1.1rem; }
.vg-pill-row { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.2rem; }
.vg-pill { border:1px solid #345176; border-radius:100px; padding:.36rem .76rem; background:#122038; font-size:.69rem; color:#C5D5F1; font-weight:750; letter-spacing:.05em; }
.vg-pill.accent { background:#D4FF24; border-color:#D4FF24; color:#030407; }
.vg-title { margin:1.5rem 0 .8rem; }
.vg-title p { margin:0 0 .2rem; font-size:.67rem; font-weight:800; color:#D4FF24; letter-spacing:.2em; text-transform:uppercase; }
.vg-title h2 { font-size:clamp(1.45rem,2vw,2rem); margin:0 0 .3rem; }
.vg-title div { font-size:.9rem; color:#93A5C0; max-width:940px; }
.vg-card-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:.7rem 0 1rem; }
.vg-card { border:1px solid #263A52; background:linear-gradient(160deg,#111F31,#0B1220); border-radius:15px; padding:17px 19px; min-width:0; }
.vg-card.featured { background:linear-gradient(145deg,#1558C9,#0B2855); border-color:#438AF0; }
.vg-card .label { color:#ADBED7; font-size:.65rem; letter-spacing:.1em; text-transform:uppercase; font-weight:750; margin-bottom:8px; }
.vg-card .value { font-family:'Space Grotesk',sans-serif; font-size:clamp(1.17rem,1.8vw,1.65rem); color:#F6F8FF; font-weight:700; line-height:1.12; overflow-wrap:anywhere; }
.vg-card.featured .value { color:#D4FF24; }
.vg-card .micro { color:#92A6C5; font-size:.73rem; margin-top:9px; }
.vg-sidebar-brand { display:flex; gap:11px; align-items:center; margin:.1rem 0 1.5rem; }
.vg-monogram { background:#D4FF24;color:#030407; font-family:'Space Grotesk'; font-weight:900; font-size:1rem; letter-spacing:-.08em; padding:10px 12px; border-radius:10px; }
.vg-sidebar-brand strong { font-family:'Space Grotesk'; font-size:1.03rem; letter-spacing:-.04em; }
.vg-sidebar-brand small { display:block; color:#91A0B9; font-size:.67rem; letter-spacing:.15em; font-weight:800; }
.vg-sidebar-box { background:#0E1A2B; border:1px solid #2C4562; border-radius:13px; padding:14px; margin-bottom:13px; }
.vg-sidebar-box .head { font-size:.65rem; color:#8FA6C5; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.vg-sidebar-box .big { margin:8px 0 6px; font-size:.92rem; color:#F3F7FF; font-weight:750; }
.vg-sidebar-box .small { color:#94A5BD; line-height:1.55; font-size:.75rem; overflow-wrap:anywhere; }
.vg-live { display:inline-block;width:8px;height:8px;border-radius:50%;background:#D4FF24;box-shadow:0 0 13px #D4FF24;margin-right:7px; }
.vg-note { border-left:3px solid #1558C9; background:#0C1728; border-radius:0 12px 12px 0; padding:12px 16px; font-size:.82rem; line-height:1.6; color:#AFC0D7; margin:.5rem 0 1.3rem; }
.vg-note b { color:#DCE7FA; }
.vg-result { border:1px solid #365575; background:linear-gradient(120deg,#10253F,#0A111D); border-radius:18px; padding:22px 25px; margin:1rem 0; }
.vg-result .kicker { color:#D4FF24; font-size:.67rem; letter-spacing:.18em; font-weight:800; text-transform:uppercase; }
.vg-result .main { font-family:'Space Grotesk',sans-serif; font-size:clamp(1.85rem,3vw,2.75rem); font-weight:700; color:#F6F8FF; line-height:1.15; margin:.5rem 0 .6rem; overflow-wrap:break-word; }
.vg-result .caption { color:#A9BBD4; font-size:.82rem; }
.vg-result.review { border-color:#A65C50; background:linear-gradient(120deg,#341E23,#0A111D); }
.vg-arch-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:15px 0; }
.vg-arch-step { border-radius:14px; border:1px solid #28415E; background:#0D1727; padding:18px; }
.vg-arch-step .n { color:#D4FF24;font-size:.66rem;font-weight:800;letter-spacing:.13em; }
.vg-arch-step h3 { font-size:1.12rem !important; margin:.5rem 0; }
.vg-arch-step p { color:#A2B3CB;font-size:.78rem;line-height:1.55; }
.vg-host-grid { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:1rem 0; }
.vg-host { border:1px solid #31557E;background:linear-gradient(125deg,#0E284F,#0B1424);border-radius:15px;padding:20px; }
.vg-host.public { border-color:#7C9436;background:linear-gradient(125deg,#23320D,#101710); }
.vg-host .tag { color:#D4FF24;font-size:.65rem;letter-spacing:.15em;font-weight:850; }
.vg-host strong { display:block;margin:.5rem 0;font-size:1.1rem; }
.vg-host p { font-size:.79rem;color:#B8C8DB;line-height:1.6;margin:0; }
.vg-prov-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin:.4rem 0 .85rem; }
.vg-prov-item { border:1px solid #24364E; background:#0D1727; border-radius:12px; padding:12px 14px; min-width:0; }
.vg-prov-item .k { color:#8FA4C2; font-size:.63rem; letter-spacing:.12em; text-transform:uppercase; font-weight:800; margin-bottom:4px; }
.vg-prov-item .v { font-family:'Space Grotesk',sans-serif; color:#F6F8FF; font-size:.95rem; font-weight:700; overflow-wrap:anywhere; }
.vg-prov-item .v.ok { color:#D4FF24; }
.vg-prov-item .v.warn { color:#FF746C; }
@media(max-width:900px) { [data-testid="stAppViewBlockContainer"] { padding:1.1rem .8rem 3rem !important; }.vg-hero {padding:1.55rem 1.25rem;}.vg-card-grid {grid-template-columns:repeat(2,minmax(0,1fr));}.vg-arch-grid {grid-template-columns:repeat(2,minmax(0,1fr));} }
@media(max-width:540px) { .vg-card-grid {grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px;}.vg-card{padding:12px;}.vg-arch-grid,.vg-host-grid,.vg-prov-grid{grid-template-columns:1fr;}.vg-hero:after{display:none;} }
</style>''', unsafe_allow_html=True)


def hero() -> None:
    st.markdown('''<section class="vg-hero">
      <div class="vg-eyebrow">VIBEGUARD / SIGNAL INTELLIGENCE SYSTEM</div>
      <h1>Every vibration<br><span>tells a story.</span></h1>
      <p>Detect seeded motor-bearing faults using a 1D convolutional neural network. Explore verified laboratory signals, locked cross-load evaluation, and a secured AWS implementation.</p>
      <div class="vg-pill-row"><span class="vg-pill accent">LIVE RESEARCH DEMO</span><span class="vg-pill">1D CNN · 15,572 PARAMETERS</span><span class="vg-pill">4 BEARING CONDITIONS</span><span class="vg-pill">AWS EC2 + PRIVATE S3</span></div>
    </section>''', unsafe_allow_html=True)


def section(eyebrow: str, title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="vg-title"><p>{escape(eyebrow)}</p><h2>{escape(title)}</h2><div>{escape(description)}</div></div>',
        unsafe_allow_html=True,
    )


def scope_note() -> None:
    st.markdown('''<div class="vg-note"><b>RESEARCH PROTOTYPE</b> — Evaluated on CWRU laboratory vibration recordings, not independently validated industrial equipment. Class scores are <b>uncalibrated softmax values</b>, not probabilities of correctness. This application does not estimate remaining useful life or replace an engineer's inspection.</div>''', unsafe_allow_html=True)


def sidebar_status(model_sha: str | None, run_id: str, epoch: Any, cnn_f1: float | None, rf_f1: float | None, s3_connected: bool, s3_message: str, confidence_threshold: float, agreement_threshold: float) -> None:
    st.markdown('''<div class="vg-sidebar-brand"><span class="vg-monogram">VG</span><div><strong>VibeGuard Cloud</strong><small>SIGNAL INTELLIGENCE</small></div></div>''', unsafe_allow_html=True)
    if model_sha:
        ckpt = f'<span class="vg-live"></span> Checkpoint verified'
        meta = f'outputs/model.pt · SHA-256 {escape(model_sha)}…'
    else:
        ckpt = 'Checkpoint missing'
        meta = 'Train or restore outputs/model.pt before inference'
    st.markdown(f'<div class="vg-sidebar-box"><div class="head">MODEL STATUS</div><div class="big">{ckpt}</div><div class="small">{meta}</div></div>',unsafe_allow_html=True)
    cloud_title = 'Private S3 connected' if s3_connected else 'Public web demo'
    cloud_desc = s3_message if s3_connected else 'Credential-free public hosting. Official CWRU demonstration files are verified by SHA-256.'
    st.markdown(f'<div class="vg-sidebar-box"><div class="head">HOSTING MODE</div><div class="big">{escape(cloud_title)}</div><div class="small">{escape(cloud_desc)}</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="vg-sidebar-box"><div class="head">EVALUATION SNAPSHOT</div><div class="big">CNN {cnn_f1:.3f} / RF {rf_f1:.3f} Macro-F1</div><div class="small">Held-out 3 HP test · Epoch {escape(str(epoch))} · Run {escape(str(run_id))}</div></div>' if cnn_f1 is not None and rf_f1 is not None else '<div class="vg-sidebar-box"><div class="head">EVALUATION SNAPSHOT</div><div class="small">Evaluation artifacts unavailable</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vg-sidebar-box"><div class="head">REVIEW GUARDRAILS</div><div class="big">Score ≥ {confidence_threshold:.2f}</div><div class="small">Segment agreement ≥ {agreement_threshold:.0%}. Thresholds are research heuristics, not calibrated risk guarantees.</div></div>',unsafe_allow_html=True)


def metrics_row(items: list[tuple[str, str, str]], featured: int | None = None) -> None:
    content = []
    for idx, (label, value, micro) in enumerate(items):
        klass = 'vg-card featured' if featured == idx else 'vg-card'
        content.append(f'<div class="{klass}"><div class="label">{escape(str(label))}</div><div class="value">{escape(str(value))}</div><div class="micro">{escape(str(micro))}</div></div>')
    st.markdown('<div class="vg-card-grid">'+''.join(content)+'</div>', unsafe_allow_html=True)


def result_hero(predicted_display: str, source_file: str, manual_review: bool, note: str = "") -> None:
    label = 'MANUAL REVIEW ADVISED' if manual_review else 'SIGNAL CLASSIFICATION COMPLETE'
    klass = 'vg-result review' if manual_review else 'vg-result'
    st.markdown(f'<div class="{klass}"><div class="kicker">{label}</div><div class="main">{escape(predicted_display)}</div><div class="caption">Source: {escape(source_file)} · {escape(note)}</div></div>',unsafe_allow_html=True)


def provenance_card(source_file: str, channel_key: str, sampling_hz: int, window_size: int, num_windows: int, manual_review: bool) -> None:
    review_text = "Flagged for review" if manual_review else "Not flagged"
    review_cls = "v warn" if manual_review else "v ok"
    khz = f"{sampling_hz / 1000:g} kHz"
    st.markdown(
        f'''<div class="vg-prov-grid">
          <div class="vg-prov-item"><div class="k">RECORDING</div><div class="v">{escape(str(source_file))}</div></div>
          <div class="vg-prov-item"><div class="k">CHANNEL</div><div class="v">{escape(str(channel_key))}</div></div>
          <div class="vg-prov-item"><div class="k">SAMPLING</div><div class="v">{escape(khz)}</div></div>
          <div class="vg-prov-item"><div class="k">PREPROCESSING</div><div class="v">Z-score · Clamped</div></div>
          <div class="vg-prov-item"><div class="k">WINDOW SIZE</div><div class="v">{window_size:,} ({num_windows} segs)</div></div>
          <div class="vg-prov-item"><div class="k">REVIEW</div><div class="{review_cls}">{review_text}</div></div>
        </div>''',
        unsafe_allow_html=True,
    )


def eval_limitation_note() -> None:
    st.markdown(
        '''<div class="vg-note"><b>BENCHMARK LIMITATION</b> — The locked 3 HP test set comprises only <b>four related held-out recordings</b> from the same CWRU laboratory test rig (one recording per class, 200 windows each). A 100% test macro-F1 demonstrates cross-load consistency across 0/1 HP → 3 HP on seeded 0.007-inch EDM faults, <b>not</b> real-world industrial reliability across different machines or natural wear.</div>''',
        unsafe_allow_html=True,
    )


def _figure_axes(n: int, size: tuple[float,float] = (13.2,3.8)):
    fig, axes = plt.subplots(1,n,figsize=size,dpi=125,facecolor=MIDNIGHT)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED,labelsize=8,length=0,pad=7)
        ax.grid(color='#29405E',linestyle=':',linewidth=.75,alpha=.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(WHITE)
    return fig,axes


def score_plot(scores: Mapping[str,float], display_names: Mapping[str,str], labels: list[str]) -> plt.Figure:
    """Readable, horizontally oriented 4-class score comparison."""
    fig,axes=_figure_axes(1,(9,3.3))
    ax=axes[0]
    values=[float(scores[k]) for k in labels]
    selected=int(np.argmax(values))
    colors=[LIME if i==selected else COBALT for i in range(len(labels))]
    ypos=np.arange(len(labels))
    ax.barh(ypos,values,color=colors,height=.54)
    ax.set_yticks(ypos,[display_names[k].split(' (')[0] for k in labels]); ax.invert_yaxis()
    ax.set_xlim(0,1.08)
    ax.set_xticks([0,.25,.5,.75,1.0],['0','.25','.50','.75','1.0'])
    for i,v in enumerate(values):
        ax.text(min(v+.018,.995),i,f'{v:.4f}',va='center',fontsize=10,fontweight='bold',color=WHITE if i!=selected else LIME)
    ax.set_xlabel('Mean uncalibrated softmax score',fontsize=9,labelpad=12)
    fig.subplots_adjust(left=.23,right=.98,top=.91,bottom=.21)
    return fig


def signal_plot(raw_signal:np.ndarray, first_window_norm:np.ndarray, sampling_hz:int=12000) -> plt.Figure:
    """Three dark signal panels with consistent colors, axes and annotations."""
    fig,axes=_figure_axes(3,(14.2,3.9))
    n=min(len(raw_signal),4096)
    t=np.arange(n)/float(sampling_hz)
    axes[0].plot(t,raw_signal[:n],color=COBALT,lw=.8)
    axes[0].set_title('01  /  RAW VIBRATION',loc='left',fontsize=11,fontweight='bold',color=WHITE,pad=15)
    axes[0].set_xlabel('Time (seconds)',fontsize=9)
    axes[0].set_ylabel('Acceleration (g)',fontsize=9)
    axes[1].plot(first_window_norm,color=LIME,lw=.75)
    axes[1].set_title('02  /  NORMALIZED WINDOW',loc='left',fontsize=11,fontweight='bold',color=WHITE,pad=15)
    axes[1].set_xlabel('Sample index',fontsize=9)
    axes[1].set_ylabel('Z-score',fontsize=9)
    freqs=np.fft.rfftfreq(len(first_window_norm),d=1.0/sampling_hz)
    mag=np.abs(np.fft.rfft(first_window_norm))/len(first_window_norm)
    axes[2].plot(freqs/1000,mag,color='#83B6FF',lw=1.1)
    axes[2].fill_between(freqs/1000,mag,alpha=.18,color=COBALT)
    axes[2].set_title('03  /  FREQUENCY SPECTRUM',loc='left',fontsize=11,fontweight='bold',color=WHITE,pad=15)
    axes[2].set_xlabel('Frequency (kHz)',fontsize=9)
    axes[2].set_ylabel('FFT magnitude',fontsize=9)
    fig.subplots_adjust(left=.055,right=.986,wspace=.29,top=.83,bottom=.21)
    return fig


def confusion_plot(cnn_matrix:list[list[int]],rf_matrix:list[list[int]],labels:list[str]) -> plt.Figure:
    fig,axes=_figure_axes(2,(12,4.4))
    cmap=LinearSegmentedColormap.from_list('vibeguard',[PANEL,COBALT,LIME])
    for ax,mat,title in zip(axes,[cnn_matrix,rf_matrix],['1D CNN · TEST','RANDOM FOREST · TEST']):
        a=np.array(mat,dtype=int)
        vmax=max(1,int(a.max()))
        ax.imshow(a,aspect='equal',vmin=0,vmax=vmax,cmap=cmap)
        ax.set_title(title,fontsize=11,fontweight='bold',pad=17)
        ax.grid(False)
        ax.set_xticks(range(len(labels)),[s.title() for s in labels]);ax.set_yticks(range(len(labels)),[s.title() for s in labels])
        ax.set_xlabel('Predicted class',fontsize=9,labelpad=10);ax.set_ylabel('Actual class',fontsize=9,labelpad=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val=int(a[i,j]);ax.text(j,i,str(val),ha='center',va='center',fontsize=11,fontweight='bold',color=MIDNIGHT if val>vmax*.65 else WHITE)
    fig.subplots_adjust(left=.065,right=.97,wspace=.28,top=.85,bottom=.17)
    return fig


def training_plot(history:dict[str,Any]) -> plt.Figure:
    rows=history.get('epochs',[])
    if not rows:
        raise ValueError('Training history has no epochs')
    ep=[int(r['epoch']) for r in rows]
    fig,axes=_figure_axes(2,(12,3.9))
    axes[0].plot(ep,[float(r['train_loss']) for r in rows],label='Train loss',color=COBALT,lw=2.5)
    axes[0].plot(ep,[float(r['val_loss']) for r in rows],label='Validation loss',color=LIME,lw=2.5)
    axes[0].set_title('LOSS / LEARNING CURVE',loc='left',fontsize=11,fontweight='bold',color=WHITE,pad=16)
    axes[0].set_xlabel('Epoch',fontsize=9)
    axes[1].plot(ep,[float(r['train_macro_f1']) for r in rows],label='Train macro-F1',color=COBALT,lw=2.5)
    axes[1].plot(ep,[float(r['val_macro_f1']) for r in rows],label='Validation macro-F1',color=LIME,lw=2.5)
    axes[1].set_title('MODEL QUALITY / MACRO-F1',loc='left',fontsize=11,fontweight='bold',color=WHITE,pad=16)
    axes[1].set_xlabel('Epoch',fontsize=9);axes[1].set_ylim(0,1.07)
    chosen=history.get('selected_epoch')
    if chosen:
        for ax in axes: ax.axvline(int(chosen),color='#95A9C7',lw=1.2,ls='--',alpha=.75)
    for ax in axes:
        ax.legend(loc='best',facecolor=PANEL,edgecolor=BORDER,labelcolor=WHITE,fontsize=8)
    fig.subplots_adjust(left=.055,right=.98,wspace=.22,top=.84,bottom=.18)
    return fig


def architecture() -> None:
    st.markdown('''<div class="vg-arch-grid">
      <div class="vg-arch-step"><div class="n">01 / SOURCE</div><h3>CWRU dataset</h3><p>16 verified MATLAB files · 12 kHz drive-end vibration · SHA-256 checksums.</p></div>
      <div class="vg-arch-step"><div class="n">02 / PROCESS</div><h3>Signal windows</h3><p>2,048 samples · disjoint motor-load splits · window-wise normalization.</p></div>
      <div class="vg-arch-step"><div class="n">03 / INFER</div><h3>Deep learning</h3><p>1D TinyFaultCNN and Random Forest baseline · four bearing classes.</p></div>
      <div class="vg-arch-step"><div class="n">04 / DELIVER</div><h3>Interactive results</h3><p>Predicted condition · class scores · signal plots · evaluation evidence.</p></div>
    </div>
    <div class="vg-host-grid">
      <div class="vg-host"><div class="tag">ENVIRONMENT A / AWS CLOUD</div><strong>EC2 + Private S3 + IAM</strong><p>Secured application deployment with an instance IAM role and private artifact storage. Accessed via SSH tunnel; EC2 may be stopped for cost protection.</p></div>
      <div class="vg-host public"><div class="tag">ENVIRONMENT B / PUBLIC DEMO</div><strong>Streamlit Community Cloud</strong><p>Public web interface, using the same checkpoint and official SHA-256-verified demo files, without AWS credentials or access to private S3.</p></div>
    </div>''',unsafe_allow_html=True)
