# VibeGuard Cloud: Load-Shift-Aware Industrial Motor Bearing Fault Classification Using 1D Deep Learning and AWS

**Final-Year Cloud Computing & Deep Learning Project**  
**Repository:** `https://github.com/shaziyasultanablr-byte/Cloud-Computing-Final`  
**Public Web App (Streamlit Community Cloud):** [`https://vibeguard-cloud.streamlit.app`](https://vibeguard-cloud.streamlit.app)

**AWS Cloud Infrastructure (`us-east-1`):** EC2 `i-0c8d53060f56c451d` (`t3.small`, Ubuntu 24.04.5 LTS) + `20 GiB gp3` EBS (`vol-09af634eea204d82d`) + Private S3 (`s3://vibeguard-cloud-2026-810995308548`) + IAM Role (`VibeGuardEC2Role`)

---

## 1. Project Overview

**VibeGuard Cloud** is a reproducible deep learning and cloud engineering research prototype that classifies electric-motor bearing condition from drive-end accelerometer vibration signals into four categories:

1. **Normal Baseline (`normal`)**
2. **Inner Race Fault — 0.007 in (`inner`)**
3. **Ball Fault — 0.007 in (`ball`)**
4. **Outer Race Fault @ 6:00 — 0.007 in (`outer`)**

Unlike random window-level splits that leak identical recording conditions across train and test sets, VibeGuard Cloud evaluates **load-shift generalization**:
- **Training Split (`0 HP` & `1 HP`):** 8 recordings (`97, 98, 105, 106, 118, 119, 130, 131`) — `673` non-overlapping windows
- **Validation Split (`2 HP`):** 4 recordings (`99, 107, 120, 132`) — `377` non-overlapping windows (used exclusively for checkpoint selection)
- **Locked Final Test Split (`3 HP`):** 4 recordings (`100, 108, 121, 133`) — `378` non-overlapping windows (evaluated once)

The project compares a lightweight PyTorch 1D Convolutional Neural Network (**`TinyFaultCNN`**, `15,572` trainable parameters) against a classical **Random Forest** baseline (`200` trees trained on `18` time- and frequency-domain features), flags uncertain or inconsistent predictions via a pre-specified **manual-review heuristic**, and serves interactive diagnostics through a **Streamlit web dashboard** designed for **AWS EC2 + EBS + Private Amazon S3** deployment over an encrypted SSH tunnel.

> **Research Scope & Safety Disclaimer:** VibeGuard Cloud analyzes laboratory recordings from the Case Western Reserve University (CWRU) Bearing Data Center (`*_DE_time`, nominal 12 kHz). Model softmax scores are **uncalibrated confidence estimates**, not true posterior probabilities. The prototype does not estimate Remaining Useful Life (RUL) or replace certified industrial vibration diagnostics.

---

## 2. Repository Structure

```text
Cloud-Computing-Final/
├── AGENTS.md                        # Mandatory operational, security, and scientific integrity rules
├── AGENT_MASTER_PLAN.md             # Authoritative master execution specification
├── README.md                        # Project overview, reproduction guide, and measured results
├── .gitignore                       # Excludes .venv/, raw .mat binaries, and private keys/secrets
├── requirements.txt                 # Runtime Python dependencies
├── requirements-dev.txt             # Testing and report/slide generation dependencies
├── requirements.lock.txt            # Exact locked package versions from verified Python 3.11 venv
├── download_data.py                 # Official CWRU .mat downloader with channel & binary verification
├── data_utils.py                    # Manifest loader, split-leakage checks, windowing, z-score & features
├── model.py                         # 1D TinyFaultCNN architecture, checkpoint I/O, manual-review heuristic
├── train.py                         # Reproducible training & locked 3 HP cross-load evaluation pipeline
├── cloud_sync.py                    # Credential-free boto3 S3 sync & verification utility (IAM Role)
├── app.py                           # Streamlit web application (upload, inference, plots, S3 status)
├── ui_design.py                     # Midnight/Cobalt/Lime presentation layer, CSS theme, and interactive Plotly charts
├── report_utils.py                  # Downloadable PDF and CSV diagnostic report generator (ReportLab)
├── apply_redesign.py                # Reproducible UI redesign integration script
├── scripts/
│   ├── validate_dataset.py          # Generates SHA-256 provenance manifest (docs/dataset_checksums.csv)
│   ├── generate_architecture_diagram.py
│   └── generate_docs.py             # Generates FINAL_REPORT.pdf and FINAL_PRESENTATION.pptx from metrics
├── tests/
│   ├── test_core.py                 # Unit & integration tests (shapes, NaN/Inf, CSV/MAT guards, checkpoint)
│   ├── test_splits.py               # Programmatic verification of recording- and load-disjoint splits
│   └── test_ui_upgrade.py           # Automated UI workflow, interactive Plotly, and PDF/CSV report export tests
├── docs/
│   ├── DATASET_MANIFEST.csv         # Immutable 16-file CWRU split and metadata specification
│   ├── dataset_checksums.csv        # Verified SHA-256 checksums, channel keys, and window counts
│   ├── architecture.png             # System architecture diagram
│   ├── data_dictionary.md           # Dataset schema, channel provenance, and CWRU 99.mat hygiene notes
│   ├── methodology.md               # Preprocessing, model architecture, and cross-load evaluation protocol
│   ├── AWS_DEPLOYMENT.md            # Step-by-step AWS Free Plan S3, IAM Role, EC2, and SSH tunnel guide
│   ├── COST_AND_CLEANUP.md          # Budget guardrails, credit monitoring, and resource teardown checklist
│   ├── runbook.md                   # Live demonstration and troubleshooting runbook
│   ├── SUBMISSION_CHECKLIST.md      # Gate-by-gate verification status and artifact manifest
│   ├── FINAL_REPORT.md              # Full academic project report source (Markdown)
│   ├── FINAL_REPORT.pdf             # Full academic project report (PDF)
│   ├── FINAL_PRESENTATION.pptx      # 11-slide university defense presentation (PowerPoint)
│   └── VibeGuard_Cloud_Project_Summary.pdf
├── outputs/
│   ├── model.pt                     # Trained 1D TinyFaultCNN checkpoint (selected on 2 HP validation)
│   ├── metrics.json                 # Window-level train/val/test metrics for 1D CNN and Random Forest
│   ├── file_level_metrics.json      # Recording-level predictions, scores, and manual-review flags
│   ├── training_history.json        # Epoch-by-epoch loss and macro-F1 history
│   ├── run_manifest.json            # Full reproducibility manifest (environment, seed, SHA-256 hashes)
│   ├── test_predictions.csv         # Traceable per-window predictions on held-out 3 HP test split
│   ├── confusion_matrix.png         # 3 HP test confusion matrices and per-class F1 comparison
│   └── training_loss.png            # Training loss and validation macro-F1 dynamics
└── .github/workflows/ci.yml         # Automated GitHub Actions CI for offline unit & split tests
```

---

## 3. Quickstart & Full Reproduction (Windows PowerShell / Linux Bash)

### 3.1 Create Isolated Python 3.11 Environment & Run Unit Tests
```powershell
git clone https://github.com/shaziyasultanablr-byte/Cloud-Computing-Final.git
cd Cloud-Computing-Final

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt

# Run offline unit and split-disjointness test suite
.\.venv\Scripts\python.exe -m pytest -q
```

Example verified test output:
```text
...............                                                          [100%]
15 passed in 4.83s
```

### 3.2 Acquire Official CWRU Dataset, Validate Checksums & Train Models
```powershell
# 1. Download the 16 official CWRU MATLAB files into data/raw/{normal,inner,ball,outer}/
.\.venv\Scripts\python.exe download_data.py

# 2. Verify channel identities (*_DE_time), sample counts, and SHA-256 checksums
.\.venv\Scripts\python.exe scripts/validate_dataset.py

# 3. Train 1D TinyFaultCNN + Random Forest baseline (seed=42, 14 epochs, batch_size=64)
.\.venv\Scripts\python.exe train.py --epochs 14 --batch-size 64
```

### 3.3 Launch the Streamlit Dashboard (Two Deployment Versions)

#### Version 1 — AWS EC2 + Private S3 + Encrypted SSH Tunnel (`us-east-1`)
When running on the provisioned AWS EC2 instance (`i-0c8d53060f56c451d`), the application connects to the private bucket `s3://vibeguard-cloud-2026-810995308548` via the attached `VibeGuardEC2Role` instance profile (zero static access keys) and is accessed over an encrypted SSH tunnel:
```powershell
# Start EC2 instance (when needed for live university demonstration)
aws ec2 start-instances --instance-ids i-0c8d53060f56c451d --region us-east-1

# Open encrypted SSH tunnel to EC2 loopback (127.0.0.1:8501)
ssh -F NUL -i "$HOME\.ssh\vibeguard-ec2-key.pem" -L 8501:127.0.0.1:8501 ubuntu@<EC2_PUBLIC_IP>
```
Then open **`http://localhost:8501`** in your browser. See [`docs/AWS_DEPLOYMENT.md`](docs/AWS_DEPLOYMENT.md) and [`docs/COST_AND_CLEANUP.md`](docs/COST_AND_CLEANUP.md) for full details.

#### Version 2 — Public Web Hosting on Streamlit Community Cloud (`$0` Cost, Zero AWS Credentials)
`app.py` is also configured for zero-credential public hosting on **Streamlit Community Cloud**:
- **Public URL:** [`https://vibeguard-cloud.streamlit.app`](https://vibeguard-cloud.streamlit.app)
- **One-Click Deployment Link:** [`https://share.streamlit.io/deploy?repository=shaziyasultanablr-byte/Cloud-Computing-Final&branch=main&mainModule=app.py`](https://share.streamlit.io/deploy?repository=shaziyasultanablr-byte/Cloud-Computing-Final&branch=main&mainModule=app.py)
- **How It Works Without AWS Keys or Git-Committed Raw Data:**
  1. Uses `.python-version` (`3.11`) and `requirements.txt` (direct PEP 508 CPU wheel URLs for `torch-2.5.1+cpu`, `numpy==1.26.4`, `pyarrow==18.1.0`) to install lightweight CPU-only wheels within Streamlit Community Cloud's resource limits without index resolution conflicts.
  2. Loads the verified `outputs/model.pt` checkpoint (`SHA-256: 3cd1ac3b823ea903...`) once into memory via `@st.cache_resource`.
  3. Automatically fetches and caches the 4 official `3 HP` demonstration recordings (`100.mat`, `108.mat`, `121.mat`, `133.mat`) on demand from the official CWRU Bearing Data Center (`https://engineering.case.edu/sites/default/files/{file_id}.mat`) and verifies their SHA-256 checksums before running inference.

#### Local Workstation Execution
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

### 3.4 Visual Design System (Midnight / Cobalt / Lime UI)
The dashboard presentation layer ([`ui_design.py`](ui_design.py) and [`.streamlit/config.toml`](.streamlit/config.toml)) implements an industrial AI signal-intelligence visual hierarchy while keeping all model checkpoints, preprocessing, and evaluation artifacts strictly unchanged:
- **Color Palette:** Midnight Black (`#030407`) primary canvas and chart backdrop, Electric Cobalt Blue (`#1558C9`) signal traces and featured metric cards, and Acid Lime (`#D4FF24`) active tab navigation, primary status chips, and winning-class highlights.
- **Typography & Layout:** `Space Grotesk` display headings/numeric metrics paired with `DM Sans` body typography, responsive CSS grid cards (`4 → 2 → 1` columns across desktop `1440px`, tablet `900px`, and mobile `390px`), and full-width untruncated fault-prediction banners (`vg-result`).
- **Verified Browser Screenshots:**
  - [`docs/evidence/redesign_desktop_tab1.png`](docs/evidence/redesign_desktop_tab1.png) — Tab `01 / Diagnose` (Desktop `1440px`, `108.mat` Inner Race Fault `0.9953` inference, 4-class score bar chart, provenance card, and 3-panel signal forensics)
  - [`docs/evidence/redesign_desktop_tab2.png`](docs/evidence/redesign_desktop_tab2.png) — Tab `02 / Evaluate` (Desktop `1440px`, cross-load macro-F1 cards, benchmark limitation banner, 4-recording audit table, dark integer confusion matrices, and training curves)
  - [`docs/evidence/redesign_desktop_tab3.png`](docs/evidence/redesign_desktop_tab3.png) — Tab `03 / Architecture` (Desktop `1440px`, 4-step pipeline cards, AWS vs. Public Streamlit Cloud environment comparison, and methodological boundaries)
  - [`docs/evidence/redesign_mobile_390px.png`](docs/evidence/redesign_mobile_390px.png) — Responsive Mobile (`390px`) layout verification

---

## 4. Actual Measured Results (`run_id: vibeguard-20261102T092858Z-s42`)

### 4.1 Window-Level Performance Summary
| Model | Split | Motor Load | Recordings | Windows | Accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|---:|
| **1D TinyFaultCNN (`15,572` params)** | Train | 0 & 1 HP | 8 | 673 | **100.00%** | **1.0000** |
| **1D TinyFaultCNN (`15,572` params)** | Validation | 2 HP | 4 | 377 | **100.00%** | **1.0000** |
| **1D TinyFaultCNN (`15,572` params)** | **Held-Out Test** | **3 HP** | **4** | **378** | **100.00%** | **1.0000** |
| **Random Forest (`200` trees, `18` features)** | Train | 0 & 1 HP | 8 | 673 | **100.00%** | **1.0000** |
| **Random Forest (`200` trees, `18` features)** | Validation | 2 HP | 4 | 377 | **100.00%** | **1.0000** |
| **Random Forest (`200` trees, `18` features)** | **Held-Out Test** | **3 HP** | **4** | **378** | **100.00%** | **1.0000** |

### 4.2 Recording-Level Predictions on Locked 3 HP Test Split (`outputs/file_level_metrics.json`)
| File ID | Filename | True Class | Windows | 1D CNN Pred | 1D CNN Top Score | 1D CNN Margin | RF Pred | RF Top Score | RF Margin | Manual Review |
|---:|---|---|---:|---|---:|---:|---|---:|---:|---|
| `100` | `100.mat` | `normal` | 200 | `normal` | `0.9869` | `0.9820` | `normal` | `0.9785` | `0.9576` | `False` |
| `108` | `108.mat` | `inner` | 60 | `inner` | `0.9953` | `0.9932` | `inner` | `0.7273` | `0.5005` | `False` |
| `121` | `121.mat` | `ball` | 59 | `ball` | `0.9890` | `0.9840` | `ball` | `0.9870` | `0.9794` | `False` |
| `133` | `133.mat` | `outer` | 59 | `outer` | `0.9851` | `0.9777` | `outer` | `0.9981` | `0.9972` | `False` |

**Key Analytical Observation:** While both models classify all 378 held-out 3 HP windows and all 4 test recordings accurately on this single-rig benchmark, the Random Forest baseline suffers substantial vote-share dilution on the 3 HP Inner Race recording (`108.mat`: `0.7273` for `inner` vs. `0.2268` for `ball`, margin `0.5005`), hovering just above the `0.70` manual-review threshold. By learning multi-scale shift-invariant impulse filters directly from normalized waveforms, `1D TinyFaultCNN` maintains a `0.9953` mean score (`0.9932` margin) on `108.mat`. See [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) for a critical discussion of CWRU single-rig limitations and `99.mat` channel-hygiene remediation.
