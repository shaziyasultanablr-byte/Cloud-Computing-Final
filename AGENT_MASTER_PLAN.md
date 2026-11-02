# VibeGuard Cloud — Master Execution Plan for an Autonomous Coding Agent

**Version:** 1.0 | **Planning date:** 2 November 2026 | **Deadline:** four calendar days | **Expected cash spend:** AED 0, only while the confirmed AWS Free Plan remains in effect and no paid-plan upgrade is made.

## 0. Mission, boundaries, and verified starting point

Build, test, document, and demonstrate a complete final-year mini-project: **VibeGuard Cloud: Load-Shift-Aware Industrial Motor Bearing Fault Classification Using 1D Deep Learning and AWS**.

**What the system does:** ingests accelerometer vibration data from the Case Western Reserve University (CWRU) bearing laboratory, converts recordings into short signal windows, uses a 1D convolutional neural network to classify **normal / inner-race fault / ball fault / outer-race fault**, tests performance when motor load differs from training conditions, compares with a Random Forest baseline, shows a demo-only manual-review flag when predictions are uncertain/inconsistent, and serves results through an EC2-hosted Streamlit dashboard. Store authorized private copies of data/model artifacts in S3, and put all *project-owned* reusable artifacts, code, reports, reproducibility manifests, test results, and sanitized evidence into GitHub.

**Verified starting state (do not infer that deployment has occurred):**

- GitHub destination is `shaziyasultanablr-byte/Cloud-Computing-Final`, on branch `main`. It was **empty** when inspected. The current assistant's connected GitHub integration has **read-only** permission; a different authenticated, write-authorized session is needed to push.
- The AWS screenshots show **Free Plan**, **USD 100 active Free Tier credit**, **USD 0 used**, with credit expiry 14 October 2027. The **Free Plan ends earlier**, normally 6 months after account creation or when credit is consumed. Recheck current account state at execution time. Never click *Upgrade Plan*.
- An earlier ZIP contains `download_data.py`, `data_utils.py`, `model.py`, `train.py`, `app.py`, `cloud_sync.py`, `requirements.txt`, and templates. This handoff package includes those **starter files only**; they have not been verified against the full real dataset or a live AWS deployment.
- University source: https://engineering.case.edu/bearingdatacenter/download-data-file ; study is not new/fault predictive equipment physics and cannot claim industrial certification or true remaining useful life prediction.

**Hard success definition:** From a clean GitHub checkout, a reviewer can follow the README, reproduce a real-data training run, inspect honest metrics, and run a web demo on an AWS EC2 instance that reads/writes to a private S3 bucket. The repo holds a final report, slide deck, screenshots, reproducibility manifest, working code and test outputs. Report and slides must contain **only actual measured performance**.

## 1. Critical security step — do before any other work

A password for the AWS account was pasted into the conversation. **Treat it as compromised.** Ask the *authorized account owner* to sign in to AWS themselves, **change the AWS root password immediately** and **enable MFA for the root user**. Review recent security activity if anything looks suspicious. **Never use or repeat that shared password**, even if present in the chat. Don't copy it into GitHub, a model prompt, a terminal, a config file, `.env`, report, screenshot or error log. Don't ask an AI agent to enter it.

Use a normal IAM identity/role for day-to-day actions; let the authorized user authenticate interactively and approve IAM/security actions. GitHub authentication must be through GitHub CLI/browser, Git Credential Manager or a properly-scoped short-lived credential **outside the repository**, never a password embedded in `git remote` URL or source code. Do not expose AWS root credentials/access keys or a `.pem` EC2 private key. Ensure repository settings allow secret scanning and push protection where available. Repo is **public**: carefully inspect all files for secrets/personal data before pushing.

If the owner has not confirmed the password was changed, **stop all AWS provisioning**. Local code/research preparation may continue without accessing AWS.

## 2. Agent operating contract

1. Act as a senior ML/MLOps/cloud engineer and final-year academic project lead. Deliver a finished, demonstrable implementation, not an outline.
2. Use the existing starter files as a starting point; validate and repair rather than trust them blindly. Read `AGENTS.md` and this file first.
3. Work in this repository only: `https://github.com/shaziyasultanablr-byte/Cloud-Computing-Final.git`. **Do not create a second repository.** Do not overwrite unrelated work. Verify write access before the first push. If denied, record the blocker and give exact owner steps to grant collaborator or authorize GitHub CLI locally.
4. Commit and push once each milestone passes. Log commit SHA and date. If no GitHub push succeeds, mark project **NOT SAVED** even if local files exist. Never claim success from a local-only folder.
5. Work without paid services or account upgrades. Before launching every AWS resource, confirm it is allowed in the account's Free Plan and budget. Stop when current plan is not Free Plan or unexpected charges arise. No GPU, NAT Gateway, load balancer, Elastic IP reservation, managed notebook, always-on SageMaker endpoint or paid database.
6. Record commands, versions, seeds, dataset hashes and experiment configuration. Use UTC timestamps in evidence files.
7. Treat the 3 HP test recordings as **frozen final-test evidence**. Never tune architecture, thresholds, epochs or preprocessing using this test set. If the experiment is rerun for infrastructure reasons, label each run separately and document the reason. Distinguish trial/smoke tests from official final runs.
8. No fabricated data, metrics, cloud screenshots, demo URL, usage charges, citations, or successful tests.
9. If any action is blocked by user login, repo permissions, data access, AWS quota, instance eligibility or resource cost, complete safe offline tasks and log the blocker plus a safe fallback.
10. Finish with the commit SHA, exact instructions for demonstrator, artifact list, actual AWS credit usage, unresolved items and cleanup status.

## 3. Four-day project schedule with gates

| Day | Focus | Mandatory exit gate | Commit suggestion |
|---|---|---|---|
| D1 | Account safety, repo, dataset, local environment, infrastructure | New clone installs + tests; 16 `.mat` files validated; private S3/IAM/eligible EC2 planned/created if authorized | `D1: reproducible dataset and AWS baseline` |
| D2 | CNN, RF baseline, split validation, reproducible final evaluation | Real-data `metrics.json`, per-class scores, confusion matrix, trained model, output hashes | `D2: frozen cross-load evaluation and model` |
| D3 | Private S3, EC2 deployment, secure tunnel, dashboard | Working 4-class inference; S3 verification; end-to-end screen captures | `D3: verified cloud demo and infrastructure docs` |
| D4 | Academic report, slides, user instructions, tests, GitHub audit, cost cleanup | Final report/slides/README validated against real run, demo rehearsed and safely preserved on GitHub | `D4: final submission and evidence` |

**Priority rule:** Never spend time on optional features until every core gate is green. If short on time: get data + honest evaluation + working EC2/S3 Streamlit + documentation, before polish.

## 4. Local setup and GitHub bootstrap (D1)

On Windows PowerShell with Git and Python 3.11 installed and access granted to the target GitHub repository:

```powershell
# Sign into GitHub interactively; do not paste a password into commands.
gh auth login
# Select GitHub.com > HTTPS > authenticate through browser.
gh auth status

git clone https://github.com/shaziyasultanablr-byte/Cloud-Computing-Final.git
cd Cloud-Computing-Final
# This repo may be empty. Copy all contents from this handoff package,
# including hidden .gitignore, into this root; do NOT copy a nested wrapper folder.

git status
# verify remote points to intended repo
git remote -v
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python -c "import torch, scipy, sklearn, streamlit, boto3; print('ENV_READY', torch.__version__)"
```

If PowerShell activation is blocked by local policy, use `.venv\Scripts\python.exe` directly rather than changing machine-wide execution policies. `gh` is optional; Git Credential Manager/browser OAuth is acceptable. Copy project starter files to the empty checkout first, then:

```powershell
git add .
git diff --cached --name-only
# Review every staged path; ensure no secrets/private data.
git commit -m "D1: scaffold VibeGuard Cloud research project"
git push -u origin main
```

**GitHub storage rule:** `src`/scripts, docs, sanitized evidence, finalized small `outputs/model.pt`, `outputs/metrics.json`, images, source manifest and CSV findings are checked in. `data/raw/` and temporary `.venv/` remain excluded: upstream university files should be fetched via source URLs and, for the cloud demo, stored in private S3. This supports reproducibility and avoids questionable redistribution or Git history bloat. If any single required **project-owned** artifact is larger than GitHub's normal 100 MiB file limit, arrange Git LFS (quota-conscious), a GitHub release asset, or commit a content-addressed manifest/reference; document the exception clearly. **Never upload a secret even if the user says “everything in GitHub.”**

Suggested final tree:

```text
Cloud-Computing-Final/
├── AGENTS.md
├── AGENT_MASTER_PLAN.md
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── requirements.lock.txt                    # generated from verified environment
├── download_data.py
├── data_utils.py
├── model.py
├── train.py
├── cloud_sync.py
├── app.py
├── tests/                                  # offline unit/integration smoke tests
├── data/raw/                               # ignored original .mat recordings
├── docs/
│   ├── DATASET_MANIFEST.csv
│   ├── dataset_checksums.csv                # generated after verified download
│   ├── architecture.png
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── AWS_DEPLOYMENT.md
│   ├── COST_AND_CLEANUP.md
│   ├── runbook.md
│   ├── FINAL_REPORT.pdf
│   ├── FINAL_PRESENTATION.pptx
│   ├── VibeGuard_Cloud_Project_Summary.pdf
│   └── evidence/                            # cropped/redacted genuine screenshots
├── outputs/
│   ├── model.pt
│   ├── metrics.json
│   ├── file_level_metrics.json
│   ├── training_history.json
│   ├── run_manifest.json
│   ├── test_predictions.csv
│   ├── confusion_matrix.png
│   └── training_loss.png
└── .github/workflows/ci.yml
```

## 5. Dataset acquisition and provenance (D1)

Use only the officially hosted **CWRU Bearing Data Center**: https://engineering.case.edu/bearingdatacenter/download-data-file. Original signal files are `.mat`. The intended files are listed in `docs/DATASET_MANIFEST.csv`:

| Class | 0 HP train | 1 HP train | 2 HP validation | 3 HP test |
|---|---:|---:|---:|---:|
| Normal | 97 | 98 | 99 | 100 |
| Inner race (0.007 in) | 105 | 106 | 107 | 108 |
| Ball (0.007 in) | 118 | 119 | 120 | 121 |
| Outer race at 6:00 (0.007 in) | 130 | 131 | 132 | 133 |

Approximate RPM by load: 0 HP 1797, 1 HP 1772, 2 HP 1750, 3 HP 1730. **Make sure both the correct class and load match each filename**. Inspect original web tables rather than trusting an unofficial mirror.

The supplied downloader points to URLs such as `https://engineering.case.edu/sites/default/files/105.mat`; it might be blocked by anti-bot controls. Run:

```bash
python download_data.py
```

If one or more downloads fail: manually click the proper `.mat` file on CWRU's website, place it into its expected `data/raw/{normal,inner,ball,outer}/` path, and rerun the validator. **Never substitute synthetic data in the official final-test run.** For each file, validate: actual MATLAB format (not an HTML error page), required `*_DE_time` channel, array length >= 2048, finite values, correct metadata and filename, file length. Confirm sampling frequency for each selected group from university documentation before any time/frequency-axis claims; if a mismatch is found, implement documented resampling consistently before training.

Run `python scripts/validate_dataset.py` to generate `docs/dataset_checksums.csv` with `file,label,load_hp,split,num_samples,bytes,sha256,original_url,checked_utc`, then commit it. Keep original binaries locally and/or privately in S3; don't commit them without verified redistribution rights. If network download is blocked on EC2, download locally and transfer through SCP/SSH to EC2, or use the S3 object upload. Avoid broad anonymous internet mirrors unless provenance is verified.

**Pass criterion:** All sixteen files load successfully, SHA-256 is recorded, each label has all four loads, same intended sensor channel, and split manifest is immutable.

## 6. Research protocol: preprocessing, leakage control and novelty (D1–D2)

**Scope:** Four-class supervised diagnosis of existing lab faults, not prediction of failure date or remaining useful life. Distinctive study question: **Does an accelerometer-based model trained on 0 and 1 HP retain correct classification at a previously unseen 3 HP operating load?** The 2 HP condition is reserved for validation.

- Use a single sensor location (drive-end `*_DE_time`) consistently.
- Check file sampling rates and channel provenance first. Nominal target 12 kHz. Never silently mix 12 kHz/48 kHz as if homogeneous.
- Deterministically segment into nonoverlapping **2,048-sample** windows (stride 2,048), maximum **200 windows per recording** for CPU feasibility. Record chosen index ranges. No leakage of windows across train, validation and test; groups are recording file + motor load.
- Standardize **per window** to zero mean/unit variance, clamp outliers to a documented range (starter uses `[-10, 10]`). Compute preprocessing identically for all splits and uploaded inference data.
- Train: 8 recordings (0/1 HP), validation: 4 recordings (2 HP), final: 4 recordings (3 HP). Lock the split before training. Validation and final test have one recording per class; report this major limitation.
- Prefer random seed **42** and log software/CPU details. Do not use 3 HP for threshold or hyperparameter tuning, even indirectly.
- One experiment can measure *load-conditional generalization* on the same lab rig; it **cannot** establish robustness to unseen machines/bearing units or general industry conditions. Avoid calling the 16-window subset statistically independent experiments.
- Optional if time remains: plot class-conditional waveform/power-spectrum from training only, with frequency axis verified by sampling metadata.

**Pass criterion:** `test_core.py` passes; a programmatic assertion proves file IDs and loads are disjoint. A trained model can only be finalized after the data pass criterion.

## 7. Model training and locked evaluation (D2)

Use the supplied `TinyFaultCNN`: Conv1D(1→16, k15) + BN/ReLU/MaxPool; Conv1D(16→32,k9) + BN/ReLU/MaxPool; Conv1D(32→64,k5) + BN/ReLU/AdaptiveAvgPool; dropout 0.2; linear four-class head. PyTorch, cross-entropy, Adam 0.001, default batch size 64, up to 14 epochs, CPU to avoid GPU cost. Select the checkpoint with best **2 HP validation macro-F1**, then evaluate once on 3 HP. Train **Random Forest** baseline on the *same training files* using hand-crafted time/frequency-domain features and test it on the *same 3 HP recordings*.

```bash
python train.py --epochs 14 --batch-size 64
python -m pytest -q
```

**Agent must audit/fix existing training script** before declaring completion:

1. Confirm all files really loaded from correct conditions; verify 12 kHz assumption for both normal/fault recordings.
2. Make final run deterministic to practical CPU tolerance, log Python/Torch/sklearn versions, UTC run ID, Git SHA, dataset checksums, preprocess config, epochs, threshold rules, and hardware.
3. Save original CNN model checkpoint under `outputs/model.pt` plus exact labels and window metadata. Avoid unsafe untrusted pickle loading; only load this trusted locally trained checkpoint.
4. Save **window-level** metrics (`accuracy`, `macro_f1`, class-wise `precision/recall/f1/support`, confusion matrix) and **recording-level** per-file predictions using mean of window softmax scores; report file-level accuracy + count (only 4 test recordings).
5. Store `outputs/test_predictions.csv` with true class, predicted class, file ID, load and honest model score (no personal data). Keep final-test outputs traceable to run ID.
6. Save graph(s): train loss, validation macro-F1, test confusion matrix, 4-class metric comparison CNN vs RF. Figures must be legible.
7. Set manual-review flag only using a **pre-specified demo heuristic** or a threshold chosen **on validation**, never on held-out test. A softmax number is **not** a calibrated probability; avoid any claim of calibrated uncertainty.
8. Run sanity tests: label ordering, batch shapes, infer one known normal and one known fault test file, handle upload errors, deterministic processing, and small-data failure cases.
9. If accuracy is low, report it honestly; discuss small sample size, seeded faults, load shift, class confusion and preprocessing. Never fabricate a performance number or train repeatedly on held-out test to improve the score.

At the end of D2 commit all **small, project-owned final run artifacts** plus `requirements.lock.txt`. Save a frozen copy of final metric JSON; never silently overwrite it in later runs. If an improved model is necessary, use `outputs/runs/<run_id>/` and document which run was official.

**Pass criterion:** From one explicit command the agent produces a valid model and correct validation/test outputs from **real CWRU data**. The code, run metadata, metrics and model are checked into the repository.

## 8. AWS cost protection and service selection (D1/D3)

Reconfirm screenshots: Free Plan and USD 100 credit from earlier account evidence. At runtime AWS Billing → **Credits** and **Home**. Create an AWS Budget: Billing → Budgets → Create budget → a monthly budget around **USD 5** with email alert. Alerts do **not** automatically stop usage. Check the cost monitor and credits before/after working. **Do not upgrade the account to Paid Plan.** Avoid AWS Organizations because that can automatically trigger a paid-plan change for Free Plan accounts.

Choose a single region (e.g., `us-east-1`) for EC2 and S3; record it in `docs/AWS_DEPLOYMENT.md`. Select a **Free Tier eligible and available x86_64** instance explicitly labelled as eligible in the EC2 launch wizard. For newer AWS accounts, current AWS documentation lists `t3.micro`, `t3.small`, `t4g.micro`, `t4g.small`, `c7i-flex.large`, `m7i-flex.large`; ARM `t4g` is not ideal for this starter (x86 Python deps assumed). Prefer `m7i-flex.large` (if eligible/available in account and acceptable credit estimate), else `c7i-flex.large`, else `t3.small`. On small RAM fall back to **train locally, serve inference on AWS** rather than buying a bigger instance. **Never use `t3.medium` without explicit Free Plan eligibility**—the old starter guide suggested it, but it is not on AWS's current list for newer accounts.

Avoid nonessential recurring services and support. Treat EC2 instance hours, EBS, network/public IPv4, S3 storage/requests as credit consumption until verified. Free Plan protects against billed cash charges while still active; credits can be used and plan can expire earlier than the displayed October 2027 credit expiry. Retain submission artifacts on GitHub so account closure cannot destroy the project.

## 9. AWS infrastructure — console click-through (D1 or D3)

**A. S3**

1. Console → S3 → Create bucket → General purpose.
2. Unique name `vibeguard-cloud-2026-<random-suffix>`; region same as EC2.
3. Object Ownership: ACLs disabled; **Block all public access ON**; default SSE-S3 encryption; bucket versioning optional (adds storage costs; leave off for minimal demo).
4. Create; record **non-secret bucket name and region** in `docs/AWS_DEPLOYMENT.md`.
5. Create prefixes naturally through upload: `raw/{normal,inner,ball,outer}` and `artifacts/`.

**B. IAM role** (owner-approved)

1. AWS → IAM → Roles → Create role → Trusted entity AWS service → Use case EC2.
2. Name `VibeGuardEC2Role`. Add only the S3 operations required for the project:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::YOUR_BUCKET"},
    {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":"arn:aws:s3:::YOUR_BUCKET/*"}
  ]
}
```

3. Replace `YOUR_BUCKET` with actual bucket; review, save policy. Do **not** create root access keys, use account password in scripts or attach `AdministratorAccess`.

**C. EC2**

1. EC2 → Instances → Launch instances.
2. Name: `VibeGuard-Cloud-EC2`; AMI: Ubuntu Server 24.04 LTS **x86_64** (eligible image); type: only console-verified Free Tier eligible choice per §8.
3. Create/download new RSA `.pem` SSH key pair **only onto the local authorized computer**. Private key must never go into GitHub or reports.
4. Default VPC, default subnet, public IPv4 enabled only for remote SSH if needed; security group **SSH port 22 from My IP**; no world-open ports. For primary secure demo, **do not open 8501 publicly**.
5. EBS root volume e.g. 20 GiB `gp3`; verify eligibility and credit usage. Advanced details → IAM instance profile `VibeGuardEC2Role`. Launch.
6. Wait for 2/2 status checks; record sanitized instance type, region, software, role and storage, not account IDs or public DNS if sensitive.

**Stop if:** Free Plan eligibility/permissions cannot be confirmed or AWS asks to upgrade to Paid Plan. Prefer offline training and local demo until authorized compatible resources exist.

## 10. Cloud setup, code sync, real training, S3 operations (D2–D3)

Windows PowerShell:

```powershell
$IP = 'EC2_PUBLIC_IPV4_FROM_CONSOLE'
$KEY = "$HOME\Downloads\vibeguard-key.pem"
ssh -i $KEY ubuntu@$IP
```

On EC2 Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip tmux
# Auth to this public repo does not require any secret for cloning:
git clone https://github.com/shaziyasultanablr-byte/Cloud-Computing-Final.git
cd Cloud-Computing-Final
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python download_data.py
# If downloader blocked: download selected originals via browser locally and scp them to corresponding data/raw/ paths.
python train.py --epochs 14
python cloud_sync.py --bucket YOUR_BUCKET --source data/raw --prefix raw
python cloud_sync.py --bucket YOUR_BUCKET --source outputs --prefix artifacts
```

**If data were already trained locally:** push committed model and results through GitHub first; on EC2 run `git pull --ff-only` and use the same trusted model for inference. If training on EC2, copy the small outputs/ and logs **back to the trusted local working tree by SCP** for secret review and GitHub commit. Avoid putting GitHub credentials on the cloud VM unless absolutely necessary. Verify S3 objects via boto3 `list_objects_v2` or AWS S3 console. Confirm `raw/` files private and `artifacts/model.pt` and metrics exist. Never make the bucket public for a demo.

Recommended artifact metadata: SHA-256 for the model, S3 object keys, byte size, region, run ID and last verified UTC time. GitHub is the source of truth; S3 is the intentionally private cloud working copy.

**Pass criterion:** EC2 Python tests pass; a validated model exists; private S3 contains all intended prefixes and is readable via the EC2 IAM role without access keys.

## 11. Streamlit web application and secure final demonstration (D3)

Use existing `app.py` as prototype. Agent must test and repair until:

- Upload accepts only `.mat` CWRU drive-end data or safe, bounded single-column CSV; validates file type, size, finite 1D numeric array, minimum length, sampling-rate notice.
- Consistent preprocessing is used; inference shows predicted class, all four model scores, segment agreement, waveform, and manual-review indication.
- App states: research lab prototype; softmax scores not calibrated probabilities; not a maintenance/safety diagnostic tool.
- If validation fails or model checkpoint missing, show friendly UI error and no server traceback/secret data.
- Add optional fixed sample/demo choices referencing held-out file IDs **100 / 108 / 121 / 133**; never silently feed them back into training.
- Include a small explanatory panel showing architecture, current run ID, source citation and evidence of AWS EC2/S3 integration.

Start securely (on EC2 with venv active):

```bash
tmux new -s vibeguard
streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
# Leave tmux session running using Ctrl+B, then D
```

On Windows **in a second terminal**, open an encrypted SSH port-forward:

```powershell
ssh -i $KEY -L 8501:127.0.0.1:8501 ubuntu@$IP
```

Open **http://localhost:8501** on the same Windows PC. **No public port 8501 is needed.** For an examiner viewing from another machine, demonstrate screen sharing or require an independently restricted and secured temporary route; never change the S3 bucket to public or enable unrestricted inbound `0.0.0.0/0` by default.

Test recordings: normal `100.mat`, inner `108.mat`, ball `121.mat`, outer `133.mat`. Save actual screenshots with timestamp and run SHA; if any class prediction is incorrect, retain the honest screenshot and discuss the error rather than pretending success. Test invalid CSV, wrong `.mat`, short signal, missing checkpoint and oversize upload.

**Pass criterion:** verified end-to-end inference at localhost:8501 through AWS SSH tunnel, with genuine S3 object evidence and no secrets in console screenshots.

## 12. Testing and GitHub CI (D3)

- Keep `tests/test_core.py` as baseline. Add a `test_splits.py` to ensure label/file/load disjointness and correct dataset manifest; mock paths or use synthetic tests only for **software** testing. Real-data final metrics must be separately verified.
- Add tests for: model input/output shape, determinism of segmentation, NaN/infinity rejection, case-sensitive `_DE_time` channel, invalid CSV formatting, file-size guards, and no dependence on AWS account credentials for offline import.
- Validate and adjust the supplied `.github/workflows/ci.yml` on push and pull request: Ubuntu, Python 3.11, CPU PyTorch, `pip install -r requirements.txt -r requirements-dev.txt`, `python -m pytest -q`. **Do not** download entire raw CWRU dataset in ordinary CI. Set a short job timeout; avoid GitHub Actions secret injection or AWS production deployments.
- Run a secret scan on staged changes and verify `git status` after commit. Add `.env.example` only if needed with placeholders and absolutely no real secrets.
- Document exact smoke-test command and one example result in README. Do not claim CI green until GitHub Actions actually reports success.

## 13. Report, diagrams, and slides (D4)

Write final academically credible document `docs/FINAL_REPORT.pdf`, and source `docs/FINAL_REPORT.md` (or DOCX) for reproducible editing. Required chapters:

1. Title page, abstract (150–250 words), problem and motivation.
2. Objectives, scope, clearly bounded originality/novelty (load shift + review heuristic + cloud deployment).
3. Literature/background: predictive maintenance vs fault classification, 1D CNN, domain/load shift, AWS architecture. Cite original CWRU source and relevant deep learning work; references formatted consistently.
4. Dataset provenance: exact 16 IDs, 4 classes, sensor, file checksums, sample rate verification, class and split counts, motor speeds and CWRU lab setup.
5. Methodology: 2048-sample windows, per-window z-scoring, train/validation/test policy, tiny CNN parameters, RF comparison, locked test protocol.
6. Cloud implementation: AWS Free Plan context, EC2/EBS/Ubuntu/CPU, private S3/IAM least privilege, user upload/SSH tunnel/Streamlit, budget and cost controls.
7. **Results copied automatically from verified metrics JSON:** confusion matrix, per-class precision/recall/F1/support, macro-F1, window-vs-file level, validation, CNN-vs-RF, illustrative model outputs, review flag and failure cases.
8. Discussion, bias/leakage, confidence-score caveat, one file/class/test load, seeded-lab-fault limitation, impact of load shift and future extension.
9. Conclusion, citations, reproducibility and Git SHA.
10. Appendix: minimal install commands, screenshots, cost/credit evidence (redacted), tests and demo procedure.

Create professional block diagram and 8–12-slide `docs/FINAL_PRESENTATION.pptx`: problem; objectives; novelty; architecture; dataset; CNN and preprocessing; experiment; real results; live demonstration; limitations; conclusion. Use legible fonts and genuine metrics. No lorem ipsum, speculative results, unrelated AI diagrams, or fabricated screenshots. Export slide deck to PDF if practical and check all pages visually.

**Expected live viva flow (5–7 minutes):** show repo/architecture → explain 16 files & held-out 3 HP → show frozen metrics + comparison → show AWS S3 private data and instance details → SSH-tunneled Streamlit demo uploading normal and fault example → explain review flag, costs and methodological limitations.

## 14. Permanent GitHub save policy, branching and evidence (every day)

GitHub repository remains the **single project index**. Git workflow at each gate:

```bash
git status --short
git diff --check
git add README.md AGENTS.md AGENT_MASTER_PLAN.md requirements* *.py tests/ docs/ outputs/
git diff --cached --name-only
# Human/agent inspect staged paths and content for keys, e-mail/password, private screenshots.
git commit -m "D2: validated 1D CNN final evaluation"
git push origin main
git rev-parse HEAD
```

Do not commit nonexistent paths; adapt `git add` to changed files. If branch protection disallows main pushes, create a feature branch and PR, merge after checks. Use clear commits (`D1`, `D2`, `D3`, `D4`). Commit summary/evaluation/model only once actual experiments complete; don't invent outputs to make checklist look complete.

Before final submission: clean clone into a **new** directory, install dependencies, run tests, verify model and artifacts load, verify report/slide files open, and ensure source URLs and demo instructions work. Save final run ID, SHA and date to `docs/SUBMISSION_CHECKLIST.md`. If repo is still empty/does not have write permission, owner needs to grant access or use authenticated GitHub CLI on their local machine; agent cannot claim push occurred.

## 15. AWS cost monitoring and cleanup (D4)

- Capture redacted Free Plan + credit balance at beginning and end. Estimate consumed credits from console once usage data refreshes. Do not equate the $100 balance with a guarantee of unlimited resources.
- Stop EC2 when not using it. **Stop does not eliminate EBS storage costs/credit usage**; public IPv4 storage/assignment rules also matter.
- Once screenshots and demo are complete and GitHub has verified final artifacts, terminate disposable EC2 instance and delete unneeded EBS volumes/snapshots, security groups, key-pair metadata, and S3 copies if account owner approves. **Confirm files and final metrics are in GitHub before deleting resources**.
- Keep S3 bucket only if demonstrator must remain live and user accepts ongoing credit usage; otherwise delete objects and bucket after final presentation, following explicit authorization. A terminated EC2 public IP will no longer host demo; documented local reproduction remains available.
- Verify AWS resource inventory in **each used region**, credit balance and final expenses in billing. Record cleanup state honestly in `docs/COST_AND_CLEANUP.md`.

## 16. Troubleshooting / fallback order

| Blocker | Safe recovery |
|---|---|
| GitHub push denied | Verify actual linked GitHub account and write permission; authorized owner grants collaborator or signs in through `gh auth login`; stage local files, do not call them uploaded. |
| CWRU URL blocked | Download the same file manually from official university table, validate MAT + checksum, SCP privately to EC2. |
| AWS password not rotated/MFA unavailable | Stop AWS provisioning. Proceed offline on local repo without credentials. |
| Free Tier instance not offered | Use any currently allowed x86 Free Tier eligible instance and document; or train locally and use a small eligible inference instance; never upgrade plan. |
| EC2 OOM or pip install fails | Use CPU-only Torch, disable unnecessary processes, move training to local computer, keep EC2 for serving. Do not increase to paid instance. |
| Normal/other sample-rate mismatch | Investigate original official metadata. Normalize sample rates with documented resampling or postpone final scoring until valid. |
| Accuracy low on 3 HP | Report honestly; inspect split and dataset, discuss load shift and limited sample sizes; don't train on final test. |
| SSH denies key | Check key path and file permissions, local source IP and security group; never paste private key into chat or source repo. |
| Streamlit unavailable publicly | Use secured `ssh -L` tunnel to `localhost:8501`, no open public 8501 required. |
| S3 access denied | Check attached instance profile, bucket ARN, IAM role, region, bucket block-public-access; don't put keys in code. |
| Report deadline near | Prioritize honest metrics, reproducible code, working demo and report before UI cosmetics. |

## 17. Final acceptance / status report to user

Do not mark project complete unless **every mandatory criterion** is supported by observable evidence:

- [ ] Authorized owner rotated compromised AWS password and enabled root MFA.
- [ ] Correct GitHub repo has real commits, confirmed SHA and clean final tree; no secrets.
- [ ] All 16 official CWRU files validated with source ID and checksums.
- [ ] Dataset/sensor/sampling verified and file/load-disjoint splits frozen.
- [ ] Actual CNN training and baseline both completed; genuine final held-out results saved.
- [ ] Per-class and recording-level metrics computed; no hidden final-test tuning.
- [ ] Model, metrics, report, slides, screenshots, tests, guide pushed to GitHub.
- [ ] Cloud uses EC2 + EBS + private S3 + IAM role, correct region and cost boundaries.
- [ ] Streamlit demo tested through encrypted tunnel with 4 classes and negative tests.
- [ ] AWS account remained on Free Plan, USD credit consumption inspected, no paid upgrade.
- [ ] AWS cleanup documented and source can be reproduced after any service is stopped.

Final response to project owner must contain: **what was implemented**, **which files were committed**, **Git SHA and branch**, **real model metrics**, **verified demo route and how to access it**, **AWS resources and credit impact**, **what remains incomplete**, and **safely completed cleanup**. Distinguish implemented from planned, screenshots from simulations, and measured from estimated values.

---
### Official references for agent to verify during build

- CWRU original dataset and data tables: https://engineering.case.edu/bearingdatacenter/download-data-file ; https://engineering.case.edu/bearingdatacenter/normal-baseline-data ; https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
- AWS account plans: https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html
- AWS Free Tier eligible EC2 types: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html
- GitHub repository: https://github.com/shaziyasultanablr-byte/Cloud-Computing-Final
- GitHub file size and LFS: https://docs.github.com/en/repositories/working-with-files/managing-large-files

**Rule of truth:** The only valid project conclusion is the one supported by validated datasets, actual executions, actual AWS evidence and a verified GitHub commit. No guessed results.
