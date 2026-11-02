# Demonstration & Operational Runbook (`VibeGuard Cloud`)

## 1. Live Viva / Examiner Demonstration Flow (5–7 Minutes)

1. **Architecture & Problem Framing (1 min):**
   - Open [`docs/architecture.png`](architecture.png) and [`README.md`](../README.md).
   - Explain the core research question: evaluating whether a 1D CNN trained on `0 & 1 HP` CWRU drive-end vibration recordings generalizes to an unseen `3 HP` motor load, compared against an 18-feature Random Forest baseline.
2. **Dataset Hygiene & Leakage Control (1 min):**
   - Show [`docs/DATASET_MANIFEST.csv`](DATASET_MANIFEST.csv) and [`docs/dataset_checksums.csv`](dataset_checksums.csv).
   - Highlight the recording- and load-disjoint split (`0/1 HP` train, `2 HP` val, `3 HP` test) and our discovery/fix of the upstream CWRU `99.mat` workspace contamination (`X098_DE_time` vs `X099_DE_time`).
3. **Locked Evaluation Metrics & Figures (1.5 min):**
   - Show [`outputs/confusion_matrix.png`](../outputs/confusion_matrix.png), [`outputs/training_loss.png`](../outputs/training_loss.png), and [`outputs/file_level_metrics.json`](../outputs/file_level_metrics.json).
   - Point out that while both models achieve `100.0%` window and recording accuracy on the 4 held-out 3 HP files, Random Forest's mean confidence drops to `0.7273` (`margin = 0.5005`) on `108.mat` (3 HP Inner Race), whereas `1D TinyFaultCNN` maintains `0.9953` (`margin = 0.9932`).
4. **Cloud Infrastructure & Private S3 Verification (1 min):**
   - Show the EC2 terminal running `python cloud_sync.py --bucket $VIBEGUARD_S3_BUCKET --prefix artifacts --action verify` via the attached `VibeGuardEC2Role` IAM instance profile (no AWS access keys).
5. **Interactive Streamlit Demo over Encrypted SSH Tunnel (2 min):**
   - Show the local SSH port-forward command (`ssh -i $KEY -L 8501:127.0.0.1:8501 ubuntu@$IP`) and open `http://localhost:8501`.
   - Select or upload all 4 held-out 3 HP recordings (`100.mat` Normal, `108.mat` Inner Race, `121.mat` Ball Fault, `133.mat` Outer Race) and inspect the predicted class, 4-class uncalibrated scores, segment agreement (`100.0%`), and time/frequency plots.
   - Upload an invalid file (e.g., multi-column CSV or short signal) to demonstrate graceful validation error handling.
   - Conclude with the methodological limitations (1 recording per class per load on a single laboratory rig; uncalibrated softmax scores).

---

## 2. Smoke Test Commands

```powershell
# Run full unit, split-leakage, and real-checkpoint end-to-end tests
.\.venv\Scripts\python.exe -m pytest -q

# Validate all 16 CWRU recordings and regenerate docs/dataset_checksums.csv
.\.venv\Scripts\python.exe scripts/validate_dataset.py

# Test dry-run S3 artifact packaging without AWS credentials
.\.venv\Scripts\python.exe cloud_sync.py --bucket vibeguard-demo --source outputs --prefix artifacts --dry-run
```
