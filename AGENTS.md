# AGENTS.md — Operational Rules for VibeGuard Cloud

## 1. Mission & Authority
Read `AGENT_MASTER_PLAN.md` completely before modifying code, running experiments, or interacting with AWS. `AGENT_MASTER_PLAN.md` is the authoritative execution contract for **VibeGuard Cloud: Load-Shift-Aware Industrial Motor Bearing Fault Classification Using 1D Deep Learning and AWS**.

## 2. Non-Negotiable Security & Cost Rules
1. **No Secrets Anywhere:** Never store or log AWS passwords, root credentials, IAM access keys, GitHub tokens, or `.pem` SSH private keys in code, configuration files, git history, screenshots, or documentation.
2. **Compromised Password Guard:** Before any AWS interaction, the authorized account owner must confirm root password rotation and MFA activation.
3. **AWS Free Plan Only:** Never click *Upgrade Plan* or provision non-eligible/paid AWS services (no GPUs, NAT Gateways, Application Load Balancers, Elastic IPs, or managed SageMaker endpoints).
4. **IAM Least Privilege:** EC2 access to S3 must use an IAM Instance Profile (`VibeGuardEC2Role`) scoped strictly to `s3:ListBucket`, `s3:GetObject`, and `s3:PutObject` on the project's private bucket.
5. **Private Demo Route:** Keep S3 *Block Public Access* ON and demonstrate the Streamlit application (`127.0.0.1:8501`) over an encrypted SSH port-forward (`ssh -L 8501:127.0.0.1:8501`).

## 3. Scientific Integrity & Leakage Prevention
1. **Official Dataset Only:** Use the 16 specified Case Western Reserve University (CWRU) Bearing Data Center `.mat` files listed in `docs/DATASET_MANIFEST.csv`. Never substitute synthetic data for final reported metrics.
2. **Locked Cross-Load Split:**
   - **Train (0 HP & 1 HP):** File IDs `97, 98, 105, 106, 118, 119, 130, 131`
   - **Validation (2 HP):** File IDs `99, 107, 120, 132`
   - **Final Held-Out Test (3 HP):** File IDs `100, 108, 121, 133`
3. **Zero Test Leakage:** Never tune model architecture, hyperparameters, training epochs, or manual-review thresholds on the 3 HP test recordings.
4. **Honest Reporting:** Report exact measured window-level and recording-level metrics from `outputs/metrics.json` and `outputs/file_level_metrics.json`. Clearly state that softmax outputs are uncalibrated confidence scores and that validation/test splits contain one recording per class.
