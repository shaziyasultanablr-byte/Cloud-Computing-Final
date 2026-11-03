# AWS Cloud Deployment Guide (`VibeGuard Cloud`)

## 1. Mandatory Pre-Flight Security & Free Plan Verification
1. **Password Rotation & Root MFA:** Before accessing AWS, the authorized account owner must confirm that any previously exposed password was rotated and MFA is enabled on the root user.
2. **Confirm AWS Free Plan Status:**
   - Sign in to the AWS Management Console $\to$ **Billing and Cost Management** $\to$ **Home** & **Credits**.
   - Verify the account plan is **Free Plan** with active Free Tier promotional credits (e.g., USD 100 balance).
   - **Never click "Upgrade Plan".** Do not enable AWS Organizations or paid support tiers.
3. **Create a USD 5.00 Monthly Budget Alert:**
   - Navigate to **Billing and Cost Management** $\to$ **Budgets** $\to$ **Create budget**.
   - Choose **Customize (advanced)** $\to$ **Cost budget** $\to$ Monthly budget amount: **`USD 5.00`**.
   - Configure email alert notifications at `50%` (`$2.50`) and `80%` (`$4.00`) of budgeted spend.

---

## 2. Target Cloud Architecture (`us-east-1`)
- **Region:** `us-east-1` (N. Virginia) for both Amazon S3 and Amazon EC2.
- **Storage:** Private Amazon S3 bucket (`vibeguard-cloud-2026-<suffix>`) with **Block all public access = ON**, ACLs disabled, and default SSE-S3 (`AES256`) encryption.
  - `s3://<bucket>/raw/{normal,inner,ball,outer}/` — 16 validated CWRU `.mat` files
  - `s3://<bucket>/artifacts/` — `model.pt`, `metrics.json`, `file_level_metrics.json`, `training_history.json`, `run_manifest.json`, `test_predictions.csv`, `confusion_matrix.png`, `training_loss.png`
- **Identity & Access Management:** EC2 Instance Profile `VibeGuardEC2Role` granting least-privilege `s3:ListBucket`, `s3:GetObject`, and `s3:PutObject` to that single bucket ARN. Zero access keys stored on disk or in code.
- **Compute:** Ubuntu Server 24.04 LTS (`x86_64`) on a console-verified Free Tier eligible instance type (`t3.small`, `t3.micro`, `c7i-flex.large`, or `m7i-flex.large` as marked *Free tier eligible* in the account's launch wizard) with `20 GiB gp3` EBS root volume.
- **Network Security:** Security Group allowing inbound **SSH (TCP port 22) from My IP only**. Port `8501` is bound strictly to loopback (`127.0.0.1:8501`) on EC2 and accessed via an encrypted SSH tunnel (`ssh -L 8501:127.0.0.1:8501`).

---

## 3. Step-by-Step AWS Console Provisioning

### Step 3A — Create Private S3 Bucket
1. Open **AWS Console $\to$ S3 $\to$ Create bucket**.
2. **Bucket name:** `vibeguard-cloud-2026-<your-random-suffix>` in `us-east-1`.
3. **Object Ownership:** ACLs disabled (recommended).
4. **Block Public Access settings:** Ensure **Block all public access** is **CHECKED (ON)**.
5. **Bucket Versioning:** Disable (prevents unnecessary duplicate storage consumption).
6. **Default encryption:** Server-side encryption with Amazon S3 managed keys (**SSE-S3**).
7. Click **Create bucket**.

### Step 3B — Create Least-Privilege IAM Role (`VibeGuardEC2Role`)
1. Open **AWS Console $\to$ IAM $\to$ Roles $\to$ Create role**.
2. **Trusted entity type:** AWS service $\to$ Use case: **EC2** $\to$ Next.
3. Click **Next** without attaching broad managed policies, name the role **`VibeGuardEC2Role`**, and create it.
4. Open `VibeGuardEC2Role` $\to$ **Permissions** $\to$ **Add permissions** $\to$ **Create inline policy** (**JSON** editor) and paste (replacing `YOUR_BUCKET_NAME` with the bucket from Step 3A):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowListProjectBucket",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME"
    },
    {
      "Sid": "AllowReadWriteProjectObjects",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/*"
    }
  ]
}
```
5. Save the inline policy as `VibeGuardS3LeastPrivilege`.

### Step 3C — Launch Free Plan Eligible EC2 Instance
1. Open **AWS Console $\to$ EC2 $\to$ Launch instances**.
2. **Name:** `VibeGuard-Cloud-EC2`.
3. **AMI:** `Ubuntu Server 24.04 LTS (HVM), SSD Volume Type` — Architecture: **`64-bit (x86)`** (Free tier eligible).
4. **Instance type:** Select an explicitly badged **Free tier eligible** `x86_64` type (`t3.small` or `t3.micro` or `m7i-flex.large`). Never choose an unbadged paid instance type.
5. **Key pair:** Create a new RSA `.pem` key pair (e.g., `vibeguard-key.pem`) downloaded only to `$HOME\Downloads\vibeguard-key.pem` on the authorized local machine. Never copy `.pem` into the repository.
6. **Network settings:**
   - Enable **Auto-assign public IP**.
   - Create security group `vibeguard-ssh-only`: allow **SSH (TCP 22)** from **My IP** only. Do **not** add port 8501 to inbound rules.
7. **Configure storage:** `1x 20 GiB gp3` root volume.
8. **Advanced details $\to$ IAM instance profile:** Select **`VibeGuardEC2Role`**.
9. Click **Launch instance** and wait for `2/2 checks passed`.

---

## 4. EC2 Application Setup, S3 Sync & Encrypted SSH Tunnel Demo

### 4.1 Connect to EC2 from Local PowerShell
```powershell
$IP = "<EC2_PUBLIC_IPV4>"
$KEY = "$HOME\Downloads\vibeguard-key.pem"
ssh -i $KEY ubuntu@$IP
```

### 4.2 Install Environment, Verify Tests, and Sync with Private S3 on EC2
```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip tmux

git clone https://github.com/shaziyasultanablr-byte/Cloud-Computing-Final.git
cd Cloud-Computing-Final

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-dev.txt

# Verify unit & split tests on EC2
python -m pytest -q

# Download & validate official CWRU .mat files
python download_data.py
python scripts/validate_dataset.py

# Upload raw dataset and verified model artifacts to private S3 via IAM Role (no keys)
export VIBEGUARD_S3_BUCKET="vibeguard-cloud-2026-<your-random-suffix>"
export AWS_DEFAULT_REGION="us-east-1"

python cloud_sync.py --bucket "$VIBEGUARD_S3_BUCKET" --source data/raw --prefix raw --action upload
python cloud_sync.py --bucket "$VIBEGUARD_S3_BUCKET" --source outputs --prefix artifacts --action upload

# Verify S3 contents via IAM Role
python cloud_sync.py --bucket "$VIBEGUARD_S3_BUCKET" --prefix raw --action verify
python cloud_sync.py --bucket "$VIBEGUARD_S3_BUCKET" --prefix artifacts --action verify
```

### 4.3 Start Streamlit Bound to Loopback (`127.0.0.1:8501`) in `tmux`
```bash
tmux new -s vibeguard
source .venv/bin/activate
export VIBEGUARD_S3_BUCKET="vibeguard-cloud-2026-<your-random-suffix>"
export AWS_DEFAULT_REGION="us-east-1"
streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
# Detach tmux session with: Ctrl+B, then D
```

### 4.4 Open Encrypted SSH Tunnel from Local Windows PC
In a second PowerShell terminal on the local PC:
```powershell
$IP = "54.162.177.175"
$KEY = "$HOME\.ssh\vibeguard-ec2-key.pem"
ssh -i $KEY -L 8501:127.0.0.1:8501 ubuntu@$IP
```
Open **`http://localhost:8501`** in the local browser to interact with the EC2-hosted dashboard securely over SSH.

---

## 5. Verified Live AWS Cloud Deployment Record (`us-east-1`)

The complete cloud architecture was provisioned and verified live in AWS Account `810995308548` (`us-east-1`). Full machine-readable telemetry is preserved in [`docs/evidence/aws_live_deployment_evidence.json`](evidence/aws_live_deployment_evidence.json).

| Component | Provisioned Resource / Identifier | Verified Configuration & Status |
|---|---|---|
| **AWS Account & Plan** | Account `810995308548` (`us-east-1`) | **AWS Free Plan (`ACTIVE`)**, `99.69 USD` remaining promotional credits |
| **Root & IAM Security** | `VibeGuard-Root-MFA` / `vibeguard-deployer` / `VibeGuardDeployerRole` | Root MFA enabled (`1`), Root access keys: `0`, Deployer permanent keys: `0`, deployed via temporary STS session (`assumed-role/VibeGuardDeployerRole/vibeguard-deployer-session`) |
| **Monthly Cost Budget** | `VibeGuard-Monthly-5USD` | `5.00 USD` monthly COST budget (`HEALTHY`, actual spend `0.195 USD`, hard ceiling `10.00 USD`) |
| **Private S3 Bucket** | `s3://vibeguard-cloud-2026-810995308548` | `BlockPublicAcls=true`, `IgnorePublicAcls=true`, `BlockPublicPolicy=true`, `RestrictPublicBuckets=true`, `BucketOwnerEnforced`, `SSE-S3 (AES256)` (`16` raw CWRU `.mat` files + `8` model artifacts = `24` objects verified) |
| **EC2 IAM Role & Profile** | `VibeGuardEC2Role` (`AROA3ZUYWLACNULYAM2DR` / `AIPA3ZUYWLACI3D3Z3CLD`) | Trusts `ec2.amazonaws.com`; inline policy `VibeGuardS3LeastPrivilege` scoped strictly to `arn:aws:s3:::vibeguard-cloud-2026-810995308548` |
| **EC2 Compute Instance** | `i-0c8d53060f56c451d` (`VibeGuard-Cloud-EC2`) | `t3.small` (Free Tier eligible), Ubuntu 24.04.5 LTS (`ami-0d6dd9e8265d48906`), AZ `us-east-1c`, Public IP `54.162.177.175`, Private IP `172.31.26.199` |
| **EBS Root Storage** | `vol-09af634eea204d82d` (`VibeGuard-Cloud-EBS`) | `20 GiB gp3`, `Encrypted=true`, `DeleteOnTermination=true` |
| **Network Security Group** | `sg-01fee40d745024ecc` (`vibeguard-ssh-sg`) | Inbound TCP `22` restricted to authorized workstation IP `/32` (`sgr-05b511659fb56cc37`); port `8501` closed publicly |
| **EC2 Automated Test Suite** | `python3 -m pytest -v` on `i-0c8d53060f56c451d` | **`15 passed in 6.10s`** (all unit, split, negative-input, S3 offline-safety, and 4-class end-to-end tests passed on EC2) |
| **EC2 4-Class `3 HP` Inference** | `100.mat`, `108.mat`, `121.mat`, `133.mat` | `4/4` (`100.00%`) exact match with local metrics (`normal=0.986942`, `inner=0.995320`, `ball=0.989040`, `outer=0.985114`) |
| **SSH Tunnel Health Check** | `http://127.0.0.1:8501/_stcore/health` over `ssh -L 8501:127.0.0.1:8501` | Verified `HTTP 200 OK` (`ok`) and dashboard HTML (`6,467` bytes) over encrypted SSH tunnel |
