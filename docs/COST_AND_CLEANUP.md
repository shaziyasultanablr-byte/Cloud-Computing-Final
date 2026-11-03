# AWS Cost Protection, Credit Tracking & Resource Cleanup (`VibeGuard Cloud`)

## 1. Cost Protection Principles
1. **Stay Strictly on AWS Free Plan:** Never click *Upgrade Plan* in the AWS Billing console. While on the Free Plan with active credits (initial balance: `USD 100.00`), cash charges are blocked unless the account owner explicitly upgrades to a Paid Plan.
2. **Budget Alert Guardrail:** Configure a monthly AWS Cost Budget of **`USD 5.00`** with email alerts at `50%` and `80%`. Note that AWS Budgets send notifications only—they do not automatically terminate resources.
3. **Prohibited Services:** Do not provision GPU instances (`g4dn`, `p3`), NAT Gateways, Application/Network Load Balancers, unattached Elastic IPv4 addresses, Amazon RDS/Aurora, or always-on SageMaker endpoints.
4. **Single Region Discipline:** Provision all resources in a single region (`us-east-1`) so no orphaned resources are hidden in secondary regions.

## 2. Credit & Resource Inventory Log

| Phase / Checkpoint | Account Plan | Promotional Credit Balance | Active EC2 Instance | Active EBS Volume | Active S3 Bucket | Notes |
|---|---|---|---|---|---|---|
| **Baseline (Pre-Deployment)** | AWS Free Plan (`ACTIVE`) | `USD 99.69` remaining (`$0.195` spend) | None | None | None | Verified via `aws freetier get-account-plan-state`; monthly budget `VibeGuard-Monthly-5USD` (`$5.00`) active |
| **Live Cloud Deployment (`us-east-1`)** | AWS Free Plan (`ACTIVE`) | `USD 99.69` (`< $0.25` total credit usage) | `i-0c8d53060f56c451d` (`t3.small`, Ubuntu 24.04.5 LTS) | `vol-09af634eea204d82d` (`20 GiB gp3`, encrypted) | `vibeguard-cloud-2026-810995308548` (Private, `24` objects) | Bound to `127.0.0.1:8501` via `VibeGuardEC2Role` and SSH tunnel (`sg-01fee40d745024ecc`) |
| **Post-Viva Cleanup** | AWS Free Plan | Record final balance after billing refresh | Terminated (`i-0c8d53060f56c451d`) | Auto-deleted (`DeleteOnTermination=true`) | Emptied & deleted (once authorized) | All permanent deliverables preserved in GitHub |

## 3. Mandatory Post-Demonstration Teardown Checklist

> **IMPORTANT:** Confirm that all code, `outputs/model.pt`, `outputs/metrics.json`, `outputs/file_level_metrics.json`, `docs/FINAL_REPORT.pdf`, `docs/FINAL_PRESENTATION.pptx`, and redacted screenshots are committed and pushed to `https://github.com/shaziyasultanablr-byte/Cloud-Computing-Final` **before** terminating cloud resources.

1. **Terminate EC2 Instance:**
   - Note: *Stopping* an EC2 instance stops compute hour consumption but **continues** consuming credits for the attached `gp3` EBS volume.
   - Once the live demonstration and screenshots are complete and the owner authorizes cleanup: go to **EC2 $\to$ Instances**, select `VibeGuard-Cloud-EC2`, and click **Instance state $\to$ Terminate (delete) instance**.
2. **Verify EBS Volumes & Snapshots:**
   - Go to **EC2 $\to$ Elastic Block Store $\to$ Volumes** and **Snapshots**. Confirm no orphaned volumes or snapshots remain.
3. **Empty and Delete S3 Bucket (If Authorized):**
   - Go to **S3**, select `vibeguard-cloud-2026-<suffix>`, click **Empty**, and then click **Delete** if the cloud demo is no longer required.
4. **Remove Security Group & Key Pair Metadata:**
   - Delete custom security group `vibeguard-ssh-only` and key pair entry in **EC2 $\to$ Key Pairs**.
5. **Final Billing & Credit Check:**
   - Open **Billing and Cost Management $\to$ Credits** and **Bills** to verify `AED 0.00 / USD 0.00` cash charges.
