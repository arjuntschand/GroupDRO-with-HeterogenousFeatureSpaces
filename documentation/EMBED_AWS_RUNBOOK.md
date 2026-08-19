# EMBED on AWS — GPU training runbook

Status 2026-08-18: pipeline validated locally end-to-end. GPU quota requested in
**us-west-2** (case REDACTED-CASE-ID, CASE_OPENED, want 8 vCPUs). Everything below
runs once the quota is approved. Data + GPU both in **us-west-2** so S3→EC2 is free/fast.

Most of the launch can be done from Arjun's Mac via the already-configured `.venv/bin/aws`
CLI (Claude can drive it with a go-ahead). Costs come out of the ~$10k credits.

## Instance & storage plan

| Item | Choice | ~Cost |
|------|--------|-------|
| Instance | `g5.xlarge` (1× A10G 24GB, 4 vCPU, 16GB RAM) | ~$1.0/hr on-demand |
| AMI | AWS Deep Learning AMI (Ubuntu, CUDA + PyTorch preinstalled) | — |
| Storage | Start ~500 GB gp3 EBS (subset), grow if we go full | ~$0.08/GB-mo |
| Region | us-west-2 (Oregon) — same as data | in-region transfer free |

Full open image set ≈ 1.9 TB. We do NOT need it all to start: use the selective
downloader with a `--per-group` cap to pull a large subset (~300–400 GB), iterate,
then scale to full only if we want the final headline at REMIND's scale.

## Steps (CLI-driven from the Mac unless noted)

```bash
AWS=.venv/bin/aws; REGION=us-west-2

# 0. confirm quota approved (Value should be >= 8)
$AWS service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region $REGION

# 1. key pair (private key saved locally, chmod 400)
$AWS ec2 create-key-pair --key-name embed-key --region $REGION \
    --query 'KeyMaterial' --output text > ~/.ssh/embed-key.pem && chmod 400 ~/.ssh/embed-key.pem

# 2. security group allowing SSH from your current IP only
MYIP=$(curl -s https://checkip.amazonaws.com)
SG=$($AWS ec2 create-security-group --group-name embed-sg --description "EMBED SSH" \
     --region $REGION --query GroupId --output text)
$AWS ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 \
     --cidr ${MYIP}/32 --region $REGION

# 3. latest Deep Learning AMI id (PyTorch, Ubuntu 22.04)
AMI=$($AWS ec2 describe-images --owners amazon --region $REGION \
   --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
   --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

# 4. launch g5.xlarge with a 500 GB gp3 root volume
$AWS ec2 run-instances --image-id $AMI --instance-type g5.xlarge \
   --key-name embed-key --security-group-ids $SG --region $REGION \
   --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]' \
   --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=embed-train}]' \
   --query 'Instances[0].InstanceId' --output text
# grab public IP:
$AWS ec2 describe-instances --filters Name=tag:Name,Values=embed-train \
   Name=instance-state-name,Values=running --region $REGION \
   --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

### On the instance (SSH in)
```bash
ssh -i ~/.ssh/embed-key.pem ubuntu@<PUBLIC_IP>

git clone <this repo>  &&  cd GroupDRO-with-HeterogenousFeatureSpaces
pip install -r dro_hetero_anchors/requirements.txt      # DLAMI already has torch/cuda
aws configure           # same EMBED-approved keys (or attach an IAM role instead)

# tables + index
aws s3 cp s3://embed-dataset-open/tables/ datasets/embed/tables/ --recursive --region us-west-2
python -m dro_hetero_anchors.src.datasets_embed          # sanity: prints group/tail

# images — start with a per-group-capped subset (fast, ~hundreds of GB), then scale
python -m dro_hetero_anchors.src.download_embed --per-group 4000 --workers 32 --aws aws

# train (GPU auto-detected). require_local_images:false for full-index runs.
python -m dro_hetero_anchors.src.train_embed --config experiments/embed_baseline.yaml
python -m dro_hetero_anchors.src.train_embed --config experiments/embed_groupdro.yaml
```

## When done (avoid charges)
```bash
# stop = keep disk, pause compute cost; terminate = delete everything
$AWS ec2 stop-instances --instance-ids <id> --region us-west-2
```
Also: delete the root access key in the console when the project's done (Security
credentials → delete). And per the DUA — publish results+code only, no released weights.

## To-dos before the real headline run
- [ ] Set `stratified_batching:true` + real head/tail (full index has 2 head groups 55/41%).
- [ ] Add class weighting for density imbalance (A 10% / D 5%) — mirror NHANES `class_weight:auto`.
- [ ] Merge/drop micro modality-combos (n<~30) so GroupDRO groups are meaningful.
- [ ] Confirm sample unit (per-breast, current) vs REMIND's exact preprocessing.
