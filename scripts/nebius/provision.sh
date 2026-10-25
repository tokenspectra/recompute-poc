#!/usr/bin/env bash
# provision.sh <a|b>: sync the repo, pull the pinned image, download weights, print the digest.
set -euo pipefail
source "$(dirname "$0")/lib.sh"; need rsync
W="${1:?usage: provision.sh <a|b>}"; source "$HERE/.$W.env"
REPO="$(cd "$HERE/../.." && pwd)"
IP=$("$HERE/vm.sh" "$W" ip)
echo "== wait for ssh on $IP"
for _ in $(seq 1 40); do ssh_vm "$IP" true 2>/dev/null && break; sleep 5; done
echo "== host facts"
ssh_vm "$IP" 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; docker --version'
echo "== sync repo"
rsync -az --delete --exclude .venv --exclude .git --exclude data --exclude '*.sqlite' --exclude '*.jsonl' \
  -e "ssh -o StrictHostKeyChecking=accept-new" "$REPO/" "$SSH_USER@$IP:rpoc/"
echo "== pull image + weights (several minutes)"
ssh_vm "$IP" "sudo mkdir -p $DATA_DIR/hf && sudo chown -R \$USER $DATA_DIR && docker pull -q '$VLLM_IMAGE'"
ssh_vm "$IP" "docker run --rm -v $DATA_DIR/hf:/root/.cache/huggingface --entrypoint python3 '$VLLM_IMAGE' -c \"from huggingface_hub import snapshot_download as s; print(s('$MODEL_ID'))\""
echo "== image digest on this host"
ssh_vm "$IP" "docker image inspect '$VLLM_IMAGE' --format '{{index .RepoDigests 0}} {{.Id}}'"
