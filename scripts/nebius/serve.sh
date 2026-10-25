#!/usr/bin/env bash
# serve.sh <a|b>: start vLLM on 127.0.0.1:8001 with the Workload Manifest flags and write manifest.json.
# VM-A and VM-B must run IDENTICAL flags; the manifest hash is the record's manifest_id.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
W="${1:?usage: serve.sh <a|b>}"; source "$HERE/.$W.env"; IP=$("$HERE/vm.sh" "$W" ip)
ssh_vm "$IP" bash -s <<REMOTE
set -euo pipefail
docker rm -f vllm >/dev/null 2>&1 || true
docker run -d --name vllm --gpus all --ipc host -p 127.0.0.1:8001:8000 \
  -v $DATA_DIR/hf:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  '$VLLM_IMAGE' --model '$MODEL_ID' --served-model-name '$SERVED_MODEL_NAME' \
  --generation-config vllm --max-logprobs 20 --max-model-len 8192 --gpu-memory-utilization 0.85 >/dev/null
for i in \$(seq 1 120); do curl -fs http://127.0.0.1:8001/health >/dev/null && break; sleep 5; done
curl -fs http://127.0.0.1:8001/health >/dev/null || { docker logs --tail 40 vllm; exit 1; }
python3 - <<'PY'
import hashlib, json, subprocess
inspect = json.loads(subprocess.check_output(["docker", "inspect", "vllm"]))[0]
logs = subprocess.run(["docker", "logs", "vllm"], capture_output=True, text=True)
text = logs.stdout + logs.stderr
m = {
  "image": inspect["Config"]["Image"], "image_id": inspect["Image"],
  "args": inspect["Args"],
  "env": sorted(e for e in inspect["Config"]["Env"] if e.startswith("VLLM_")),
  "model": "$MODEL_ID",
  "v2_runner_logged": "model runner v2" in text.lower(),
}
m["manifest_id"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]
open("manifest.json", "w").write(json.dumps(m, indent=2))
print(json.dumps(m, indent=2))
PY
REMOTE
