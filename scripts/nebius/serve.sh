#!/usr/bin/env bash
# serve.sh <a|b> [ACTUAL_MODEL]: start vLLM on 127.0.0.1:8001 and write manifest.json.
# Honest run: no second argument. VM-A and VM-B must run IDENTICAL flags; the manifest hash is
# the record's manifest_id.
# Negative control: `serve.sh a Qwen/Qwen3-1.7B` serves ACTUAL_MODEL under the DECLARED name
# (SERVED_MODEL_NAME) with the declared manifest id, i.e. an operator that lies about the model.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
W="${1:?usage: serve.sh <a|b> [ACTUAL_MODEL]}"; ACTUAL="${2:-$MODEL_ID}"
source "$HERE/.$W.env"; IP=$("$HERE/vm.sh" "$W" ip)
ssh_vm "$IP" bash -s <<REMOTE
set -euo pipefail
docker rm -f vllm >/dev/null 2>&1 || true
if [ '$ACTUAL' != '$MODEL_ID' ]; then
  docker run --rm -v $DATA_DIR/hf:/root/.cache/huggingface --entrypoint python3 '$VLLM_IMAGE' \
    -c "from huggingface_hub import snapshot_download as s; print(s('$ACTUAL'))"
fi
docker run -d --name vllm --gpus all --ipc host -p 127.0.0.1:8001:8000 \
  -v $DATA_DIR/hf:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  '$VLLM_IMAGE' --model '$ACTUAL' --served-model-name '$SERVED_MODEL_NAME' \
  --generation-config vllm --max-logprobs 20 --max-model-len 8192 --gpu-memory-utilization 0.85 >/dev/null
for i in \$(seq 1 120); do curl -fs http://127.0.0.1:8001/health >/dev/null && break; sleep 5; done
curl -fs http://127.0.0.1:8001/health >/dev/null || { docker logs --tail 40 vllm; exit 1; }
ACTUAL='$ACTUAL' DECLARED='$MODEL_ID' python3 - <<'PY'
import hashlib, json, os, subprocess
inspect = json.loads(subprocess.check_output(["docker", "inspect", "vllm"]))[0]
logs = subprocess.run(["docker", "logs", "vllm"], capture_output=True, text=True)
text = logs.stdout + logs.stderr
actual, declared = os.environ["ACTUAL"], os.environ["DECLARED"]
args = [declared if a == actual else a for a in inspect["Args"]]  # hash the DECLARED view
m = {
  "image": inspect["Config"]["Image"], "image_id": inspect["Image"], "args": args,
  "env": sorted(e for e in inspect["Config"]["Env"] if e.startswith("VLLM_")),
  "model": declared,
  "v2_runner_logged": "using v2 model runner" in text.lower(),
}
m["manifest_id"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]
m["actual_model"] = actual  # not part of the hash: this is what a lying operator would hide
open("manifest.json", "w").write(json.dumps(m, indent=2))
print(json.dumps(m, indent=2))
PY
REMOTE
