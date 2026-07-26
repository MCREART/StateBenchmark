#!/usr/bin/env bash
set -euo pipefail

ROOT="${STATEBENCH_ROOT:-/root/statebench}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
HF="${HF:-/root/miniconda3/bin/hf}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
MAX_JOBS="${MAX_JOBS:-3}"
SEEDS=(20260504 20260505 20260506)
DATASETS=(openpi_c_nl trip_nl_known_only alfworld_nl_no_goto)
MODELS=(Qwen3-Embedding-0.6B bge-large-en-v1.5 e5-large-v2)

cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
mkdir -p "$MODEL_ROOT" outputs/review_core logs/review_core

safe_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

model_repo() {
  case "$1" in
    Qwen3-Embedding-0.6B) echo "Qwen/Qwen3-Embedding-0.6B" ;;
    bge-large-en-v1.5) echo "BAAI/bge-large-en-v1.5" ;;
    e5-large-v2) echo "intfloat/e5-large-v2" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

write_model_config() {
  "$PYTHON" - "$ROOT/configs/models.yaml" "$MODEL_ROOT" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = Path(sys.argv[2])
models = [
    ("Qwen3-Embedding-0.6B", "", 32),
    ("bge-large-en-v1.5", "", 64),
    ("e5-large-v2", "passage: ", 64),
]
lines = ["models:"]
for name, prefix, batch_size in models:
    lines.extend([
        f"  - name: {name}",
        f"    path: {root / name}",
        "    enabled: true",
        f'    prefix: "{prefix}"',
        f"    batch_size: {batch_size}",
        "",
    ])
path.write_text("\n".join(lines), encoding="utf-8")
PY
}

download_model() {
  local model="$1"
  local target="$MODEL_ROOT/$model"
  if [[ -f "$target/config.json" ]]; then
    return
  fi
  local repo
  repo=$(model_repo "$model")
  echo "Downloading $repo to $target"
  HF_HUB_DISABLE_XET=1 "$HF" download "$repo" \
    --exclude "onnx/*" --exclude "openvino/*" --exclude "*.onnx" \
    --exclude "*.bin" --local-dir "$target"
}

train_one() {
  local dataset="$1" model="$2" seed="$3" cache_exp="$4"
  local safe exp log
  safe=$(safe_name "$model")
  exp="review_core_${dataset}_${safe}_seed${seed}"
  log="logs/review_core/${exp}.log"
  if [[ -f "outputs/$exp/runs/${safe}_next_state_vector_metrics.json" ]]; then
    echo "cached $exp"
    return
  fi
  mkdir -p "outputs/$exp"
  if [[ "$exp" != "$cache_exp" ]]; then
    ln -sfn "../$cache_exp/embeddings" "outputs/$exp/embeddings"
  fi
  "$PYTHON" scripts/next_state_vector_prediction.py \
    --no-generate \
    --data-prefix "$dataset" \
    --experiment-name "$exp" \
    --architecture gated_residual_film \
    --models "$model" \
    --models-yaml configs/models.yaml \
    --seed "$seed" \
    --false-negative-mask same_next_text \
    --epochs 30 --patience 5 \
    --train-batch-size 256 --eval-batch-size 512 --encode-batch-size 64 \
    --max-prediction-rows 200 \
    $([[ "$exp" != "$cache_exp" ]] && echo "--skip-cache") \
    >"$log" 2>&1
}

evaluate_one() {
  local dataset="$1" model="$2" seed="$3" cache_exp="$4"
  local safe exp output
  safe=$(safe_name "$model")
  exp="review_core_${dataset}_${safe}_seed${seed}"
  output="outputs/review_core/${dataset}_${safe}_seed${seed}.json"
  if [[ -f "$output" ]]; then
    echo "cached diagnostics $output"
    return
  fi
  "$PYTHON" scripts/eval_review_diagnostics.py \
    --embeddings "outputs/$cache_exp/embeddings/${safe}_test.npz" \
    --checkpoint "outputs/$exp/runs/${safe}_next_state_vector.pt" \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --hard-candidates 50 \
    --random-repeats 10 \
    --output "$output"
}

write_model_config
for model in "${MODELS[@]}"; do
  download_model "$model"
  safe=$(safe_name "$model")
  for dataset in "${DATASETS[@]}"; do
    cache_exp="review_core_${dataset}_${safe}_seed${SEEDS[0]}"
    train_one "$dataset" "$model" "${SEEDS[0]}" "$cache_exp"
    for seed in "${SEEDS[@]:1}"; do
      while [[ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]]; do
        wait -n
      done
      train_one "$dataset" "$model" "$seed" "$cache_exp" &
    done
    wait
    for seed in "${SEEDS[@]}"; do
      evaluate_one "$dataset" "$model" "$seed" "$cache_exp"
    done
  done
done

"$PYTHON" - <<'PY'
import csv
import json
from pathlib import Path

root = Path("outputs/review_core")
rows = []

def flatten(prefix, value, row):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, row)
    else:
        row[prefix] = value

for path in sorted(root.glob("*.json")):
    flat = {}
    flatten("", json.loads(path.read_text(encoding="utf-8")), flat)
    rows.append(flat)

fields = sorted({key for row in rows for key in row})
with (root / "review_core_all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(root / "review_core_all_runs.csv")
PY

echo "review core experiments complete"
