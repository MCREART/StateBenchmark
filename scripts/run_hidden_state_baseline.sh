#!/usr/bin/env bash
set -euo pipefail

ROOT="${STATEBENCH_ROOT:-/root/statebench}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
HF="${HF:-/root/miniconda3/bin/hf}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
MAX_JOBS="${MAX_JOBS:-3}"
MODEL_NAME="Qwen3-0.6B-Base-meanpool"
SEEDS=(20260504 20260505 20260506)
DATASETS=(openpi_c_nl trip_nl_known_only)

cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
mkdir -p "$MODEL_ROOT" outputs/review_hidden_state logs/review_hidden_state

base_model="$MODEL_ROOT/Qwen3-0.6B-Base"
if [[ ! -f "$base_model/config.json" ]]; then
  HF_HUB_DISABLE_XET=1 "$HF" download Qwen/Qwen3-0.6B-Base \
    --exclude "onnx/*" --exclude "openvino/*" --exclude "*.onnx" \
    --exclude "*.bin" --local-dir "$base_model"
fi

"$PYTHON" - "$ROOT/configs/models.yaml" "$base_model" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    f"""models:
  - name: Qwen3-0.6B-Base-meanpool
    path: {sys.argv[2]}
    enabled: true
    prefix: ""
    batch_size: 16
""",
    encoding="utf-8",
)
PY

train_one() {
  local dataset="$1" seed="$2" cache_exp="$3"
  local exp="review_hidden_${dataset}_seed${seed}"
  local log="logs/review_hidden_state/${exp}.log"
  [[ -f "outputs/$exp/runs/${MODEL_NAME}_next_state_vector_metrics.json" ]] && return
  mkdir -p "outputs/$exp"
  ln -sfn "../$cache_exp/embeddings" "outputs/$exp/embeddings"
  "$PYTHON" scripts/next_state_vector_prediction.py \
    --no-generate --skip-cache \
    --data-prefix "$dataset" \
    --experiment-name "$exp" \
    --architecture gated_residual_film \
    --models "$MODEL_NAME" \
    --models-yaml configs/models.yaml \
    --seed "$seed" \
    --false-negative-mask same_next_text \
    --epochs 30 --patience 5 \
    --train-batch-size 256 --eval-batch-size 512 \
    >"$log" 2>&1
}

for dataset in "${DATASETS[@]}"; do
  cache_exp="review_hidden_${dataset}_cache"
  cache_dir="outputs/$cache_exp/embeddings"
  "$PYTHON" scripts/cache_hidden_state_embeddings.py \
    --model-path "$base_model" \
    --data-dir data \
    --data-prefix "$dataset" \
    --output-dir "$cache_dir" \
    --cache-name "$MODEL_NAME" \
    >"logs/review_hidden_state/cache_${dataset}.log" 2>&1
  for seed in "${SEEDS[@]}"; do
    while [[ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]]; do
      wait -n
    done
    train_one "$dataset" "$seed" "$cache_exp" &
  done
  wait
done

"$PYTHON" - <<'PY'
import csv
import json
import re
from pathlib import Path

rows = []
for path in sorted(Path("outputs").glob("review_hidden_*/runs/*_metrics.json")):
    match = re.match(
        r"review_hidden_(openpi_c_nl|trip_nl_known_only)_seed(\d+)",
        path.parents[1].name,
    )
    if not match:
        continue
    dataset, seed = match.groups()
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = metrics["test"]
    rows.append({
        "dataset": dataset,
        "seed": seed,
        "top1": test["retrieval_top1"],
        "top5": test["retrieval_top5"],
        "top10": test["retrieval_top10"],
        "row_top1": test["row_retrieval_top1"],
        "cosine": test["cosine_mean"],
    })
output = Path("outputs/review_hidden_state/all_runs.csv")
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset"])
    writer.writeheader()
    writer.writerows(rows)
print(output)
PY

echo "hidden-state baseline complete"
