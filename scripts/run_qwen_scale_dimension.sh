#!/usr/bin/env bash
set -euo pipefail

ROOT="${STATEBENCH_ROOT:-/root/statebench}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
HF="${HF:-/root/miniconda3/bin/hf}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
MAX_JOBS="${MAX_JOBS:-2}"
SEEDS=(20260504 20260505 20260506)
DATASETS=(openpi_c_nl trip_nl_known_only)
MODELS=(Qwen3-Embedding-0.6B Qwen3-Embedding-4B Qwen3-Embedding-8B)
DIMENSIONS=(256 512 1024 2048 4096)

cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
mkdir -p "$MODEL_ROOT" outputs/review_qwen_scale_dimension logs/review_qwen_scale_dimension

safe_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

model_repo() {
  case "$1" in
    Qwen3-Embedding-0.6B) echo "Qwen/Qwen3-Embedding-0.6B" ;;
    Qwen3-Embedding-4B) echo "Qwen/Qwen3-Embedding-4B" ;;
    Qwen3-Embedding-8B) echo "Qwen/Qwen3-Embedding-8B" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

download_model() {
  local model="$1" target="$MODEL_ROOT/$1"
  [[ -f "$target/config.json" ]] && return
  HF_HUB_DISABLE_XET=1 "$HF" download "$(model_repo "$model")" --local-dir "$target"
}

write_model_config() {
  "$PYTHON" - "$ROOT/configs/models.yaml" "$MODEL_ROOT" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = Path(sys.argv[2])
models = [
    ("Qwen3-Embedding-0.6B", 32),
    ("Qwen3-Embedding-4B", 12),
    ("Qwen3-Embedding-8B", 6),
]
lines = ["models:"]
for name, batch_size in models:
    lines.extend([
        f"  - name: {name}",
        f"    path: {root / name}",
        "    enabled: true",
        '    prefix: ""',
        f"    batch_size: {batch_size}",
        "",
    ])
path.write_text("\n".join(lines), encoding="utf-8")
PY
}

train_one() {
  local dataset="$1" model="$2" seed="$3" cache_exp="$4" dimension="$5"
  local safe exp dim_arg log
  safe=$(safe_name "$model")
  if [[ "$dimension" == "full" ]]; then
    exp="review_scale_${dataset}_${safe}_seed${seed}"
    dim_arg=()
  else
    exp="review_dimension_${dataset}_${safe}_d${dimension}_seed${seed}"
    dim_arg=(--embedding-dim "$dimension")
  fi
  log="logs/review_qwen_scale_dimension/${exp}.log"
  if [[ -f "outputs/$exp/runs/${safe}_next_state_vector_metrics.json" ]]; then
    return
  fi
  mkdir -p "outputs/$exp"
  if [[ "$exp" != "$cache_exp" ]]; then
    ln -sfn "../$cache_exp/embeddings" "outputs/$exp/embeddings"
  fi
  "$PYTHON" scripts/next_state_vector_prediction.py \
    --no-generate --skip-cache \
    --data-prefix "$dataset" \
    --experiment-name "$exp" \
    --architecture gated_residual_film \
    --models "$model" \
    --models-yaml configs/models.yaml \
    --seed "$seed" \
    --false-negative-mask same_next_text \
    --epochs 30 --patience 5 \
    --train-batch-size 256 --eval-batch-size 512 \
    "${dim_arg[@]}" \
    >"$log" 2>&1
}

cache_and_train_first_seed() {
  local dataset="$1" model="$2"
  local safe exp log core_exp
  local cache_args=()
  safe=$(safe_name "$model")
  exp="review_scale_${dataset}_${safe}_seed${SEEDS[0]}"
  log="logs/review_qwen_scale_dimension/${exp}.log"
  if [[ -f "outputs/$exp/runs/${safe}_next_state_vector_metrics.json" ]]; then
    return
  fi
  core_exp="review_core_${dataset}_${safe}_seed${SEEDS[0]}"
  if [[ -d "outputs/$core_exp/embeddings" ]]; then
    mkdir -p "outputs/$exp"
    ln -sfn "../$core_exp/embeddings" "outputs/$exp/embeddings"
    cache_args=(--skip-cache)
  fi
  "$PYTHON" scripts/next_state_vector_prediction.py \
    --no-generate \
    --data-prefix "$dataset" \
    --experiment-name "$exp" \
    --architecture gated_residual_film \
    --models "$model" \
    --models-yaml configs/models.yaml \
    --seed "${SEEDS[0]}" \
    --false-negative-mask same_next_text \
    --epochs 30 --patience 5 \
    --train-batch-size 256 --eval-batch-size 512 \
    "${cache_args[@]}" \
    >"$log" 2>&1
}

write_model_config
for model in "${MODELS[@]}"; do
  download_model "$model"
  safe=$(safe_name "$model")
  for dataset in "${DATASETS[@]}"; do
    cache_exp="review_scale_${dataset}_${safe}_seed${SEEDS[0]}"
    cache_and_train_first_seed "$dataset" "$model"
    for seed in "${SEEDS[@]:1}"; do
      train_one "$dataset" "$model" "$seed" "$cache_exp" full &
    done
    wait
    if [[ "$model" == "Qwen3-Embedding-8B" ]]; then
      for dimension in "${DIMENSIONS[@]}"; do
        [[ "$dimension" == "4096" ]] && continue
        for seed in "${SEEDS[@]}"; do
          while [[ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]]; do
            wait -n
          done
          train_one "$dataset" "$model" "$seed" "$cache_exp" "$dimension" &
        done
      done
      wait
    fi
  done
done

"$PYTHON" - <<'PY'
import csv
import json
import re
from pathlib import Path

root = Path("outputs")
rows = []
for path in sorted(root.glob("review_scale_*/runs/*_metrics.json")) + sorted(
    root.glob("review_dimension_*/runs/*_metrics.json")
):
    match = re.match(
        r"review_(scale|dimension)_(openpi_c_nl|trip_nl_known_only)_"
        r"(Qwen3-Embedding-[^_]+)(?:_d(\d+))?_seed(\d+)",
        path.parents[1].name,
    )
    if not match:
        continue
    kind, dataset, model, requested_dim, seed = match.groups()
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = metrics["test"]
    rows.append({
        "kind": kind,
        "dataset": dataset,
        "model": model,
        "requested_dim": requested_dim or metrics["embedding_dim"],
        "seed": seed,
        "best_epoch": metrics["best_epoch"],
        "top1": test["retrieval_top1"],
        "top5": test["retrieval_top5"],
        "top10": test["retrieval_top10"],
        "row_top1": test["row_retrieval_top1"],
        "cosine": test["cosine_mean"],
    })
output = Path("outputs/review_qwen_scale_dimension/all_runs.csv")
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["kind"])
    writer.writeheader()
    writer.writerows(rows)
print(output)
PY

echo "Qwen scale and dimension experiments complete"
