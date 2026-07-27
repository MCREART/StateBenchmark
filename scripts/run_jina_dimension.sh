#!/usr/bin/env bash
set -euo pipefail

ROOT="${STATEBENCH_ROOT:-/root/statebench}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
MAX_JOBS="${MAX_JOBS:-6}"
SEEDS=(20260504 20260505 20260506)
DATASETS=(openpi_c_nl trip_nl_known_only)
DIMENSIONS=(32 64 128 256 512 768)
MODEL="jina-embeddings-v3-none"
CONFIG="outputs/review_jina_dimension/models.yaml"

cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
mkdir -p outputs/review_jina_dimension logs/review_jina_dimension

"$PYTHON" - "$CONFIG" "$MODEL_ROOT" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
model_path = Path(sys.argv[2]) / "jina-embeddings-v3"
path.write_text(
    f"""models:
  - name: jina-embeddings-v3-none
    path: {model_path}
    enabled: true
    prefix: ""
    batch_size: 32
""",
    encoding="utf-8",
)
PY

train_one() {
  local dataset="$1" dimension="$2" seed="$3" cache_exp="$4"
  local exp log
  exp="review_jina_dimension_${dataset}_d${dimension}_seed${seed}"
  log="logs/review_jina_dimension/${exp}.log"
  if [[ -f "outputs/$exp/runs/${MODEL}_next_state_vector_metrics.json" ]]; then
    return
  fi
  mkdir -p "outputs/$exp"
  ln -sfn "../$cache_exp/embeddings" "outputs/$exp/embeddings"
  "$PYTHON" scripts/next_state_vector_prediction.py \
    --no-generate --skip-cache \
    --data-prefix "$dataset" \
    --experiment-name "$exp" \
    --architecture gated_residual_film \
    --models "$MODEL" \
    --models-yaml "$CONFIG" \
    --embedding-dim "$dimension" \
    --seed "$seed" \
    --false-negative-mask same_next_text \
    --epochs 30 --patience 5 \
    --train-batch-size 256 --eval-batch-size 512 \
    >"$log" 2>&1
}

for dataset in "${DATASETS[@]}"; do
  cache_exp="review_jina_${dataset}_none_seed${SEEDS[0]}"
  if [[ ! -d "outputs/$cache_exp/embeddings" ]]; then
    echo "missing cached Jina embeddings: outputs/$cache_exp/embeddings" >&2
    exit 1
  fi
  for dimension in "${DIMENSIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      while [[ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]]; do
        wait -n
      done
      train_one "$dataset" "$dimension" "$seed" "$cache_exp" &
    done
  done
done
wait

"$PYTHON" - <<'PY'
import csv
import json
import re
from pathlib import Path

rows = []
root = Path("outputs")
for path in sorted(root.glob("review_jina_dimension_*/runs/*_metrics.json")):
    match = re.match(
        r"review_jina_dimension_(openpi_c_nl|trip_nl_known_only)_"
        r"d(\d+)_seed(\d+)",
        path.parents[1].name,
    )
    if not match:
        continue
    dataset, dimension, seed = match.groups()
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = metrics["test"]
    rows.append({
        "dataset": dataset,
        "dimension": dimension,
        "seed": seed,
        "best_epoch": metrics["best_epoch"],
        "top1": test["retrieval_top1"],
        "top5": test["retrieval_top5"],
        "top10": test["retrieval_top10"],
        "row_top1": test["row_retrieval_top1"],
        "cosine": test["cosine_mean"],
    })

# The previously completed no-task Jina runs are the 1024-dimensional endpoint.
for path in sorted(root.glob("review_jina_*_none_seed*/runs/*_metrics.json")):
    match = re.match(
        r"review_jina_(openpi_c_nl|trip_nl_known_only)_none_seed(\d+)",
        path.parents[1].name,
    )
    if not match:
        continue
    dataset, seed = match.groups()
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics["embedding_dim"] != 1024:
        continue
    test = metrics["test"]
    rows.append({
        "dataset": dataset,
        "dimension": "1024",
        "seed": seed,
        "best_epoch": metrics["best_epoch"],
        "top1": test["retrieval_top1"],
        "top5": test["retrieval_top5"],
        "top10": test["retrieval_top10"],
        "row_top1": test["row_retrieval_top1"],
        "cosine": test["cosine_mean"],
    })

rows.sort(key=lambda row: (
    row["dataset"], int(row["dimension"]), int(row["seed"])
))
output = Path("outputs/review_jina_dimension/all_runs.csv")
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(rows[0]) if rows else ["dataset"],
    )
    writer.writeheader()
    writer.writerows(rows)
print(output)
PY

echo "Jina dimension experiments complete"
