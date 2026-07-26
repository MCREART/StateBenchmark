#!/usr/bin/env bash
set -euo pipefail

ROOT="${STATEBENCH_ROOT:-/root/statebench}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
MODELS=(Qwen3-Embedding-0.6B bge-large-en-v1.5 e5-large-v2)
DATASETS=(propara_nl propara_deepseek_nl scone_nl_full)

cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
mkdir -p outputs/review_extension_baselines logs/review_extension_baselines

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

safe_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

for model in "${MODELS[@]}"; do
  safe=$(safe_name "$model")
  for dataset in "${DATASETS[@]}"; do
    exp="review_extension_cache_${dataset}_${safe}"
    embeddings="outputs/$exp/embeddings/${safe}_test.npz"
    if [[ ! -f "$embeddings" ]]; then
      "$PYTHON" scripts/next_state_vector_prediction.py \
        --no-generate --cache-only \
        --data-prefix "$dataset" \
        --experiment-name "$exp" \
        --models "$model" \
        --models-yaml configs/models.yaml \
        >"logs/review_extension_baselines/cache_${dataset}_${safe}.log" 2>&1
    fi
    output="outputs/review_extension_baselines/${dataset}_${safe}.json"
    if [[ ! -f "$output" ]]; then
      "$PYTHON" scripts/eval_review_diagnostics.py \
        --embeddings "$embeddings" \
        --dataset "$dataset" \
        --model "$model" \
        --seed 20260504 \
        --hard-candidates 50 \
        --random-repeats 10 \
        --output "$output"
    fi
  done
done

"$PYTHON" - <<'PY'
import csv
import json
from pathlib import Path

rows = []
for path in sorted(Path("outputs/review_extension_baselines").glob("*.json")):
    result = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "dataset": result["dataset"],
        "model": result["model"],
        "num_test": result["num_test"],
        "num_unique_next_states": result["num_unique_next_states"],
        "identity_top1": result["identity"]["top1"],
        "identity_row_top1": result["identity"]["row_top1"],
        "tfidf_top1": result["tfidf_state_action"]["top1"],
        "tfidf_row_top1": result["tfidf_state_action"]["row_top1"],
        "random_top1": result["random"]["top1_mean"],
        "identity_hard_top1": result["identity_hard"]["top1"],
        "tfidf_hard_top1": result["tfidf_hard"]["top1"],
    })
output = Path("outputs/review_extension_baselines/all_runs.csv")
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset"])
    writer.writeheader()
    writer.writerows(rows)
print(output)
PY

echo "extension baselines complete"
