#!/usr/bin/env bash
set -euo pipefail

ROOT="${STATEBENCH_ROOT:-/root/statebench}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
SEEDS=(20260504 20260505 20260506)
DATASETS=(openpi_c_nl trip_nl_known_only alfworld_nl_no_goto)
MODELS=(Qwen3-Embedding-0.6B bge-large-en-v1.5 e5-large-v2)

cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
mkdir -p data/review_action_paraphrases outputs/review_action_paraphrases \
  logs/review_action_paraphrases

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

for dataset in "${DATASETS[@]}"; do
  paraphrased="data/review_action_paraphrases/${dataset}_test.jsonl"
  if [[ ! -f "$paraphrased" ]]; then
    "$PYTHON" scripts/rewrite_actions_llm.py \
      --input "data/${dataset}_test.jsonl" \
      --output "$paraphrased" \
      --model deepseek-v4-flash \
      --concurrency 20 \
      >"logs/review_action_paraphrases/rewrite_${dataset}.log" 2>&1
  fi
  for model in "${MODELS[@]}"; do
    safe=$(safe_name "$model")
    cache_exp="review_core_${dataset}_${safe}_seed${SEEDS[0]}"
    embeddings="outputs/$cache_exp/embeddings/${safe}_test.npz"
    for seed in "${SEEDS[@]}"; do
      exp="review_core_${dataset}_${safe}_seed${seed}"
      checkpoint="outputs/$exp/runs/${safe}_next_state_vector.pt"
      output="outputs/review_action_paraphrases/${dataset}_${safe}_seed${seed}.json"
      [[ -f "$output" ]] && continue
      "$PYTHON" scripts/eval_action_paraphrases.py \
        --embeddings "$embeddings" \
        --checkpoint "$checkpoint" \
        --paraphrased-data "$paraphrased" \
        --models-yaml configs/models.yaml \
        --model "$model" \
        --dataset "$dataset" \
        --seed "$seed" \
        --output "$output"
    done
  done
done

"$PYTHON" - <<'PY'
import csv
import json
from pathlib import Path

rows = []
for path in sorted(Path("outputs/review_action_paraphrases").glob("*.json")):
    result = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "dataset": result["dataset"],
        "model": result["model"],
        "seed": result["seed"],
        "original_top1": result["original"]["top1"],
        "paraphrased_top1": result["paraphrased"]["top1"],
        "delta_top1": result["delta_top1"],
        "original_row_top1": result["original"]["row_top1"],
        "paraphrased_row_top1": result["paraphrased"]["row_top1"],
        "delta_row_top1": result["delta_row_top1"],
        "prediction_cosine": result[
            "prediction_cosine_original_vs_paraphrased"
        ],
        "action_jaccard": result["action_original_paraphrase_jaccard"],
        "target_original_jaccard": result["target_original_action_jaccard"],
        "target_paraphrased_jaccard": result[
            "target_paraphrased_action_jaccard"
        ],
    })
output = Path("outputs/review_action_paraphrases/all_runs.csv")
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset"])
    writer.writeheader()
    writer.writerows(rows)
print(output)
PY

echo "action paraphrase evaluation complete"
