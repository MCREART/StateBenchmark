#!/usr/bin/env bash
set -euo pipefail

ROOT="${STATEBENCH_ROOT:-/root/statebench}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
HF="${HF:-/root/miniconda3/bin/hf}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
MAX_JOBS="${MAX_JOBS:-3}"
SEEDS=(20260504 20260505 20260506)
DATASETS=(openpi_c_nl trip_nl_known_only)
MODES=(none text_matching asymmetric_retrieval)

cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
mkdir -p "$MODEL_ROOT" outputs/review_jina_modes logs/review_jina_modes

model_name() {
  case "$1" in
    none) echo "jina-embeddings-v3-none" ;;
    text_matching) echo "jina-embeddings-v3-text-matching" ;;
    asymmetric_retrieval) echo "jina-embeddings-v3-asymmetric-retrieval" ;;
  esac
}

write_model_config() {
  "$PYTHON" - "$ROOT/configs/models.yaml" "$MODEL_ROOT" <<'PY'
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

  - name: jina-embeddings-v3-text-matching
    path: {model_path}
    enabled: true
    prefix: ""
    batch_size: 32
    encode_kwargs:
      task: text-matching
      prompt_name: text-matching

  - name: jina-embeddings-v3-asymmetric-retrieval
    path: {model_path}
    enabled: true
    prefix: ""
    batch_size: 32
    state_encode_kwargs:
      task: retrieval.query
      prompt_name: retrieval.query
    action_encode_kwargs:
      task: retrieval.query
      prompt_name: retrieval.query
    next_encode_kwargs:
      task: retrieval.passage
      prompt_name: retrieval.passage
""",
    encoding="utf-8",
)
PY
}

train_one() {
  local dataset="$1" mode="$2" seed="$3" cache_exp="$4"
  local model exp log
  model=$(model_name "$mode")
  exp="review_jina_${dataset}_${mode}_seed${seed}"
  log="logs/review_jina_modes/${exp}.log"
  [[ -f "outputs/$exp/runs/${model}_next_state_vector_metrics.json" ]] && return
  mkdir -p "outputs/$exp"
  local cache_args=()
  if [[ "$exp" != "$cache_exp" ]]; then
    ln -sfn "../$cache_exp/embeddings" "outputs/$exp/embeddings"
    cache_args=(--skip-cache)
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
    --train-batch-size 256 --eval-batch-size 512 \
    "${cache_args[@]}" \
    >"$log" 2>&1
}

target="$MODEL_ROOT/jina-embeddings-v3"
if [[ ! -f "$target/config.json" ]]; then
  HF_HUB_DISABLE_XET=1 "$HF" download jinaai/jina-embeddings-v3 \
    --exclude "onnx/*" --exclude "openvino/*" --exclude "*.onnx" \
    --exclude "*.bin" --local-dir "$target"
fi
write_model_config

for mode in "${MODES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    cache_exp="review_jina_${dataset}_${mode}_seed${SEEDS[0]}"
    train_one "$dataset" "$mode" "${SEEDS[0]}" "$cache_exp"
    for seed in "${SEEDS[@]:1}"; do
      while [[ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]]; do
        wait -n
      done
      train_one "$dataset" "$mode" "$seed" "$cache_exp" &
    done
    wait
  done
done

"$PYTHON" - <<'PY'
import csv
import json
import re
from pathlib import Path

rows = []
for path in sorted(Path("outputs").glob("review_jina_*/runs/*_metrics.json")):
    match = re.match(
        r"review_jina_(openpi_c_nl|trip_nl_known_only)_"
        r"(none|text_matching|asymmetric_retrieval)_seed(\d+)",
        path.parents[1].name,
    )
    if not match:
        continue
    dataset, mode, seed = match.groups()
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = metrics["test"]
    rows.append({
        "dataset": dataset,
        "mode": mode,
        "seed": seed,
        "best_epoch": metrics["best_epoch"],
        "top1": test["retrieval_top1"],
        "top5": test["retrieval_top5"],
        "top10": test["retrieval_top10"],
        "row_top1": test["row_retrieval_top1"],
        "cosine": test["cosine_mean"],
    })
output = Path("outputs/review_jina_modes/all_runs.csv")
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset"])
    writer.writeheader()
    writer.writerows(rows)
print(output)
PY

echo "Jina task-mode experiments complete"
