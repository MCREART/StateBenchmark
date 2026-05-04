#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
SCRIPT=scripts/eval_multistep_rollout.py
OUT_ROOT=outputs/table5_teacher_forcing_3model_3seed
LOG_DIR=logs/table5_teacher_forcing_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_ROOT" "$LOG_DIR"
STATUS="$LOG_DIR/status.tsv"
printf "dataset\tmodel\tseed\tsteps\tstatus\n" > "$STATUS"

models=("Qwen3-Embedding-0.6B" "bge-large-en-v1.5" "e5-large-v2")
seeds=("20260504" "20260505" "20260506")

run_one() {
  local dataset="$1"
  local schema="$2"
  local steps="$3"
  local ckpt_prefix="$4"
  local emb_dir="$5"
  local jsonl="$6"
  local model="$7"
  local seed="$8"
  local out_dir="$OUT_ROOT/${dataset}_${model}_seed${seed}_${steps}step"
  local log="$LOG_DIR/${dataset}_${model}_seed${seed}_${steps}step.log"
  local ckpt="outputs/${ckpt_prefix}_${model}_seed${seed}/runs/${model}_next_state_vector.pt"
  local emb="${emb_dir}/${model}_test.npz"

  if "$PY" "$SCRIPT" \
    --checkpoint "$ckpt" \
    --embeddings "$emb" \
    --jsonl "$jsonl" \
    --out-dir "$out_dir" \
    --dataset-name "${dataset}_${model}_seed${seed}" \
    --schema "$schema" \
    --mode teacher_forcing \
    --steps "$steps" > "$log" 2>&1; then
    printf "%s\t%s\t%s\t%s\tok\n" "$dataset" "$model" "$seed" "$steps" >> "$STATUS"
  else
    printf "%s\t%s\t%s\t%s\tfailed\n" "$dataset" "$model" "$seed" "$steps" >> "$STATUS"
  fi
}

launch_dataset() {
  local dataset="$1"
  local schema="$2"
  local steps="$3"
  local ckpt_prefix="$4"
  local emb_dir="$5"
  local jsonl="$6"
  local max_jobs="${MAX_JOBS:-3}"

  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
      run_one "$dataset" "$schema" "$steps" "$ckpt_prefix" "$emb_dir" "$jsonl" "$model" "$seed" &
      while (( $(jobs -rp | wc -l) >= max_jobs )); do
        sleep 2
      done
    done
  done
}

launch_dataset "scone_full" "scone" 3 \
  "paper_gated_resfilm_3seed_scone_full" \
  "outputs/scone_nl_full_residual_film_infonce/embeddings" \
  "data/scone_nl_full_test.jsonl"

launch_dataset "scone_full" "scone" 5 \
  "paper_gated_resfilm_3seed_scone_full" \
  "outputs/scone_nl_full_residual_film_infonce/embeddings" \
  "data/scone_nl_full_test.jsonl"

launch_dataset "alfworld_nogoto" "alfworld" 3 \
  "paper_gated_resfilm_3seed_alfworld" \
  "outputs/alfworld_nl_no_goto_residual_film_infonce/embeddings" \
  "data/alfworld_nl_no_goto_test.jsonl"

launch_dataset "propara_deepseek" "propara" 3 \
  "paper_gated_resfilm_3seed_propara_deepseek" \
  "outputs/propara_deepseek_nl_residual_film_infonce/embeddings" \
  "data/propara_deepseek_nl_test.jsonl"

wait
echo "$STATUS"
