#!/usr/bin/env bash
set -euo pipefail

cd /root/statebench

PY=/root/miniconda3/bin/python
SCRIPT=scripts/eval_scone_rollout.py
MAX_JOBS="${MAX_JOBS:-6}"
LOG_DIR=logs/multistep_dupaware_mask_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/status.tsv"
printf "mode\tdataset\tmodel\tseed\tsteps\tstatus\n" > "$STATUS"

models=("Qwen3-Embedding-0.6B" "bge-large-en-v1.5" "e5-large-v2")
seeds=("20260504" "20260505" "20260506")

run_one() {
  local mode="$1"
  local out_root="$2"
  local dataset="$3"
  local schema="$4"
  local steps="$5"
  local ckpt_prefix="$6"
  local emb_dir="$7"
  local jsonl="$8"
  local model="$9"
  local seed="${10}"
  local out_dir="$out_root/${dataset}_${model}_seed${seed}_${steps}step"
  local log="$LOG_DIR/${mode}_${dataset}_${model}_seed${seed}_${steps}step.log"
  local ckpt="outputs/${ckpt_prefix}_${model}_seed${seed}/runs/${model}_next_state_vector.pt"
  local emb="${emb_dir}/${model}_test.npz"
  local mode_args=()

  if [[ "$mode" == "teacher_forcing" ]]; then
    mode_args=(--mode teacher_forcing)
  fi
  if [[ ! -f "$ckpt" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\tmissing_ckpt\n" "$mode" "$dataset" "$model" "$seed" "$steps" >> "$STATUS"
    return 0
  fi
  if [[ ! -f "$emb" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\tmissing_embeddings\n" "$mode" "$dataset" "$model" "$seed" "$steps" >> "$STATUS"
    return 0
  fi
  mkdir -p "$out_root"
  if "$PY" "$SCRIPT" \
    --checkpoint "$ckpt" \
    --embeddings "$emb" \
    --jsonl "$jsonl" \
    --out-dir "$out_dir" \
    --dataset-name "${dataset}_${model}_seed${seed}" \
    --schema "$schema" \
    "${mode_args[@]}" \
    --steps "$steps" > "$log" 2>&1; then
    printf "%s\t%s\t%s\t%s\t%s\tok\n" "$mode" "$dataset" "$model" "$seed" "$steps" >> "$STATUS"
  else
    printf "%s\t%s\t%s\t%s\t%s\tfailed\n" "$mode" "$dataset" "$model" "$seed" "$steps" >> "$STATUS"
  fi
}

launch_dataset() {
  local mode="$1"
  local out_root="$2"
  local dataset="$3"
  local schema="$4"
  local steps="$5"
  local ckpt_prefix="$6"
  local emb_dir="$7"
  local jsonl="$8"

  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
      while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
        wait -n || true
      done
      run_one "$mode" "$out_root" "$dataset" "$schema" "$steps" "$ckpt_prefix" "$emb_dir" "$jsonl" "$model" "$seed" &
    done
  done
}

launch_dataset "autoregressive" "outputs/multistep_dupaware_mask_3model_3seed" "trip_known" "trip" 3 \
  "paper_gated_dupaware_mask_3seed_trip" \
  "outputs/trip_nl_known_only_residual_film_infonce/embeddings" \
  "data/trip_nl_known_only_test.jsonl"
launch_dataset "autoregressive" "outputs/multistep_dupaware_mask_3model_3seed" "propara_deepseek" "propara" 3 \
  "paper_gated_dupaware_mask_3seed_propara_deepseek" \
  "outputs/propara_deepseek_nl_residual_film_infonce/embeddings" \
  "data/propara_deepseek_nl_test.jsonl"
launch_dataset "autoregressive" "outputs/multistep_dupaware_mask_3model_3seed" "alfworld_nogoto" "alfworld" 3 \
  "paper_gated_dupaware_mask_3seed_alfworld" \
  "outputs/alfworld_nl_no_goto_residual_film_infonce/embeddings" \
  "data/alfworld_nl_no_goto_test.jsonl"
launch_dataset "autoregressive" "outputs/multistep_dupaware_mask_3model_3seed" "scone_full" "scone" 5 \
  "paper_gated_dupaware_mask_3seed_scone_full" \
  "outputs/scone_nl_full_residual_film_infonce/embeddings" \
  "data/scone_nl_full_test.jsonl"

launch_dataset "teacher_forcing" "outputs/table5_teacher_forcing_dupaware_mask_3model_3seed" "scone_full" "scone" 3 \
  "paper_gated_dupaware_mask_3seed_scone_full" \
  "outputs/scone_nl_full_residual_film_infonce/embeddings" \
  "data/scone_nl_full_test.jsonl"
launch_dataset "teacher_forcing" "outputs/table5_teacher_forcing_dupaware_mask_3model_3seed" "scone_full" "scone" 5 \
  "paper_gated_dupaware_mask_3seed_scone_full" \
  "outputs/scone_nl_full_residual_film_infonce/embeddings" \
  "data/scone_nl_full_test.jsonl"
launch_dataset "teacher_forcing" "outputs/table5_teacher_forcing_dupaware_mask_3model_3seed" "alfworld_nogoto" "alfworld" 3 \
  "paper_gated_dupaware_mask_3seed_alfworld" \
  "outputs/alfworld_nl_no_goto_residual_film_infonce/embeddings" \
  "data/alfworld_nl_no_goto_test.jsonl"
launch_dataset "teacher_forcing" "outputs/table5_teacher_forcing_dupaware_mask_3model_3seed" "propara_deepseek" "propara" 3 \
  "paper_gated_dupaware_mask_3seed_propara_deepseek" \
  "outputs/propara_deepseek_nl_residual_film_infonce/embeddings" \
  "data/propara_deepseek_nl_test.jsonl"

while [ "$(jobs -pr | wc -l)" -gt 0 ]; do
  wait -n || true
done

echo "$STATUS"
