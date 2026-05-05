#!/usr/bin/env bash
set -euo pipefail

cd /root/statebench
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

EXP_ROOT=paper_gated_dupaware_mask_3seed
ARCH=gated_residual_film
SEEDS=(20260504 20260505 20260506)
MAX_JOBS="${MAX_JOBS:-6}"
LOG_DIR="/root/statebench/logs/${EXP_ROOT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/status.tsv"
printf 'status\tdataset\tmodel\tseed\texperiment\tseconds\n' > "$STATUS"

safe_model() { echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'; }
short_data() {
  case "$1" in
    openpi_c_nl) echo openpi ;;
    trip_nl_known_only) echo trip ;;
    alfworld_nl_no_goto) echo alfworld ;;
    propara_nl) echo propara ;;
    propara_deepseek_nl) echo propara_deepseek ;;
    scone_nl_full) echo scone_full ;;
  esac
}
shared_cache() {
  local DATA="$1" MODEL="$2"
  case "$DATA:$MODEL" in
    openpi_c_nl:Qwen3-Embedding-0.6B|openpi_c_nl:bge-large-en-v1.5|openpi_c_nl:e5-large-v2) echo "/root/statebench/outputs/ablation_5arch_3seed_openpi_c_nl_shared_cache/embeddings" ;;
    trip_nl_known_only:Qwen3-Embedding-0.6B|trip_nl_known_only:bge-large-en-v1.5|trip_nl_known_only:e5-large-v2) echo "/root/statebench/outputs/ablation_5arch_3seed_trip_nl_known_only_shared_cache/embeddings" ;;
    alfworld_nl_no_goto:Qwen3-Embedding-0.6B|alfworld_nl_no_goto:bge-large-en-v1.5|alfworld_nl_no_goto:e5-large-v2) echo "/root/statebench/outputs/ablation_5arch_3seed_alfworld_nl_no_goto_shared_cache/embeddings" ;;
    openpi_c_nl:Qwen3-Embedding-8B|openpi_c_nl:jina-embeddings-v3|openpi_c_nl:all-MiniLM-L6-v2) echo "/root/statebench/outputs/openpi_c_nl_encoder_sweep_extra_residual_film_infonce/embeddings" ;;
    trip_nl_known_only:Qwen3-Embedding-8B|trip_nl_known_only:jina-embeddings-v3|trip_nl_known_only:all-MiniLM-L6-v2) echo "/root/statebench/outputs/trip_nl_known_only_encoder_sweep_extra_residual_film_infonce/embeddings" ;;
    alfworld_nl_no_goto:Qwen3-Embedding-8B|alfworld_nl_no_goto:jina-embeddings-v3|alfworld_nl_no_goto:all-MiniLM-L6-v2) echo "/root/statebench/outputs/alfworld_nl_no_goto_encoder_sweep_extra_residual_film_infonce/embeddings" ;;
    propara_nl:*) echo "/root/statebench/outputs/propara_nl_changed_residual_film_infonce/embeddings" ;;
    propara_deepseek_nl:*) echo "/root/statebench/outputs/propara_deepseek_nl_residual_film_infonce/embeddings" ;;
    scone_nl_full:*) echo "/root/statebench/outputs/scone_nl_full_residual_film_infonce/embeddings" ;;
    *) return 1 ;;
  esac
}

run_one() {
  local DATA="$1" MODEL="$2" SEED="$3"
  local SAFE SHORT EXP SHARED LOG START END SECS
  SAFE=$(safe_model "$MODEL")
  SHORT=$(short_data "$DATA")
  SHARED=$(shared_cache "$DATA" "$MODEL")
  EXP="${EXP_ROOT}_${SHORT}_${SAFE}_seed${SEED}"
  LOG="$LOG_DIR/${EXP}.log"
  mkdir -p "/root/statebench/outputs/${EXP}"
  ln -sfn "$SHARED" "/root/statebench/outputs/${EXP}/embeddings"
  if [[ -f "/root/statebench/outputs/${EXP}/runs/${SAFE}_next_state_vector_metrics.json" ]]; then
    printf 'cached\t%s\t%s\t%s\t%s\t0\n' "$DATA" "$MODEL" "$SEED" "$EXP" >> "$STATUS"
    return 0
  fi
  START=$(date +%s)
  if /root/miniconda3/bin/python /root/statebench/scripts/next_state_vector_prediction.py \
      --no-generate --skip-cache \
      --data-prefix "$DATA" \
      --experiment-name "$EXP" \
      --architecture "$ARCH" \
      --models "$MODEL" \
      --seed "$SEED" \
      --false-negative-mask same_next_text \
      --epochs 30 --patience 5 \
      --train-batch-size 256 --eval-batch-size 512 --encode-batch-size 64 \
      --max-prediction-rows 200 > "$LOG" 2>&1; then
    END=$(date +%s); SECS=$((END-START))
    printf 'ok\t%s\t%s\t%s\t%s\t%s\n' "$DATA" "$MODEL" "$SEED" "$EXP" "$SECS" >> "$STATUS"
  else
    END=$(date +%s); SECS=$((END-START))
    printf 'fail\t%s\t%s\t%s\t%s\t%s\n' "$DATA" "$MODEL" "$SEED" "$EXP" "$SECS" >> "$STATUS"
    return 1
  fi
}

TASKS=/tmp/${EXP_ROOT}_tasks.tsv
: > "$TASKS"
CORE_MODELS=(Qwen3-Embedding-0.6B Qwen3-Embedding-8B jina-embeddings-v3 all-MiniLM-L6-v2 bge-large-en-v1.5 e5-large-v2)
for DATA in openpi_c_nl trip_nl_known_only alfworld_nl_no_goto; do
  for MODEL in "${CORE_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      printf '%s\t%s\t%s\n' "$DATA" "$MODEL" "$SEED" >> "$TASKS"
    done
  done
done
EXT_MODELS=(Qwen3-Embedding-0.6B bge-large-en-v1.5 e5-large-v2)
for DATA in propara_nl propara_deepseek_nl scone_nl_full; do
  for MODEL in "${EXT_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      printf '%s\t%s\t%s\n' "$DATA" "$MODEL" "$SEED" >> "$TASKS"
    done
  done
done

FAIL=0
while IFS=$'\t' read -r DATA MODEL SEED; do
  while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n || FAIL=1
  done
  run_one "$DATA" "$MODEL" "$SEED" &
done < "$TASKS"
while [ "$(jobs -pr | wc -l)" -gt 0 ]; do
  wait -n || FAIL=1
done

/root/miniconda3/bin/python - <<'PY'
import csv, json, re, statistics
from pathlib import Path
root=Path('/root/statebench')
exp_root='paper_gated_dupaware_mask_3seed'
seeds=['20260504','20260505','20260506']
core_models=['Qwen3-Embedding-0.6B','Qwen3-Embedding-8B','jina-embeddings-v3','all-MiniLM-L6-v2','bge-large-en-v1.5','e5-large-v2']
ext_models=['Qwen3-Embedding-0.6B','bge-large-en-v1.5','e5-large-v2']
tasks=[]
for d in ['openpi_c_nl','trip_nl_known_only','alfworld_nl_no_goto']:
  for m in core_models:
    for seed in seeds: tasks.append((d,m,seed))
for d in ['propara_nl','propara_deepseek_nl','scone_nl_full']:
  for m in ext_models:
    for seed in seeds: tasks.append((d,m,seed))
def safe(x): return re.sub(r'[^A-Za-z0-9_.-]+','_',x)
def short(d): return {'openpi_c_nl':'openpi','trip_nl_known_only':'trip','alfworld_nl_no_goto':'alfworld','propara_nl':'propara','propara_deepseek_nl':'propara_deepseek','scone_nl_full':'scone_full'}[d]
rows=[]
for d,m,seed in tasks:
  exp=f'{exp_root}_{short(d)}_{safe(m)}_seed{seed}'
  path=root/'outputs'/exp/'runs'/f'{safe(m)}_next_state_vector_metrics.json'
  if not path.exists():
    rows.append({'dataset':d,'model':m,'seed':seed,'status':'missing'})
    continue
  j=json.loads(path.read_text(encoding='utf-8'))
  test=j['test']
  rows.append({'dataset':d,'model':m,'seed':seed,'status':'ok','best_epoch':j['best_epoch'],'false_negative_mask':j.get('false_negative_mask'),'top1':test['retrieval_top1'],'top5':test['retrieval_top5'],'top10':test['retrieval_top10'],'row_top1':test.get('row_retrieval_top1'),'row_top5':test.get('row_retrieval_top5'),'row_top10':test.get('row_retrieval_top10'),'cosine':test['cosine_mean'],'rank':test['mean_target_rank']})
out=root/'outputs'/f'{exp_root}_summary.csv'
with out.open('w', newline='', encoding='utf-8') as f:
  fields=['dataset','model','seed','status','best_epoch','false_negative_mask','top1','top5','top10','row_top1','row_top5','row_top10','cosine','rank']
  w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
out2=root/'outputs'/f'{exp_root}_mean_std.csv'
with out2.open('w', newline='', encoding='utf-8') as f:
  fields=['dataset','model','n_seeds','top1_mean','top1_std','top5_mean','top5_std','top10_mean','top10_std','row_top1_mean','row_top1_std','cosine_mean','cosine_std','rank_mean']
  w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
  for d,m in sorted(set((r['dataset'],r['model']) for r in rows)):
    vals=[r for r in rows if r['dataset']==d and r['model']==m and r['status']=='ok']
    if not vals:
      w.writerow({'dataset':d,'model':m,'n_seeds':0}); continue
    def mean(k): return statistics.mean(float(x[k]) for x in vals if x.get(k) not in (None, ''))
    def std(k):
      xs=[float(x[k]) for x in vals if x.get(k) not in (None, '')]
      return statistics.stdev(xs) if len(xs)>1 else 0.0
    w.writerow({'dataset':d,'model':m,'n_seeds':len(vals),'top1_mean':mean('top1'),'top1_std':std('top1'),'top5_mean':mean('top5'),'top5_std':std('top5'),'top10_mean':mean('top10'),'top10_std':std('top10'),'row_top1_mean':mean('row_top1'),'row_top1_std':std('row_top1'),'cosine_mean':mean('cosine'),'cosine_std':std('cosine'),'rank_mean':mean('rank')})
print(out)
print(out2)
PY

echo "ALL DONE fail=$FAIL"
exit "$FAIL"
