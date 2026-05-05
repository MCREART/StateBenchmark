#!/usr/bin/env bash
set -euo pipefail

cd /root/statebench
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

MODELS=(Qwen3-Embedding-0.6B bge-large-en-v1.5 e5-large-v2)
DATASETS=(openpi_c_nl trip_nl_known_only alfworld_nl_no_goto)
ARCHS=(concat film residual_film gated_residual_film state_only action_only)
SEEDS=(20260504 20260505 20260506)
MAX_JOBS="${MAX_JOBS:-6}"
EPOCHS=30
PATIENCE=5
TRAIN_BS=256
EVAL_BS=512
ENCODE_BS=64
EXP_ROOT="ablation_dupaware_mask_3seed"
LOG_DIR="/root/statebench/logs/${EXP_ROOT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/status.tsv"
printf 'status\tdataset\tseed\tarch\tmodel\texperiment\tseconds\n' > "$STATUS"

shared_cache() {
  local DATA="$1"
  echo "/root/statebench/outputs/ablation_5arch_3seed_${DATA}_shared_cache/embeddings"
}

run_one() {
  local DATA="$1" SEED="$2" ARCH="$3" MODEL="$4"
  local EXP="${EXP_ROOT}_${DATA}_${ARCH}_seed${SEED}"
  local SAFE_MODEL
  SAFE_MODEL=$(echo "$MODEL" | sed 's/[^A-Za-z0-9_.-]/_/g')
  local LOG="$LOG_DIR/${EXP}_${SAFE_MODEL}.log"
  local START END SECS
  mkdir -p "/root/statebench/outputs/${EXP}"
  ln -sfn "$(shared_cache "$DATA")" "/root/statebench/outputs/${EXP}/embeddings"
  if [[ -f "/root/statebench/outputs/${EXP}/runs/${SAFE_MODEL}_next_state_vector_metrics.json" ]]; then
    printf 'cached\t%s\t%s\t%s\t%s\t%s\t0\n' "$DATA" "$SEED" "$ARCH" "$MODEL" "$EXP" >> "$STATUS"
    return 0
  fi
  START=$(date +%s)
  echo "===== RUN DATA=$DATA SEED=$SEED ARCH=$ARCH MODEL=$MODEL EXP=$EXP =====" > "$LOG"
  if /root/miniconda3/bin/python /root/statebench/scripts/next_state_vector_prediction.py \
      --no-generate \
      --skip-cache \
      --data-prefix "$DATA" \
      --experiment-name "$EXP" \
      --architecture "$ARCH" \
      --models "$MODEL" \
      --seed "$SEED" \
      --false-negative-mask same_next_text \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --train-batch-size "$TRAIN_BS" \
      --eval-batch-size "$EVAL_BS" \
      --encode-batch-size "$ENCODE_BS" \
      --max-prediction-rows 200 >> "$LOG" 2>&1; then
    END=$(date +%s); SECS=$((END-START))
    printf 'ok\t%s\t%s\t%s\t%s\t%s\t%s\n' "$DATA" "$SEED" "$ARCH" "$MODEL" "$EXP" "$SECS" >> "$STATUS"
  else
    END=$(date +%s); SECS=$((END-START))
    printf 'fail\t%s\t%s\t%s\t%s\t%s\t%s\n' "$DATA" "$SEED" "$ARCH" "$MODEL" "$EXP" "$SECS" >> "$STATUS"
    return 1
  fi
}

FAIL=0
for DATA in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
      for MODEL in "${MODELS[@]}"; do
        while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
          wait -n || FAIL=1
        done
        run_one "$DATA" "$SEED" "$ARCH" "$MODEL" &
      done
    done
  done
done

while [ "$(jobs -pr | wc -l)" -gt 0 ]; do
  wait -n || FAIL=1
done

/root/miniconda3/bin/python - <<'PY'
import csv, json, statistics
from pathlib import Path
root = Path('/root/statebench')
exp_root = 'ablation_dupaware_mask_3seed'
datasets = ['openpi_c_nl','trip_nl_known_only','alfworld_nl_no_goto']
archs = ['concat','film','residual_film','gated_residual_film','state_only','action_only']
seeds = ['20260504','20260505','20260506']
models = ['Qwen3-Embedding-0.6B','bge-large-en-v1.5','e5-large-v2']
rows = []
def safe(name):
    import re
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
for data in datasets:
  for seed in seeds:
    for arch in archs:
      exp = f'{exp_root}_{data}_{arch}_seed{seed}'
      for model in models:
        path = root/'outputs'/exp/'runs'/f'{safe(model)}_next_state_vector_metrics.json'
        if not path.exists():
          rows.append({'dataset':data,'seed':seed,'architecture':arch,'model':model,'status':'missing'})
          continue
        m = json.loads(path.read_text(encoding='utf-8'))
        test = m['test']
        rows.append({
          'dataset':data,'seed':seed,'architecture':arch,'model':model,'status':'ok',
          'best_epoch':m['best_epoch'], 'embedding_dim':m['embedding_dim'],
          'false_negative_mask':m.get('false_negative_mask'),
          'test_cosine_mean':test['cosine_mean'],
          'test_retrieval_top1':test['retrieval_top1'],
          'test_retrieval_top5':test['retrieval_top5'],
          'test_retrieval_top10':test['retrieval_top10'],
          'test_row_retrieval_top1':test.get('row_retrieval_top1'),
          'test_row_retrieval_top5':test.get('row_retrieval_top5'),
          'test_row_retrieval_top10':test.get('row_retrieval_top10'),
          'test_mean_target_rank':test['mean_target_rank'],
        })
out = root/'outputs'/f'{exp_root}_summary.csv'
fields = ['dataset','seed','architecture','model','status','embedding_dim','best_epoch','false_negative_mask','test_cosine_mean','test_retrieval_top1','test_retrieval_top5','test_retrieval_top10','test_row_retrieval_top1','test_row_retrieval_top5','test_row_retrieval_top10','test_mean_target_rank']
with out.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
out2 = root/'outputs'/f'{exp_root}_mean_std.csv'
with out2.open('w', newline='', encoding='utf-8') as f:
    fields = ['dataset','architecture','model','n_seeds','top1_mean','top1_std','top5_mean','top5_std','top10_mean','top10_std','row_top1_mean','row_top1_std','cosine_mean','cosine_std']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for data in datasets:
      for arch in archs:
        for model in models:
          vals=[r for r in rows if r['dataset']==data and r['architecture']==arch and r['model']==model and r['status']=='ok']
          if not vals:
            w.writerow({'dataset':data,'architecture':arch,'model':model,'n_seeds':0}); continue
          def xs(k): return [float(v[k]) for v in vals if v.get(k) not in (None, '')]
          def mean(k): return statistics.mean(xs(k))
          def std(k):
            vals_k=xs(k)
            return statistics.stdev(vals_k) if len(vals_k)>1 else 0.0
          w.writerow({'dataset':data,'architecture':arch,'model':model,'n_seeds':len(vals),'top1_mean':mean('test_retrieval_top1'),'top1_std':std('test_retrieval_top1'),'top5_mean':mean('test_retrieval_top5'),'top5_std':std('test_retrieval_top5'),'top10_mean':mean('test_retrieval_top10'),'top10_std':std('test_retrieval_top10'),'row_top1_mean':mean('test_row_retrieval_top1'),'row_top1_std':std('test_row_retrieval_top1'),'cosine_mean':mean('test_cosine_mean'),'cosine_std':std('test_cosine_mean')})
print(out)
print(out2)
PY

echo "ALL DONE fail=$FAIL"
exit "$FAIL"
