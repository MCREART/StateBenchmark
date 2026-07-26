# StateBenchmark

Code and result summaries for next-state vector prediction with frozen text
embedding models. Each example is represented as a triple:

```json
{"text_t": "current state", "action_text": "action or event", "text_t1": "next state"}
```

The transition probe encodes `text_t`, `action_text`, and `text_t1` with a
frozen embedding model, trains a lightweight action-conditioned transition
head with InfoNCE, and evaluates whether the predicted vector retrieves an
equivalent next-state embedding. Exact duplicate triples are removed during
preprocessing; if multiple examples share the same normalized `text_t1`, the
main retrieval metric counts any matching next-state text as correct and the
trainer can mask same-next-state in-batch false negatives.

## Contents

- `scripts/next_state_vector_prediction.py`: synthetic-data generation,
  embedding caching, one-step training, and retrieval evaluation.
- `scripts/process_*.py`: converters for public state-tracking datasets.
- `scripts/eval_multistep_rollout.py`: autoregressive and teacher-forced
  multi-step evaluation.
- `scripts/run_paper_gated_dupaware_mask.sh`: three-seed gated residual FiLM
  runner for the paper main tables.
- `scripts/run_ablation_dupaware_mask.sh`: three-seed architecture ablation
  runner for the paper ablation table.
- `scripts/run_multistep_dupaware_mask.sh`: autoregressive and teacher-forced
  multi-step runner for the paper rollout table.
- `scripts/eval_review_diagnostics.py`: identity, random, lexical, hard-negative,
  action-shuffle, counterfactual-action, and metric-sensitivity diagnostics.
- `scripts/run_review_core.sh`: three-seed reviewer-diagnostic runner over the
  three core datasets and the Qwen3-0.6B, BGE-large, and E5-large encoders.
- `scripts/run_review_extension_baselines.sh`: cache-only identity, random,
  lexical, and hard-negative diagnostics on ProPara, ProPara-DeepSeek, and
  SCONE.
- `scripts/run_qwen_scale_dimension.sh`: Qwen 0.6B/4B/8B scale comparison and
  Qwen-8B Matryoshka dimension ablation.
- `scripts/run_jina_task_modes.sh`: Jina no-adapter, text-matching, and
  asymmetric query/passage task-mode comparison.
- `scripts/run_hidden_state_baseline.sh`: mean-pooled final hidden-state
  baseline using a next-token-pretrained Qwen base model.
- `scripts/run_action_paraphrase_eval.sh`: low-lexical-overlap action
  paraphrase robustness evaluation using existing three-seed checkpoints.
- `results/summary/`: CSV summaries used for the paper tables.
- `processed_metadata/`: split-size and conversion metadata. Processed JSONL
  splits are not bundled in this anonymous repository; recreate them from the
  public source datasets using the processing scripts and respect original
  dataset licenses and terms.
- `paper/`: LaTeX source for the anonymous draft.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `configs/models.example.yaml` to `configs/models.yaml` and edit the
`path` fields to point to local SentenceTransformers-compatible model
directories or Hugging Face model IDs.

```bash
cp configs/models.example.yaml configs/models.yaml
```

By default, the trainer sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
to match the cached-model experiments. If you want Hugging Face downloads,
unset those environment variables before running or edit the script.

## Quick Smoke Test

This command generates the controlled synthetic benchmark and trains one
small probe:

```bash
python scripts/next_state_vector_prediction.py \
  --models all-MiniLM-L6-v2 \
  --models-yaml configs/models.yaml \
  --experiment-name smoke_synthetic \
  --false-negative-mask same_next_text \
  --epochs 2 \
  --train-batch-size 128 \
  --eval-batch-size 256
```

Outputs are written under `outputs/smoke_synthetic/`.

## Reviewer Diagnostics

On a fresh single-GPU server with the processed JSONL files linked under
`data/`, run:

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/run_review_core.sh
```

The runner downloads missing public models, caches each dataset/model embedding
set once, trains gated residual FiLM with three seeds, and writes flattened
diagnostic results to `outputs/review_core/review_core_all_runs.csv`. The
state-matched hard-negative protocol forms each candidate set from next states
that are most lexically similar to the current state, while always retaining
the gold candidate. This controls for shared entities and scene context.

## Public Dataset Workflow

The public-dataset experiments expect JSONL files under `data/` with the common
schema `text_t`, `action_text`, and `text_t1`. Use the conversion scripts as
templates for recreating processed splits from local copies of ProPara,
OpenPI-C, SCONE, ALFWorld, and TRIP.

Example one-step run after creating `data/openpi_c_nl_train.jsonl`,
`data/openpi_c_nl_dev.jsonl`, and `data/openpi_c_nl_test.jsonl`:

```bash
python scripts/next_state_vector_prediction.py \
  --no-generate \
  --data-prefix openpi_c_nl \
  --experiment-name openpi_c_nl_gated_resfilm \
  --architecture gated_residual_film \
  --models Qwen3-Embedding-0.6B bge-large-en-v1.5 e5-large-v2 \
  --models-yaml configs/models.yaml \
  --seed 20260504 \
  --false-negative-mask same_next_text
```

Repeat with seeds `20260505` and `20260506` for the reported three-seed
protocol.

## Paper Experiment Runners

The paper runners are designed for the cached server layout used in the
experiments (`/root/statebench`). They reuse embedding caches when present and
write outputs under `outputs/`. On a fresh machine, first recreate the `data/`
JSONL splits and `configs/models.yaml`, then either adjust the cache paths in
the runner scripts or let `scripts/next_state_vector_prediction.py` regenerate
embeddings. The official duplicate-aware runs are:

```bash
bash scripts/run_paper_gated_dupaware_mask.sh
bash scripts/run_ablation_dupaware_mask.sh
bash scripts/run_multistep_dupaware_mask.sh
```

For the current server backup, the ignored local file `configs/models.yaml`
has been copied from the server so the same model names and paths are
available for rerunning on that server image.

## Notes on Data Release

The experiments use public datasets and open embedding models with different
licenses. This repository includes code, metadata, and result summaries, but
does not redistribute full processed derivative splits for the anonymous
submission. For assets whose license is not clearly specified in public
metadata, redistribution permissions should be verified before releasing
processed files.
