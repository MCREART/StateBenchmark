# StateBenchmark

Code and result summaries for next-state vector prediction with frozen text
embedding models. Each example is represented as a triple:

```json
{"text_t": "current state", "action_text": "action or event", "text_t1": "next state"}
```

The transition probe encodes `text_t`, `action_text`, and `text_t1` with a
frozen embedding model, trains a lightweight action-conditioned transition
head with InfoNCE, and evaluates whether the predicted vector retrieves the
correct row-level next-state embedding.

## Contents

- `scripts/next_state_vector_prediction.py`: synthetic-data generation,
  embedding caching, one-step training, and retrieval evaluation.
- `scripts/process_*.py`: converters for public state-tracking datasets.
- `scripts/eval_multistep_rollout.py`: autoregressive and teacher-forced
  multi-step evaluation.
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
  --epochs 2 \
  --train-batch-size 128 \
  --eval-batch-size 256
```

Outputs are written under `outputs/smoke_synthetic/`.

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
  --seed 20260504
```

Repeat with seeds `20260505` and `20260506` for the reported three-seed
protocol.

## Notes on Data Release

The experiments use public datasets and open embedding models with different
licenses. This repository includes code, metadata, and result summaries, but
does not redistribute full processed derivative splits for the anonymous
submission. For assets whose license is not clearly specified in public
metadata, redistribution permissions should be verified before releasing
processed files.
