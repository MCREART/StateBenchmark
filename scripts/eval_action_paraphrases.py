#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

import eval_review_diagnostics as diagnostics
import next_state_vector_prediction as nsvp


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokens(text):
    return set(re.findall(r"[A-Za-z0-9]+", str(text).lower()))


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--paraphrased-data", required=True)
    parser.add_argument("--models-yaml", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=False)
    rows = read_jsonl(Path(args.paraphrased_data))
    if len(rows) != len(data["ids"]):
        raise ValueError(
            f"paraphrase rows ({len(rows)}) != embedding rows ({len(data['ids'])})"
        )
    for index, row in enumerate(rows):
        if str(row["id"]) != str(data["ids"][index]):
            raise ValueError(f"id mismatch at row {index}")

    model_cfg = nsvp.load_models([args.model], args.models_yaml)[0]
    encoder = nsvp.load_encoder(model_cfg)
    try:
        common_kwargs = model_cfg.get("encode_kwargs") or {}
        action_kwargs = model_cfg.get("action_encode_kwargs") or common_kwargs
        paraphrased_actions, used_batch_size = nsvp.encode_with_fallback(
            encoder,
            [row["action_text"] for row in rows],
            int(model_cfg.get("batch_size") or args.encode_batch_size),
            model_cfg.get("prefix", "") or "",
            f"{args.model}:{args.dataset}:paraphrased_action",
            action_kwargs,
        )
    finally:
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    embedding_dim = int(checkpoint["dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = diagnostics.build_model(checkpoint, device)
    z_t = diagnostics.normalize_rows(data["z_t"][:, :embedding_dim])
    z_action = diagnostics.normalize_rows(data["z_action"][:, :embedding_dim])
    z_next = diagnostics.normalize_rows(data["z_next"][:, :embedding_dim])
    paraphrased_actions = diagnostics.normalize_rows(
        paraphrased_actions[:, :embedding_dim]
    )
    original_pred = diagnostics.predict(
        model, z_t, z_action, args.eval_batch_size, device
    )
    paraphrased_pred = diagnostics.predict(
        model, z_t, paraphrased_actions, args.eval_batch_size, device
    )
    targets = data["text_t1"].astype(str)
    original_metrics = diagnostics.retrieval_from_scores(
        original_pred @ z_next.T, targets
    )
    paraphrased_metrics = diagnostics.retrieval_from_scores(
        paraphrased_pred @ z_next.T, targets
    )

    action_overlaps = [
        jaccard(tokens(row["original_action_text"]), tokens(row["action_text"]))
        for row in rows
    ]
    target_original_overlaps = [
        jaccard(tokens(row["original_action_text"]), tokens(row["text_t1"]))
        for row in rows
    ]
    target_paraphrase_overlaps = [
        jaccard(tokens(row["action_text"]), tokens(row["text_t1"]))
        for row in rows
    ]
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "num_test": len(rows),
        "encode_batch_size": used_batch_size,
        "original": original_metrics,
        "paraphrased": paraphrased_metrics,
        "delta_top1": paraphrased_metrics["top1"] - original_metrics["top1"],
        "delta_row_top1": (
            paraphrased_metrics["row_top1"] - original_metrics["row_top1"]
        ),
        "prediction_cosine_original_vs_paraphrased": float(
            np.mean(np.sum(original_pred * paraphrased_pred, axis=1))
        ),
        "action_original_paraphrase_jaccard": float(np.mean(action_overlaps)),
        "target_original_action_jaccard": float(np.mean(target_original_overlaps)),
        "target_paraphrased_action_jaccard": float(
            np.mean(target_paraphrase_overlaps)
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
