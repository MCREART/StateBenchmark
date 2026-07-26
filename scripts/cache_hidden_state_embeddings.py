#!/usr/bin/env python3
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(1e-8)
    return F.normalize(summed / denominator, dim=1)


def encode(model, tokenizer, texts, batch_size, max_length, label):
    chunks = []
    start = 0
    active_batch_size = batch_size
    while start < len(texts):
        batch = texts[start:start + active_batch_size]
        try:
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(model.device) for key, value in tokens.items()}
            with torch.inference_mode():
                output = model(**tokens)
                pooled = mean_pool(output.last_hidden_state, tokens["attention_mask"])
            chunks.append(pooled.float().cpu().numpy())
            start += len(batch)
            if start % max(active_batch_size * 20, 1) == 0 or start == len(texts):
                print(f"{label}: {start}/{len(texts)}", flush=True)
        except torch.cuda.OutOfMemoryError:
            if active_batch_size <= 1:
                raise
            active_batch_size = max(1, active_batch_size // 2)
            torch.cuda.empty_cache()
            gc.collect()
            print(f"{label}: reducing batch size to {active_batch_size}", flush=True)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-name", default="Qwen3-0.6B-Base-meanpool")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        split for split in ("train", "dev", "test")
        if not (output_dir / f"{args.cache_name}_{split}.npz").exists()
    ]
    if not missing:
        print("all hidden-state embedding caches already exist")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    for split in missing:
        rows = read_jsonl(Path(args.data_dir) / f"{args.data_prefix}_{split}.jsonl")
        z_t = encode(
            model, tokenizer, [row["text_t"] for row in rows],
            args.batch_size, args.max_length, f"{split}:state",
        )
        z_action = encode(
            model, tokenizer, [row["action_text"] for row in rows],
            args.batch_size, args.max_length, f"{split}:action",
        )
        z_next = encode(
            model, tokenizer, [row["text_t1"] for row in rows],
            args.batch_size, args.max_length, f"{split}:next",
        )
        path = output_dir / f"{args.cache_name}_{split}.npz"
        np.savez_compressed(
            path,
            z_t=z_t,
            z_action=z_action,
            z_next=z_next,
            ids=np.asarray([row["id"] for row in rows], dtype=str),
            ranges=np.asarray([row["range"] for row in rows], dtype=str),
            transition_types=np.asarray(
                [row["transition_type"] for row in rows], dtype=str
            ),
            text_t=np.asarray([row["text_t"] for row in rows], dtype=str),
            action_text=np.asarray([row["action_text"] for row in rows], dtype=str),
            text_t1=np.asarray([row["text_t1"] for row in rows], dtype=str),
        )
        print(f"wrote {path} {z_t.shape}", flush=True)


if __name__ == "__main__":
    main()
