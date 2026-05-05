#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re


class NextStateVectorGatedResidualFiLM(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.action_to_params = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim * 3),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z_t, z_action):
        gamma, beta, gate_logits = self.action_to_params(z_action).chunk(3, dim=1)
        delta = gamma * z_t + beta
        return self.output(z_t + torch.sigmoid(gate_logits) * delta)


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def normalize_next_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def next_text_group_ids(texts):
    group_by_text = {}
    ids = []
    for text in texts:
        key = normalize_next_text(text)
        if key not in group_by_text:
            group_by_text[key] = len(group_by_text)
        ids.append(group_by_text[key])
    return np.asarray(ids, dtype=np.int64)


def batched_ranks(pred, candidates, target_indices, batch_size, candidate_groups=None):
    pred = normalize(pred.astype(np.float32, copy=False))
    candidates = normalize(candidates.astype(np.float32, copy=False))
    ranks = np.empty(pred.shape[0], dtype=np.int32)
    row_ranks = np.empty(pred.shape[0], dtype=np.int32)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    candidate_groups = np.asarray(candidate_groups, dtype=np.int64) if candidate_groups is not None else None
    for start in range(0, pred.shape[0], batch_size):
        end = min(start + batch_size, pred.shape[0])
        sims = pred[start:end] @ candidates.T
        target_scores = sims[np.arange(end - start), target_indices[start:end]]
        row_ranks[start:end] = (sims > target_scores[:, None]).sum(axis=1) + 1
        if candidate_groups is None:
            ranks[start:end] = row_ranks[start:end]
        else:
            target_groups = candidate_groups[target_indices[start:end]]
            for offset, group in enumerate(target_groups):
                same = candidate_groups == group
                best_equivalent_score = sims[offset, same].max()
                ranks[start + offset] = int((sims[offset] > best_equivalent_score).sum()) + 1
    return ranks, row_ranks


def summarize_ranks(ranks):
    ranks = np.asarray(ranks)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_and_sort(rows, schema):
    by_group = defaultdict(list)
    if schema == "scone":
        for row in rows:
            by_group[row["source_id"]].append(row)
        order_key = lambda r: int(r["step"])
    elif schema == "trip":
        for row in rows:
            key = (row["story_id"], row["entity"], row["attribute"])
            by_group[key].append(row)
        order_key = lambda r: int(r["sentence_idx"])
    elif schema == "propara":
        for row in rows:
            key = (row["para_id"], row["participant"])
            by_group[key].append(row)
        order_key = lambda r: int(r["step_index"])
    elif schema == "alfworld":
        for row in rows:
            by_group[row["source_file"]].append(row)
        order_key = lambda r: int(r["high_idx"])
    else:
        raise ValueError(f"unknown schema: {schema}")
    for key in by_group:
        by_group[key].sort(key=order_key)
    return by_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-name", default="scone")
    ap.add_argument("--schema", choices=["scone", "trip", "propara", "alfworld"], default="scone")
    ap.add_argument("--mode", choices=["autoregressive", "teacher_forcing"], default="autoregressive")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--rank-batch-size", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    dim = int(ckpt["dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    model = NextStateVectorGatedResidualFiLM(dim, hidden_dim, dropout=0.0)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    emb = np.load(args.embeddings, allow_pickle=False)
    ids = emb["ids"].astype(str)
    id_to_idx = {item_id: i for i, item_id in enumerate(ids.tolist())}
    z_t = emb["z_t"].astype(np.float32)
    z_action = emb["z_action"].astype(np.float32)
    z_next = emb["z_next"].astype(np.float32)

    rows = load_jsonl(args.jsonl)
    row_by_id = {str(row["id"]): row for row in rows}
    next_texts = [row_by_id[str(item_id)]["text_t1"] for item_id in ids]
    candidate_groups = next_text_group_ids(next_texts)
    by_source = group_and_sort(rows, args.schema)

    windows = []
    for source_id, seq in by_source.items():
        for i in range(0, len(seq) - args.steps + 1):
            chunk = seq[i : i + args.steps]
            if any(chunk[j]["id"] not in id_to_idx for j in range(args.steps)):
                continue
            windows.append(chunk)

    if not windows:
        raise RuntimeError("No rollout windows found.")

    horizon_preds = [[] for _ in range(args.steps)]
    horizon_targets = [[] for _ in range(args.steps)]
    domains = []
    start_ids = []
    final_ids = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for start in range(0, len(windows), args.eval_batch_size):
        batch = windows[start : start + args.eval_batch_size]
        state_idx = np.asarray([id_to_idx[w[0]["id"]] for w in batch], dtype=np.int64)
        state = torch.from_numpy(z_t[state_idx]).to(device)
        for h in range(args.steps):
            if args.mode == "teacher_forcing":
                state_idx = np.asarray([id_to_idx[w[h]["id"]] for w in batch], dtype=np.int64)
                state = torch.from_numpy(z_t[state_idx]).to(device)
            action_idx = np.asarray([id_to_idx[w[h]["id"]] for w in batch], dtype=np.int64)
            action = torch.from_numpy(z_action[action_idx]).to(device)
            with torch.no_grad():
                state = F.normalize(model(state, action), dim=1)
            horizon_preds[h].append(state.cpu().numpy())
            horizon_targets[h].extend([id_to_idx[w[h]["id"]] for w in batch])

        domains.extend([w[0]["range"] for w in batch])
        start_ids.extend([w[0]["id"] for w in batch])
        final_ids.extend([w[-1]["id"] for w in batch])

    domains = np.asarray(domains)
    summary_rows = []
    detailed = []

    for h in range(args.steps):
        pred = np.concatenate(horizon_preds[h], axis=0)
        targets = np.asarray(horizon_targets[h], dtype=np.int64)
        ranks, row_ranks = batched_ranks(pred, z_next, targets, args.rank_batch_size, candidate_groups)
        metrics = summarize_ranks(ranks)
        row_metrics = {f"row_{key}": value for key, value in summarize_ranks(row_ranks).items()}
        summary_rows.append({
            "scope": "all",
            "horizon": h + 1,
            "n": len(ranks),
            **metrics,
            **row_metrics,
        })
        if h == args.steps - 1:
            for i, (rank, row_rank) in enumerate(zip(ranks.tolist(), row_ranks.tolist())):
                detailed.append({
                    "start_id": start_ids[i],
                    "final_id": final_ids[i],
                    "domain": domains[i],
                    "rank": rank,
                    "row_rank": row_rank,
                    "top1": int(rank <= 1),
                    "top5": int(rank <= 5),
                    "top10": int(rank <= 10),
                    "row_top1": int(row_rank <= 1),
                    "row_top5": int(row_rank <= 5),
                    "row_top10": int(row_rank <= 10),
                })
        for domain in sorted(set(domains.tolist())):
            mask = domains == domain
            dm = summarize_ranks(ranks[mask])
            row_dm = {f"row_{key}": value for key, value in summarize_ranks(row_ranks[mask]).items()}
            summary_rows.append({
                "scope": f"domain:{domain}",
                "horizon": h + 1,
                "n": int(mask.sum()),
                **dm,
                **row_dm,
            })

    summary_path = out_dir / f"{args.dataset_name}_autoregressive_rollout_{args.steps}step_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_path = out_dir / f"{args.dataset_name}_autoregressive_rollout_{args.steps}step_details.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed[0].keys()))
        writer.writeheader()
        writer.writerows(detailed)

    print(json.dumps({
        "windows": len(windows),
        "steps": args.steps,
        "mode": args.mode,
        "summary_path": str(summary_path),
        "detail_path": str(detail_path),
        "final_horizon": summary_rows[(args.steps - 1) * (1 + len(set(domains.tolist())))],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
