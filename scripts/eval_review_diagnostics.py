#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader

import next_state_vector_prediction as nsvp


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def retrieval_from_scores(scores, target_texts, topks=(1, 5, 10)):
    ranks = np.argsort(-scores, axis=1)
    groups = nsvp.next_text_group_ids(target_texts)
    rows = np.arange(scores.shape[0])
    out = {}
    for k in topks:
        row_hits = np.asarray([rows[i] in ranks[i, :k] for i in rows])
        group_hits = np.asarray([
            groups[i] in groups[ranks[i, :k]] for i in rows
        ])
        out[f"row_top{k}"] = float(row_hits.mean())
        out[f"top{k}"] = float(group_hits.mean())
    group_ranks = []
    row_ranks = []
    for i in rows:
        ordered = ranks[i]
        row_ranks.append(int(np.where(ordered == i)[0][0]) + 1)
        group_ranks.append(int(np.where(groups[ordered] == groups[i])[0][0]) + 1)
    out["row_mean_rank"] = float(np.mean(row_ranks))
    out["mean_rank"] = float(np.mean(group_ranks))
    return out


def retrieval_from_vectors(pred, true, target_texts):
    return retrieval_from_scores(normalize_rows(pred) @ normalize_rows(true).T, target_texts)


def unique_target_metrics(scores, target_texts):
    normalized = [nsvp.normalize_next_text(text) for text in target_texts]
    counts = Counter(normalized)
    keep = np.asarray([counts[text] == 1 for text in normalized])
    if not keep.any():
        return {"n": 0}
    subset_scores = scores[keep]
    subset_targets = np.flatnonzero(keep)
    ranks = np.argsort(-subset_scores, axis=1)
    out = {"n": int(keep.sum())}
    for k in (1, 5, 10):
        hits = [
            int(target) in ranks[i, :k]
            for i, target in enumerate(subset_targets)
        ]
        out[f"top{k}"] = float(np.mean(hits))
    out["mean_rank"] = float(np.mean([
        int(np.where(ranks[i] == target)[0][0]) + 1
        for i, target in enumerate(subset_targets)
    ]))
    return out


def top1_hit_arrays(scores, target_texts):
    nearest = np.argmax(scores, axis=1)
    rows = np.arange(scores.shape[0])
    groups = nsvp.next_text_group_ids(target_texts)
    return {
        "top1": groups[nearest] == groups,
        "row_top1": nearest == rows,
    }


def paired_bootstrap(first_scores, second_scores, target_texts, seed, repeats=2000):
    first_hits = top1_hit_arrays(first_scores, target_texts)
    second_hits = top1_hit_arrays(second_scores, target_texts)
    rng = np.random.default_rng(seed)
    result = {"bootstrap_repeats": repeats}
    for metric in ("top1", "row_top1"):
        differences = (
            first_hits[metric].astype(np.float32)
            - second_hits[metric].astype(np.float32)
        )
        observed = float(differences.mean())
        indices = rng.integers(
            0, len(differences), size=(repeats, len(differences))
        )
        samples = differences[indices].mean(axis=1)
        lower, upper = np.percentile(samples, [2.5, 97.5])
        two_sided_p = min(
            1.0,
            2 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))),
        )
        result[metric] = {
            "difference": observed,
            "ci95": [float(lower), float(upper)],
            "bootstrap_p_two_sided": two_sided_p,
        }
    return result


def lexical_scores(states, actions, next_states):
    corpus = list(next_states)
    queries = [f"{state} {action}" for state, action in zip(states, actions)]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(corpus + queries)
    candidates = vectorizer.transform(corpus)
    query_matrix = vectorizer.transform(queries)
    state_matrix = vectorizer.transform(states)
    return (
        (query_matrix @ candidates.T).toarray().astype(np.float32),
        (state_matrix @ candidates.T).toarray().astype(np.float32),
    )


def random_baselines(true, target_texts, repeats, seed):
    rng = np.random.default_rng(seed)
    true_n = normalize_rows(true)
    retrievals = []
    pair_cosines = []
    for _ in range(repeats):
        perm = rng.permutation(true_n.shape[0])
        retrievals.append(retrieval_from_vectors(true_n[perm], true_n, target_texts))
        pair_cosines.extend(np.sum(true_n * true_n[perm], axis=1).tolist())
    out = {
        "repeats": repeats,
        "random_pair_cosine_mean": float(np.mean(pair_cosines)),
        "random_pair_cosine_std": float(np.std(pair_cosines)),
    }
    for key in ("top1", "top5", "top10", "row_top1", "row_top5", "row_top10"):
        values = [metrics[key] for metrics in retrievals]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def hard_candidate_metrics(scores, state_lexical_scores, target_texts, candidate_count):
    n = scores.shape[0]
    groups = nsvp.next_text_group_ids(target_texts)
    hits = {1: [], 5: [], 10: []}
    row_hits = {1: [], 5: [], 10: []}
    candidate_sizes = []
    for i in range(n):
        order = np.argsort(-state_lexical_scores[i])
        selected = order[:candidate_count].tolist()
        if i not in selected:
            selected[-1] = i
        selected = np.asarray(sorted(set(selected)), dtype=np.int64)
        candidate_sizes.append(len(selected))
        ranked = selected[np.argsort(-scores[i, selected])]
        for k in hits:
            top = ranked[:k]
            row_hits[k].append(i in top)
            hits[k].append(groups[i] in groups[top])
    out = {
        "n": n,
        "candidate_count_requested": candidate_count,
        "candidate_count_mean": float(np.mean(candidate_sizes)),
    }
    for k in hits:
        out[f"top{k}"] = float(np.mean(hits[k]))
        out[f"row_top{k}"] = float(np.mean(row_hits[k]))
    return out


def build_model(checkpoint, device):
    dim = int(checkpoint["dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    architecture = checkpoint["architecture"]
    dropout = 0.0
    if architecture == "concat":
        model = nsvp.NextStateVectorConcatMLP(dim, hidden_dim, dropout)
    elif architecture in {"state_only", "action_only"}:
        model = nsvp.NextStateVectorSingleInputMLP(dim, hidden_dim, dropout, architecture)
    elif architecture in {"film", "residual_film"}:
        model = nsvp.NextStateVectorFiLM(
            dim, hidden_dim, dropout, residual=architecture == "residual_film"
        )
    elif architecture == "gated_residual_film":
        model = nsvp.NextStateVectorGatedResidualFiLM(dim, hidden_dim, dropout)
    elif architecture == "pair_gated_residual":
        model = nsvp.NextStateVectorPairGatedResidual(dim, hidden_dim, dropout)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def predict(model, z_t, z_action, batch_size, device):
    loader = DataLoader(
        nsvp.TensorDataset(
            torch.from_numpy(np.asarray(z_t, dtype=np.float32)),
            torch.from_numpy(np.asarray(z_action, dtype=np.float32)),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks = []
    with torch.no_grad():
        for state, action in loader:
            output = model(state.to(device), action.to(device))
            chunks.append(F.normalize(output, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def derangement(size, seed):
    rng = np.random.default_rng(seed)
    base = np.arange(size)
    for _ in range(100):
        perm = rng.permutation(size)
        if np.all(perm != base):
            return perm
    return np.roll(base, 1)


def counterfactual_metrics(
    model, data, z_t, z_action, z_next, correct_pred, batch_size, device, seed
):
    groups = defaultdict(list)
    for i, state in enumerate(data["text_t"]):
        groups[nsvp.normalize_next_text(state)].append(i)
    eligible = []
    swap_for = {}
    rng = random.Random(seed)
    for indices in groups.values():
        by_action = defaultdict(list)
        for index in indices:
            by_action[nsvp.normalize_next_text(data["action_text"][index])].append(index)
        if len(by_action) < 2:
            continue
        for index in indices:
            alternatives = [
                other for other in indices
                if nsvp.normalize_next_text(data["action_text"][other])
                != nsvp.normalize_next_text(data["action_text"][index])
                and nsvp.normalize_next_text(data["text_t1"][other])
                != nsvp.normalize_next_text(data["text_t1"][index])
            ]
            if alternatives:
                eligible.append(index)
                swap_for[index] = rng.choice(alternatives)
    if not eligible:
        return {"n": 0}

    eligible = np.asarray(eligible, dtype=np.int64)
    swapped_actions = np.asarray([z_action[swap_for[i]] for i in eligible])
    swapped_pred = predict(
        model, z_t[eligible], swapped_actions, batch_size, device
    )
    correct_subset = normalize_rows(correct_pred[eligible])
    true_n = normalize_rows(z_next)
    groups_by_state = defaultdict(list)
    for index in eligible:
        groups_by_state[nsvp.normalize_next_text(data["text_t"][index])].append(index)

    correct_hits = []
    swapped_still_hits_original = []
    swapped_hits_counterfactual = []
    for local_i, index in enumerate(eligible):
        candidates = np.asarray(groups_by_state[nsvp.normalize_next_text(data["text_t"][index])])
        correct_ranked = candidates[np.argsort(-(correct_subset[local_i] @ true_n[candidates].T))]
        swapped_ranked = candidates[np.argsort(-(swapped_pred[local_i] @ true_n[candidates].T))]
        correct_hits.append(
            nsvp.normalize_next_text(data["text_t1"][correct_ranked[0]])
            == nsvp.normalize_next_text(data["text_t1"][index])
        )
        swapped_still_hits_original.append(
            nsvp.normalize_next_text(data["text_t1"][swapped_ranked[0]])
            == nsvp.normalize_next_text(data["text_t1"][index])
        )
        counterfactual_index = swap_for[int(index)]
        swapped_hits_counterfactual.append(
            nsvp.normalize_next_text(data["text_t1"][swapped_ranked[0]])
            == nsvp.normalize_next_text(data["text_t1"][counterfactual_index])
        )
    return {
        "n": int(len(eligible)),
        "correct_action_local_top1": float(np.mean(correct_hits)),
        "swapped_action_original_top1": float(np.mean(swapped_still_hits_original)),
        "swapped_action_counterfactual_top1": float(np.mean(swapped_hits_counterfactual)),
        "correct_swapped_prediction_cosine": float(
            np.mean(np.sum(correct_subset * swapped_pred, axis=1))
        ),
    }


def flatten(prefix, value, row):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, row)
    else:
        row[prefix] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--hard-candidates", type=int, default=50)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=False)
    checkpoint = None
    embedding_dim = int(data["z_next"].shape[1])
    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        embedding_dim = int(checkpoint["dim"])
    states = data["text_t"].astype(str)
    actions = data["action_text"].astype(str)
    targets = data["text_t1"].astype(str)
    z_t = normalize_rows(data["z_t"][:, :embedding_dim])
    z_action = normalize_rows(data["z_action"][:, :embedding_dim])
    z_next = normalize_rows(data["z_next"][:, :embedding_dim])

    tfidf_scores, state_tfidf_scores = lexical_scores(states, actions, targets)
    identity_scores = z_t @ z_next.T
    action_scores = z_action @ z_next.T
    additive_scores = normalize_rows(z_t + z_action) @ z_next.T

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "num_test": int(len(targets)),
        "num_unique_next_states": int(len(set(map(nsvp.normalize_next_text, targets)))),
        "identity": retrieval_from_scores(identity_scores, targets),
        "action_embedding": retrieval_from_scores(action_scores, targets),
        "state_action_additive": retrieval_from_scores(additive_scores, targets),
        "tfidf_state_action": retrieval_from_scores(tfidf_scores, targets),
        "random": random_baselines(z_next, targets, args.random_repeats, args.seed),
        "identity_unique_targets": unique_target_metrics(identity_scores, targets),
        "tfidf_unique_targets": unique_target_metrics(tfidf_scores, targets),
        "identity_hard": hard_candidate_metrics(
            identity_scores, state_tfidf_scores, targets, args.hard_candidates
        ),
        "tfidf_hard": hard_candidate_metrics(
            tfidf_scores, state_tfidf_scores, targets, args.hard_candidates
        ),
    }

    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(checkpoint, device)
        correct_pred = predict(model, z_t, z_action, args.batch_size, device)
        learned_scores = correct_pred @ z_next.T
        permutation = derangement(len(targets), args.seed + 1701)
        shuffled_pred = predict(
            model, z_t, z_action[permutation], args.batch_size, device
        )
        shuffled_scores = shuffled_pred @ z_next.T
        results.update({
            "architecture": checkpoint["architecture"],
            "learned": retrieval_from_scores(learned_scores, targets),
            "learned_unique_targets": unique_target_metrics(learned_scores, targets),
            "learned_hard": hard_candidate_metrics(
                learned_scores, state_tfidf_scores, targets, args.hard_candidates
            ),
            "shuffled_action": retrieval_from_scores(shuffled_scores, targets),
            "correct_shuffled_prediction_cosine": float(
                np.mean(np.sum(correct_pred * shuffled_pred, axis=1))
            ),
            "counterfactual_same_state": counterfactual_metrics(
                model,
                data,
                z_t,
                z_action,
                z_next,
                correct_pred,
                args.batch_size,
                device,
                args.seed + 2903,
            ),
            "paired_comparisons": {
                "learned_minus_identity": paired_bootstrap(
                    learned_scores,
                    identity_scores,
                    targets,
                    args.seed + 4101,
                ),
                "learned_minus_tfidf": paired_bootstrap(
                    learned_scores,
                    tfidf_scores,
                    targets,
                    args.seed + 4102,
                ),
                "learned_minus_shuffled_action": paired_bootstrap(
                    learned_scores,
                    shuffled_scores,
                    targets,
                    args.seed + 4103,
                ),
            },
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    flat = {}
    flatten("", results, flat)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    print(output)
    print(csv_path)


if __name__ == "__main__":
    main()
