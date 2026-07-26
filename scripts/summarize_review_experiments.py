#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def mean_std(values):
    values = [float(value) for value in values if value not in (None, "")]
    if not values:
        return None, None
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def add_record(records, experiment, dataset, model, setting, seed, metrics, note=""):
    records.append({
        "experiment": experiment,
        "dataset": dataset,
        "model": model,
        "setting": setting,
        "seed": seed,
        "top1": metrics.get("top1"),
        "top5": metrics.get("top5"),
        "top10": metrics.get("top10"),
        "row_top1": metrics.get("row_top1"),
        "cosine": metrics.get("cosine"),
        "n": metrics.get("n"),
        "note": note,
    })


def collect_core(root, records):
    for path in sorted((root / "review_core").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        dataset = result["dataset"]
        model = result["model"]
        seed = result["seed"]
        for setting in (
            "identity",
            "action_embedding",
            "state_action_additive",
            "tfidf_state_action",
            "learned",
            "shuffled_action",
            "identity_hard",
            "tfidf_hard",
            "learned_hard",
        ):
            if setting in result:
                add_record(
                    records, "core_diagnostics", dataset, model, setting,
                    seed, result[setting],
                )
        counterfactual = result.get("counterfactual_same_state", {})
        if counterfactual.get("n", 0):
            add_record(
                records,
                "core_diagnostics",
                dataset,
                model,
                "counterfactual_correct_action",
                seed,
                {
                    "top1": counterfactual["correct_action_local_top1"],
                    "n": counterfactual["n"],
                },
                "Candidate set contains next states paired with the same current state.",
            )
            add_record(
                records,
                "core_diagnostics",
                dataset,
                model,
                "counterfactual_swapped_action_target",
                seed,
                {
                    "top1": counterfactual["swapped_action_counterfactual_top1"],
                    "n": counterfactual["n"],
                },
                "Action is swapped within the same-current-state group.",
            )
        random_result = result["random"]
        add_record(
            records,
            "core_diagnostics",
            dataset,
            model,
            "random_next_vector",
            seed,
            {
                "top1": random_result["top1_mean"],
                "top5": random_result["top5_mean"],
                "top10": random_result["top10_mean"],
                "row_top1": random_result["row_top1_mean"],
                "cosine": random_result["random_pair_cosine_mean"],
                "n": result["num_test"],
            },
        )


def collect_paraphrases(root, records):
    directory = root / "review_action_paraphrases"
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".summary.json"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        for setting in ("original", "paraphrased"):
            add_record(
                records,
                "action_paraphrase",
                result["dataset"],
                result["model"],
                setting,
                result["seed"],
                {**result[setting], "n": result["num_test"]},
            )


def collect_extension(root, records):
    for path in sorted((root / "review_extension_baselines").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for setting in (
            "identity",
            "action_embedding",
            "state_action_additive",
            "tfidf_state_action",
            "identity_hard",
            "tfidf_hard",
        ):
            add_record(
                records,
                "extension_baselines",
                result["dataset"],
                result["model"],
                setting,
                result["seed"],
                result[setting],
            )
        random_result = result["random"]
        add_record(
            records,
            "extension_baselines",
            result["dataset"],
            result["model"],
            "random_next_vector",
            result["seed"],
            {
                "top1": random_result["top1_mean"],
                "top5": random_result["top5_mean"],
                "top10": random_result["top10_mean"],
                "row_top1": random_result["row_top1_mean"],
                "cosine": random_result["random_pair_cosine_mean"],
                "n": result["num_test"],
            },
        )


def collect_csv(root, relative_path, experiment, records, setting_column, model_default):
    path = root / relative_path
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            setting = row.get(setting_column) or ""
            model = row.get("model") or model_default
            add_record(
                records,
                experiment,
                row["dataset"],
                model,
                setting,
                row.get("seed", ""),
                {
                    "top1": row.get("top1"),
                    "top5": row.get("top5"),
                    "top10": row.get("top10"),
                    "row_top1": row.get("row_top1"),
                    "cosine": row.get("cosine"),
                },
            )


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        key = (
            row["experiment"], row["dataset"], row["model"],
            row["setting"], row["note"],
        )
        grouped[key].append(row)
    summaries = []
    for key, rows in sorted(grouped.items()):
        experiment, dataset, model, setting, note = key
        summary = {
            "experiment": experiment,
            "dataset": dataset,
            "model": model,
            "setting": setting,
            "n_runs": len(rows),
            "n_examples": next(
                (row["n"] for row in rows if row["n"] is not None), ""
            ),
            "note": note,
        }
        for metric in ("top1", "top5", "top10", "row_top1", "cosine"):
            mean, std = mean_std([row[metric] for row in rows])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summaries.append(summary)
    return summaries


def write_markdown(path, summaries):
    lines = [
        "# Reviewer Experiment Summary",
        "",
        "| Experiment | Dataset | Model | Setting | Runs | Top-1 | Row Top-1 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        def display(metric):
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            if mean is None:
                return ""
            return f"{100 * mean:.2f} +/- {100 * std:.2f}"
        lines.append(
            f"| {row['experiment']} | {row['dataset']} | {row['model']} | "
            f"{row['setting']} | {row['n_runs']} | {display('top1')} | "
            f"{display('row_top1')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/review_summary")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    records = []
    collect_core(root, records)
    collect_paraphrases(root, records)
    collect_extension(root, records)
    collect_csv(
        root,
        "review_qwen_scale_dimension/all_runs.csv",
        "qwen_scale_dimension",
        records,
        "requested_dim",
        "",
    )
    collect_csv(
        root,
        "review_jina_modes/all_runs.csv",
        "jina_task_mode",
        records,
        "mode",
        "jina-embeddings-v3",
    )
    collect_csv(
        root,
        "review_hidden_state/all_runs.csv",
        "hidden_state_baseline",
        records,
        "pooling",
        "Qwen3-0.6B-Base-meanpool",
    )
    summaries = summarize(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "review_experiment_summary.csv"
    fields = list(summaries[0]) if summaries else ["experiment"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    markdown_path = output_dir / "review_experiment_summary.md"
    write_markdown(markdown_path, summaries)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
