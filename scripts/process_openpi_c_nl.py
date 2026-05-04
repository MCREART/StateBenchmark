#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def clean_text(value):
    return " ".join(str(value).strip().split())


def title_from_url(url):
    slug = str(url).rstrip("/").split("/")[-1]
    return clean_text(slug.replace("-", " "))


def make_state(entity, attribute, value, when):
    entity = clean_text(entity)
    attribute = clean_text(attribute)
    value = clean_text(value)
    return f"{when}, the {attribute} of {entity} is {value}."


def iter_transitions(path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            url, step = row.get("id", ["unknown", line_no])
            task = title_from_url(url)
            action = clean_text(row.get("query", ""))
            for answer_idx, answer in enumerate(row.get("answers", [])):
                if len(answer) != 4:
                    continue
                entity, attribute, pre_state, post_state = [clean_text(x) for x in answer]
                if not entity or not attribute or not pre_state or not post_state:
                    continue
                item_id = f"{Path(str(url)).name}_step{step}_change{answer_idx}"
                yield {
                    "id": item_id,
                    "source": str(url),
                    "step_index": int(step) if str(step).isdigit() else step,
                    "change_index": answer_idx,
                    "task": task,
                    "entity": entity,
                    "attribute": attribute,
                    "pre_state": pre_state,
                    "post_state": post_state,
                    "text_t": make_state(entity, attribute, pre_state, "Before the step"),
                    "action_text": f"In the task '{task}', perform this step: {action}",
                    "text_t1": make_state(entity, attribute, post_state, "After the step"),
                    "range": attribute,
                    "transition_type": "openpi_c_state_change",
                }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="public_datasets/openpi_c_repo/data")
    parser.add_argument("--output-dir", default="processed_datasets/openpi_c_nl")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    sample_rows = []
    all_seen = set()
    duplicate_count = 0

    for split in ["train", "dev", "test"]:
        rows = []
        split_seen = set()
        for row in iter_transitions(input_dir / f"{split}.jsonl"):
            key = (row["text_t"], row["action_text"], row["text_t1"])
            if key in split_seen:
                duplicate_count += 1
                continue
            split_seen.add(key)
            all_seen.add(key)
            rows.append(row)
        write_jsonl(output_dir / f"openpi_c_nl_{split}.jsonl", rows)
        summary[split] = {
            "rows": len(rows),
            "unique_triples": len(split_seen),
            "attributes": len(Counter(row["attribute"] for row in rows)),
            "top_attributes": Counter(row["attribute"] for row in rows).most_common(15),
            "tasks": len(Counter(row["source"] for row in rows)),
        }
        sample_rows.extend(rows[:5])

    summary["duplicates_removed_within_split"] = duplicate_count
    summary["unique_triples_all_splits"] = len(all_seen)
    (output_dir / "openpi_c_nl_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = ["# OpenPI-C Natural-Language Samples", ""]
    for row in sample_rows[:15]:
        lines.extend([
            f"## {row['id']}",
            f"- state_t: {row['text_t']}",
            f"- action: {row['action_text']}",
            f"- state_t1: {row['text_t1']}",
            f"- attribute: {row['attribute']}",
            "",
        ])
    (output_dir / "openpi_c_nl_samples.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
