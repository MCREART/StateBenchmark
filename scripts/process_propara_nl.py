#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def clean_text(value):
    return " ".join(str(value).strip().split())


def state_phrase(entity, value):
    entity = clean_text(entity)
    value = clean_text(value)
    if value == "-":
        return f"{entity}: not present."
    if value == "?":
        return f"{entity} location: unknown."
    return f"{entity} location: {value}."


def transition_type(before, after):
    if before == after:
        return "unchanged"
    if before == "-" and after not in {"-", "?"}:
        return "created"
    if before not in {"-", "?"} and after == "-":
        return "destroyed"
    if before == "?" or after == "?":
        return "unknown_location_change"
    return "moved"


def read_prompts(tsv_path):
    prompts = {}
    if not tsv_path.exists():
        return prompts
    with tsv_path.open("r", encoding="utf-8") as f:
        current_para = None
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            current_para = parts[0]
            if len(parts) >= 3 and parts[2].startswith("PROMPT:"):
                prompts[current_para] = clean_text(parts[2].replace("PROMPT:", "", 1))
    return prompts


def iter_paragraphs(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_rows(json_path, prompt_map, split, changed_only=False):
    rows = []
    skipped_unchanged = 0
    seen = set()
    for para in iter_paragraphs(json_path):
        para_id = str(para["para_id"])
        prompt = prompt_map.get(para_id, "the described process")
        sentences = [clean_text(x) for x in para["sentence_texts"]]
        participants = [clean_text(x) for x in para["participants"]]
        states = para["states"]
        for participant_idx, participant in enumerate(participants):
            participant_states = [clean_text(x) for x in states[participant_idx]]
            for step_idx, sentence in enumerate(sentences, start=1):
                before = participant_states[step_idx - 1]
                after = participant_states[step_idx]
                kind = transition_type(before, after)
                if changed_only and kind == "unchanged":
                    skipped_unchanged += 1
                    continue
                row = {
                    "id": f"propara_{split}_{para_id}_p{participant_idx}_step{step_idx}",
                    "para_id": para_id,
                    "step_index": step_idx,
                    "participant": participant,
                    "pre_state": before,
                    "post_state": after,
                    "prompt": prompt,
                    "text_t": state_phrase(participant, before),
                    "action_text": sentence,
                    "text_t1": state_phrase(participant, after),
                    "range": "location_existence",
                    "transition_type": kind,
                }
                key = (row["text_t"], row["action_text"], row["text_t1"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows, skipped_unchanged


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset(input_dir, output_dir, changed_only):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"changed_only": changed_only}
    sample_rows = []
    for split in ["train", "dev", "test"]:
        prompt_map = read_prompts(input_dir / f"grids.v1.{split}.tsv")
        rows, skipped_unchanged = build_rows(
            input_dir / f"grids.v1.{split}.json",
            prompt_map,
            split,
            changed_only=changed_only,
        )
        write_jsonl(output_dir / f"propara_nl_{split}.jsonl", rows)
        summary[split] = {
            "rows": len(rows),
            "paragraphs": len({row["para_id"] for row in rows}),
            "participants": len({(row["para_id"], row["participant"]) for row in rows}),
            "skipped_unchanged": skipped_unchanged,
            "transition_types": Counter(row["transition_type"] for row in rows),
        }
        summary[split]["transition_types"] = dict(summary[split]["transition_types"])
        sample_rows.extend(rows[:5])

    (output_dir / "propara_nl_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = ["# ProPara Natural-Language Samples", ""]
    for row in sample_rows[:15]:
        lines.extend([
            f"## {row['id']}",
            f"- state_t: {row['text_t']}",
            f"- action: {row['action_text']}",
            f"- state_t1: {row['text_t1']}",
            f"- transition_type: {row['transition_type']}",
            "",
        ])
    (output_dir / "propara_nl_samples.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="public_datasets/propara_repo/data/emnlp18")
    parser.add_argument("--output-root", default="processed_datasets")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    all_summary = write_dataset(input_dir, output_root / "propara_nl_all", changed_only=False)
    changed_summary = write_dataset(input_dir, output_root / "propara_nl_changed", changed_only=True)

    print(json.dumps({
        "all": all_summary,
        "changed": changed_summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
