#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "public_datasets" / "trip_official_repo" / "all_data" / "www.json"
DEFAULT_OUTPUT = ROOT / "processed_datasets" / "trip_nl"

ATTRS = [
    "h_location", "conscious", "wearing", "h_wet", "hygiene",
    "location", "exist", "clean", "power", "functional", "pieces",
    "wet", "open", "temperature", "solid", "contain", "running",
    "moveable", "mixed", "edible",
]

DEFAULT_VALUES = {
    "h_location": 0, "conscious": 2, "wearing": 0, "h_wet": 0, "hygiene": 0,
    "location": 0, "exist": 2, "clean": 0, "power": 0, "functional": 2,
    "pieces": 0, "wet": 0, "open": 0, "temperature": 0, "solid": 0,
    "contain": 0, "running": 0, "moveable": 2, "mixed": 0, "edible": 0,
}

DEFAULT_TRANSITIONS = {
    0: (-1, -1),
    1: (0, 0),
    2: (1, 1),
    3: (1, 0),
    4: (0, 1),
    5: (-1, 0),
    6: (-1, 1),
    7: (0, -1),
    8: (1, -1),
}

ADJECTIVES = {
    "conscious": ("unconscious", "conscious"),
    "wearing": ("undressed", "dressed"),
    "h_wet": ("dry", "wet"),
    "hygiene": ("dirty", "clean"),
    "exist": ("not present", "present"),
    "clean": ("dirty", "clean"),
    "power": ("unpowered", "powered"),
    "functional": ("broken", "functional"),
    "pieces": ("whole", "in pieces"),
    "wet": ("dry", "wet"),
    "open": ("closed", "open"),
    "temperature": ("cold", "hot"),
    "solid": ("fluid", "solid"),
    "contain": ("empty", "occupied"),
    "running": ("turned off", "turned on"),
    "moveable": ("stuck", "moveable"),
    "mixed": ("separated", "mixed"),
    "edible": ("inedible", "edible"),
}

LOCATION_TRANSITIONS = {
    "h_location": {
        1: ("present somewhere", "not present"),
        2: ("in one location", "in a new location"),
    },
    "location": {
        1: ("present somewhere", "not present"),
        2: ("not held", "picked up"),
        3: ("held", "put down"),
        4: ("not on something", "put on something"),
        5: ("attached or contained", "removed"),
        6: ("outside a container", "inside a container"),
        7: ("inside a container", "out of a container"),
        8: ("in one location", "in a new location"),
    },
}

SPLIT_MAP = {"train": "train", "dev": "dev", "test": "test"}


def clean_text(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def entity_phrase(entity):
    entity = clean_text(entity)
    if not entity:
        return "the entity"
    if entity[0].isupper():
        return entity
    return f"the {entity}"


def state_sentence(entity, attr, value):
    subj = entity_phrase(entity)
    if attr in {"location", "h_location"}:
        return f"{subj} is {value}."
    if attr == "wearing":
        return f"{subj} is {value}."
    if attr == "h_wet":
        return f"{subj} is {value}."
    if attr == "hygiene":
        return f"{subj} is {value}."
    return f"{subj} is {value}."


def transition_to_states(attr, code):
    if attr in LOCATION_TRANSITIONS:
        return LOCATION_TRANSITIONS[attr].get(code)
    if attr not in ADJECTIVES:
        return None
    neg, pos = ADJECTIVES[attr]
    pre_post = DEFAULT_TRANSITIONS.get(code)
    if not pre_post:
        return None
    pre, post = pre_post
    if pre == post:
        return None
    def label(v):
        if v == 0:
            return neg
        if v == 1:
            return pos
        return "unknown"
    return label(pre), label(post)


def is_informative(attr, code):
    if attr in LOCATION_TRANSITIONS:
        return code not in {0}
    return code not in {0, 1, 2}


def iter_story_rows(split, story, include_implausible):
    if not include_implausible and not story.get("plausible", True):
        return
    story_id = story.get("example_id", "")
    plausible = bool(story.get("plausible", True))
    states = story.get("states") or []
    sentences = story.get("sentences") or []
    for sent_idx, sent_state in enumerate(states):
        if sent_idx >= len(sentences):
            continue
        action = clean_text(sentences[sent_idx])
        for attr in ATTRS:
            for entity, code in sent_state.get(attr, []):
                if not is_informative(attr, int(code)):
                    continue
                pair = transition_to_states(attr, int(code))
                if not pair:
                    continue
                pre, post = pair
                row = {
                    "id": f"trip_{split}_{story_id}_s{sent_idx}_{clean_text(entity).replace(' ', '_')}_{attr}_{code}",
                    "story_id": story_id,
                    "sentence_idx": sent_idx,
                    "entity": clean_text(entity),
                    "attribute": attr,
                    "change_code": int(code),
                    "plausible": plausible,
                    "text_t": state_sentence(entity, attr, pre),
                    "action_text": action,
                    "text_t1": state_sentence(entity, attr, post),
                    "range": attr,
                    "transition_type": attr,
                    "story_context": " ".join(sentences[:sent_idx]),
                    "full_story": " ".join(sentences),
                }
                yield row


def load_www(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized = {}
    for split, stories_by_id in data.items():
        normalized[split] = list(stories_by_id.values())
    return normalized


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-implausible", action="store_true")
    parser.add_argument("--known-only", action="store_true")
    args = parser.parse_args()

    data = load_www(args.source)
    rows_by_split = defaultdict(list)
    raw_counts = Counter()
    duplicate_count = 0
    seen = {}

    for source_split, out_split in SPLIT_MAP.items():
        for story in data.get(source_split, []):
            for row in iter_story_rows(out_split, story, args.include_implausible):
                if args.known_only and ("unknown" in row["text_t"] or "unknown" in row["text_t1"]):
                    continue
                raw_counts[out_split] += 1
                key = (row["text_t"], row["action_text"], row["text_t1"])
                if key in seen:
                    duplicate_count += 1
                    continue
                seen[key] = row["id"]
                rows_by_split[out_split].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        write_jsonl(args.output_dir / f"trip_nl_{split}.jsonl", rows_by_split[split])

    sample_rows = rows_by_split["train"][:25] + rows_by_split["dev"][:5] + rows_by_split["test"][:5]
    with (args.output_dir / "trip_nl_samples.md").open("w", encoding="utf-8") as f:
        f.write("# TRIP Next-State Samples\n\n")
        for i, row in enumerate(sample_rows, 1):
            f.write(f"## {i}. {row['id']}\n")
            f.write(f"- plausible: {row['plausible']}\n")
            f.write(f"- attribute: {row['attribute']}\n")
            f.write(f"- state_t: {row['text_t']}\n")
            f.write(f"- action: {row['action_text']}\n")
            f.write(f"- state_t1: {row['text_t1']}\n\n")

    attr_counts = Counter()
    split_attr_counts = defaultdict(Counter)
    for split, rows in rows_by_split.items():
        for row in rows:
            attr_counts[row["attribute"]] += 1
            split_attr_counts[split][row["attribute"]] += 1

    summary = {
        "source": str(args.source),
        "output": str(args.output_dir),
        "include_implausible": args.include_implausible,
        "known_only": args.known_only,
        "raw_rows_by_split": dict(raw_counts),
        "deduplicated_rows_by_split": {k: len(v) for k, v in rows_by_split.items()},
        "duplicates_removed": duplicate_count,
        "attribute_counts": dict(attr_counts),
        "split_attribute_counts": {k: dict(v) for k, v in split_attr_counts.items()},
        "note": "Rows are sentence-level TRIP physical state changes converted to natural language; no-change/default annotations are omitted.",
    }
    (args.output_dir / "trip_nl_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
