#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "public_datasets" / "alfworld_data" / "json_2.1.1"
DEFAULT_OUTPUT = ROOT / "processed_datasets" / "alfworld_nl"

SPLIT_MAP = {
    "train": "train",
    "valid_seen": "dev",
    "valid_unseen": "test",
}

COMPOUND_NAMES = {
    "alarmclock": "alarm clock",
    "baseballbat": "baseball bat",
    "bathtubbason": "bathtub basin",
    "bathtubbasin": "bathtub basin",
    "butterknife": "butter knife",
    "cd": "CD",
    "cellphone": "cell phone",
    "coffeemachine": "coffee machine",
    "countertop": "counter top",
    "creditcard": "credit card",
    "desklamp": "desk lamp",
    "diningtable": "dining table",
    "dishsponge": "dish sponge",
    "floorlamp": "floor lamp",
    "garbagecan": "garbage can",
    "glassbottle": "glass bottle",
    "handtowel": "hand towel",
    "houseplant": "house plant",
    "keychain": "key chain",
    "lightswitch": "light switch",
    "newspaper": "newspaper",
    "papertowelroll": "paper towel roll",
    "sidetable": "side table",
    "sinkbasin": "sink basin",
    "soapbottle": "soap bottle",
    "spraybottle": "spray bottle",
    "stoveburner": "stove burner",
    "stoveknob": "stove knob",
    "tennisracket": "tennis racket",
    "toiletpaper": "toilet paper",
    "towelholder": "towel holder",
    "wateringcan": "watering can",
    "winebottle": "wine bottle",
}


def clean_token(value):
    if not value:
        return ""
    value = str(value).split("|", 1)[0]
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value).strip().lower()
    value = COMPOUND_NAMES.get(value, value)
    return value


def article(noun):
    if not noun:
        return "the object"
    return f"the {noun}"


def get_receptacle(planner_action, fallback=""):
    rec = planner_action.get("coordinateReceptacleObjectId")
    if isinstance(rec, list) and rec:
        return clean_token(rec[0])
    rec = planner_action.get("receptacleObjectId")
    if rec:
        return clean_token(rec)
    return clean_token(fallback)


def get_object(discrete_action, planner_action, fallback=""):
    obj = planner_action.get("coordinateObjectId")
    if isinstance(obj, list) and obj:
        return clean_token(obj[0])
    for key in ("objectId", "cleanObjectId"):
        if planner_action.get(key):
            return clean_token(planner_action[key])
    args = discrete_action.get("args") or []
    if args:
        return clean_token(args[0])
    return clean_token(fallback)


def fallback_action(action, obj="", rec="", previous="", dest=""):
    obj_a = article(obj)
    rec_a = article(rec or dest)
    if action == "GotoLocation":
        return f"Go to {rec_a}."
    if action == "PickupObject":
        return f"Pick up {obj_a} from {rec_a}."
    if action == "PutObject":
        return f"Put {obj_a} on or in {rec_a}."
    if action == "CleanObject":
        return f"Clean {obj_a}."
    if action == "HeatObject":
        return f"Heat {obj_a}."
    if action == "CoolObject":
        return f"Cool {obj_a}."
    if action == "ToggleObject":
        return f"Toggle {obj_a}."
    if action == "SliceObject":
        return f"Slice {obj_a}."
    return action


def action_text(high_descs, high_idx, action, obj="", rec="", dest=""):
    if 0 <= high_idx < len(high_descs):
        text = re.sub(r"\s+", " ", high_descs[high_idx]).strip()
        if text:
            return text[0].upper() + text[1:]
    return fallback_action(action, obj=obj, rec=rec, dest=dest)


def emit_row(path, split, data, high_step, state_t, action, state_t1, range_name, transition_type):
    task = data.get("task_type", "")
    params = data.get("pddl_params", {})
    return {
        "id": f"alfworld_{split}_{path.parent.parent.name}_{path.parent.name}_h{high_step.get('high_idx', 0)}",
        "source_file": str(path.relative_to(ROOT)),
        "task_type": task,
        "object_target": clean_token(params.get("object_target", "")),
        "parent_target": clean_token(params.get("parent_target", "")),
        "toggle_target": clean_token(params.get("toggle_target", "")),
        "high_idx": high_step.get("high_idx", 0),
        "action_type": high_step.get("discrete_action", {}).get("action", ""),
        "text_t": state_t,
        "action_text": action,
        "text_t1": state_t1,
        "range": range_name,
        "transition_type": transition_type,
    }


def process_trial(path, split):
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("pddl_params", {})
    target_obj = clean_token(params.get("object_target", ""))
    target_parent = clean_token(params.get("parent_target", ""))
    toggle_target = clean_token(params.get("toggle_target", ""))
    high_descs = []
    anns = data.get("turk_annotations", {}).get("anns") or []
    if anns:
        high_descs = anns[0].get("high_descs") or []

    agent_location = ""
    held_object = ""
    object_locations = {}
    clean = set()
    hot = set()
    cool = set()
    sliced = set()
    toggled_on = set()
    rows = []

    for high in data.get("plan", {}).get("high_pddl", []):
        discrete = high.get("discrete_action", {})
        planner = high.get("planner_action", {})
        action = discrete.get("action", "")
        if action == "NoOp":
            continue

        high_idx = int(high.get("high_idx", 0))
        args = [clean_token(x) for x in (discrete.get("args") or [])]
        obj = get_object(discrete, planner, target_obj)
        rec = get_receptacle(planner, args[1] if len(args) > 1 else "")
        text_action = action_text(high_descs, high_idx, action, obj=obj, rec=rec, dest=rec)

        if action == "GotoLocation":
            dest = args[0] if args else rec
            if not dest:
                continue
            if agent_location:
                state_t = f"The agent is at {article(agent_location)}."
            else:
                state_t = f"The agent is away from {article(dest)}."
            state_t1 = f"The agent is at {article(dest)}."
            rows.append(emit_row(path, split, data, high, state_t, text_action, state_t1, "agent_location", "moved"))
            agent_location = dest

        elif action == "PickupObject":
            source = rec or object_locations.get(obj, "")
            if not obj:
                continue
            if source:
                object_locations[obj] = source
                state_t = f"{article(obj).capitalize()} is on or in {article(source)}. The agent is not holding {article(obj)}."
            else:
                state_t = f"{article(obj).capitalize()} is reachable. The agent is not holding {article(obj)}."
            state_t1 = f"The agent is holding {article(obj)}."
            rows.append(emit_row(path, split, data, high, state_t, text_action, state_t1, "object_possession", "picked_up"))
            held_object = obj
            object_locations.pop(obj, None)

        elif action == "PutObject":
            dest = rec or (args[1] if len(args) > 1 else target_parent)
            obj = obj or held_object or target_obj
            if not obj or not dest:
                continue
            state_t = f"The agent is holding {article(obj)}."
            state_t1 = f"{article(obj).capitalize()} is on or in {article(dest)}. The agent is not holding {article(obj)}."
            rows.append(emit_row(path, split, data, high, state_t, text_action, state_t1, "object_location", "placed"))
            object_locations[obj] = dest
            if held_object == obj:
                held_object = ""

        elif action == "CleanObject":
            obj = obj or held_object or target_obj
            if not obj:
                continue
            state_t = f"{article(obj).capitalize()} is not clean. The agent is holding {article(obj)}."
            state_t1 = f"{article(obj).capitalize()} is clean. The agent is holding {article(obj)}."
            rows.append(emit_row(path, split, data, high, state_t, text_action, state_t1, "object_attribute", "cleaned"))
            clean.add(obj)

        elif action == "HeatObject":
            obj = obj or held_object or target_obj
            if not obj:
                continue
            state_t = f"{article(obj).capitalize()} is not hot. The agent is holding {article(obj)}."
            state_t1 = f"{article(obj).capitalize()} is hot. The agent is holding {article(obj)}."
            rows.append(emit_row(path, split, data, high, state_t, text_action, state_t1, "object_attribute", "heated"))
            hot.add(obj)
            cool.discard(obj)

        elif action == "CoolObject":
            obj = obj or held_object or target_obj
            if not obj:
                continue
            state_t = f"{article(obj).capitalize()} is not cool. The agent is holding {article(obj)}."
            state_t1 = f"{article(obj).capitalize()} is cool. The agent is holding {article(obj)}."
            rows.append(emit_row(path, split, data, high, state_t, text_action, state_t1, "object_attribute", "cooled"))
            cool.add(obj)
            hot.discard(obj)

        elif action == "ToggleObject":
            obj = obj or toggle_target
            if not obj:
                continue
            was_on = obj in toggled_on
            state_t = f"{article(obj).capitalize()} is {'on' if was_on else 'off'}."
            state_t1 = f"{article(obj).capitalize()} is {'off' if was_on else 'on'}."
            rows.append(emit_row(path, split, data, high, state_t, text_action, state_t1, "object_attribute", "toggled"))
            if was_on:
                toggled_on.discard(obj)
            else:
                toggled_on.add(obj)

        elif action == "SliceObject":
            obj = obj or target_obj
            if not obj:
                continue
            state_t = f"{article(obj).capitalize()} is not sliced."
            state_t1 = f"{article(obj).capitalize()} is sliced."
            rows.append(emit_row(path, split, data, high, state_t, text_action, state_t1, "object_attribute", "sliced"))
            sliced.add(obj)

    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-valid-train", action="store_true")
    args = parser.parse_args()

    split_map = dict(SPLIT_MAP)
    if args.include_valid_train:
        split_map["valid_train"] = "dev"

    rows_by_split = defaultdict(list)
    seen = {}
    duplicate_count = 0
    raw_counts = Counter()
    action_counts = Counter()
    task_counts = Counter()

    for source_split, out_split in split_map.items():
        split_dir = args.source_dir / source_split
        for path in sorted(split_dir.rglob("traj_data.json")):
            for row in process_trial(path, out_split):
                raw_counts[out_split] += 1
                key = (row["text_t"], row["action_text"], row["text_t1"])
                if key in seen:
                    duplicate_count += 1
                    continue
                seen[key] = row["id"]
                rows_by_split[out_split].append(row)
                action_counts[row["action_type"]] += 1
                task_counts[row["task_type"]] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        write_jsonl(args.output_dir / f"alfworld_nl_{split}.jsonl", rows_by_split[split])
        write_jsonl(
            args.output_dir / f"alfworld_nl_no_goto_{split}.jsonl",
            [row for row in rows_by_split[split] if row["action_type"] != "GotoLocation"],
        )

    sample_path = args.output_dir / "alfworld_nl_samples.md"
    with sample_path.open("w", encoding="utf-8") as f:
        f.write("# ALFWorld Next-State Samples\n\n")
        sample_rows = rows_by_split["train"][:20] + rows_by_split["dev"][:5] + rows_by_split["test"][:5]
        for i, row in enumerate(sample_rows, 1):
            f.write(f"## {i}. {row['id']}\n")
            f.write(f"- task_type: {row['task_type']}\n")
            f.write(f"- action_type: {row['action_type']}\n")
            f.write(f"- state_t: {row['text_t']}\n")
            f.write(f"- action: {row['action_text']}\n")
            f.write(f"- state_t1: {row['text_t1']}\n\n")

    summary = {
        "source": str(args.source_dir),
        "output": str(args.output_dir),
        "split_map": split_map,
        "raw_rows_by_split": dict(raw_counts),
        "deduplicated_rows_by_split": {k: len(v) for k, v in rows_by_split.items()},
        "no_goto_rows_by_split": {
            k: sum(1 for row in v if row["action_type"] != "GotoLocation")
            for k, v in rows_by_split.items()
        },
        "duplicates_removed": duplicate_count,
        "action_counts": dict(action_counts),
        "task_counts": dict(task_counts),
        "note": "Rows are high-level ALFWorld state-action-next-state transitions; exact triples are globally deduplicated.",
    }
    (args.output_dir / "alfworld_nl_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
