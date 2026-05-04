#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://api.deepseek.com/chat/completions"
_THREAD_LOCAL = threading.local()


def get_ssl_context(insecure_skip_verify):
    if not insecure_skip_verify:
        return None
    if not hasattr(_THREAD_LOCAL, "ssl_context"):
        _THREAD_LOCAL.ssl_context = ssl._create_unverified_context()
    return _THREAD_LOCAL.ssl_context


def call_deepseek(api_key, model, state_t, action, state_t1, timeout=60, ssl_context=None):
    prompt = {
        "state_t": state_t,
        "action": action,
        "state_t1": state_t1,
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite only the compact state_t and state_t1 fields into concise natural English. "
                    "Use the action only as context. Do not rewrite the action. "
                    "Preserve the entity and location/existence meaning exactly. Do not add facts. "
                    "Return only JSON with keys text_t and text_t1. Keep each state short, one sentence each."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    rewritten = json.loads(content)
    return {
        "text_t": rewritten["text_t"].strip(),
        "text_t1": rewritten["text_t1"].strip(),
    }


def rewrite_one(idx, row, args, api_key):
    row = dict(row)
    original = {
        "text_t": row["text_t"],
        "action_text": row["action_text"],
        "text_t1": row["text_t1"],
    }
    for attempt in range(args.retries):
        try:
            rewritten = call_deepseek(
                api_key,
                args.model,
                row["text_t"],
                row["action_text"],
                row["text_t1"],
                timeout=args.timeout,
                ssl_context=get_ssl_context(args.insecure_skip_verify),
            )
            row["original_text_t"] = original["text_t"]
            row["original_action_text"] = original["action_text"]
            row["original_text_t1"] = original["text_t1"]
            row.update(rewritten)
            row["action_text"] = original["action_text"]
            return idx, row, original, None
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as exc:
            if attempt == args.retries - 1:
                return idx, row, original, repr(exc)
            time.sleep(args.retry_sleep * (attempt + 1))
    return idx, row, original, "unreachable retry state"


def comparison_block(display_idx, row, original):
    return [
        f"## {display_idx}. {row['id']}",
        f"- original state_t: {original['text_t']}",
        f"- rewritten state_t: {row['text_t']}",
        f"- action: {row['action_text']}",
        f"- original state_t1: {original['text_t1']}",
        f"- rewritten state_t1: {row['text_t1']}",
        "",
    ]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="processed_datasets/propara_nl_changed")
    parser.add_argument("--output-dir", default="processed_datasets/propara_nl_changed_deepseek100_stateonly")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--insecure-skip-verify", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required")
    if args.smoke_test:
        args.limit = 1
        args.concurrency = 1
        args.splits = [args.splits[0]]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_outputs = {}
    split_originals = {}
    comparisons = ["# DeepSeek Rewritten ProPara Samples", ""]
    failures = []
    requested_by_split = {}

    for split in args.splits:
        rows = list(read_jsonl(input_dir / f"propara_nl_{split}.jsonl"))
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
        requested_by_split[split] = len(rows)
        rewritten_rows = [None] * len(rows)
        originals = [None] * len(rows)
        max_workers = max(1, args.concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(rewrite_one, idx, row, args, api_key)
                for idx, row in enumerate(rows)
            ]
            done_count = 0
            for future in concurrent.futures.as_completed(futures):
                idx, row, original, error = future.result()
                done_count += 1
                rewritten_rows[idx] = row
                originals[idx] = original
                label = f"{split} {done_count}/{len(rows)}"
                if error:
                    failures.append({"split": split, "index": idx, "id": row.get("id"), "error": error})
                    print(f"failed {label}: {row.get('id')} {error}", flush=True)
                else:
                    print(f"rewrote {label}: {row['id']}", flush=True)
        split_outputs[split] = [row for row in rewritten_rows if row is not None]
        split_originals[split] = originals

    sample_count = 0
    for split in args.splits:
        for idx, row in enumerate(split_outputs.get(split, [])):
            original = split_originals[split][idx]
            if row is not None and original is not None and sample_count < 120:
                comparisons.extend(comparison_block(sample_count + 1, row, original))
                sample_count += 1

    for split in ["train", "dev", "test"]:
        if split in split_outputs:
            write_jsonl(output_dir / f"propara_nl_{split}.jsonl", split_outputs[split])
        else:
            rows = list(read_jsonl(input_dir / f"propara_nl_{split}.jsonl"))
            write_jsonl(output_dir / f"propara_nl_{split}.jsonl", rows)

    summary = {
        "source": str(input_dir),
        "model": args.model,
        "splits": args.splits,
        "requested_by_split": requested_by_split,
        "rewritten_rows": sum(len(v) for v in split_outputs.values()) - len(failures),
        "failed_rows": len(failures),
        "failures": failures,
        "output_rows_by_split": {split: len(rows) for split, rows in split_outputs.items()},
        "concurrency": args.concurrency,
        "smoke_test": args.smoke_test,
        "note": "Only state_t/text_t1 were rewritten; actions copied unchanged.",
    }
    (output_dir / "propara_nl_deepseek100_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "propara_nl_deepseek100_samples.md").write_text(
        "\n".join(comparisons),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
