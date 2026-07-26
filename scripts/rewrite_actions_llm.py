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
THREAD_LOCAL = threading.local()


def ssl_context(insecure):
    if not insecure:
        return None
    if not hasattr(THREAD_LOCAL, "ssl_context"):
        THREAD_LOCAL.ssl_context = ssl._create_unverified_context()
    return THREAD_LOCAL.ssl_context


def call_model(api_key, model, row, timeout, context):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite the action as one concise, natural English sentence. "
                    "Preserve its physical meaning, participants, attributes, direction, "
                    "and location exactly. Use noticeably different wording and minimize "
                    "non-essential lexical overlap with the supplied next state. Do not "
                    "change the current or next state and do not add facts. Return only "
                    "JSON with the key action_text."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "current_state": row["text_t"],
                    "original_action": row["action_text"],
                    "next_state": row["text_t1"],
                }, ensure_ascii=False),
            },
        ],
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(
        request, timeout=timeout, context=context
    ) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = json.loads(result["choices"][0]["message"]["content"])
    action = str(content["action_text"]).strip()
    if not action:
        raise ValueError("empty action rewrite")
    return action


def rewrite(index, row, args, api_key):
    for attempt in range(args.retries):
        try:
            action = call_model(
                api_key, args.model, row, args.timeout,
                ssl_context(args.insecure_skip_verify),
            )
            output = dict(row)
            output["original_action_text"] = row["action_text"]
            output["action_text"] = action
            return index, output, None
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as error:
            if attempt + 1 == args.retries:
                output = dict(row)
                output["original_action_text"] = row["action_text"]
                return index, output, repr(error)
            time.sleep(args.retry_sleep * (attempt + 1))
    raise AssertionError("unreachable")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--insecure-skip-verify", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required")
    rows = read_jsonl(Path(args.input))
    if args.limit is not None:
        rows = rows[:args.limit]
    rewritten = [None] * len(rows)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.concurrency)
    ) as pool:
        futures = [
            pool.submit(rewrite, index, row, args, api_key)
            for index, row in enumerate(rows)
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            index, row, error = future.result()
            rewritten[index] = row
            if error:
                failures.append({"index": index, "id": row.get("id"), "error": error})
            print(
                f"{completed}/{len(rows)} {row.get('id')} "
                f"{'failed' if error else 'ok'}",
                flush=True,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rewritten)
    summary = {
        "input": args.input,
        "output": args.output,
        "model": args.model,
        "rows": len(rows),
        "successful_rewrites": len(rows) - len(failures),
        "failed_rewrites": len(failures),
        "failures": failures,
        "concurrency": args.concurrency,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
