import argparse
import json
import math
from pathlib import Path


def estimate_tokens(text):
    return math.ceil(len(str(text or "")) / 4)


def load_entries(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    if not isinstance(data, list):
        return []
    return data


def summarize_log_dir(log_dir):
    rows = []
    totals = {
        "requests": 0,
        "prompt_est": 0,
        "completion_est": 0,
        "total_est": 0,
        "prompt_exact": 0,
        "completion_exact": 0,
        "total_exact": 0,
        "exact_entries": 0,
    }

    for path in sorted(log_dir.glob("*.json")):
        if path.name in {"usage_events.json", "usage_summary.json"}:
            continue

        entries = load_entries(path)
        row = {
            "file": path.name,
            "requests": len(entries),
            "prompt_est": 0,
            "completion_est": 0,
            "total_est": 0,
            "prompt_exact": 0,
            "completion_exact": 0,
            "total_exact": 0,
            "exact_entries": 0,
        }

        for item in entries:
            prompt_est = estimate_tokens(item.get("prompt", ""))
            completion_est = estimate_tokens(item.get("resp_content", ""))
            row["prompt_est"] += prompt_est
            row["completion_est"] += completion_est
            row["total_est"] += prompt_est + completion_est

            usage = item.get("usage") or {}
            if usage.get("total_tokens") is not None:
                row["exact_entries"] += 1
                row["prompt_exact"] += usage.get("prompt_tokens") or 0
                row["completion_exact"] += usage.get("completion_tokens") or 0
                row["total_exact"] += usage.get("total_tokens") or 0

        for key in totals:
            totals[key] += row.get(key, 0)
        rows.append(row)

    return rows, totals


def print_table(rows, totals):
    header = (
        f"{'log file':28} {'req':>5} {'est prompt':>12} {'est resp':>10} "
        f"{'est total':>11} {'exact total':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        exact = str(row["total_exact"]) if row["exact_entries"] else "-"
        print(
            f"{row['file'][:28]:28} {row['requests']:5d} {row['prompt_est']:12d} "
            f"{row['completion_est']:10d} {row['total_est']:11d} {exact:>12}"
        )
    print("-" * len(header))
    exact_total = str(totals["total_exact"]) if totals["exact_entries"] else "-"
    print(
        f"{'TOTAL':28} {totals['requests']:5d} {totals['prompt_est']:12d} "
        f"{totals['completion_est']:10d} {totals['total_est']:11d} {exact_total:>12}"
    )
    if not totals["exact_entries"]:
        print("\nNo provider token usage was found. Estimated tokens use chars/4 from saved prompts/responses.")


def write_json_report(output_path, rows, totals):
    output = {"files": rows, "total": totals}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=4), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize VideoLingo LLM request logs and token estimates.")
    parser.add_argument("--log-dir", default="output/gpt_log", help="Path to VideoLingo gpt_log directory.")
    parser.add_argument("--json-out", help="Optional path to write a JSON report.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    rows, totals = summarize_log_dir(log_dir)
    print_table(rows, totals)
    if args.json_out:
        write_json_report(Path(args.json_out), rows, totals)
        print(f"\nWrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
