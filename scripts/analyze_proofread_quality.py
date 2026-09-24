"""Read-only human evaluation summary across current and archived videos."""

import argparse
import importlib.util
import json
from pathlib import Path


def discover_workbooks(paths):
    found = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            found.add(path.resolve())
        else:
            found.update(p.resolve() for p in path.rglob("translation_results_proofread.xlsx"))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description="Summarize human-labeled proofreading; never treats model changes as verified fixes.")
    parser.add_argument("paths", nargs="+", help="Workbook paths or video/history directories")
    args = parser.parse_args()
    # Avoid importing core/__init__ (ASR, GPU and Streamlit are irrelevant here).
    spec = importlib.util.spec_from_file_location(
        "proofread_evaluation", Path(__file__).resolve().parents[1] / "core/utils/proofread_evaluation.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    import pandas as pd
    videos, errors = [], []
    try:
        files = discover_workbooks(args.paths)
    except FileNotFoundError as error:
        parser.error(f"Path does not exist: {error}")
    if not files:
        parser.error("No proofreading workbooks found")
    for path in files:
        try:
            frame = pd.read_excel(path)
            if "LLM Proofread" not in frame or not {"Translation", "Original Translation"}.intersection(frame.columns):
                raise ValueError("Missing original/candidate translation columns")
            summary = evaluation.summarize_records(frame.to_dict("records"))
            videos.append({"file": str(path), **summary})
        except Exception as error:
            errors.append({"file": str(path), "error": str(error)})
    print(json.dumps({
        "note": "Human labels only establish sampled row outcomes, not whole-video accuracy. Blank labels are pending, never successful fixes. mixed means both a fix and a regression; report it separately. Counts include invalid labels and unreadable files.",
        "videos": videos, "total": evaluation.combine_summaries(videos), "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
