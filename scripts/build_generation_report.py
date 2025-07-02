# scripts/build_generation_report.py
"""Build an Excel + JSON report from *_out.txt logs right after generation.

Usage:  python scripts/build_generation_report.py --log_dir out/tatqa_logs/20240630_1230 --dataset tatqa [--gold path]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

_summary_delim = "=== FINAL SUMMARY ==="
_logs_delim = "=== LOGS"


def parse_out_file(path: Path):
    text = path.read_text()
    summary_part, logs_part = text, ""
    if _summary_delim in text and _logs_delim in text:
        parts = text.split(_logs_delim, 1)
        summary_part = parts[0]
        logs_part = parts[1]
        if _summary_delim in summary_part:
            summary_part = summary_part.split(_summary_delim, 1)[1]
    summary = summary_part.strip()
    answer_echoes = ""
    idx = summary.find("Answer Echoes:")
    if idx != -1:
        answer_echoes = summary[idx + len("Answer Echoes:"):].strip()
    agent_cols: Dict[str, str] = {}
    if logs_part:
        try:
            logs_dict = ast.literal_eval(logs_part.strip())
            for agent, payload in logs_dict.items():
                agent_cols[f"agent_{agent}"] = payload.get("result", "")
        except Exception:
            pass
    return summary, answer_echoes, agent_cols

def load_gold(dataset: str, path: Path | None = None):
    if path is None:
        default_paths = {
            "tatqa": Path("data/tatqa/tatqa_dataset_dev.json"),
            "finqa": Path("data/finqa/finqa_dev.json"),
            "mmqa": Path("data/mmqa/mmqa_dev.jsonl"),
        }
        path = default_paths.get(dataset)
    if not path or not path.exists():
        return {}, {}
    import json as _json

    if dataset == "mmqa":
        gold_ans: Dict[str, str] = {}
        q_text: Dict[str, str] = {}
        with path.open() as f:
            for line in f:
                ex = _json.loads(line)
                qid = ex.get("qid")
                q_text[qid] = ex.get("question", "")
                gold_ans[qid] = ""  # MMQA answers not used for now
        return q_text, gold_ans
    data = _json.loads(path.read_text())
    q_text: Dict[str, str] = {}
    gold_ans: Dict[str, str] = {}
    if dataset == "tatqa":
        from evaluation.tatqa.preprocessor import _normalise as norm
        for ex in data:
            for qa in ex.get("questions", []):
                qid = qa.get("uid") or qa.get("id")
                q_text[qid] = qa.get("question", "")
                gold_ans[qid] = norm(qa.get("answer"), qa.get("scale", ""))
    elif dataset == "finqa":
        from evaluation.finqa.preprocessor import _normalise as norm
        for ex in data:
            for qa in ex.get("questions", ex.get("qa_pairs", [])):
                qid = qa.get("id") or qa.get("uid") or qa.get("question_id")
                q_text[qid] = qa.get("question", "")
                gold_ans[qid] = norm(qa.get("answer"))
    return q_text, gold_ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=["tatqa", "finqa", "mmqa"])
    ap.add_argument("--gold", type=Path, help="Optional gold file path override")
    args = ap.parse_args()

    q_text, gold_ans = load_gold(args.dataset, args.gold)

    rows: List[Dict] = []
    for file in args.log_dir.glob("*_out.txt"):
        qid = file.stem.replace("_out", "")
        summary, echoes, agents = parse_out_file(file)
        row = {
            "qid": qid,
            "question": q_text.get(qid, ""),
            "gold": gold_ans.get(qid, ""),
            "summary": summary,
            "answer_echoes": echoes,
        }
        row.update(agents)
        rows.append(row)

    if not rows:
        print("No *_out.txt files found in", args.log_dir)
        return

    df = pd.DataFrame(rows)
    # write JSON + Excel
    (args.log_dir / "run.json").write_text(json.dumps(rows, indent=2))
    df.to_excel(args.log_dir / "run.xlsx", index=False)
    print(f"Wrote {len(rows)} rows to run.xlsx & run.json in {args.log_dir}")


if __name__ == "__main__":
    main() 