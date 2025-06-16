import argparse
import json
import math
import re
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Helper utilities

num_re = re.compile(r"-?\d+(?:\.\d+)?")


def normalise(s: str) -> str:
    """Normalise numeric strings for fuzzy matching."""
    return re.sub(r"[,$% ]", "", str(s).lower())


def numeric_hit(ans: str, summary: str) -> bool:
    """Return True if the numeric answer appears in the summary."""
    if re.search(re.escape(normalise(ans)), normalise(summary)):
        return True
    if num_re.fullmatch(ans.strip()):
        val = float(ans)
        for m in num_re.findall(summary):
            try:
                if math.isclose(float(m), val, rel_tol=0.01):
                    return True
            except ValueError:
                pass
    return False

# ---------------------------------------------------------------------------
# Metric calculations

def tatqa_metrics(ds_path: Path, log_dir: Path) -> None:
    """Compute ROUGE-L and BLEU against answer statements."""
    with ds_path.open(encoding="utf-8") as f:
        ds = json.load(f)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1

    rouge_scores = []
    bleu_scores = []

    for ex in ds:
        uid = ex.get("id") or ex.get("table", {}).get("uid")
        if not uid:
            continue
        summ_path = log_dir / f"{uid}_out.txt"
        if not summ_path.exists():
            continue
        summary = summ_path.read_text().split("=== FINAL SUMMARY ===")[-1].strip()

        refs = []
        for qa in ex.get("questions", []):
            q = qa.get("question", "").rstrip("?")
            ans = qa.get("answer")
            if isinstance(ans, list):
                ans = " ".join(str(a) for a in ans)
            refs.append(f"The answer to {q} is {ans}.")
        reference = " ".join(refs)
        rouge_scores.append(scorer.score(reference, summary)["rougeL"].fmeasure)
        bleu_scores.append(sentence_bleu([reference.split()], summary.split(), smoothing_function=smooth))

    if rouge_scores:
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        print(f"TAT-QA ROUGE-L: {avg_rouge:.4f}")
        print(f"TAT-QA BLEU:    {avg_bleu:.4f}")
    else:
        print("No matching TAT-QA summaries found.")


def finqa_accuracy(ds_path: Path, log_dir: Path) -> None:
    """Compute FinQA QA accuracy as the fraction of answers appearing in summaries."""
    with ds_path.open(encoding="utf-8") as f:
        ds = json.load(f)

    total = hits = 0
    for ex in ds:
        uid = ex.get("id") or ex.get("table", {}).get("uid")
        if not uid:
            continue
        summ_path = log_dir / f"{uid}_out.txt"
        if not summ_path.exists():
            continue
        summary = summ_path.read_text().split("=== FINAL SUMMARY ===")[-1].lower()
        for qa in ex.get("questions", []):
            ans = qa.get("answer")
            if isinstance(ans, list):
                ans = " ".join(str(a) for a in ans)
            ans = str(ans).strip()
            if not ans:
                continue
            total += 1
            if numeric_hit(ans, summary):
                hits += 1

    if total:
        print(f"FinQA QA accuracy: {hits}/{total} -> {hits/total:.2%}")
    else:
        print("No FinQA examples evaluated.")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute summarization metrics.")
    ap.add_argument("--tatqa", type=Path, default=Path("data/TATQA/tatqa_dataset_dev.json"), help="TAT-QA dataset JSON")
    ap.add_argument("--tatqa_logs", type=Path, default=Path("out/tatqa_logs/15_06_2025"), help="Directory with TAT-QA summaries")
    ap.add_argument("--finqa", type=Path, help="FinQA dataset JSON")
    ap.add_argument("--finqa_logs", type=Path, help="Directory with FinQA summaries")
    args = ap.parse_args()

    if args.tatqa:
        tatqa_metrics(args.tatqa, args.tatqa_logs)
    if args.finqa and args.finqa_logs:
        finqa_accuracy(args.finqa, args.finqa_logs)


if __name__ == "__main__":
    main()