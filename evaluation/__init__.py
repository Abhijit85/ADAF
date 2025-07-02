# evaluation/__init__.py
# Public dispatcher: evaluate(dataset, gold_path, pred_path, ...)
# Provides function that defers to dataset-specific Evaluator classes.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

from .engine import get_evaluator

__all__ = ["evaluate"]


def evaluate(
    dataset: str,
    gold_path: str | Path,
    pred_path: str | Path,
    mode: str = "summary+echoes",
    pred_kind: str = "summary",  # "summary" or "answer"
    openai_key: str | None = None,
    run_id: str | None = None,
    eval_id: str | None = None,
) -> Dict[str, Any]:
    """Run evaluation for *dataset* and return aggregate scores & per-question table.

    Parameters
    ----------
    dataset: str
        One of "tatqa", "finqa" (case-insensitive).
    gold_path: path
        Path to dataset ground-truth JSON.
    pred_path: path
        Either directory with *_out.txt files or JSON mapping qid→prediction.
    mode: str
        summary+echoes | summary (ignored for pred_kind=="answer").
    pred_kind: str
        "summary" → run full summarisation metrics (REMS / CAE / HCS)
        "answer"   → treat predictions as direct answers; only EM / token-F1.
    openai_key: str | None
        Provide to enable CAE GPT metric.
    run_id: str | None
        Optional run ID for saving results.
    eval_id: str | None
        Optional evaluation ID for saving results.
    """

    evaluator_cls = get_evaluator(dataset)
    evaluator = evaluator_cls(
        gold_path=Path(gold_path),
        pred_path=Path(pred_path),
        mode=mode,
        pred_kind=pred_kind,
        openai_key=openai_key,
    )
    result = evaluator.evaluate()

    if run_id:
        if eval_id is None:
            from datetime import datetime

            eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / dataset.lower() / run_id / eval_id
        out_dir.mkdir(parents=True, exist_ok=True)

        fname_stub = f"{run_id}_{eval_id}"
        (out_dir / f"report_{fname_stub}.json").write_text(json.dumps(result, indent=2))

        try:
            import pandas as pd

            df = pd.DataFrame(result.get("per_question", []))
            df.to_excel(out_dir / f"detailed_{fname_stub}.xlsx", index=False)
        except ImportError:
            pass
    return result 