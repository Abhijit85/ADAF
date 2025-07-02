# evaluation/tatqa/scorer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple
from ast import literal_eval
import json as _json
import logging

from ..engine import EvalEngine, register_dataset
from .preprocessor import load_gold, _normalise


# ---------------------------------------------------------------------------
# logging setup --------------------------------------------------------------
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # basicConfig does nothing if root handlers are already configured by user
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

# Ensure our module logger emits INFO even if root was already configured higher
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# helpers --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _norm_uid(uid: str | None) -> str | None:
    """Strip dashes so 8e7f-... matches 8e7f... . Returns None if uid None."""
    return uid.replace("-", "") if uid else uid


@register_dataset("tatqa")
class TatqaEvaluator(EvalEngine):
    def _load_gold(self):
        """Load gold answers & questions, keep mapping table_uid → list[qid].

        Prediction files are named after TABLE uids, but evaluation metrics are
        reported per-question.  Therefore we build both a question-level gold
        dict and an index from table_uid to the list of question ids it owns.
        We only parse tables that appear in the prediction folder to avoid
        wasting memory/time.
        """

        # 1) collect table-uids present in prediction artefacts
        needed_tables: set[str] = set()
        p = self.pred_path
        if p.is_dir():
            needed_tables = {_norm_uid(f.stem.replace("_out", "")) for f in p.glob("*_out.txt")}
        else:
            try:
                needed_tables = {_norm_uid(t) for t in _json.loads(p.read_text()).keys()}
            except Exception:
                pass  # keep all tables

        # ADDED LOGS ---------------------------------------------------------
        logger.info(
            "Predictions reference %d distinct tables (dir=%s). First 5 normalised ids: %s",
            len(needed_tables), p.is_dir(), list(needed_tables)[:5],
        )
        if not needed_tables:
            logger.warning(
                "No table UIDs were discovered from prediction artefacts – gold will not be filtered."
            )
        # -------------------------------------------------------------------

        # 2) parse gold file
        data = _json.loads(self.gold_path.read_text())
        self._q_text: Dict[str, str] = {}
        self._table_to_qids: Dict[str, list[str]] = {}
        answers: Dict[str, str] = {}

        total_tables, retained_tables = 0, 0  # ADDED counters

        for ex in data:
            total_tables += 1  # ADDED
            table_uid_raw = ex.get("id") or ex.get("table", {}).get("uid")
            table_uid = _norm_uid(table_uid_raw)
            if needed_tables and table_uid not in needed_tables:
                logger.debug(
                    "Skipping gold table %s (norm %s) – not present in predictions",
                    table_uid_raw,
                    table_uid,
                )
                continue
            retained_tables += 1  # ADDED

            qids_for_table: list[str] = []
            for qa in ex.get("questions", []):
                qid = qa.get("uid") or qa.get("id")
                self._q_text[qid] = qa.get("question", "")
                answers[qid] = _normalise(qa.get("answer"), qa.get("scale", ""))
                qids_for_table.append(qid)

            if not qids_for_table:
                logger.warning(
                    "Gold table %s (norm %s) retained but contains 0 questions after processing",
                    table_uid_raw,
                    table_uid,
                )

            self._table_to_qids[table_uid] = qids_for_table

        # ADDED summary LOG --------------------------------------------------
        logger.info(
            "Gold file parsing finished: %d/%d tables retained after filtering; %d questions total",
            retained_tables,
            total_tables,
            len(answers),
        )
        # -------------------------------------------------------------------

        logger.info("TAT-QA gold loaded: %d questions across %d tables", len(answers), len(self._table_to_qids))
        logger.debug("First 5 table→qids: %s", {k: v[:3] for k, v in list(self._table_to_qids.items())[:5]})
        return answers

    def _iter_predictions(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """Yield (qid, summary_text, meta). Accept dir or JSON mapping."""
        p = self.pred_path
        if p.is_dir():
            for file in p.glob("*_out.txt"):
                table_uid = file.stem.replace("_out", "")
                norm_uid = _norm_uid(table_uid)

                raw = file.read_text()
                summary_part, logs_text = raw, ""
                if "=== FINAL SUMMARY ===" in raw and "=== LOGS" in raw:
                    summary_part = raw.split("=== LOGS", 1)[0]
                    logs_text = raw.split("=== LOGS", 1)[1]
                summary = self._extract_summary(summary_part)

                # parse agent logs once
                agent_cols: Dict[str, str] = {}
                if logs_text:
                    import ast
                    try:
                        logs_dict = ast.literal_eval(logs_text.strip())
                        for agent, payload in logs_dict.items():
                            agent_cols[f"agent_{agent}"] = payload.get("result", "")
                    except Exception:
                        pass

                answer_echoes = ""
                if "Answer Echoes:" in summary:
                    answer_echoes = summary.split("Answer Echoes:", 1)[1].strip()

                qids = self._table_to_qids.get(norm_uid, [])
                if not qids:
                    logger.warning("No gold questions for table %s (norm %s)", table_uid, norm_uid)
                else:
                    logger.info(
                        "Table %s matched %d gold questions; first 5 qids: %s",
                        table_uid,
                        len(qids),
                        qids[:5],
                    )
                for qid in qids:
                    meta = {
                        "question": self._q_text.get(qid, ""),
                        "answer_echoes": answer_echoes,
                    }
                    meta.update(agent_cols)
                    yield qid, summary, meta
        else:
            import json

            data = json.loads(p.read_text())
            for table_uid, summary in data.items():
                norm_uid = _norm_uid(table_uid)
                for qid in self._table_to_qids.get(norm_uid, []):
                    yield qid, str(summary), {}

    def _extract_summary(self, text: str) -> str:
        if "=== FINAL SUMMARY ===" in text:
            text = text.split("=== FINAL SUMMARY ===", 1)[1]
        if "=== LOGS" in text:
            text = text.split("=== LOGS", 1)[0]
        if self.mode == "summary":
            idx = text.find("Answer Echoes:")
            if idx != -1:
                text = text[:idx]
        return text.strip() 