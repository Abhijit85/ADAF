#!/usr/bin/env python
"""One-off helper: fix wrong table-uids in a TAT-QA gold file.

Usage
-----
python scripts/fix_tatqa_uids.py \
       --gold data/tatqa/tatqa_dataset_test_gold.json \
       --test data/tatqa/tatqa_dataset_test.json \
       --out  data/tatqa/tatqa_dataset_test_gold_fixed.json

Strategy
~~~~~~~~
1.  Build a lookup for the *test* file keyed by a cheap signature:
      (sorted question texts, first table row joined)
2.  For each entry in the *gold* file find the matching signature and
    copy over the `id` / `table.uid` from the test file.
3.  Write the patched list to --out.

The script logs how many replacements were made and warns about
mismatches so you can inspect remaining issues manually.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger("fix_tatqa_uids")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

Sig = Tuple[Tuple[str, ...], str]  # (sorted question texts, first table row)

def _signature(ex: Dict) -> Sig:
    q_texts = tuple(sorted(q.get("question", "").strip().lower() for q in ex.get("questions", [])))
    first_row = "|".join(str(cell).strip().lower() for cell in ex.get("table", {}).get("table", [[""]])[0])
    return q_texts, first_row

def main():
    ap = argparse.ArgumentParser(description="Fix wrong uids in TAT-QA gold JSON by matching entries from test JSON")
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    gold_data = json.loads(args.gold.read_text())
    test_data = json.loads(args.test.read_text())

    # build signature → uid mapping for test examples
    sig_to_uid: Dict[Sig, str] = {}
    sig_to_ex: Dict[Sig, Dict] = {}
    for ex in test_data:
        sig = _signature(ex)
        table_uid = ex.get("id") or ex.get("table", {}).get("uid")
        sig_to_uid[sig] = table_uid
        sig_to_ex[sig] = ex

    replaced = missed = 0
    for ex in gold_data:
        sig = _signature(ex)
        correct_uid = sig_to_uid.get(sig)
        if not correct_uid:
            logger.warning("No matching test entry for gold example starting with question '%s'", sig[0][0] if sig[0] else "<no question>")
            missed += 1
            continue
        # 3) copy uid at table level ------------------------------------
        current_uid = ex.get("id") or ex.get("table", {}).get("uid")
        if current_uid != correct_uid:
            ex["id"] = correct_uid
            if "table" in ex and isinstance(ex["table"], dict):
                ex["table"]["uid"] = correct_uid
            replaced += 1

        # 4) copy each question uid -------------------------------------
        test_ex = sig_to_ex.get(sig)
        if test_ex:
            test_q_lookup = {q.get("question", "").strip().lower(): q for q in test_ex.get("questions", [])}
        else:
            test_q_lookup = {}
        for q_gold in ex.get("questions", []):
            txt = q_gold.get("question", "").strip().lower()
            q_test = test_q_lookup.get(txt)
            if not q_test:
                continue
            gold_uid = q_gold.get("uid") or q_gold.get("id")
            test_uid = q_test.get("uid") or q_test.get("id")
            if gold_uid != test_uid and test_uid:
                if "uid" in q_gold:
                    q_gold["uid"] = test_uid
                elif "id" in q_gold:
                    q_gold["id"] = test_uid
    logger.info("Replaced %d uids; %d examples unmatched", replaced, missed)

    args.out.write_text(json.dumps(gold_data, indent=2))
    logger.info("Wrote fixed file → %s", args.out)


if __name__ == "__main__":
    main() 