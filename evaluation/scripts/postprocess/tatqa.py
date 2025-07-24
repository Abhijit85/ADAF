from __future__ import annotations

import json
import re
from typing import Dict, Any, List


class TatqaPostProcessor:
    """Dataset-specific post-processing rules for TatQA consolidated JSON."""

    _NUMERIC_TYPES = {"arithmetic", "count"}
    _MAYBE_NUMERIC = {"span", "multi-span"}

    _CURRENCY_TOKENS = {"£", "$", "€", "eur", "euro", "usd", "dollars", "pounds"}
    _UNIT_TOKENS = {"million", "thousand", "billion", "bn", "m", "b", "k"}
    _SCALE_WORDS = _UNIT_TOKENS | {"percent", "%", "bps"}

    def __init__(self):
        self._atype_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in records:
            qid = rec["qid"]
            atype = rec.get("answer_type") or self._atype_map.get(qid, "")
            rec["answer_type"] = atype

            # skip record if gold answer empty
            if not rec.get("pp_gold_answer") and not rec.get("gold_answer"):
                continue

            # derive pp answers
            gold_raw = rec["gold_answer"]
            pred_raw = rec["pred_answer"]

            # special textual multi-span join
            if atype == "multi-span" and not self._is_numeric_list(gold_raw):
                pp_gold = ", ".join(sorted(str(x).strip() for x in gold_raw)) if isinstance(gold_raw, list) else str(gold_raw)
                if isinstance(pred_raw, list):
                    pp_pred = ", ".join(sorted(str(x).strip() for x in pred_raw))
                else:
                    pp_pred = str(pred_raw).strip()
            else:
                # always attempt numeric cleaning for arithmetic type
                numeric_flag = self._is_numeric(atype, gold_raw)

                g_first = self._first_elem(gold_raw)
                pp_gold_num = self._clean_numeric(str(g_first)) if numeric_flag else ""

                # prediction handling
                if atype in self._NUMERIC_TYPES:
                    pp_pred_clean = self._clean_numeric(str(pred_raw))
                    if pp_pred_clean and pp_pred_clean != "0":
                        pp_pred = pp_pred_clean
                    else:
                        pp_pred = str(pred_raw).strip()
                    pp_gold = pp_gold_num if pp_gold_num else self._textify(gold_raw)
                else:
                    if numeric_flag:
                        p_first = self._first_elem(pred_raw)
                        pp_pred_clean = self._clean_numeric(str(p_first))
                        if (pp_pred_clean in {"", "0"}) and (not self._looks_numeric(str(p_first))):
                            pp_pred = str(p_first).strip()
                        else:
                            pp_pred = pp_pred_clean
                        pp_gold = pp_gold_num
                    else:
                        # textual path
                        pp_gold = self._textify(gold_raw)
                        pp_pred = self._textify(pred_raw)

                # truncate prediction to match significant digits of gold
                if pp_gold.isdigit() and pp_pred.isdigit():
                    sig = len(pp_gold)
                    if len(pp_pred) > sig:
                        pp_pred = pp_pred[:sig]

            pp_gold = self._sanitize_text(pp_gold)
            pp_pred = self._sanitize_text(pp_pred)

            rec["pp_gold_answer"] = pp_gold
            rec["pp_pred_answer"] = pp_pred
            out.append(rec)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _first_elem(val: Any):
        if isinstance(val, list):
            return val[0] if val else ""
        return val

    @staticmethod
    def _textify(val: Any) -> str:
        if isinstance(val, list):
            return str(val[0]) if val else ""
        return str(val)

    # ------------------------------------------------------------------
    def _is_numeric(self, answer_type: str, gold_value: Any) -> bool:
        if answer_type in self._NUMERIC_TYPES:
            return True
        if answer_type in self._MAYBE_NUMERIC:
            first = self._first_elem(gold_value)
            if self._looks_numeric(str(first)):
                return True
            # fallback: try cleaning and check digits
            cleaned = self._clean_numeric(str(first))
            return cleaned.isdigit() and cleaned != "0"
        return False

    @classmethod
    def _looks_numeric(cls, s: str) -> bool:
        s_low = s.lower().strip()
        for tok in cls._CURRENCY_TOKENS | cls._SCALE_WORDS:
            s_low = s_low.replace(tok, "")
        s_low = s_low.replace(",", "").replace("%", "")
        s_low = s_low.strip()  # remove any spaces left after token removal
        return bool(re.match(r"^[+-]?\d*\.?\d+$", s_low))

    @classmethod
    def _is_numeric_list(cls, lst: Any) -> bool:
        if not isinstance(lst, list):
            return False
        return all(cls._looks_numeric(str(x)) for x in lst)

    # ------------------------------------------------------------------
    def _clean_numeric(self, s: str) -> str:
        s_low = s.lower()
        s_low = s_low.replace("(", "").replace(")", "")
        # remove currency and scale tokens
        for tok in self._CURRENCY_TOKENS | self._SCALE_WORDS:
            s_low = s_low.replace(tok, "")
        # remove commas and thousands dots
        s_low = s_low.replace(",", "")
        s_low = re.sub(r"\.(?=\d{3}\b)", "", s_low)

        # split on whitespace, take first token
        token = s_low.strip().split()[0] if s_low.strip() else ""

        # drop sign and plus
        token = token.lstrip("+-")

        if not token:
            return ""

        # unit phrase conversion before split
        unit_match = re.match(r"^(\d+(?:\.\d+)?)\s*(million|billion|thousand|bn|m|k)$", token)
        if unit_match:
            num_str, unit = unit_match.groups()
            try:
                num_val = float(num_str)
            except ValueError:
                num_val = 0
            factor = 1
            if unit in {"million", "m"}:
                factor = 1e6
            elif unit in {"billion", "bn"}:
                factor = 1e9
            elif unit in {"thousand", "k"}:
                factor = 1e3
            token = str(int(num_val * factor))
        else:
            # existing decimal handling
            if "." in token:
                int_part, dec_part = token.split(".", 1)
                if len(int_part) >= 2:
                    token = int_part
                else:
                    token = int_part  # drop decimal completely for 1-digit int part

        # remove any remaining non-digit chars
        token = re.sub(r"\D", "", token)

        # strip leading zeros
        token = token.lstrip("0") or "0"
        return token

    # ------------------------------------------------------------------
    @staticmethod
    def _round_to_digits(num_str: str, digits: int) -> str:
        if not num_str.isdigit():
            return num_str
        if len(num_str) <= digits:
            return num_str
        n = int(num_str)
        factor = 10 ** (len(num_str) - digits)
        rounded = int(round(n / factor)) * factor
        # ensure no scientific notation
        return str(rounded) 

    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_text(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9 .-]+", "", str(s)).strip() 