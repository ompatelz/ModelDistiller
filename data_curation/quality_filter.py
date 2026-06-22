"""
Quality filtering step for the Forge curation pipeline.

Reads deduplicated examples and removes degenerate or low-quality ones before
the train/val/eval split.

Filters applied (in order, with individual stats tracked)
---------------------------------------------------------
1.  DOCUMENT LENGTH: document_text must be between MIN_DOC_CHARS and MAX_DOC_CHARS
    characters.  Too short → probably a failed/garbled generation.
    Too long → probably a model that ignored formatting guidance.

2.  TOTAL AMOUNT SANITY: total_amount must be > 0.  A $0.00 invoice is
    theoretically possible (fully comped), but in practice indicates a
    generation failure where the model couldn't produce a plausible amount.

3.  LINE ITEMS PRESENCE: line_items must be non-empty.  (This should already
    be caught by schema validation, but we double-check here.)

4.  MINIMUM FIELD POPULATION: at least MIN_NON_NULL_OPTIONAL_FIELDS of the
    optional fields must be non-null.  An example where every optional field
    is null is likely degenerate even if it passes schema validation — it
    provides almost no useful training signal for those fields.  Exception:
    the 'missing_fields_hard' scenario is explicitly designed to have many
    nulls, so we exempt examples from that scenario.

5.  DOCUMENT TEXT SANITY: basic heuristics on the document_text string:
    - Must contain at least one digit (real invoices have numbers)
    - Must not be repetitive (repetition ratio check)

Each filter writes a separate count to the output stats so the curation report
can state precisely why each example was removed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema.extraction_schema import InvoiceExtraction

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable thresholds
# ---------------------------------------------------------------------------

MIN_DOC_CHARS: int = 100
MAX_DOC_CHARS: int = 8000
MIN_NON_NULL_OPTIONAL_FIELDS: int = 3

# Scenarios explicitly designed to have many null fields — exempt from filter 4
NULL_FIELD_EXEMPT_SCENARIOS: set[str] = {"missing_fields_hard"}

# Optional fields (everything except vendor_name, line_items, total_amount, currency)
OPTIONAL_FIELDS: list[str] = [
    "vendor_address", "invoice_number", "invoice_date", "due_date",
    "bill_to", "subtotal", "tax_amount", "tax_rate", "payment_terms", "notes",
]


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def _filter_doc_length(example: dict) -> tuple[bool, str]:
    doc = example.get("document_text", "")
    n = len(doc)
    if n < MIN_DOC_CHARS:
        return False, f"doc_too_short ({n} < {MIN_DOC_CHARS} chars)"
    if n > MAX_DOC_CHARS:
        return False, f"doc_too_long ({n} > {MAX_DOC_CHARS} chars)"
    return True, ""


def _filter_total_amount(example: dict) -> tuple[bool, str]:
    gt = example.get("ground_truth_json", {})
    total = gt.get("total_amount")
    if total is None:
        return False, "total_amount_missing"
    try:
        total = float(total)
    except (TypeError, ValueError):
        return False, "total_amount_not_numeric"
    if total <= 0:
        return False, f"total_amount_zero_or_negative ({total})"
    return True, ""


def _filter_line_items(example: dict) -> tuple[bool, str]:
    gt = example.get("ground_truth_json", {})
    items = gt.get("line_items")
    if not items or not isinstance(items, list) or len(items) == 0:
        return False, "line_items_empty_or_missing"
    return True, ""


def _filter_min_fields(example: dict) -> tuple[bool, str]:
    scenario_id = example.get("scenario_id", "")
    if scenario_id in NULL_FIELD_EXEMPT_SCENARIOS:
        return True, ""

    gt = example.get("ground_truth_json", {})
    non_null_count = sum(
        1 for field in OPTIONAL_FIELDS if gt.get(field) is not None
    )
    if non_null_count < MIN_NON_NULL_OPTIONAL_FIELDS:
        return False, (
            f"too_few_non_null_fields "
            f"({non_null_count} < {MIN_NON_NULL_OPTIONAL_FIELDS}, scenario={scenario_id})"
        )
    return True, ""


def _filter_doc_sanity(example: dict) -> tuple[bool, str]:
    doc = example.get("document_text", "")

    # Must contain at least one digit
    if not any(c.isdigit() for c in doc):
        return False, "doc_contains_no_digits"

    # Repetition check: if the most-common 20-char substring appears > 10% of
    # the document's length, it's probably a looping / garbled generation.
    if len(doc) > 200:
        chunk_size = 20
        chunks = [doc[i : i + chunk_size] for i in range(0, len(doc) - chunk_size, chunk_size)]
        if chunks:
            from collections import Counter
            most_common_count = Counter(chunks).most_common(1)[0][1]
            if most_common_count / len(chunks) > 0.3:
                return False, "doc_repetitive_content"

    return True, ""


# ---------------------------------------------------------------------------
# Main filter pipeline
# ---------------------------------------------------------------------------

FILTERS = [
    ("doc_length",   _filter_doc_length),
    ("total_amount", _filter_total_amount),
    ("line_items",   _filter_line_items),
    ("min_fields",   _filter_min_fields),
    ("doc_sanity",   _filter_doc_sanity),
]


def apply_quality_filters(example: dict) -> tuple[bool, str]:
    """
    Apply all filters in sequence; return (passed, first_failure_reason).
    Short-circuits on first failure.
    """
    for _name, filter_fn in FILTERS:
        passed, reason = filter_fn(example)
        if not passed:
            return False, reason
    return True, ""


def run_quality_filter(
    input_path: Path,
    output_path: Path,
    removed_path: Path,
) -> dict:
    """
    Read deduplicated JSONL, apply quality filters, write kept and removed examples.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict = {
        "n_input": 0,
        "n_kept": 0,
        "n_removed": 0,
        "filter_counts": {name: 0 for name, _ in FILTERS},
    }

    log.info("Applying quality filters to %s ...", input_path)

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
        removed_path.open("w", encoding="utf-8") as ffail,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["n_input"] += 1
            passed, reason = apply_quality_filters(example)

            if passed:
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats["n_kept"] += 1
            else:
                removed_example = {**example, "quality_filter_reason": reason}
                ffail.write(json.dumps(removed_example, ensure_ascii=False) + "\n")
                stats["n_removed"] += 1
                # Attribute to the specific filter
                for filter_name, _ in FILTERS:
                    if reason.startswith(filter_name) or any(
                        reason.startswith(r) for r in [
                            "doc_too", "total_amount", "line_items", "too_few", "doc_contains", "doc_repetitive"
                        ]
                    ):
                        # Simple prefix matching is sufficient for the report
                        break

                # Bucket by first token of reason
                bucket = reason.split("_")[0] + "_" + reason.split("_")[1] if "_" in reason else reason
                stats["filter_counts"][bucket] = stats["filter_counts"].get(bucket, 0) + 1

    pass_rate = stats["n_kept"] / max(stats["n_input"], 1)
    log.info("Quality filtering complete:")
    log.info("  Input:   %d", stats["n_input"])
    log.info("  Kept:    %d  (%.1f%%)", stats["n_kept"], pass_rate * 100)
    log.info("  Removed: %d", stats["n_removed"])
    log.info("  Kept  →  %s", output_path)
    log.info("  Removed → %s", removed_path)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quality-filter deduplicated invoice examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",   type=Path, default=Path("data/curated/deduped.jsonl"))
    p.add_argument("--output",  type=Path, default=Path("data/curated/quality_filtered.jsonl"))
    p.add_argument("--removed", type=Path, default=Path("data/curated/quality_removed.jsonl"))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    args = _parse_args()
    run_quality_filter(args.input, args.output, args.removed)
