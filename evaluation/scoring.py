"""
Evaluation scoring for Forge.

This module defines ALL scoring logic used across all three model comparisons
(base Qwen2.5, fine-tuned Qwen2.5, Claude teacher).  The same functions are
called for every model — the only thing that varies between models is the
inference call.  This is what makes the comparison table legitimate.

Metrics computed
----------------
1.  schema_valid      : Can the model's output be parsed into InvoiceExtraction?
                        (0 or 1 per example)

2.  field_accuracy    : For each schema field, is the extracted value correct?
                        - Numeric fields (floats): exact match after rounding to
                          2 decimal places.
                        - String fields: case-insensitive exact match after
                          stripping leading/trailing whitespace.
                        - Date fields (invoice_date, due_date): exact match on
                          YYYY-MM-DD string (normalization is the model's job).
                        - currency: case-insensitive 3-letter match.
                        - Optional fields where both predicted and ground truth
                          are None: counts as a match.
                        - Optional field where one is None and the other isn't:
                          counts as a miss.

3.  line_items_accuracy : Fraction of ground-truth line items correctly matched
                          to a predicted line item.  Matching: greedy by
                          description similarity + total match.

4.  full_record_exact_match : All scored fields match simultaneously (0 or 1).

Aggregate scores (over the full eval set)
------------------------------------------
- field_accuracy_by_field : mean accuracy per field (useful for diagnosing
  which specific fields the fine-tuned model struggles with)
- overall_field_accuracy   : macro-average across all fields
- schema_validity_rate     : fraction of outputs that are parseable
- full_record_exact_match_rate : fraction where every field matched

Explicit non-LLM-as-judge note
---------------------------------
This task has objective, checkable ground truth.  We use deterministic field
comparison, not an LLM judge.  This is both cheaper and more defensible in an
interview — "field X either matches or it doesn't" is a harder claim to dispute
than "GPT-4o said it looked correct."
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from pydantic import ValidationError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema.extraction_schema import InvoiceExtraction, LineItem

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-field scoring configuration
# ---------------------------------------------------------------------------

# Numeric precision: compare floats rounded to N decimal places
NUMERIC_DECIMAL_PLACES: int = 2

# String fields: how similar two strings must be to count as a match
# (1.0 = exact match required after strip + lowercase)
STRING_EXACT_MATCH: bool = True

# Line item matching: minimum description similarity to attempt a total match
LINE_ITEM_DESC_THRESHOLD: float = 0.5

# Fields that are always scored (present in InvoiceExtraction)
SCORED_TOP_LEVEL_FIELDS: list[str] = [
    "vendor_name",
    "vendor_address",
    "invoice_number",
    "invoice_date",
    "due_date",
    "bill_to",
    "subtotal",
    "tax_amount",
    "tax_rate",
    "total_amount",
    "currency",
    "payment_terms",
    "notes",
]

# Fields scored within each line item
SCORED_LINE_ITEM_FIELDS: list[str] = [
    "description",
    "quantity",
    "unit_price",
    "total",
]


# ---------------------------------------------------------------------------
# Low-level comparison helpers
# ---------------------------------------------------------------------------

def _round(v: float | None) -> float | None:
    if v is None:
        return None
    return round(float(v), NUMERIC_DECIMAL_PLACES)


def _normalize_str(s: str | None) -> str | None:
    if s is None:
        return None
    return str(s).strip().lower()


def _compare_numeric(pred: Any, truth: Any) -> bool:
    """Compare two numeric (float) values after rounding."""
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False
    try:
        return _round(float(pred)) == _round(float(truth))
    except (TypeError, ValueError):
        return False


def _compare_string(pred: Any, truth: Any) -> bool:
    """Case-insensitive stripped string comparison."""
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False
    return _normalize_str(str(pred)) == _normalize_str(str(truth))


def _compare_field(field_name: str, pred_value: Any, truth_value: Any) -> bool:
    """
    Compare a single field using the appropriate comparison strategy.

    Numeric fields: float comparison after rounding.
    Everything else: string comparison.
    """
    numeric_fields = {
        "subtotal", "tax_amount", "tax_rate", "total_amount",
        "quantity", "unit_price", "total",
    }
    if field_name in numeric_fields:
        return _compare_numeric(pred_value, truth_value)
    return _compare_string(pred_value, truth_value)


# ---------------------------------------------------------------------------
# Line item matching
# ---------------------------------------------------------------------------

def _description_similarity(a: str | None, b: str | None) -> float:
    """Sequence-based similarity ratio for line item description matching."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    a_norm = _normalize_str(a) or ""
    b_norm = _normalize_str(b) or ""
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _match_line_items(
    predicted: list[LineItem],
    ground_truth: list[LineItem],
) -> dict[str, float]:
    """
    Greedy bipartite matching of predicted line items to ground-truth line items.

    Returns per-field accuracy scores across all matched line items.

    Strategy
    --------
    For each ground-truth item, find the best-matching predicted item by:
    1. Description similarity ≥ LINE_ITEM_DESC_THRESHOLD
    2. Among those, prefer the one with closest total.
    Each predicted item can only be matched once.

    Unmatched ground-truth items count as misses for all line item fields.
    """
    field_correct: dict[str, int] = {f: 0 for f in SCORED_LINE_ITEM_FIELDS}
    field_total: dict[str, int] = {f: 0 for f in SCORED_LINE_ITEM_FIELDS}

    pred_remaining = list(predicted)

    for gt_item in ground_truth:
        for fname in SCORED_LINE_ITEM_FIELDS:
            field_total[fname] += 1

        # Find best matching predicted item
        best_match_idx = None
        best_score = -1.0

        for idx, pred_item in enumerate(pred_remaining):
            desc_sim = _description_similarity(pred_item.description, gt_item.description)
            if desc_sim < LINE_ITEM_DESC_THRESHOLD:
                continue
            total_match = 1.0 if _compare_numeric(pred_item.total, gt_item.total) else 0.0
            score = desc_sim * 0.5 + total_match * 0.5
            if score > best_score:
                best_score = score
                best_match_idx = idx

        if best_match_idx is None:
            # No match found — all fields miss
            continue

        matched_pred = pred_remaining.pop(best_match_idx)

        # Score matched pair field by field
        for fname in SCORED_LINE_ITEM_FIELDS:
            pred_val = getattr(matched_pred, fname, None)
            gt_val = getattr(gt_item, fname, None)
            if _compare_field(fname, pred_val, gt_val):
                field_correct[fname] += 1

    # Aggregate: accuracy per line item field
    return {
        fname: field_correct[fname] / max(field_total[fname], 1)
        for fname in SCORED_LINE_ITEM_FIELDS
    }


# ---------------------------------------------------------------------------
# Per-example scoring
# ---------------------------------------------------------------------------

@dataclass
class ExampleScore:
    """Scoring result for a single eval example."""

    example_id: str
    schema_valid: bool                          # Could the output be parsed?

    # Top-level field matches (True/False per field)
    field_matches: dict[str, bool] = field(default_factory=dict)

    # Line item field accuracies (float 0–1 per field)
    line_item_field_accuracy: dict[str, float] = field(default_factory=dict)

    full_record_exact_match: bool = False       # All scored fields match?

    # Raw model output (for debugging)
    raw_output: str = ""
    parse_error: str = ""

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "schema_valid": self.schema_valid,
            "field_matches": self.field_matches,
            "line_item_field_accuracy": self.line_item_field_accuracy,
            "full_record_exact_match": self.full_record_exact_match,
            "parse_error": self.parse_error,
        }


def score_example(
    example_id: str,
    raw_model_output: str,
    ground_truth: dict,
    *,
    include_raw_output: bool = False,
) -> ExampleScore:
    """
    Score one model output against its ground truth.

    Parameters
    ----------
    example_id         : Identifier for logging.
    raw_model_output   : The model's raw text output (will be JSON-parsed).
    ground_truth       : The ground_truth_json dict from eval_locked.jsonl.
    include_raw_output : Whether to store the raw output in the result (large).

    Returns
    -------
    ExampleScore with all metrics populated.
    """
    result = ExampleScore(
        example_id=example_id,
        schema_valid=False,
        raw_output=raw_model_output if include_raw_output else "",
    )

    # ------------------------------------------------------------------
    # 1. Parse model output
    # ------------------------------------------------------------------
    raw_output = raw_model_output.strip()

    # Strip markdown fences if present
    if raw_output.startswith("```"):
        lines = raw_output.split("\n")
        inner = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        raw_output = inner.strip()

    try:
        pred_dict = json.loads(raw_output)
        if not isinstance(pred_dict, dict):
            raise ValueError("Model output is not a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        # Try to extract the first {...} block
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1:
            try:
                pred_dict = json.loads(raw_output[start : end + 1])
            except json.JSONDecodeError:
                result.parse_error = f"JSON parse failed: {exc}"
                return result
        else:
            result.parse_error = f"No JSON object found: {exc}"
            return result

    # ------------------------------------------------------------------
    # 2. Validate against schema
    # ------------------------------------------------------------------
    try:
        predicted = InvoiceExtraction.model_validate(pred_dict)
        result.schema_valid = True
    except ValidationError as exc:
        result.parse_error = f"Schema validation failed: {exc.errors()[:2]}"
        # Still attempt field-level scoring with raw dict values for partial credit
        # (so we can see *which* fields the model got right even if schema invalid)
        predicted = None

    # ------------------------------------------------------------------
    # 3. Parse ground truth
    # ------------------------------------------------------------------
    try:
        truth = InvoiceExtraction.model_validate(ground_truth)
    except ValidationError as exc:
        log.error("[%s] Ground truth fails schema validation: %s — this is a curation bug!", example_id, exc)
        result.parse_error = f"GROUND TRUTH ERROR: {exc}"
        return result

    # ------------------------------------------------------------------
    # 4. Field-level scoring
    # ------------------------------------------------------------------
    if predicted is not None:
        for fname in SCORED_TOP_LEVEL_FIELDS:
            pred_val = getattr(predicted, fname, None)
            truth_val = getattr(truth, fname, None)
            result.field_matches[fname] = _compare_field(fname, pred_val, truth_val)

        # Line items
        result.line_item_field_accuracy = _match_line_items(
            predicted.line_items, truth.line_items
        )

        # Full record exact match: all top-level fields + all line item fields
        top_level_ok = all(result.field_matches.values())
        line_items_ok = all(
            v >= 1.0 for v in result.line_item_field_accuracy.values()
        )
        result.full_record_exact_match = top_level_ok and line_items_ok

    else:
        # Schema invalid — all fields count as missed
        for fname in SCORED_TOP_LEVEL_FIELDS:
            result.field_matches[fname] = False
        result.line_item_field_accuracy = {f: 0.0 for f in SCORED_LINE_ITEM_FIELDS}
        result.full_record_exact_match = False

    return result


# ---------------------------------------------------------------------------
# Aggregate scoring over the full eval set
# ---------------------------------------------------------------------------

@dataclass
class AggregateScores:
    """Summary statistics over all eval examples."""

    n_examples: int = 0
    schema_validity_rate: float = 0.0
    full_record_exact_match_rate: float = 0.0
    field_accuracy_by_field: dict[str, float] = field(default_factory=dict)
    line_item_field_accuracy_by_field: dict[str, float] = field(default_factory=dict)
    overall_field_accuracy: float = 0.0    # macro-average across all top-level fields
    parse_failures: int = 0

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "schema_validity_rate": self.schema_validity_rate,
            "full_record_exact_match_rate": self.full_record_exact_match_rate,
            "field_accuracy_by_field": self.field_accuracy_by_field,
            "line_item_field_accuracy_by_field": self.line_item_field_accuracy_by_field,
            "overall_field_accuracy": self.overall_field_accuracy,
            "parse_failures": self.parse_failures,
        }


def aggregate_scores(example_scores: list[ExampleScore]) -> AggregateScores:
    """Compute dataset-level aggregate metrics from per-example scores."""
    n = len(example_scores)
    if n == 0:
        return AggregateScores()

    agg = AggregateScores(n_examples=n)

    agg.schema_validity_rate = sum(s.schema_valid for s in example_scores) / n
    agg.full_record_exact_match_rate = sum(
        s.full_record_exact_match for s in example_scores
    ) / n
    agg.parse_failures = sum(1 for s in example_scores if s.parse_error)

    # Per-field accuracy
    for fname in SCORED_TOP_LEVEL_FIELDS:
        matches = [
            s.field_matches.get(fname, False)
            for s in example_scores
        ]
        agg.field_accuracy_by_field[fname] = sum(matches) / max(len(matches), 1)

    # Line item field accuracy
    for fname in SCORED_LINE_ITEM_FIELDS:
        accuracies = [
            s.line_item_field_accuracy.get(fname, 0.0)
            for s in example_scores
        ]
        agg.line_item_field_accuracy_by_field[fname] = (
            sum(accuracies) / max(len(accuracies), 1)
        )

    # Overall field accuracy: macro-average across top-level fields
    agg.overall_field_accuracy = (
        sum(agg.field_accuracy_by_field.values())
        / max(len(agg.field_accuracy_by_field), 1)
    )

    return agg
