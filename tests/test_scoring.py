"""
Tests for evaluation scoring logic.

Verifies that scoring.py correctly handles:
- Perfect matches (all fields correct)
- Partial matches (some fields wrong)
- Null field handling (both-null = match, one-null = miss)
- Numeric precision (rounding to 2 decimal places)
- Schema-invalid model outputs (graceful degradation)
- Line item matching (greedy bipartite matching)
- Aggregate score computation

Run with pytest.  No API calls — purely deterministic scoring logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.scoring import (
    AggregateScores,
    ExampleScore,
    aggregate_scores,
    score_example,
    _compare_numeric,
    _compare_string,
    _compare_field,
    _match_line_items,
)
from schema.extraction_schema import InvoiceExtraction, LineItem


# ---------------------------------------------------------------------------
# Ground truth fixture (reused across tests)
# ---------------------------------------------------------------------------

GROUND_TRUTH_SIMPLE: dict = {
    "vendor_name": "Acme Corp",
    "vendor_address": "123 Main St, Anytown, CA",
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-06-15",
    "due_date": "2024-07-15",
    "bill_to": "Bob Smith",
    "line_items": [
        {"description": "Widget A", "quantity": 10, "unit_price": 5.00, "total": 50.00},
        {"description": "Widget B", "quantity": 5,  "unit_price": 20.00, "total": 100.00},
    ],
    "subtotal": 150.00,
    "tax_amount": 12.00,
    "tax_rate": 0.08,
    "total_amount": 162.00,
    "currency": "USD",
    "payment_terms": "Net 30",
    "notes": None,
}


# ---------------------------------------------------------------------------
# Field comparison helper tests
# ---------------------------------------------------------------------------

class TestCompareHelpers:

    def test_numeric_exact_match(self):
        assert _compare_numeric(100.00, 100.00) is True

    def test_numeric_rounding(self):
        """Values that differ only at the 3rd decimal place should match."""
        assert _compare_numeric(100.001, 100.00) is True  # rounds to 100.00

    def test_numeric_significant_difference(self):
        assert _compare_numeric(100.01, 100.00) is False

    def test_numeric_both_none(self):
        assert _compare_numeric(None, None) is True

    def test_numeric_one_none(self):
        assert _compare_numeric(None, 100.00) is False
        assert _compare_numeric(100.00, None) is False

    def test_string_exact_match(self):
        assert _compare_string("Acme Corp", "Acme Corp") is True

    def test_string_case_insensitive(self):
        assert _compare_string("acme corp", "Acme Corp") is True

    def test_string_strip_whitespace(self):
        assert _compare_string("  Acme Corp  ", "Acme Corp") is True

    def test_string_both_none(self):
        assert _compare_string(None, None) is True

    def test_string_one_none(self):
        assert _compare_string(None, "Acme") is False

    def test_string_different_values(self):
        assert _compare_string("Acme Corp", "Beta Corp") is False

    def test_compare_field_numeric_field(self):
        assert _compare_field("total_amount", 162.00, 162.00) is True
        assert _compare_field("total_amount", 162.00, 163.00) is False

    def test_compare_field_string_field(self):
        assert _compare_field("vendor_name", "ACME CORP", "Acme Corp") is True

    def test_compare_field_date_field(self):
        assert _compare_field("invoice_date", "2024-06-15", "2024-06-15") is True
        assert _compare_field("invoice_date", "2024-06-14", "2024-06-15") is False


# ---------------------------------------------------------------------------
# score_example tests
# ---------------------------------------------------------------------------

class TestScoreExample:

    def test_perfect_match(self):
        """Model output identical to ground truth → all fields match."""
        raw_output = json.dumps(GROUND_TRUTH_SIMPLE)
        result = score_example("test-001", raw_output, GROUND_TRUTH_SIMPLE)

        assert result.schema_valid is True
        assert result.full_record_exact_match is True
        assert all(result.field_matches.values())
        assert result.parse_error == ""

    def test_schema_invalid_output(self):
        """Model outputs non-JSON → schema_valid=False, all fields missed."""
        raw_output = "Sorry, I could not extract that invoice."
        result = score_example("test-002", raw_output, GROUND_TRUTH_SIMPLE)

        assert result.schema_valid is False
        assert result.full_record_exact_match is False
        assert result.parse_error != ""

    def test_wrong_vendor_name(self):
        """Single field wrong → full_record_exact_match=False but other fields still correct."""
        gt = dict(GROUND_TRUTH_SIMPLE)
        pred = dict(GROUND_TRUTH_SIMPLE)
        pred["vendor_name"] = "Wrong Company"
        raw_output = json.dumps(pred)

        result = score_example("test-003", raw_output, gt)

        assert result.schema_valid is True
        assert result.field_matches["vendor_name"] is False
        assert result.field_matches["total_amount"] is True
        assert result.full_record_exact_match is False

    def test_wrong_total_amount(self):
        pred = dict(GROUND_TRUTH_SIMPLE)
        pred["total_amount"] = 200.00   # Wrong
        raw_output = json.dumps(pred)

        result = score_example("test-004", raw_output, GROUND_TRUTH_SIMPLE)

        assert result.field_matches["total_amount"] is False
        assert result.field_matches["vendor_name"] is True   # Other fields still correct

    def test_null_optional_field_matches_null_truth(self):
        """Both predicted and ground truth have notes=null → should match."""
        pred = dict(GROUND_TRUTH_SIMPLE)   # notes is already None
        raw_output = json.dumps(pred)

        result = score_example("test-005", raw_output, GROUND_TRUTH_SIMPLE)

        assert result.field_matches.get("notes") is True

    def test_predicted_none_truth_string(self):
        """Predicted null but ground truth has a value → miss."""
        gt = dict(GROUND_TRUTH_SIMPLE)
        gt["notes"] = "Please pay by check"
        pred = dict(GROUND_TRUTH_SIMPLE)
        pred["notes"] = None   # Predicted null when there's a value
        raw_output = json.dumps(pred)

        result = score_example("test-006", raw_output, gt)

        assert result.field_matches["notes"] is False

    def test_markdown_fence_stripped(self):
        """Model wraps output in ```json fences → should still parse."""
        raw_output = "```json\n" + json.dumps(GROUND_TRUTH_SIMPLE) + "\n```"
        result = score_example("test-007", raw_output, GROUND_TRUTH_SIMPLE)

        assert result.schema_valid is True
        assert result.full_record_exact_match is True

    def test_currency_case_insensitive_match(self):
        """Predicted 'usd' should match truth 'USD'."""
        pred = dict(GROUND_TRUTH_SIMPLE)
        pred["currency"] = "usd"
        raw_output = json.dumps(pred)

        result = score_example("test-008", raw_output, GROUND_TRUTH_SIMPLE)

        assert result.field_matches["currency"] is True

    def test_numeric_rounding_tolerance(self):
        """Float precision quirks shouldn't cause false misses."""
        pred = dict(GROUND_TRUTH_SIMPLE)
        pred["total_amount"] = 162.001   # Tiny float imprecision
        raw_output = json.dumps(pred)

        result = score_example("test-009", raw_output, GROUND_TRUTH_SIMPLE)

        assert result.field_matches["total_amount"] is True


# ---------------------------------------------------------------------------
# Line item matching tests
# ---------------------------------------------------------------------------

class TestLineItemMatching:

    def test_exact_line_item_match(self):
        gt = [
            LineItem(description="Widget A", quantity=10, unit_price=5.0, total=50.0),
        ]
        pred = [
            LineItem(description="Widget A", quantity=10, unit_price=5.0, total=50.0),
        ]
        scores = _match_line_items(pred, gt)
        assert scores["total"] == 1.0
        assert scores["description"] == 1.0

    def test_description_mismatch_prevents_total_credit(self):
        gt = [LineItem(description="Widget A", total=50.0)]
        pred = [LineItem(description="Completely different item", total=50.0)]
        scores = _match_line_items(pred, gt)
        # Description similarity too low → no match → all zeros
        assert scores["total"] == 0.0

    def test_extra_predicted_items_not_penalised(self):
        """Predicted has more items than GT — GT items should still match."""
        gt = [LineItem(description="Widget A", total=50.0)]
        pred = [
            LineItem(description="Widget A", total=50.0),
            LineItem(description="Extra Item", total=9.99),
        ]
        scores = _match_line_items(pred, gt)
        assert scores["total"] == 1.0   # The one GT item was matched

    def test_missing_predicted_item_penalised(self):
        """GT has more items than predicted — unmatched GT items count as misses."""
        gt = [
            LineItem(description="Widget A", total=50.0),
            LineItem(description="Widget B", total=100.0),
        ]
        pred = [
            LineItem(description="Widget A", total=50.0),
        ]
        scores = _match_line_items(pred, gt)
        # 1 out of 2 GT items matched
        assert scores["total"] == 0.5


# ---------------------------------------------------------------------------
# Aggregate scoring tests
# ---------------------------------------------------------------------------

class TestAggregateScores:

    def _make_score(
        self,
        schema_valid: bool,
        full_match: bool,
        field_matches: dict[str, bool] | None = None,
    ) -> ExampleScore:
        s = ExampleScore(example_id="test", schema_valid=schema_valid)
        s.full_record_exact_match = full_match
        s.field_matches = field_matches or {}
        s.line_item_field_accuracy = {}
        return s

    def test_empty_list_returns_zero_scores(self):
        agg = aggregate_scores([])
        assert agg.n_examples == 0

    def test_all_valid_all_match(self):
        scores = [
            self._make_score(True, True, {"vendor_name": True, "total_amount": True})
            for _ in range(10)
        ]
        agg = aggregate_scores(scores)
        assert agg.schema_validity_rate == 1.0
        assert agg.full_record_exact_match_rate == 1.0
        assert agg.field_accuracy_by_field["vendor_name"] == 1.0

    def test_mixed_results(self):
        scores = [
            self._make_score(True, True, {"vendor_name": True, "total_amount": True}),
            self._make_score(True, False, {"vendor_name": True, "total_amount": False}),
            self._make_score(False, False, {}),
        ]
        agg = aggregate_scores(scores)
        assert agg.n_examples == 3
        assert abs(agg.schema_validity_rate - 2 / 3) < 0.01
        assert abs(agg.full_record_exact_match_rate - 1 / 3) < 0.01
