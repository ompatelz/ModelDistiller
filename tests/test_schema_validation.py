"""
Tests for schema validation logic.

These tests verify that InvoiceExtraction correctly accepts valid inputs,
rejects invalid ones, and applies normalisation rules.  Run with pytest.

These tests use only stdlib + pydantic — no API calls, no model inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema.extraction_schema import InvoiceExtraction, LineItem


# ---------------------------------------------------------------------------
# Valid examples
# ---------------------------------------------------------------------------

class TestValidExamples:

    def test_minimal_valid_invoice(self):
        """Minimal valid invoice: only required fields."""
        inv = InvoiceExtraction(
            vendor_name="Acme Corp",
            line_items=[LineItem(description="Widget", total=99.99)],
            total_amount=99.99,
            currency="USD",
        )
        assert inv.vendor_name == "Acme Corp"
        assert inv.total_amount == 99.99
        assert inv.currency == "USD"
        assert len(inv.line_items) == 1

    def test_full_invoice_all_fields(self):
        """Invoice with every optional field populated."""
        inv = InvoiceExtraction(
            vendor_name="Globex Solutions Ltd.",
            vendor_address="42 Industrial Blvd, Springfield, IL 62701",
            invoice_number="INV-2024-0087",
            invoice_date="2024-03-15",
            due_date="2024-04-14",
            bill_to="Homer Simpson, 742 Evergreen Terrace",
            line_items=[
                LineItem(description="Enterprise License", quantity=5, unit_price=199.00, total=995.00),
                LineItem(description="Support Package", quantity=None, unit_price=None, total=250.00),
            ],
            subtotal=1245.00,
            tax_amount=99.60,
            tax_rate=0.08,
            total_amount=1344.60,
            currency="USD",
            payment_terms="Net 30",
            notes="PO#: 88-2024-HOMER",
        )
        assert inv.invoice_number == "INV-2024-0087"
        assert inv.tax_rate == 0.08
        assert len(inv.line_items) == 2

    def test_currency_uppercased_automatically(self):
        """currency should be normalised to uppercase."""
        inv = InvoiceExtraction(
            vendor_name="Test",
            line_items=[LineItem(description="Item", total=10.0)],
            total_amount=10.0,
            currency="eur",   # lowercase input
        )
        assert inv.currency == "EUR"

    def test_currency_stripped(self):
        inv = InvoiceExtraction(
            vendor_name="Test",
            line_items=[LineItem(description="Item", total=10.0)],
            total_amount=10.0,
            currency="  GBP  ",
        )
        assert inv.currency == "GBP"

    def test_optional_fields_default_none(self):
        inv = InvoiceExtraction(
            vendor_name="Test",
            line_items=[LineItem(description="Item", total=10.0)],
            total_amount=10.0,
            currency="USD",
        )
        assert inv.vendor_address is None
        assert inv.invoice_number is None
        assert inv.invoice_date is None
        assert inv.due_date is None
        assert inv.bill_to is None
        assert inv.subtotal is None
        assert inv.tax_amount is None
        assert inv.tax_rate is None
        assert inv.payment_terms is None
        assert inv.notes is None

    def test_zero_tax_amount_valid(self):
        """Explicitly $0 tax (e.g. tax-exempt) is valid."""
        inv = InvoiceExtraction(
            vendor_name="Non-Profit Org",
            line_items=[LineItem(description="Donation Processing", total=500.00)],
            total_amount=500.00,
            tax_amount=0.0,
            currency="USD",
        )
        assert inv.tax_amount == 0.0

    def test_eur_currency(self):
        inv = InvoiceExtraction(
            vendor_name="Deutsche GmbH",
            line_items=[LineItem(description="Beratungsleistung", total=2500.00)],
            total_amount=2500.00,
            currency="EUR",
        )
        assert inv.currency == "EUR"

    def test_line_item_without_qty_unit_price(self):
        """Service invoice: no quantity or unit_price."""
        item = LineItem(description="Consulting services — Phase 1", total=8000.00)
        assert item.quantity is None
        assert item.unit_price is None
        assert item.total == 8000.00


# ---------------------------------------------------------------------------
# Date field validation
# ---------------------------------------------------------------------------

class TestDateValidation:

    def test_valid_iso_date(self):
        inv = InvoiceExtraction(
            vendor_name="Test",
            line_items=[LineItem(description="Item", total=1.0)],
            total_amount=1.0,
            currency="USD",
            invoice_date="2024-12-31",
        )
        assert inv.invoice_date == "2024-12-31"

    def test_non_iso_date_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[LineItem(description="Item", total=1.0)],
                total_amount=1.0,
                currency="USD",
                invoice_date="December 31, 2024",   # Not ISO format
            )

    def test_slash_date_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[LineItem(description="Item", total=1.0)],
                total_amount=1.0,
                currency="USD",
                invoice_date="31/12/2024",   # DD/MM/YYYY not accepted
            )

    def test_us_slash_date_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[LineItem(description="Item", total=1.0)],
                total_amount=1.0,
                currency="USD",
                due_date="12/31/2024",   # MM/DD/YYYY not accepted
            )


# ---------------------------------------------------------------------------
# Tax rate validation
# ---------------------------------------------------------------------------

class TestTaxRateValidation:

    def test_valid_tax_rate_decimal(self):
        inv = InvoiceExtraction(
            vendor_name="Test",
            line_items=[LineItem(description="Item", total=100.0)],
            total_amount=108.0,
            tax_rate=0.08,
            currency="USD",
        )
        assert inv.tax_rate == 0.08

    def test_tax_rate_as_percentage_rejected(self):
        """Common mistake: passing 8.0 instead of 0.08."""
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[LineItem(description="Item", total=100.0)],
                total_amount=108.0,
                tax_rate=8.0,    # Should be 0.08
                currency="USD",
            )

    def test_tax_rate_zero_valid(self):
        inv = InvoiceExtraction(
            vendor_name="Test",
            line_items=[LineItem(description="Item", total=100.0)],
            total_amount=100.0,
            tax_rate=0.0,
            currency="USD",
        )
        assert inv.tax_rate == 0.0

    def test_tax_rate_none_valid(self):
        inv = InvoiceExtraction(
            vendor_name="Test",
            line_items=[LineItem(description="Item", total=100.0)],
            total_amount=100.0,
            tax_rate=None,
            currency="USD",
        )
        assert inv.tax_rate is None


# ---------------------------------------------------------------------------
# Required fields and constraint violations
# ---------------------------------------------------------------------------

class TestConstraintViolations:

    def test_negative_total_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[LineItem(description="Item", total=100.0)],
                total_amount=-10.0,
                currency="USD",
            )

    def test_empty_line_items_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[],
                total_amount=100.0,
                currency="USD",
            )

    def test_missing_vendor_name_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                line_items=[LineItem(description="Item", total=100.0)],
                total_amount=100.0,
                currency="USD",
            )

    def test_missing_currency_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[LineItem(description="Item", total=100.0)],
                total_amount=100.0,
            )

    def test_missing_total_amount_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[LineItem(description="Item", total=100.0)],
                currency="USD",
            )

    def test_currency_too_short_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[LineItem(description="Item", total=100.0)],
                total_amount=100.0,
                currency="US",   # Must be 3 chars
            )

    def test_currency_too_long_rejected(self):
        with pytest.raises(Exception):
            InvoiceExtraction(
                vendor_name="Test",
                line_items=[LineItem(description="Item", total=100.0)],
                total_amount=100.0,
                currency="USDC",   # Must be exactly 3 chars
            )


# ---------------------------------------------------------------------------
# model_validate (dict input — what the pipeline actually uses)
# ---------------------------------------------------------------------------

class TestModelValidate:

    def test_validate_from_dict(self):
        data = {
            "vendor_name": "Tech Solutions Inc.",
            "invoice_date": "2024-06-15",
            "line_items": [
                {"description": "Software License", "quantity": 10, "unit_price": 50.0, "total": 500.0},
            ],
            "total_amount": 540.0,
            "tax_amount": 40.0,
            "tax_rate": 0.08,
            "currency": "usd",   # Should be uppercased
        }
        inv = InvoiceExtraction.model_validate(data)
        assert inv.currency == "USD"
        assert inv.tax_rate == 0.08
        assert inv.vendor_address is None
