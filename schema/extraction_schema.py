"""
Extraction schema for invoice/receipt documents.

This module is the single source of truth for the Forge distillation pipeline.
Every other module that touches structured invoice data — data generation prompts,
curation validation, training data formatting, evaluation scoring — imports from
here. Do NOT redefine these fields anywhere else.

If the schema needs to change mid-project that is a flag-to-user moment: it
likely means regenerating training data, and the change must be documented.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class LineItem(BaseModel):
    """A single line item on an invoice or receipt."""

    description: str = Field(
        ...,
        description="Description of the product or service.",
        min_length=1,
    )
    quantity: Optional[float] = Field(
        None,
        description=(
            "Quantity ordered or delivered. None when not stated "
            "(common on service invoices, e.g. 'Consulting — $5,000')."
        ),
    )
    unit_price: Optional[float] = Field(
        None,
        description=(
            "Price per unit in the invoice currency. "
            "None when not explicitly broken out in the document."
        ),
    )
    total: float = Field(
        ...,
        description=(
            "Total amount for this line item in the invoice currency. "
            "Always required — the only checkable number if "
            "quantity / unit_price are absent."
        ),
    )

    @field_validator("total")
    @classmethod
    def total_must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"LineItem.total must be a finite number, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Date format helper (shared by field validators below)
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_iso_date(v: Optional[str], field_name: str) -> Optional[str]:
    """Return v unchanged if it matches YYYY-MM-DD, raise otherwise."""
    if v is None:
        return v
    v = str(v).strip()
    if not _DATE_RE.match(v):
        raise ValueError(
            f"{field_name} must be in YYYY-MM-DD format, got {v!r}. "
            "Normalize source formats (e.g. 'Jan 15, 2024' → '2024-01-15') "
            "before assigning to this field."
        )
    return v


# ---------------------------------------------------------------------------
# Main extraction model
# ---------------------------------------------------------------------------

class InvoiceExtraction(BaseModel):
    """
    Structured extraction output for invoice / receipt documents.

    Schema design decisions
    -----------------------
    *  ``total_amount`` is the only required non-list field.  Every real
       invoice has a bottom-line amount; making it required makes failing on
       this field unambiguous in eval scoring.

    *  Dates are stored as ISO 8601 strings (YYYY-MM-DD), not datetime objects.
       This avoids timezone / parsing ambiguity during scoring and keeps the
       model output directly comparable without silent normalization side-effects.

    *  ``tax_rate`` and ``tax_amount`` are separate optional fields because
       real invoices are inconsistent: some show the amount only, some the rate
       only, some both.  Having both fields lets the eval catch whether the
       model infers the missing one — a deliberate hard case.

    *  ``quantity`` and ``unit_price`` on line items are Optional: service
       invoices (consulting, legal, SaaS) frequently omit them.  Forcing them
       would make most real-world invoices schema-invalid.

    *  ``currency`` uses ISO 4217 three-letter codes (USD, EUR, GBP, CAD …).
    """

    vendor_name: str = Field(
        ...,
        description="Name of the vendor or seller.",
    )
    vendor_address: Optional[str] = Field(
        None,
        description="Vendor's full address as it appears in the document, if present.",
    )
    invoice_number: Optional[str] = Field(
        None,
        description=(
            "Invoice or receipt identifier, e.g. 'INV-2024-0042', 'Receipt #8823'. "
            "None if the document does not include one."
        ),
    )
    invoice_date: Optional[str] = Field(
        None,
        description=(
            "Date of invoice in YYYY-MM-DD format. "
            "Normalize from any source format "
            "(e.g. 'January 15, 2024' → '2024-01-15', '15/01/2024' → '2024-01-15')."
        ),
    )
    due_date: Optional[str] = Field(
        None,
        description="Payment due date in YYYY-MM-DD format. None if no due date is stated.",
    )
    bill_to: Optional[str] = Field(
        None,
        description="Customer name and / or address being billed, if present.",
    )
    line_items: list[LineItem] = Field(
        ...,
        description=(
            "List of individual products or services. "
            "Must contain at least one item."
        ),
        min_length=1,
    )
    subtotal: Optional[float] = Field(
        None,
        description=(
            "Sum of line item totals before tax. "
            "None if not explicitly stated in the document."
        ),
    )
    tax_amount: Optional[float] = Field(
        None,
        description="Absolute tax amount in the document's currency.",
    )
    tax_rate: Optional[float] = Field(
        None,
        description=(
            "Tax rate as a decimal fraction, e.g. 0.08 for 8%, 0.10 for 10%. "
            "None if not stated in the document."
        ),
    )
    total_amount: float = Field(
        ...,
        description=(
            "Final total — the bottom-line amount due or paid. "
            "Always required.  Must be ≥ 0."
        ),
    )
    currency: str = Field(
        ...,
        description="ISO 4217 currency code, e.g. 'USD', 'EUR', 'GBP', 'CAD'.",
        min_length=3,
        max_length=3,
    )
    payment_terms: Optional[str] = Field(
        None,
        description="e.g. 'Net 30', 'Due on receipt', 'Net 60', '2/10 Net 30'.",
    )
    notes: Optional[str] = Field(
        None,
        description="Any additional notes, memo text, or comments on the invoice.",
    )

    # ------------------------------------------------------------------
    # Field-level validators
    # ------------------------------------------------------------------

    @field_validator("invoice_date", mode="before")
    @classmethod
    def validate_invoice_date(cls, v: Optional[str]) -> Optional[str]:
        return _validate_iso_date(v, "invoice_date")

    @field_validator("due_date", mode="before")
    @classmethod
    def validate_due_date(cls, v: Optional[str]) -> Optional[str]:
        return _validate_iso_date(v, "due_date")

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency_code(cls, v: str) -> str:
        """Strip whitespace and uppercase the ISO 4217 code."""
        return str(v).strip().upper()

    @field_validator("tax_rate", mode="before")
    @classmethod
    def validate_tax_rate_range(cls, v: Optional[float]) -> Optional[float]:
        """Reject rates outside [0, 1] — catch the common bug of passing 8.0 instead of 0.08."""
        if v is None:
            return v
        v = float(v)
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"tax_rate must be a decimal between 0 and 1 (got {v}). "
                "Use 0.08 for 8%, not 8.0."
            )
        return v

    @field_validator("total_amount", mode="before")
    @classmethod
    def validate_total_non_negative(cls, v: float) -> float:
        v = float(v)
        if v < 0:
            raise ValueError(f"total_amount must be ≥ 0, got {v}")
        return v

    # ------------------------------------------------------------------
    # Model-level validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_line_items_have_totals(self) -> "InvoiceExtraction":
        """Every line item must have a finite, non-negative total."""
        for i, item in enumerate(self.line_items):
            if item.total < 0:
                raise ValueError(
                    f"line_items[{i}].total must be ≥ 0, got {item.total}"
                )
        return self


# ---------------------------------------------------------------------------
# Schema description string — injected into generation/eval prompts
# ---------------------------------------------------------------------------

SCHEMA_DESCRIPTION = """
Return a JSON object with the following fields:

{
  "vendor_name":     string  (required),
  "vendor_address":  string | null,
  "invoice_number":  string | null,
  "invoice_date":    string | null   (YYYY-MM-DD format),
  "due_date":        string | null   (YYYY-MM-DD format),
  "bill_to":         string | null,
  "line_items": [                    (required, at least 1)
    {
      "description": string  (required),
      "quantity":    number | null,
      "unit_price":  number | null,
      "total":       number  (required)
    }
  ],
  "subtotal":        number | null,
  "tax_amount":      number | null,
  "tax_rate":        number | null   (decimal: 0.08 = 8%),
  "total_amount":    number  (required, ≥ 0),
  "currency":        string  (ISO 4217: "USD", "EUR", "GBP", "CAD" …),
  "payment_terms":   string | null,
  "notes":           string | null
}

Important rules:
- Dates MUST be in YYYY-MM-DD format regardless of the source format.
- tax_rate MUST be a decimal fraction (0.08 for 8%), never a percentage.
- currency MUST be a 3-letter ISO 4217 code.
- If a field is not present in the document, use null (not an empty string).
- line_items must have at least one entry.
""".strip()
