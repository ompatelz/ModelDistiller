"""
Synthetic data generation prompts for Forge.

This module defines the 20 scenario configurations used to generate diverse
invoice / receipt training examples.  Each scenario is a distinct document
archetype — different industry, format, currency, difficulty, and edge cases —
so that the final training set contains genuinely varied examples rather than
1,000 near-identical invoices that happen to have different vendor names.

Design principles
-----------------
1.  Diversity first: 20 scenario archetypes × parameterised seed values
    (random vendor names, amounts, dates) = structural variety built in.
2.  Explicit difficulty levels: easy / medium / hard map directly to the
    real-world distribution we want the student model to handle.
3.  Edge cases are first-class: at least 6 of the 20 scenarios deliberately
    test something that will trip up a model trained on clean data only
    (missing fields, unusual date formats, discounts, partial payments, etc.).
4.  The prompts explicitly instruct Claude to produce output matching the
    fixed schema in schema/extraction_schema.py — the schema description is
    embedded in every prompt.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal


Difficulty = Literal["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Scenario configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """
    All parameters needed to build one generation prompt.

    Fields
    ------
    scenario_id    : Unique stable identifier (used in raw output for traceability).
    scenario_name  : Human-readable description.
    difficulty     : easy | medium | hard (used for stratified splitting).
    currency       : ISO 4217 code injected into the prompt.
    edge_cases     : List of specific things the model should include/omit;
                     injected as bullet points into the generation prompt.
    style_notes    : Additional formatting / style instructions for this scenario.
    """

    scenario_id: str
    scenario_name: str
    difficulty: Difficulty
    currency: str
    edge_cases: list[str] = field(default_factory=list)
    style_notes: str = ""


# ---------------------------------------------------------------------------
# 20 scenario configurations — genuinely different archetypes
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS: list[ScenarioConfig] = [

    # ------------------------------------------------------------------
    # EASY tier: all fields present, clean formatting, standard USD
    # ------------------------------------------------------------------
    ScenarioConfig(
        scenario_id="saas_easy",
        scenario_name="SaaS / software subscription invoice",
        difficulty="easy",
        currency="USD",
        style_notes=(
            "Formal letterhead layout. Three subscription tiers on separate "
            "line items (e.g. Pro, Enterprise, Add-on seats). All fields present. "
            "Monthly billing period. Net 30 payment terms."
        ),
        edge_cases=[],
    ),

    ScenarioConfig(
        scenario_id="retail_easy",
        scenario_name="Retail store purchase receipt",
        difficulty="easy",
        currency="USD",
        style_notes=(
            "Point-of-sale style receipt with store name at top, date and time, "
            "4–6 physical items with quantity, unit price, and line total. "
            "Sales tax shown. Cash or card tender and change given."
        ),
        edge_cases=[],
    ),

    ScenarioConfig(
        scenario_id="consulting_easy",
        scenario_name="IT consulting services invoice",
        difficulty="easy",
        currency="USD",
        style_notes=(
            "Professional services firm. Two or three service line items "
            "(e.g. 'System architecture review', 'Implementation support', "
            "'Documentation'). No quantity or unit price — service totals only. "
            "Invoice number, dates, Net 30 terms, all fields populated."
        ),
        edge_cases=[
            "line items have no quantity or unit_price — service billing only",
        ],
    ),

    ScenarioConfig(
        scenario_id="construction_easy",
        scenario_name="Building materials supplier invoice",
        difficulty="easy",
        currency="USD",
        style_notes=(
            "Lumber yard or hardware supplier. Eight or more line items with "
            "product codes, quantities (e.g. 24 units, 150 sq ft), unit prices, "
            "and line totals. Subtotal, 8% sales tax, and total clearly shown."
        ),
        edge_cases=[],
    ),

    ScenarioConfig(
        scenario_id="restaurant_easy",
        scenario_name="Restaurant dine-in receipt",
        difficulty="easy",
        currency="USD",
        style_notes=(
            "Casual restaurant receipt. Five to eight food/drink items. "
            "Subtotal, 8.5% tax, suggested tip amounts shown (but tip not part "
            "of the invoice total — total_amount is pre-tip). Table number and "
            "server name in notes."
        ),
        edge_cases=[
            "tip suggestions printed but NOT included in total_amount",
        ],
    ),

    ScenarioConfig(
        scenario_id="uk_vat_easy",
        scenario_name="UK company invoice with VAT",
        difficulty="easy",
        currency="GBP",
        style_notes=(
            "British B2B invoice from a UK limited company. Company number and "
            "VAT registration number in header. Five line items. VAT at 20% "
            "shown as both rate and amount. Payment by BACS. Net 30 terms."
        ),
        edge_cases=[
            "date format in source document uses DD/MM/YYYY — must normalize to YYYY-MM-DD",
            "VAT rate is 20% (tax_rate = 0.20)",
        ],
    ),

    # ------------------------------------------------------------------
    # MEDIUM tier: some fields missing, moderate complexity, varied currencies
    # ------------------------------------------------------------------
    ScenarioConfig(
        scenario_id="saas_annual_medium",
        scenario_name="SaaS annual plan with discount line item",
        difficulty="medium",
        currency="USD",
        style_notes=(
            "Annual SaaS renewal with an annual-commit discount applied as a "
            "negative line item (e.g. 'Annual discount — ($240.00)'). "
            "No vendor address in the document. Invoice number present. "
            "Due date absent — just says 'Due upon receipt'."
        ),
        edge_cases=[
            "discount appears as a negative-total line item",
            "vendor_address is null — not present in document",
            "due_date is null — document says 'Due upon receipt' only",
            "payment_terms extracted from body text, not a dedicated field",
        ],
    ),

    ScenarioConfig(
        scenario_id="consulting_milestone_medium",
        scenario_name="Multi-phase consulting invoice with expenses",
        difficulty="medium",
        currency="USD",
        style_notes=(
            "Management consulting firm billing for Phase 2 of a project. "
            "Three milestone line items plus a 'Reimbursable expenses' line "
            "(travel, lodging). No quantity or unit_price on any line. "
            "Invoice number present but no due date. Notes field contains "
            "project code and PO reference."
        ),
        edge_cases=[
            "all line items are service / milestone billing — no quantity or unit_price",
            "due_date is null",
            "notes contains structured metadata (project code, PO number)",
        ],
    ),

    ScenarioConfig(
        scenario_id="eur_german_medium",
        scenario_name="German company invoice in EUR with European date format",
        difficulty="medium",
        currency="EUR",
        style_notes=(
            "German SME invoicing a client. Document written in English but "
            "uses European conventions: date in DD.MM.YYYY format in the source "
            "document (must be normalized to YYYY-MM-DD in extraction), comma "
            "as decimal separator in amounts (must be converted to standard "
            "float), 19% MwSt (VAT). IBAN payment details in notes."
        ),
        edge_cases=[
            "source date format is DD.MM.YYYY — normalize to YYYY-MM-DD",
            "European decimal separator: amounts use comma (e.g. '1.234,50 €') — normalize to float",
            "tax labeled 'MwSt' at 19% (tax_rate = 0.19)",
        ],
    ),

    ScenarioConfig(
        scenario_id="canadian_gst_medium",
        scenario_name="Canadian invoice with GST and PST shown separately",
        difficulty="medium",
        currency="CAD",
        style_notes=(
            "Canadian service provider (Ontario). Invoice shows GST (5%) and "
            "Ontario HST harmonized into a single HST line (13% total). "
            "Business number (BN) in header. Some line items have quantity and "
            "unit price; others (consulting hours) just have a total. "
            "Vendor address present. Bill-to present."
        ),
        edge_cases=[
            "tax is labeled HST (Harmonized Sales Tax) at 13% — tax_rate = 0.13",
            "mix of product line items (with qty/unit_price) and service lines (without)",
        ],
    ),

    ScenarioConfig(
        scenario_id="utilities_medium",
        scenario_name="Telecommunications / internet service bill",
        difficulty="medium",
        currency="USD",
        style_notes=(
            "Monthly phone/internet bill from a telecom provider. "
            "Line items include: base plan, data overage charge, equipment "
            "rental, and regulatory fees. Account number in invoice_number field. "
            "Prior balance and payments shown in notes — total_amount is "
            "current charges only, not the running balance. "
            "No due date field — bill cycles are stated in body text."
        ),
        edge_cases=[
            "invoice_number is the account number",
            "total_amount = current period charges only (not cumulative balance)",
            "regulatory fees / surcharges appear as separate line items",
            "due_date absent from structured fields",
        ],
    ),

    ScenarioConfig(
        scenario_id="healthcare_medium",
        scenario_name="Medical office patient invoice with insurance adjustment",
        difficulty="medium",
        currency="USD",
        style_notes=(
            "Physician's office billing statement. Shows billed charges, "
            "insurance adjustment (negative line), insurance payment (negative "
            "line), and patient balance due. total_amount = patient balance due. "
            "Procedure codes (CPT codes) in line item descriptions. "
            "Date of service differs from invoice date. No tax."
        ),
        edge_cases=[
            "negative line items for insurance adjustment and insurance payment",
            "total_amount is patient balance (not gross billed amount)",
            "tax_amount and tax_rate are null (medical services not taxed)",
            "bill_to is the patient, vendor_name is the medical practice",
        ],
    ),

    ScenarioConfig(
        scenario_id="freight_medium",
        scenario_name="Freight / logistics invoice with surcharges",
        difficulty="medium",
        currency="USD",
        style_notes=(
            "Freight carrier invoice for a shipment. Line items: base freight "
            "charge, fuel surcharge (percentage-based), residential delivery "
            "fee, and declared value charge. Freight bill number as invoice_number. "
            "Origin and destination in notes. Net 15 terms."
        ),
        edge_cases=[
            "fuel surcharge is a percentage-based accessorial (show as dollar amount)",
            "notes contains shipment origin/destination addresses",
        ],
    ),

    # ------------------------------------------------------------------
    # HARD tier: messy format, missing critical optional fields,
    # unusual conventions, or structurally tricky scenarios
    # ------------------------------------------------------------------
    ScenarioConfig(
        scenario_id="legal_services_hard",
        scenario_name="Law firm invoice with retainer drawdown",
        difficulty="hard",
        currency="USD",
        style_notes=(
            "Law firm billing for litigation matter. Line items are time entries: "
            "attorney name, date of service, hours, hourly rate, and amount. "
            "Running retainer balance shown: 'Previous retainer balance: $X, "
            "Less fees this month: $Y, Retainer balance remaining: $Z.' "
            "total_amount = fees this invoice (not retainer balance). "
            "No invoice number — matter number used instead. "
            "No due date. No tax (legal services)."
        ),
        edge_cases=[
            "matter number used as invoice_number",
            "no due_date",
            "no tax",
            "time-entry line items: quantity = hours (decimal), unit_price = hourly rate",
            "retainer balance info appears in notes, NOT in totals",
            "total_amount = current fees billed, not retainer balance",
        ],
    ),

    ScenarioConfig(
        scenario_id="retail_discount_hard",
        scenario_name="Retail receipt with coupon, store credit, and partial return",
        difficulty="hard",
        currency="USD",
        style_notes=(
            "Complex retail transaction: multiple items, one coupon discount "
            "(negative line), one store credit applied (negative line), "
            "loyalty points earned in notes. Returned item from prior visit "
            "shown as a negative line item. Sales tax applied only to eligible "
            "items (subtotal before credits). total_amount = final amount charged."
        ),
        edge_cases=[
            "multiple negative line items (coupon, store credit, return)",
            "tax calculated on pre-discount subtotal — tax_amount may seem inconsistent with total",
            "notes contains loyalty points balance",
            "total_amount can be very small (after all credits)",
        ],
    ),

    ScenarioConfig(
        scenario_id="partial_payment_hard",
        scenario_name="Invoice with prior payment history and balance due",
        difficulty="hard",
        currency="USD",
        style_notes=(
            "A contractor invoice showing: original invoice total, prior "
            "payment received (30% deposit), and balance due. "
            "total_amount = BALANCE DUE (not original total). "
            "The original invoice amount appears in a 'Payment history' section. "
            "Unusual: invoice date is when originally issued, due date is for "
            "balance payment. No quantity on any line item."
        ),
        edge_cases=[
            "total_amount = balance due (NOT the original invoice total)",
            "original total and deposit appear in notes / payment history section",
            "model must not confuse original total with total_amount",
            "all line items are service billing (no quantity or unit_price)",
        ],
    ),

    ScenarioConfig(
        scenario_id="tax_exempt_hard",
        scenario_name="Non-profit organization purchase order / invoice (tax-exempt)",
        difficulty="hard",
        currency="USD",
        style_notes=(
            "Invoice issued to a non-profit. Vendor has noted tax-exempt status "
            "with a tax-exempt certificate number in the document. "
            "tax_amount = 0.0 and tax_rate = null (not zero — absent). "
            "Invoice formatted as a purchase order receipt, not a standard invoice. "
            "Fields appear in unusual order (totals at top, items below). "
            "Some field labels use non-standard terminology."
        ),
        edge_cases=[
            "tax_amount = 0.0 (explicitly zero, not null — vendor listed it as $0.00)",
            "tax_rate = null (not stated in document)",
            "tax-exempt certificate number appears in notes",
            "field layout is non-standard (totals appear before line items in document)",
            "payment_terms uses non-standard language ('Payable within 45 days of receipt')",
        ],
    ),

    ScenarioConfig(
        scenario_id="international_aud_hard",
        scenario_name="Australian invoice with GST, ABN, and informal layout",
        difficulty="hard",
        currency="AUD",
        style_notes=(
            "Australian sole trader invoice. Document is semi-informal — "
            "created in a text editor, not a proper invoicing tool. "
            "ABN (Australian Business Number) shown in header. "
            "GST at 10% included in all prices (prices are GST-inclusive, "
            "not added on top). Date written as 'Monday, 12 August 2024' — "
            "must normalize to YYYY-MM-DD. No invoice number (freelancer omitted it). "
            "Payment by bank transfer — BSB and account number in notes."
        ),
        edge_cases=[
            "date written as full day-of-week + month name — normalize to YYYY-MM-DD",
            "invoice_number is null (omitted by freelancer)",
            "GST is included in prices (not added on top) — model must extract stated amounts",
            "currency is AUD",
            "informal layout: no consistent alignment, hand-typed appearance",
        ],
    ),

    ScenarioConfig(
        scenario_id="messy_restaurant_hard",
        scenario_name="Food delivery / takeout receipt with fees and tips",
        difficulty="hard",
        currency="USD",
        style_notes=(
            "Third-party delivery app receipt (e.g. DoorDash-style). "
            "Line items include: food items, delivery fee, service fee, "
            "'small order fee', and tip. Tip IS included in total_amount "
            "(unlike the easy restaurant scenario). Subtotal = food only. "
            "Fees appear as separate line items. Tax applied to food only. "
            "Vendor name is the restaurant, not the delivery platform. "
            "No invoice number, no due date, no payment terms."
        ),
        edge_cases=[
            "tip IS included in total_amount (different from easy restaurant scenario)",
            "multiple fee line items (delivery, service, small-order)",
            "tax applies to food subtotal only, not fees",
            "vendor_name = restaurant name (not delivery platform)",
            "invoice_number, due_date, payment_terms all null",
        ],
    ),

    ScenarioConfig(
        scenario_id="missing_fields_hard",
        scenario_name="Minimal / stripped-down invoice with maximum null fields",
        difficulty="hard",
        currency="USD",
        style_notes=(
            "Extremely bare-bones invoice — perhaps a small independent contractor "
            "who just typed something up. Present: vendor_name, one or two line "
            "items with totals only (no qty/unit_price), total_amount, currency. "
            "Missing: vendor_address, invoice_number, invoice_date, due_date, "
            "bill_to, subtotal, tax_amount, tax_rate, payment_terms, notes. "
            "Document is 6–10 lines total. Make it plausibly real."
        ),
        edge_cases=[
            "maximum null fields: vendor_address, invoice_number, invoice_date, due_date, "
            "bill_to, subtotal, tax_amount, tax_rate, payment_terms, notes are ALL null",
            "tests the model's ability to output null rather than hallucinate missing fields",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_GENERATION_SYSTEM_PROMPT = """\
You are a synthetic data generator for an AI training pipeline. Your task is to
generate realistic invoice and receipt documents, together with their correctly
extracted structured data.

The structured output must conform exactly to the schema provided in each prompt.
Pay close attention to:
- Date normalization (always YYYY-MM-DD in the output, regardless of source format)
- tax_rate as a decimal fraction (0.08 for 8%, never 8.0)
- currency as a 3-letter ISO 4217 code
- Using null (not empty string) for absent optional fields
- line_items having at least one entry

You will return a JSON object with exactly two top-level keys:
1. "document_text" — the full plain-text document, formatted realistically with
   spaces and newlines but no HTML. Make it look like a real document.
2. "ground_truth_json" — the correctly extracted structured data conforming to
   the schema below.

Be creative with vendor names, addresses, amounts, and dates so each generated
document is genuinely different from a template. Use plausible but fictional
business names and realistic dollar amounts for the scenario type.
""".strip()


def build_generation_prompt(
    config: ScenarioConfig,
    seed: int | None = None,
) -> tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) pair for a generation API call.

    Parameters
    ----------
    config : ScenarioConfig
        The scenario to generate an example for.
    seed : int | None
        Optional seed for reproducible prompt variations (not used for RNG here,
        but can be included in the prompt to nudge variety).

    Returns
    -------
    (system_prompt, user_prompt) : tuple[str, str]
    """
    # Import here to avoid circular dependency at module load time
    from schema.extraction_schema import SCHEMA_DESCRIPTION

    edge_case_block = ""
    if config.edge_cases:
        edge_case_block = "\n\nIMPORTANT edge cases / constraints for this document:\n" + "\n".join(
            f"  • {ec}" for ec in config.edge_cases
        )

    user_prompt = f"""\
Generate a synthetic {config.scenario_name} document.

Difficulty level: {config.difficulty.upper()}
Currency: {config.currency}

Document style and content guidance:
{config.style_notes}{edge_case_block}

Use realistic but fictional business names, addresses, and amounts appropriate
for this scenario type.  Vary the amounts, dates, and specifics — do not use
round numbers or obvious placeholders like '$100.00' or 'Company Name'.

The schema you must conform to:
{SCHEMA_DESCRIPTION}

Return ONLY a valid JSON object with "document_text" and "ground_truth_json" keys.
Do not include any explanation, markdown fences, or text outside the JSON object.
""".strip()

    return _GENERATION_SYSTEM_PROMPT, user_prompt


def sample_scenario(
    rng: random.Random | None = None,
    difficulty_weights: dict[Difficulty, float] | None = None,
) -> ScenarioConfig:
    """
    Sample a scenario config, optionally weighted by difficulty.

    Default weights produce roughly: 30% easy, 40% medium, 30% hard —
    matching a realistic real-world distribution where most documents are
    routine but a meaningful fraction are messy or unusual.

    Parameters
    ----------
    rng : random.Random | None
        Pass an seeded RNG for reproducibility.
    difficulty_weights : dict | None
        Override the default sampling weights by difficulty.

    Returns
    -------
    ScenarioConfig
    """
    if rng is None:
        rng = random.Random()

    if difficulty_weights is None:
        difficulty_weights = {"easy": 0.30, "medium": 0.40, "hard": 0.30}

    # Assign weight to each config based on its difficulty
    weights = [difficulty_weights[cfg.difficulty] for cfg in SCENARIO_CONFIGS]
    return rng.choices(SCENARIO_CONFIGS, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)
    for _ in range(5):
        cfg = sample_scenario(rng)
        sys_p, usr_p = build_generation_prompt(cfg)
        print(f"[{cfg.difficulty:6s}] {cfg.scenario_id}: {cfg.scenario_name}")
    print(f"\nTotal scenarios: {len(SCENARIO_CONFIGS)}")
    print("Difficulty breakdown:")
    for d in ("easy", "medium", "hard"):
        count = sum(1 for c in SCENARIO_CONFIGS if c.difficulty == d)
        print(f"  {d}: {count}")
