"""
Tests for FastAPI serving, interactive web endpoints, and multi-engine extraction.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from pathlib import Path

from schema.extraction_schema import InvoiceExtraction
from serving.api import app
from serving.inference_engine import SmartExtractor, engine_manager


client = TestClient(app)


def test_root_serves_web_ui():
    """Verify that GET / returns the interactive web HTML interface."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "FORGE" in resp.text
    assert "Structured Invoice Extraction" in resp.text


def test_app_route_serves_web_ui():
    """Verify that GET /app also serves the web interface."""
    resp = client.get("/app")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_health_endpoint():
    """Verify /health returns ok status and engine details."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "active_engine" in data
    assert "base_model" in data


def test_schema_endpoint():
    """Verify /schema returns the full InvoiceExtraction JSON schema."""
    resp = client.get("/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "description" in data
    assert "schema" in data
    assert data["schema"]["title"] == "InvoiceExtraction"


def test_samples_endpoint():
    """Verify /api/samples returns sample documents."""
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    data = resp.json()
    assert "samples" in data
    assert isinstance(data["samples"], list)
    if len(data["samples"]) > 0:
        assert "document_text" in data["samples"][0]


def test_metrics_endpoint():
    """Verify /api/metrics returns benchmark comparisons."""
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "benchmark" in data
    assert "dataset" in data
    assert data["benchmark"]["finetuned"]["overall_field_accuracy"] >= 0.85
    assert data["dataset"]["locked_eval"] == 91


def test_extract_endpoint_basic():
    """Verify POST /extract returns a valid InvoiceExtraction payload."""
    doc = (
        "Nexus Tech Solutions\n"
        "100 Cyber Way, Suite 500, Austin TX 78701\n"
        "INVOICE #INV-2025-772\n"
        "Date: 2025-02-10\n"
        "Due Date: 2025-03-10\n"
        "Bill To: Stellar Dynamics, 500 Tech Blvd\n\n"
        "Description              Qty    Rate     Total\n"
        "Cloud Migration            1    5000.00  5000.00\n"
        "Security Audit             1    2500.00  2500.00\n\n"
        "Subtotal: $7500.00\n"
        "Tax (8%): $600.00\n"
        "Total Due: $8100.00\n"
        "Payment Terms: Net 30\n"
    )
    resp = client.post("/extract", json={"document": doc})
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_valid"] is True
    assert data["latency_ms"] > 0
    ext = data["extraction"]

    # Verify Pydantic parses output cleanly
    parsed = InvoiceExtraction.model_validate(ext)
    assert parsed.vendor_name == "Nexus Tech Solutions"
    assert parsed.total_amount == 8100.0
    assert parsed.currency == "USD"
    assert len(parsed.line_items) >= 2


def test_extract_rejects_empty():
    """Verify POST /extract rejects blank documents with 400 or 422."""
    resp = client.post("/extract", json={"document": "      "})
    assert resp.status_code in (400, 422)


def test_smart_extractor_unit():
    """Unit test for SmartExtractor date normalization and currency parsing."""
    extractor = SmartExtractor()
    doc = (
        "Cafe Bistro Paris\n"
        "12 Rue de Rivoli, Paris\n"
        "Rechnung: RE-9941\n"
        "Datum: 15.04.2025\n"
        "Total: 45.50 EUR\n"
        "- 2x Espresso & Croissant - 45.50\n"
    )
    res = extractor.extract(doc)
    assert res["vendor_name"] == "Cafe Bistro Paris"
    assert res["invoice_date"] == "2025-04-15"
    assert res["total_amount"] == 45.50
    assert res["currency"] == "EUR"
