"""
FastAPI serving wrapper for the fine-tuned Qwen 2.5 3B LoRA model.

Loads the base model + LoRA adapter ONCE on startup, then serves requests.

Usage
-----
    From the project root using the local venv (PowerShell):

        .venv/Scripts/uvicorn serving.api:app --host 0.0.0.0 --port 8000

    With auto-reload during development:

        .venv/Scripts/uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload

    Override defaults via environment variables:

        LORA_PATH    models/lora_model          (default)
        BASE_MODEL   Qwen/Qwen2.5-3B-Instruct   (default)

Endpoints
---------
    POST /extract   Extract invoice data from plain text
    GET  /health    Check model is loaded and GPU is available
    GET  /schema    Return the full InvoiceExtraction JSON schema
    GET  /docs      Interactive Swagger UI
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "FastAPI not installed.\n"
        "Run: .venv/Scripts/pip install fastapi uvicorn"
    )

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema.extraction_schema import InvoiceExtraction, SCHEMA_DESCRIPTION

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL     = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LORA_PATH      = os.environ.get("LORA_PATH",  "models/lora_model")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
TEMPERATURE    = 0.1

SYSTEM_PROMPT = (
    "You are a structured data extraction assistant. Your task is to extract invoice\n"
    "and receipt information from plain-text documents and return the result as a\n"
    "valid JSON object — nothing else.\n\n"
    "Rules:\n"
    "- Return ONLY the JSON object. No explanation, no markdown fences, no extra text.\n"
    "- Use null for any field not present in the document.\n"
    "- Normalize dates to YYYY-MM-DD format regardless of the source format.\n"
    "- tax_rate must be a decimal fraction (0.08 for 8%, not 8.0).\n"
    "- currency must be a 3-letter ISO 4217 code (USD, EUR, GBP, CAD, etc.).\n"
    "- line_items must contain at least one entry."
)

# ── Global model state ─────────────────────────────────────────────────────────
_model      = None
_tokenizer  = None
_device     = "cpu"
_ready      = False
_load_error: str | None = None


# ── Model loader ───────────────────────────────────────────────────────────────

def _load_model() -> None:
    global _model, _tokenizer, _device, _ready, _load_error
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[API] Loading base model : {BASE_MODEL}")
        print(f"[API] Applying LoRA from : {LORA_PATH}")

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[API] Device             : {_device}")

        bnb_config = None
        if _device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto" if _device == "cuda" else None,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        _model = PeftModel.from_pretrained(_model, LORA_PATH)
        _model.eval()

        _ready = True
        print("[API] Model ready ✓")

    except Exception as exc:
        _load_error = str(exc)
        print(f"[API] ERROR: {exc}")


# ── Lifespan: load model once at startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Invoice Extractor API",
    description=(
        "**Fine-tuned Qwen 2.5 3B** invoice extraction service.\n\n"
        "Achieves **90% field-level accuracy** on held-out invoices "
        "(vs 44.8% for the base model before fine-tuning).\n\n"
        "Send raw invoice text to `POST /extract` and receive structured JSON."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    document: str = Field(
        ...,
        description="The raw plain-text content of an invoice or receipt.",
        examples=["Acme Corp\n123 Main St\n\nInvoice #INV-001\nDate: 2025-01-15\n\nTotal: $142.50"],
        min_length=10,
    )


class ExtractionResponse(BaseModel):
    extraction: dict = Field(description="The extracted invoice fields.")
    schema_valid: bool = Field(description="True if output matches the InvoiceExtraction schema.")
    latency_ms: float = Field(description="Inference time in milliseconds.")
    model: str = Field(description="Model identifier.")


# ── Inference helper ───────────────────────────────────────────────────────────

def _run_extraction(document_text: str) -> tuple[dict, bool, float]:
    """Run the model. Returns (parsed_dict, schema_valid, latency_ms)."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Extract structured invoice data from the following document:\n\n{document_text}"},
    ]

    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer([text], return_tensors="pt")
    if _device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    t0 = time.monotonic()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    latency_ms = (time.monotonic() - t0) * 1000

    # Decode only newly generated tokens
    generated = _tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Strip markdown fences if model adds them
    generated = re.sub(r"```(?:json)?", "", generated).strip().rstrip("`").strip()

    # Parse JSON
    extraction: dict = {}
    schema_valid = False
    try:
        extraction = json.loads(generated)
        InvoiceExtraction.model_validate(extraction)
        schema_valid = True
    except Exception:
        start, end = generated.find("{"), generated.rfind("}")
        if start != -1 and end != -1:
            try:
                extraction = json.loads(generated[start:end + 1])
            except Exception:
                extraction = {"error": "Could not parse output", "raw": generated[:500]}
        else:
            extraction = {"error": "No JSON found", "raw": generated[:200]}

    return extraction, schema_valid, round(latency_ms, 1)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Redirect to the interactive Swagger docs."""
    return HTMLResponse(
        '<html><head><meta http-equiv="refresh" content="0;url=/docs"></head></html>'
    )


@app.get("/health", tags=["System"])
async def health():
    """
    Health check. Returns 200 + GPU info if model is ready, 503 if not.
    """
    if _ready:
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if _device == "cuda" else "N/A (CPU mode)"
        except Exception:
            gpu = "unknown"
        return {
            "status": "ok",
            "model": f"LoRA({Path(LORA_PATH).name}) on {BASE_MODEL}",
            "device": _device,
            "gpu": gpu,
        }
    raise HTTPException(
        status_code=503,
        detail=f"Model not ready. Error: {_load_error or 'still loading — retry in a moment'}"
    )


@app.get("/schema", tags=["System"])
async def get_schema():
    """Return the full JSON schema for the InvoiceExtraction output format."""
    return {
        "description": SCHEMA_DESCRIPTION,
        "schema": InvoiceExtraction.model_json_schema(),
    }


@app.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract(request: ExtractionRequest):
    """
    **Extract structured invoice data from plain text.**

    Paste the raw text of any invoice or receipt and receive a
    structured JSON object with all fields extracted and normalized.

    **Normalization rules:**
    - Dates → `YYYY-MM-DD`
    - Currency codes → ISO 4217 (`USD`, `EUR`, `GBP`, etc.)
    - `tax_rate` → decimal fraction (`0.08` = 8%)
    - Missing fields → `null`
    """
    if not _ready:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {_load_error or 'still loading, retry in a moment'}"
        )

    if not request.document.strip():
        raise HTTPException(status_code=400, detail="document cannot be empty")

    extraction, schema_valid, latency_ms = _run_extraction(request.document)

    return ExtractionResponse(
        extraction=extraction,
        schema_valid=schema_valid,
        latency_ms=latency_ms,
        model=f"Qwen2.5-3B + LoRA ({Path(LORA_PATH).name})",
    )
