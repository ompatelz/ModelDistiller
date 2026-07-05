# Forge — Invoice Extraction via Model Distillation
## Product Requirements Document (Post-Completion)

> **Status:** ✅ Complete  
> **Completed:** June 2026  
> **Stack:** Python · Qwen 2.5 3B · QLoRA · Unsloth · FastAPI · OpenRouter

---

## What Is This Project?

Forge is an **end-to-end model distillation pipeline** that proves a single, high-value idea:

> *You can take a narrow AI task currently being solved by an expensive frontier model, generate synthetic training data from it, fine-tune a tiny open-weight model on that data, and get comparable accuracy at a fraction of the cost.*

**The task:** Structured JSON extraction from raw invoice and receipt text.  
**The teacher:** DeepSeek (via OpenRouter API) — generates training data.  
**The student:** Qwen 2.5 3B Instruct — the fine-tuned model.  
**The proof:** 90.0% field-level accuracy on a locked eval set vs. 44.8% for the base model.

---

## The Full Pipeline — How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FORGE PIPELINE                              │
│                                                                     │
│  Phase 1        Phase 2         Phase 3        Phase 4    Phase 5  │
│  Schema    ──▶  Synthetic   ──▶ Curation   ──▶ Train  ──▶  Eval   │
│  Design         Data Gen        Pipeline        (LoRA)   & Serve   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Schema Design

The extraction target is defined as a strict Pydantic model in `schema/extraction_schema.py`.
Every invoice field the model must extract is declared with its type, constraints, and normalization rules:

| Field | Type | Normalization Rule |
|---|---|---|
| `vendor_name` | str | As-is |
| `vendor_address` | str | As-is |
| `invoice_number` | str or null | As-is |
| `invoice_date` | str or null | **YYYY-MM-DD** |
| `due_date` | str or null | **YYYY-MM-DD** |
| `bill_to` | str or null | As-is |
| `line_items` | List[LineItem] | At least 1 required |
| `subtotal` | float or null | Numeric |
| `tax_amount` | float or null | Numeric |
| `tax_rate` | float or null | **Decimal fraction** (0.08 = 8%) |
| `total_amount` | float or null | Numeric |
| `currency` | str or null | **ISO 4217** (USD, EUR...) |
| `payment_terms` | str or null | As-is |
| `notes` | str or null | As-is |

The schema is the **single source of truth** for the entire pipeline — it drives data generation prompts, validation, evaluation scoring, and the API response model.

---

## Phase 2 — Synthetic Data Generation

**Script:** `data_generation/generate_synthetic_data.py`  
**Model used:** DeepSeek via OpenRouter API  
**Output:** `data/synthetic_data.jsonl`

### How it works

The script generates diverse, realistic invoice scenarios by asking DeepSeek to:
1. Create a realistic invoice/receipt *document* (the raw text)
2. Provide the corresponding correct JSON extraction (the label)

Scenarios are deliberately varied across:
- **Document types:** restaurant receipts, B2B invoices, freelance invoices, SaaS subscriptions, utility bills, hotel folios, medical bills
- **Difficulty levels:** easy (clean, well-formatted) / medium (unusual formats, multiple taxes) / hard (messy, mixed currencies, ambiguous fields)
- **Vendor styles:** US, EU, UK, Canadian — different date formats, currency symbols, tax terminology (VAT, GST, HST)

### Resumability
The script writes in **append mode** — if your laptop dies mid-run, it picks up exactly where it left off. No data is lost and no duplicates are created.

### Results
- **926 valid examples** generated
- ~93% success rate (some examples failed schema validation at generation time)
- Total API cost: **~$0.40** using DeepSeek pricing on OpenRouter

---

## Phase 3 — Data Curation Pipeline

Four sequential scripts in `data_curation/`, each refining the dataset:

```
raw_synthetic_data.jsonl  (926 examples)
        │
        ▼
  validate_schema.py      Remove malformed JSON / missing required fields
        │
        ▼
  deduplicate.py          Remove near-duplicate examples (same scenario_id)
        │
        ▼
  quality_filter.py       Remove low-quality: too short, trivial, edge cases
        │
        ▼
  split_dataset.py        Lock eval set BEFORE training (91 examples held out)
        │
        ▼
  format_dataset.py       Convert to ShareGPT ChatML format for SFTTrainer
        │
   ┌────┴────────┬──────────────┐
   ▼             ▼              ▼
train_formatted  val_formatted  eval_locked
  (600 ex.)       (67 ex.)      (91 ex.)
```

### The Eval-Lock Discipline
The 91-example eval set was locked **before any training began**. It was never used to make decisions during training. This is what makes the final accuracy number credible — it is not a number that was optimized against.

---

## Phase 4 — Fine-Tuning (QLoRA on Google Colab)

**Notebook:** `training/Unsloth_Finetune_Qwen2_5_3B.ipynb`  
**Platform:** Google Colab T4 GPU (free tier, 16GB VRAM)  
**Library:** Unsloth (2x faster training, 70% less memory than standard HuggingFace)

### Why QLoRA?
Full fine-tuning a 3B parameter model requires 24GB+ VRAM. QLoRA (4-bit quantization + Low-Rank Adapters) reduces this to ~6GB, making it feasible on a free Colab T4.

Instead of modifying all 3 billion parameters, LoRA trains only ~30 million parameters (the adapter matrices). The base model stays frozen. The resulting adapter file is only **~114MB** vs the full model's 6GB+.

### Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` | Best-in-class 3B with strong instruction following |
| Quantization | 4-bit NF4 | Fits in 8GB VRAM, minimal accuracy loss |
| LoRA rank (r) | 16 | Standard tradeoff between capacity and size |
| LoRA alpha | 16 | Scales gradient updates |
| Target modules | q/k/v/o/gate/up/down projections | All attention + MLP layers |
| Effective batch size | 8 (2 per device x 4 grad accum) | Stable within VRAM budget |
| Epochs | 3 | Sufficient for SFT on clean synthetic data |
| Learning rate | 2e-4 | Standard LoRA LR |
| Optimizer | AdamW 8-bit | Memory-efficient |
| Scheduler | Linear with 10 warmup steps | Smooth ramp-up |

### Training Loss Curve

| Step | Train Loss | Val Loss |
|---|---|---|
| 25 | 0.526 | 0.491 |
| 75 | 0.372 | 0.553 |
| 125 | 0.293 | 0.321 |
| 175 | 0.269 | 0.309 |
| **225** | **0.263** | **0.303** |

Loss dropped from **0.526 → 0.263** over 225 steps (~2 hours on T4).

### Output
The LoRA adapter is saved at `models/lora_model/` (~114MB total):
- `adapter_config.json` — LoRA architecture config
- `adapter_model.safetensors` — the actual trained weights
- `tokenizer.json` + `tokenizer_config.json` + `chat_template.jinja`

---

## Phase 5 — Evaluation Results

**Eval set:** 91 locked examples (never seen during training)  
**Metric:** Field-level accuracy — fraction of fields correctly extracted, averaged across all examples and all 14 fields.

```
═══════════════════════════════════════════════════════════════════════
  METRIC                           BASE       FINE-TUNED      DELTA
═══════════════════════════════════════════════════════════════════════
  Overall Accuracy               44.8%          90.0%       +45.2pp
  JSON Parse Rate                98.9%          98.9%        +0.0pp
───────────────────────────────────────────────────────────────────────
  BY DIFFICULTY
  Easy                           43.4%          89.5%       +46.0pp
  Medium                         43.7%          93.9%       +50.2pp
  Hard                           47.4%          85.8%       +38.4pp
───────────────────────────────────────────────────────────────────────
  BY FIELD
  vendor_name                     9.9%          92.3%       +82.4pp
  vendor_address                 16.5%          71.4%       +54.9pp
  invoice_number                 84.6%          95.6%       +11.0pp
  invoice_date                   78.0%          98.9%       +20.9pp
  due_date                       93.4%          96.7%        +3.3pp
  bill_to                        22.0%          81.3%       +59.3pp
  subtotal                       26.4%          91.2%       +64.8pp
  tax_amount                     49.5%          94.5%       +45.1pp
  tax_rate                       63.7%          96.7%       +33.0pp
  total_amount                   25.3%          97.8%       +72.5pp
  currency                       36.3%          98.9%       +62.6pp
  payment_terms                  79.1%          91.2%       +12.1pp
  notes                          39.6%          64.8%       +25.3pp
  line_items                      2.5%          88.5%       +86.0pp
═══════════════════════════════════════════════════════════════════════
```

### Key Observations

- **`line_items` +86pp** — The single biggest win. The base model had essentially no ability to produce correct structured nested arrays. Fine-tuning completely solved this.
- **`vendor_name` +82pp** — The base model was hallucinating vendor names. Fine-tuning taught it to extract faithfully.
- **`total_amount` +72.5pp** — The base model frequently computed totals itself instead of extracting them. Fine-tuning corrected this.
- **`notes` weakest** (64.8%) — Notes are inherently ambiguous; what counts as a "note" vs. incidental text is subjective even in ground truth labels.
- **JSON parse rate unchanged** — Both models already output valid JSON at 98.9%. The base model's failure was in *what* it extracted, not whether it could output valid JSON.

---

## Phase 6 — Serving

### Option A — CLI / Direct Script
```powershell
# Interactive mode (paste text, Enter twice to run)
.\.venv\Scripts\python.exe infer.py --pretty

# File mode
.\.venv\Scripts\python.exe infer.py --input invoice.txt --pretty
```

### Option B — FastAPI REST Endpoint
```powershell
.\.venv\Scripts\python.exe -m uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

| Method | Path | Description |
|---|---|---|
| `POST` | `/extract` | Extract invoice data from plain text |
| `GET` | `/health` | Model status + GPU info |
| `GET` | `/schema` | Full InvoiceExtraction JSON schema |
| `GET` | `/docs` | Interactive Swagger UI |

**Sample request/response:**
```bash
# Request
POST /extract
{"document": "Acme Corp\nInvoice #001\nDate: 2025-01-15\nTotal: $142.50"}

# Response
{
  "extraction": {
    "vendor_name": "Acme Corp",
    "invoice_number": "001",
    "invoice_date": "2025-01-15",
    "total_amount": 142.50,
    "currency": "USD",
    ...
  },
  "schema_valid": true,
  "latency_ms": 1840,
  "model": "Qwen2.5-3B + LoRA (lora_model)"
}
```

---

## Project File Structure

```
Distillation/
├── schema/
│   └── extraction_schema.py         Single source of truth (Pydantic)
├── data_generation/
│   └── generate_synthetic_data.py   DeepSeek -> JSONL (resumable)
├── data_curation/
│   ├── validate_schema.py
│   ├── deduplicate.py
│   ├── quality_filter.py
│   ├── split_dataset.py             Locks eval set
│   └── format_dataset.py            -> ChatML format
├── data/
│   ├── synthetic_data.jsonl         Raw (926 examples)
│   ├── train_formatted.jsonl        Training (600)
│   ├── val_formatted.jsonl          Validation (67)
│   └── eval_locked.jsonl            Locked eval (91)
├── training/
│   ├── Unsloth_Finetune_Qwen2_5_3B.ipynb
│   └── Eval_Qwen2_5_3B.ipynb
├── models/
│   └── lora_model/                  Trained LoRA adapter (~114MB)
├── serving/
│   └── api.py                       FastAPI server
├── infer.py                         Local CLI inference
└── .venv/                           Python 3.12 + PyTorch CUDA
```

---

## Cost Analysis

| | Teacher (DeepSeek API) | Fine-Tuned (Local) |
|---|---|---|
| **Inference cost** | ~$0.003 / document | **$0.00** (runs locally) |
| **Latency (p50)** | 3-5s (network round trip) | ~1.8s (local GPU) |
| **VRAM required** | None (cloud) | ~6GB (4-bit quant) |
| **Vendor lock-in** | Yes | **None** |
| **Data privacy** | Data leaves device | **Fully on-device** |
| **Field accuracy** | ~95%+ | **90.0%** |
| **One-time training cost** | — | **~$0.40** (data gen API calls) |

**The tradeoff:** 90% of the accuracy at 0% ongoing cost per document.

---

## Technical Decisions

| Decision | Alternative Considered | Why This Was Chosen |
|---|---|---|
| Qwen 2.5 3B | Llama 3.2 3B, Phi-3.5 | Leads 3B benchmarks; best instruction following in class |
| QLoRA 4-bit | Full fine-tune, 8-bit | Fits in 8GB VRAM; negligible accuracy loss for SFT |
| Unsloth | HuggingFace TRL directly | 2x faster, 70% less VRAM on same hardware |
| Google Colab | Local Windows GPU | Unsloth requires Linux; Colab T4 is free and clean |
| DeepSeek for data gen | GPT-4o, Claude | Cheapest frontier model; ~$0.40 total for 926 examples |
| FastAPI | Flask, Gradio | Async, auto-docs (/docs), Pydantic-native, production-grade |
| transformers + peft locally | Unsloth locally | Unsloth is Linux-only; peft runs natively on Windows |

---

## Resume Bullet

> *"Built Forge, an end-to-end model distillation pipeline for structured invoice extraction: generated 926 synthetic training examples via DeepSeek on OpenRouter, curated a 600-example QLoRA training set with a locked eval discipline, fine-tuned Qwen 2.5 3B on Google Colab (free T4 GPU) using Unsloth, and validated on 91 held-out examples — achieving 90.0% field-level accuracy vs. 44.8% for the base model at $0 marginal inference cost per document, served via FastAPI."*

---

## Potential Extensions

| Extension | Effort | Impact |
|---|---|---|
| HuggingFace Hub upload + model card | Low (1hr) | Public artifact, discoverable by recruiters |
| GGUF export + Ollama | Medium (4hr) | `ollama run invoice-extractor` from any terminal |
| Streamlit demo UI | Medium (1 day) | Visual demo, shareable link |
| PDF/image input via OCR | Medium (1 day) | Closes gap to real-world use case |
| Per-field confidence scores | Medium (1 day) | Production-realistic: flags low-confidence for review |
| 7B model comparison | Low (re-run notebook) | Quantifies the accuracy/cost scaling tradeoff |
| Modal.com cloud deployment | Medium (1 day) | Live public API URL on resume |
| Active learning loop | High (2-3 days) | Continuous improvement pipeline |
