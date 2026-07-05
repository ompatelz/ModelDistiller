# Forge — Frontier-to-Small-Model Distillation for Structured Data Extraction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.5%20CUDA%2012.1-ee4c2c.svg)](https://pytorch.org/)
[![Model: Qwen 2.5 3B](https://img.shields.io/badge/Model-Qwen%202.5%203B-7b2cbf.svg)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![PEFT: QLoRA](https://img.shields.io/badge/PEFT-QLoRA%20(4--bit%20NF4)-00b4d8.svg)](https://github.com/huggingface/peft)
[![Training: Unsloth](https://img.shields.io/badge/Training-Unsloth%202x%20Faster-00bb00.svg)](https://github.com/unslothai/unsloth)
[![Serving: FastAPI](https://img.shields.io/badge/Serving-FastAPI%20%2B%20Uvicorn-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An end-to-end knowledge distillation pipeline that proves the unit economics of AI:**  
> By distilling a frontier teacher model (**DeepSeek**) into a specialized open-weight small language model (**Qwen 2.5 3B**) via **QLoRA**, Forge achieves **90.0% field-level accuracy** on complex invoice extraction (matching frontier quality) while reducing inference costs by **100% to $0 marginal cost** and executing entirely on-device with sub-2s latency.

---

## ⚡ Executive Summary & Key Results

Frontier LLMs (GPT-4o, Claude 3.5, DeepSeek) are phenomenal at general reasoning, but deploying them for high-volume, repetitive structured extraction tasks creates unsustainable cloud bills, latency bottlenecks, data privacy risks, and strict vendor lock-in. 

**Forge answers the fundamental MLOps question:** *Why pay frontier API prices forever for a narrow, well-defined task?*

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             FORGE PERFORMANCE DELTA                              │
│                                                                                  │
│   BASE MODEL (Qwen 2.5 3B)             DISTILLED MODEL (Forge LoRA)              │
│   ❌ Overall Accuracy: 44.8%    ───▶   ✅ Overall Accuracy: 90.0%  (+45.2pp)    │
│   ❌ Line Items Array: 2.5%     ───▶   ✅ Line Items Array: 88.5%  (+86.0pp)    │
│   ❌ Vendor Name:      9.9%     ───▶   ✅ Vendor Name:      92.3%  (+82.4pp)    │
│   ❌ Marginal Cost:    $0.003   ───▶   ✅ Marginal Cost:    $0.00  (100% Free)   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ System Architecture & Distillation Pipeline

Forge is built on a strict **eval-first, data-disciplined architecture** across five sequential phases:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — SYNTHETIC DATA GENERATION (LLM-as-a-Generator)                        │
│                                                                                  │
│  20+ Scenario Templates ──▶ DeepSeek API (OpenRouter) ──▶ 926 Raw Examples      │
│  (Varied across US/EU/UK/CAD vendors, multi-currency, tax types, and layouts)   │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — CURATION PIPELINE & EVAL-LOCK DISCIPLINE                              │
│                                                                                  │
│  Pydantic Schema Validation ──▶ Jaccard Deduplication ──▶ Quality Filtering      │
│  ──▶ Stratified Split: Train (600) / Val (67) / Eval Locked (91 — NEVER touched) │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — PARAMETER-EFFICIENT FINE-TUNING (PEFT)                                │
│                                                                                  │
│  Qwen 2.5 3B Instruct + QLoRA (4-bit NF4 Quantization, r=16, alpha=16)           │
│  Trained via Unsloth on Google Colab T4 GPU (Free Tier) in ~2 hours              │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — QUANTITATIVE EVALUATION & BENCHMARKING                                │
│                                                                                  │
│  Base Model vs. Distilled Model evaluated against 91 locked held-out invoices    │
│  Strict field-by-field scoring across 14 Pydantic schema attributes              │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5 — PRODUCTION SERVING & EDGE DEPLOYMENT                                  │
│                                                                                  │
│  114MB LoRA Adapter loaded dynamically over base weights via HuggingFace PEFT    │
│  Served locally via CUDA-accelerated FastAPI / Uvicorn REST API with Swagger UI  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Quantitative Benchmark Results

All evaluations were conducted on the **locked 91-example held-out test set** (`data/eval_locked.jsonl`), which was strictly separated before any training or hyperparameter tuning began.

### Overall Performance
| Metric | Base Model (Qwen 2.5 3B) | Distilled Model (Forge LoRA) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Overall Field Accuracy** | **44.8%** | **90.0%** | **+45.2 pp** |
| **JSON Parse / Valid Syntax Rate** | 98.9% | 98.9% | +0.0 pp |
| **Easy Scenarios Accuracy** | 43.4% | 89.5% | +46.0 pp |
| **Medium Scenarios Accuracy** | 43.7% | 93.9% | +50.2 pp |
| **Hard Scenarios Accuracy** | 47.4% | 85.8% | +38.4 pp |

### Detailed Field-by-Field Accuracy
The most dramatic improvements occurred in complex structured data types (nested JSON arrays) and semantic entity extraction where small base models traditionally hallucinate:

| Schema Attribute | Base Accuracy | Distilled Accuracy | Delta | Why Distillation Made the Difference |
| :--- | :---: | :---: | :---: | :--- |
| `line_items` (Nested Array) | **2.5%** | **88.5%** | **+86.0 pp** | Taught the model how to format multi-item nested arrays with quantities and unit prices without breaking syntax. |
| `vendor_name` | **9.9%** | **92.3%** | **+82.4 pp** | Eliminated hallucinations; forced faithful extraction from document header. |
| `total_amount` | 25.3% | 97.8% | +72.5 pp | Stopped the model from attempting internal math; trained it to extract explicit totals. |
| `subtotal` | 26.4% | 91.2% | +64.8 pp | Correctly distinguished between pre-tax subtotals and final balance dues. |
| `currency` | 36.3% | 98.9% | +62.6 pp | Learned strict normalization to 3-letter ISO 4217 codes (`USD`, `EUR`, `GBP`, `CAD`). |
| `bill_to` | 22.0% | 81.3% | +59.3 pp | Correctly separated recipient name/company from vendor address blocks. |
| `vendor_address` | 16.5% | 71.4% | +54.9 pp | Captured multi-line international address formats accurately. |
| `tax_amount` | 49.5% | 94.5% | +45.1 pp | Distinguishes between VAT, GST, HST, and standard US sales tax. |
| `tax_rate` | 63.7% | 96.7% | +33.0 pp | Learned strict normalization to decimal fractions (`0.08` instead of `8%`). |
| `notes` | 39.6% | 64.8% | +25.3 pp | Improved extraction of payment instructions and wire details (inherently ambiguous). |
| `invoice_date` | 78.0% | 98.9% | +20.9 pp | Normalizes all textual dates (`14 Jan 2025`, `01/14/25`) to `YYYY-MM-DD`. |
| `payment_terms` | 79.1% | 91.2% | +12.1 pp | Accurately extracts `Net 30`, `Due on receipt`, etc. |
| `invoice_number` | 84.6% | 95.6% | +11.0 pp | Near-perfect identification of alphanumeric invoice/transaction IDs. |
| `due_date` | 93.4% | 96.7% | +3.3 pp | High baseline; fine-tuning brought it to near-perfection. |

---

## 💰 Unit Economics & ROI Analysis

By replacing cloud API endpoints with a fine-tuned local SLM, Forge alters the operational economics of document processing:

| Metric | Teacher Model (DeepSeek API) | Distilled Model (Forge Local SLM) | Business Impact |
| :--- | :---: | :---: | :--- |
| **Inference Cost / Doc** | ~$0.0030 | **$0.0000** | **100% cost reduction** (runs on owned hardware). |
| **Latency (p50)** | 3,500ms – 5,000ms | **~1,800ms** | **2x–3x faster** (no network roundtrip or queue times). |
| **Data Privacy** | Sensitive financial data leaves network | **100% On-Device / Local** | Compliant with GDPR, HIPAA, and SOC2 requirements. |
| **Vendor Lock-in** | High (API rate limits & pricing changes) | **Zero** (Open-weight weights owned forever) | Complete operational independence. |
| **VRAM Footprint** | N/A (Cloud) | **~6 GB VRAM** | Runs on commodity consumer GPUs (RTX 3060/4060). |
| **Upfront Training Cost**| N/A | **~$0.40** (926 API calls) | Payback period achieved after processing just ~135 invoices. |

---

## 🚀 Quick Start & How to Run

### 1. Prerequisites & Installation
Ensure you have **Python 3.12** and an NVIDIA GPU with CUDA support installed.

```powershell
# Clone the repository
git clone https://github.com/<your-username>/Forge.git
cd Forge

# Create and activate virtual environment using uv or standard venv
python -m venv .venv
.\.venv\Scripts\activate

# Install PyTorch with CUDA 12.1 support + project dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes fastapi uvicorn pydantic
```

---

### 2. Run Local CLI Inference
Test the distilled model directly from your terminal using `infer.py`. The script dynamically loads the base Qwen 2.5 3B model in 4-bit quantization and attaches the trained 114MB LoRA adapter (`models/lora_model`).

```powershell
# Interactive mode (paste invoice text, press Enter twice to extract)
.\.venv\Scripts\python.exe infer.py --pretty

# File mode (extract from a saved document)
.\.venv\Scripts\python.exe infer.py --input sample_invoice.txt --pretty
```

---

### 3. Launch FastAPI REST Service & Interactive Docs
Spin up a production-ready asynchronous HTTP server that keeps the model loaded in GPU memory for sub-2s inference.

```powershell
# Start the FastAPI server on port 8000
.\.venv\Scripts\python.exe -m uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

Once running, open your browser to navigate to the **Interactive OpenAPI Swagger UI**:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

#### Example cURL Request:
```bash
curl -X POST "http://localhost:8000/extract" \
     -H "Content-Type: application/json" \
     -d '{
       "document": "Acme Corp\n123 Tech Way, Silicon Valley, CA\n\nINVOICE #INV-2025-089\nDate: 14 Jan 2025\nDue: 14 Feb 2025\n\nDescription           Qty   Price\nCloud Consultation     1    $1,200.00\nArchitecture Audit     2      $450.00\n\nSubtotal: $2,100.00\nTax (8%):   $168.00\nTotal Due: $2,268.00\n\nPayment Terms: Net 30"
     }'
```

#### JSON Response:
```json
{
  "extraction": {
    "vendor_name": "Acme Corp",
    "vendor_address": "123 Tech Way, Silicon Valley, CA",
    "invoice_number": "INV-2025-089",
    "invoice_date": "2025-01-14",
    "due_date": "2025-02-14",
    "bill_to": null,
    "line_items": [
      {
        "description": "Cloud Consultation",
        "quantity": 1,
        "unit_price": 1200.0,
        "total": 1200.0
      },
      {
        "description": "Architecture Audit",
        "quantity": 2,
        "unit_price": 450.0,
        "total": 900.0
      }
    ],
    "subtotal": 2100.0,
    "tax_amount": 168.0,
    "tax_rate": 0.08,
    "total_amount": 2268.0,
    "currency": "USD",
    "payment_terms": "Net 30",
    "notes": null
  },
  "schema_valid": true,
  "latency_ms": 1784.2,
  "model": "Qwen2.5-3B + LoRA (lora_model)"
}
```

---

## 📁 Repository Structure

```
Forge/
├── schema/
│   └── extraction_schema.py       # Single source of truth: Pydantic models & JSON schema definitions
│
├── data_generation/
│   └── generate_synthetic_data.py # Resumable DeepSeek LLM-as-a-generator script (OpenRouter API)
│
├── data_curation/
│   ├── validate_schema.py         # Pydantic validation pass (removes malformed/incomplete JSON)
│   ├── deduplicate.py             # Jaccard shingle similarity & scenario_id deduplication
│   ├── quality_filter.py          # Discards trivial, overly short, or degenerate examples
│   ├── split_dataset.py           # Stratified train/val/eval splitter (locks 91 eval examples)
│   └── format_dataset.py          # Converts curated data into ShareGPT / ChatML format for training
│
├── data/
│   ├── synthetic_data.jsonl       # Raw generated dataset (926 examples)
│   ├── train_formatted.jsonl      # Curated training set (600 examples)
│   ├── val_formatted.jsonl        # Validation set (67 examples)
│   └── eval_locked.jsonl          # Untouched benchmark test set (91 examples)
│
├── training/
│   ├── Unsloth_Finetune_Qwen2_5_3B.ipynb  # Colab T4 notebook: QLoRA 4-bit SFT via Unsloth
│   ├── Eval_Qwen2_5_3B.ipynb              # Automated benchmark scoring notebook (Base vs LoRA)
│   └── config.py                          # Hyperparameter configuration
│
├── models/
│   └── lora_model/                # Trained PEFT LoRA adapter weights (~114MB safetensors)
│
├── serving/
│   ├── api.py                     # Asynchronous FastAPI / Uvicorn REST server
│   └── cli.py                     # Command-line utility tools
│
├── tests/                         # Unit tests for schema validation and evaluation scoring
├── infer.py                       # Standalone local CUDA inference script
├── PRD.md                         # Detailed Product Requirements Document & engineering notes
└── README.md                      # Project documentation
```

---

## 🧠 Core Engineering & MLOps Concepts Demonstrated

1. **Knowledge Distillation & SLM Specialization**: Transferring structured domain capability from an expensive 671B+ parameter frontier model into a highly efficient 3B parameter open-weight student model.
2. **Parameter-Efficient Fine-Tuning (PEFT) via QLoRA**: Quantizing base weights to 4-bit NormalFloat (NF4) and freezing them while training low-rank adapter matrices ($r=16, \alpha=16$), reducing training VRAM by 70%.
3. **Data Discipline & Curation Pipelines**: Implementing automated validation gates (Pydantic schema checks, Jaccard similarity deduplication) to ensure high-signal training data without manual labeling.
4. **Benchmark Rigor (Locked Held-out Test Sets)**: Isolating test data before training begins to guarantee zero data leakage and produce defensible, real-world accuracy metrics.
5. **Modern AI Serving Infrastructure**: Deploying local GPU-accelerated endpoints via FastAPI with structured JSON adherence, CORS middleware, and automated OpenAPI documentation.

---

## 📜 License
This project is open-source and available under the **MIT License**.
