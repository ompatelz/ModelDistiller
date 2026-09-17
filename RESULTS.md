# Forge Evaluation & Distillation Benchmark Results

This file documents the evidence-backed benchmark comparing **Base Qwen 2.5 3B**, **Distilled Qwen 2.5 3B (QLoRA)**, and the **DeepSeek V4 Flash Frontier Teacher** evaluated on the locked held-out test set.

All figures below correspond to committed evaluation artifacts in `evaluation/results/` and dataset splits in `data/`.

---

## Executive Summary

- **Task**: Structured JSON extraction from unformatted plain-text invoices and receipts.
- **Accuracy Jump**: **+45.2 percentage points** (Base 44.8% → Distilled 90.0%).
- **Highest Field Gains**:
  - `line_items`: **+86.0pp** (2.5% → 88.5%)
  - `vendor_name`: **+82.4pp** (9.9% → 92.3%)
  - `total_amount`: **+72.5pp** (25.3% → 97.8%)
  - `subtotal`: **+64.8pp** (26.4% → 91.2%)
- **Latency Advantage**: **~1.78s local** vs **4.12s cloud frontier API** (**2.3x faster**).
- **Economics**: Total pipeline build cost of **~$0.40** breaks even after processing just **2,857 documents** (~3.4 days at 25k/mo volume).

---

## Dataset & Split Evidence

| Artifact | Count / Hash | Description |
|---|---|---|
| **Raw Synthetic Generation** | 926 examples | DeepSeek via OpenRouter (`data/raw/generation_run_20260621T195959.json`) |
| **Training Split** | 600 examples | Curated, validated, deduplicated (`data/train.jsonl` & `train_formatted.jsonl`) |
| **Validation Split** | 67 examples | Held-out validation split for checkpointing (`data/val.jsonl`) |
| **Locked Eval Set** | 91 examples | Isolated before training began (`data/eval_locked.jsonl`) |
| **Locked Eval SHA-256** | `0ec6e5175885a61467757403a6fe0f9207b5378b175f0563b08871c5beba9264` | Mechanical checksum verification enforced in `evaluation/run_eval.py` |

---

## Model Quality Comparison

| Metric | Base Qwen 2.5 3B | Fine-Tuned Qwen 2.5 3B (LoRA) | Teacher (DeepSeek V4 Flash) | Delta (LoRA vs Base) |
|---|:---:|:---:|:---:|:---:|
| **Overall Field Accuracy** | 44.8% | **90.0%** | 96.5% | **+45.2pp** |
| **JSON Parse Rate** | 98.9% | **98.9%** | 100.0% | +0.0pp |
| **Full-Record Exact Match** | 12.1% | **72.5%** | 86.8% | **+60.4pp** |
| **Easy Scenarios** | 43.4% | **89.5%** | 97.8% | **+46.0pp** |
| **Medium Scenarios** | 43.7% | **93.9%** | 97.0% | **+50.2pp** |
| **Hard / Messy Scenarios** | 47.4% | **85.8%** | 94.6% | **+38.4pp** |

---

## Field-by-Field Accuracy Breakdown

Scored across all 14 schema dimensions on the 91 locked test invoices:

| Field Name | Base Qwen 2.5 | Fine-Tuned Qwen 2.5 | Teacher | Delta | Key Extraction Challenge |
|---|:---:|:---:|:---:|:---:|---|
| **`line_items`** | 2.5% | **88.5%** | 95.6% | **+86.0pp** | Nested array structure, quantity/unit price inference |
| **`vendor_name`** | 9.9% | **92.3%** | 98.9% | **+82.4pp** | Distinguishing seller from customer & carrier |
| **`total_amount`** | 25.3% | **97.8%** | 98.9% | **+72.5pp** | Extracting bottom-line rather than calculating |
| **`subtotal`** | 26.4% | **91.2%** | 97.8% | **+64.8pp** | Handling pre-tax sums and multi-rate sub-lines |
| **`currency`** | 36.3% | **98.9%** | 100.0% | **+62.6pp** | Normalizing symbols (`$`, `€`, `£`, `A$`) to ISO 4217 |
| **`bill_to`** | 22.0% | **81.3%** | 94.5% | **+59.3pp** | Segmenting recipient name and street address |
| **`vendor_address`** | 16.5% | **71.4%** | 92.3% | **+54.9pp** | Multi-line addresses, suite/postal codes |
| **`tax_amount`** | 49.5% | **94.5%** | 97.8% | **+45.1pp** | Separating GST, VAT, state, and provincial tax |
| **`tax_rate`** | 63.7% | **96.7%** | 98.9% | **+33.0pp** | Converting `8.5%` → decimal fraction `0.085` |
| **`notes`** | 39.6% | **64.8%** | 83.5% | **+25.3pp** | Identifying memo text vs footer disclaimers |
| **`invoice_date`** | 78.0% | **98.9%** | 100.0% | **+20.9pp** | Converting diverse date styles → `YYYY-MM-DD` |
| **`payment_terms`** | 79.1% | **91.2%** | 96.7% | **+12.1pp** | Parsing Net 30, Due on Receipt, 2/10 Net 30 |
| **`invoice_number`** | 84.6% | **95.6%** | 98.9% | **+11.0pp** | Alphanumeric IDs and receipt numbers |
| **`due_date`** | 93.4% | **96.7%** | 98.9% | **+3.3pp** | Parsing payment deadlines |

---

## Latency Profile

Measured on NVIDIA T4 (CUDA) / Local Hardware:

| Metric | Base Model | Distilled LoRA Model | Teacher API | Advantage |
|---|:---:|:---:|:---:|---|
| **p50 Latency** | 1.820s | **1.784s** | 4.120s | **2.3x faster than API** |
| **p95 Latency** | 2.450s | **2.340s** | 6.850s | **2.9x faster than API** |
| **Network Variability** | None (Local) | None (Local) | ±1.5s queue jitter | Predictable SLA |

---

## Cost & ROI Analysis

```powershell
# Reproduce this analysis with:
python -m cost_analysis.compute_cost_comparison 0.40
```

| Parameter | Frontier API (DeepSeek/Claude) | Self-Hosted Distilled Model |
|---|:---:|:---:|
| **Cost per 1,000 documents** | $0.1400 – $0.3500 | **~$0.00** (self-hosted) |
| **Cost per successful extraction** | $0.000145 | **~$0.00** |
| **Pipeline Build Cost** | N/A | **~$0.40** (926 teacher API calls) |
| **Break-Even Volume** | N/A | **2,857 documents** |
| **Payback at 10k docs/month** | N/A | **~0.3 months (~9 days)** |
| **Payback at 25k docs/month** | N/A | **~0.1 months (~3.4 days)** |

---

## Limitations & Production Recommendations

1. **Synthetic Training Distribution**: Data was generated by prompting teacher models across 20 scenario templates. While these cover 95%+ of structural variants, production deployments should fine-tune with a 10% sample of de-identified real business invoices.
2. **Ambiguity in Notes**: The `notes` field achieved 64.8% accuracy. Real invoices often mix legalese, payment wire details, and customer messages; decoupling terms from notes is recommended for stricter parsing.
3. **Adapter Checkpoint**: The repo contains the training notebook (`training/Unsloth_Finetune_Qwen2_5_3B.ipynb`) and evaluation artifacts (`evaluation/results/*.json`). Weights can be regenerated via Colab or served via the built-in multi-engine adapter.
