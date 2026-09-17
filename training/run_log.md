# Training Run Log

Record every training run here, including failed and completed runs. This file maintains the audit trail for model selection, hyperparameters, wall-clock time, loss curves, and evaluation results.

---

## Completed Run 001 (Distilled Qwen 2.5 3B QLoRA)

- **Date**: June 2026
- **Model**: `unsloth/Qwen2.5-3B-Instruct`
- **Framework**: Unsloth + TRL `SFTTrainer` + PEFT + BitsAndBytes
- **Hardware**: Google Colab Free Tier (NVIDIA Tesla T4 16GB VRAM)
- **Wall-Clock Training Time**: ~2 hours (225 steps)
- **Dataset**:
  - Training: 600 examples (`data/train_formatted.jsonl`)
  - Validation: 67 examples (`data/val_formatted.jsonl`)
  - Locked Held-Out Eval: 91 examples (`data/eval_locked.jsonl`)

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Quantization | 4-bit NF4 (NormalFloat4) | Reduces 3B model memory footprint from ~6GB to ~2.2GB |
| LoRA Rank ($r$) | 16 | Optimal capacity for single-domain structured formatting |
| LoRA Alpha ($\alpha$) | 16 | Scaling factor = 1.0 ($\alpha/r$) |
| LoRA Dropout | 0.05 | Regularization to prevent overfitting |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Adapts all attention projection and MLP feed-forward layers |
| Optimizer | `adamw_8bit` | 8-bit AdamW optimizer via bitsandbytes |
| Learning Rate | `2e-4` | Standard Unsloth SFT learning rate |
| LR Scheduler | Linear with 10 warmup steps | Smooth learning rate ramp |
| Batch Size | 2 per device | Fits within T4 16GB VRAM budget |
| Gradient Accumulation | 4 steps | Effective batch size = 8 ($2 \times 4$) |
| Epochs | 3 | Sufficient convergence without overfitting |
| Precision | FP16 | Native T4 floating point precision |

### Loss Progression

| Step | Train Loss | Validation Loss |
|---|:---:|:---:|
| 25 | 0.526 | 0.491 |
| 75 | 0.372 | 0.553 |
| 125 | 0.293 | 0.321 |
| 175 | 0.269 | 0.309 |
| **225** (Final) | **0.263** | **0.303** |

### Evaluation Results (Locked Test Set: 91 examples)

| System | Overall Field Accuracy | Full-Record Exact Match | Schema Validity | p50 Latency |
|---|:---:|:---:|:---:|:---:|
| **Base Model** | 44.8% | 12.1% | 98.9% | 1.820s |
| **Fine-Tuned Run 001** | **90.0%** | **72.5%** | **98.9%** | **1.784s** |
| **Teacher Model** | 96.5% | 86.8% | 100.0% | 4.120s |

- **Artifacts Generated**:
  - `evaluation/results/base_model_results.json`
  - `evaluation/results/finetuned_model_results.json`
  - `evaluation/results/teacher_model_results.json`
  - Checksum verified: `0ec6e5175885a61467757403a6fe0f9207b5378b175f0563b08871c5beba9264`
