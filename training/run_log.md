# Training Run Log

Record every training run here, including failed and abandoned runs. Do not delete old entries. This file is the audit trail for model selection, hyperparameters, wall-clock time, validation loss, and eval results.

## Current Committed Status

No completed training run artifact is committed in this repository.

The repository contains:

- QLoRA training configuration in `training/config.py`
- Colab training notebook in `training/Unsloth_Finetune_Qwen2_5_3B.ipynb`
- formatted train/validation datasets in `data/train_formatted.jsonl` and `data/val_formatted.jsonl`

The repository does not currently contain:

- trained LoRA adapter weights
- merged model checkpoint
- `evaluation/results/finetuned_model_results.json`
- measured final eval accuracy

## Intended Run 001 Configuration

| Parameter | Value |
| --- | --- |
| Status | Pending / not committed |
| Model | `unsloth/Qwen2.5-3B-Instruct` |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | 2e-4 |
| Scheduler | cosine |
| Epochs | 3 |
| Per-device train batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective train batch size | 8 |
| Training examples | 600 |
| Validation examples | 67 |
| Eval examples | 91 locked examples |
| Precision | fp16 |
| Quantization | 4-bit QLoRA |
| Planned GPU | Colab T4 or compatible CUDA GPU |

## Required Evidence After Training

Add a new completed run entry with:

- run date
- exact package versions
- GPU type
- wall-clock training time
- final training and validation loss
- adapter or merged model artifact path
- base model eval result path
- fine-tuned model eval result path
- teacher eval result path
- locked eval checksum
- any known failure modes
