"""
Training hyperparameter configuration for Forge.

All tuneable parameters in one place.  The Colab training notebook imports from
here so there's a single source of truth for what hyperparameters were used.

IMPORTANT — read before modifying
-----------------------------------
This file is the canonical record of training configuration.  When you run a
training job, do not change these values mid-run.  Instead:
1. Copy the current values to training/run_log.md with the run's results.
2. Update the values here for the next run.

This creates a clean audit trail: run_log.md has the full history, and this
file always reflects the *current* intended configuration.

Version-confirm note (from PRD Section 5)
------------------------------------------
The fine-tuning ecosystem (Unsloth, trl, peft, transformers) changes fast.
Before running the training notebook, confirm the following against the current
Unsloth documentation:
- The correct pip install command for Colab (it changes with each release)
- The SFTTrainer argument names (some have been renamed across trl versions)
- The train_on_responses_only implementation (approach changes with Unsloth version)
- The GGUF export method (llama.cpp integration has changed across versions)

These are flagged with # CONFIRM-AT-TRAINING-TIME comments below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Student model selection."""

    # CONFIRM-AT-TRAINING-TIME: verify the HuggingFace model ID is current
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct"
    # Stretch option if 1.5B results are unconvincing (flag to user first):
    # model_name: str = "unsloth/Qwen2.5-3B-Instruct"

    max_seq_length: int = 2048    # Maximum context length during training
    dtype: str = "float16"        # bfloat16 on Ampere+; float16 on T4 (older)
    load_in_4bit: bool = True     # QLoRA: load base model in 4-bit


@dataclass(frozen=True)
class LoraConfig:
    """QLoRA adapter configuration."""

    # LoRA rank: higher = more parameters trained, more VRAM, slower convergence
    # r=16 is a reasonable starting point for a 1.5B model on this task
    r: int = 16

    # LoRA alpha: typically 2*r or equal to r
    # alpha=16 with r=16 gives an effective scale of 1.0 (lora_alpha/r)
    lora_alpha: int = 16

    # Dropout on LoRA weights (small values help generalization)
    lora_dropout: float = 0.05

    # Which weight matrices to adapt.  For instruction-following tasks,
    # adapting attention + MLP projection matrices works well.
    # CONFIRM-AT-TRAINING-TIME: check current Unsloth recommended target_modules
    # for Qwen2.5 (the set changes as Unsloth optimises for each architecture)
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    bias: str = "none"            # Don't train bias terms
    use_rslora: bool = False      # Rank-stabilised LoRA (try True if training is unstable)
    use_gradient_checkpointing: str = "unsloth"  # Unsloth's optimised GC


@dataclass(frozen=True)
class TrainingConfig:
    """SFTTrainer / training loop configuration."""

    # Batch size: 2 is safe for T4 (16GB VRAM) with 1.5B model + 4-bit
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2

    # Gradient accumulation: effective batch size = 2 * 4 = 8
    gradient_accumulation_steps: int = 4

    # Epochs: 2–3 for a few hundred examples; watch for overfitting on val loss
    num_train_epochs: int = 3

    # Learning rate: 2e-4 is Unsloth's recommended starting point for QLoRA
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05    # 5% of steps for warm-up

    # Optimiser: adamw_8bit from bitsandbytes reduces VRAM vs. full adamw
    # CONFIRM-AT-TRAINING-TIME: check if Unsloth recommends a different optimiser
    optim: str = "adamw_8bit"

    # Logging and evaluation
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50          # Evaluate on val set every N steps

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2     # Keep only the 2 most recent checkpoints

    # Output
    output_dir: str = "training/checkpoints"

    # Precision
    fp16: bool = True             # Use FP16 on T4 (no BF16 support on T4)
    bf16: bool = False

    # Sequence packing: Unsloth can pack multiple short examples into one
    # sequence for efficiency.  Good for this task (invoices are shorter than 2K)
    # CONFIRM-AT-TRAINING-TIME: check current Unsloth packing API
    packing: bool = True

    # Random seed for reproducibility
    seed: int = 42

    # Dataset column
    dataset_text_field: str = "text"    # After Unsloth applies chat template


@dataclass(frozen=True)
class DataConfig:
    """Dataset paths (relative to repo root)."""

    train_formatted: str = "data/train_formatted.jsonl"
    val_formatted: str = "data/val_formatted.jsonl"


# ---------------------------------------------------------------------------
# Convenience: instantiate defaults
# ---------------------------------------------------------------------------

MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoraConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()


if __name__ == "__main__":
    import dataclasses
    print("=== Model Config ===")
    for k, v in dataclasses.asdict(MODEL_CONFIG).items():
        print(f"  {k}: {v}")
    print("\n=== LoRA Config ===")
    for k, v in dataclasses.asdict(LORA_CONFIG).items():
        print(f"  {k}: {v}")
    print("\n=== Training Config ===")
    for k, v in dataclasses.asdict(TRAINING_CONFIG).items():
        print(f"  {k}: {v}")
