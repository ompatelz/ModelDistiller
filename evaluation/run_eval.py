"""
Evaluation runner for Forge.

Runs a given model (base Qwen2.5 / fine-tuned Qwen2.5 / DeepSeek V4 Flash teacher)
against the locked eval set and writes results to evaluation/results/.

Usage
-----
    # Evaluate the teacher (DeepSeek via OpenRouter) model:
    python -m evaluation.run_eval \\
        --model-type teacher \\
        --output evaluation/results/teacher_model_results.json

    # Evaluate the base Qwen2.5 model (after training is set up):
    python -m evaluation.run_eval \\
        --model-type base \\
        --hf-model Qwen/Qwen2.5-1.5B-Instruct \\
        --output evaluation/results/base_model_results.json

    # Evaluate the fine-tuned model (after training completes):
    python -m evaluation.run_eval \\
        --model-type finetuned \\
        --hf-model path/to/merged_model \\
        --output evaluation/results/finetuned_model_results.json

Eval set integrity check
------------------------
Before any eval run, this script verifies the SHA-256 hash of eval_locked.jsonl
against the stored .sha256 file.  If the file has been modified, the run aborts.
This is a mechanical enforcement of the PRD's eval-set integrity rule — not just
a convention.

Model interface
---------------
All three model types implement the same interface: given a document_text string,
return a raw string (expected to be JSON).  The scoring code in scoring.py then
parses and scores the output identically for all three.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Protocol

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "openai package is required for OpenRouter teacher eval.\n"
        "Install with: pip install openai"
    ) from exc

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_curation.split_dataset import verify_checksum
from evaluation.scoring import (
    AggregateScores,
    ExampleScore,
    aggregate_scores,
    score_example,
)

log = logging.getLogger(__name__)

EVAL_LOCKED_PATH = Path("data/eval_locked.jsonl")
RESULTS_DIR = Path("evaluation/results")

# The extraction instruction injected into every prompt
_EXTRACTION_INSTRUCTION = """\
Extract structured invoice data from the document below and return ONLY a valid JSON object.
Do not include any explanation, markdown formatting, or text outside the JSON object.

The JSON must have these fields:
  vendor_name (required), vendor_address, invoice_number, invoice_date (YYYY-MM-DD),
  due_date (YYYY-MM-DD), bill_to, line_items (list, required, each with description,
  quantity, unit_price, total), subtotal, tax_amount, tax_rate (decimal, e.g. 0.08 for 8%),
  total_amount (required), currency (ISO 4217, e.g. USD), payment_terms, notes.

Use null for any field not present in the document.

Document:
"""


# ---------------------------------------------------------------------------
# Model interface protocol
# ---------------------------------------------------------------------------

class ModelInterface(Protocol):
    """Common interface for all three model types."""

    def extract(self, document_text: str) -> tuple[str, float]:
        """
        Run extraction on document_text.

        Returns
        -------
        (raw_output, latency_seconds)
        """
        ...


# ---------------------------------------------------------------------------
# OpenRouter / DeepSeek teacher model
# ---------------------------------------------------------------------------

class OpenRouterModel:
    """
    Calls the OpenRouter API (OpenAI-compatible) for extraction.
    Used as the teacher model in eval.

    Reads OPENROUTER_API_KEY and OPENROUTER_MODEL from the environment
    (or accepts them as constructor arguments for explicit control).
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        import os
        self.model_name = model_name or os.environ.get(
            "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"
        )
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set.  Set it in .env or export it."
            )
        self.client = OpenAI(
            api_key=resolved_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def extract(self, document_text: str) -> tuple[str, float]:
        import time as time_module
        prompt = _EXTRACTION_INSTRUCTION + document_text

        t0 = time_module.monotonic()
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        latency = time_module.monotonic() - t0
        return response.choices[0].message.content or "", latency


# ---------------------------------------------------------------------------
# HuggingFace / local model (base + fine-tuned)
# ---------------------------------------------------------------------------

class HuggingFaceModel:
    """
    Loads a Qwen2.5-Instruct model locally via transformers and runs inference.

    Works for both the base model (Qwen/Qwen2.5-1.5B-Instruct) and the merged
    fine-tuned model (path/to/merged_model).

    NOTE: This class loads the model in full precision by default.
    For Colab / GPU inference, pass device_map="auto" and use 4-bit quantization
    via BitsAndBytesConfig if VRAM is constrained.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        load_in_4bit: bool = False,
    ) -> None:
        log.info("Loading model: %s  (device=%s, 4bit=%s)", model_name_or_path, device, load_in_4bit)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for HuggingFaceModel. "
                "Install with: pip install transformers torch"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device,
            quantization_config=bnb_config,
        )
        self.model.eval()
        log.info("Model loaded successfully.")

    def extract(self, document_text: str) -> tuple[str, float]:
        import time as time_module
        import torch

        # Format as a chat message using the model's chat template
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a structured data extraction assistant. "
                    "Extract invoice fields from documents and return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": _EXTRACTION_INSTRUCTION + document_text,
            },
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        t0 = time_module.monotonic()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,          # Greedy decoding for reproducibility
                temperature=1.0,          # Required when do_sample=False
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time_module.monotonic() - t0

        # Decode only the generated tokens (not the prompt)
        generated = outputs[0][input_ids.shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)

        return raw_output.strip(), latency


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(
    model: ModelInterface,
    eval_path: Path,
    output_path: Path,
    model_label: str,
    inter_request_delay: float = 0.3,
    verify_eval_checksum: bool = True,
) -> AggregateScores:
    """
    Run evaluation against the locked eval set and write results JSON.

    Parameters
    ----------
    model                 : ModelInterface instance (ClaudeModel or HuggingFaceModel).
    eval_path             : Path to eval_locked.jsonl.
    output_path           : Where to write the results JSON.
    model_label           : Label for logging (e.g. "base", "finetuned", "teacher").
    inter_request_delay   : Seconds to sleep between API calls (rate limiting).
    verify_eval_checksum  : Abort if eval set has been modified (default True).

    Returns
    -------
    AggregateScores for this model run.
    """
    # ------------------------------------------------------------------
    # Integrity check
    # ------------------------------------------------------------------
    if verify_eval_checksum:
        if not verify_checksum(eval_path):
            raise RuntimeError(
                "Eval set integrity check failed — eval_locked.jsonl has been "
                "modified since it was locked.  Per PRD Section 6, this eval "
                "run cannot proceed.  See the error above for details."
            )
        log.info("✓ Eval set integrity verified (SHA-256 matches)")

    # ------------------------------------------------------------------
    # Load eval set
    # ------------------------------------------------------------------
    examples: list[dict] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    log.info("Loaded %d eval examples from %s", len(examples), eval_path)
    log.info("Model: %s  (label=%s)", type(model).__name__, model_label)

    # ------------------------------------------------------------------
    # Inference + scoring loop
    # ------------------------------------------------------------------
    example_scores: list[ExampleScore] = []
    latencies: list[float] = []

    for i, example in enumerate(examples):
        example_id = example.get("id", f"eval-{i:05d}")
        doc_text = example.get("document_text", "")
        ground_truth = example.get("ground_truth_json", {})

        try:
            raw_output, latency = model.extract(doc_text)
            latencies.append(latency)
        except Exception as exc:
            log.error("[%s] Inference failed: %s", example_id, exc)
            raw_output = ""
            latency = 0.0

        score = score_example(
            example_id=example_id,
            raw_model_output=raw_output,
            ground_truth=ground_truth,
        )
        example_scores.append(score)

        if (i + 1) % 25 == 0:
            running_schema_rate = sum(s.schema_valid for s in example_scores) / len(example_scores)
            log.info(
                "Progress: %d/%d  (schema valid so far: %.1f%%)",
                i + 1, len(examples), running_schema_rate * 100,
            )

        if inter_request_delay > 0 and i < len(examples) - 1:
            time.sleep(inter_request_delay)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    agg = aggregate_scores(example_scores)

    # Add latency stats
    if latencies:
        latencies_sorted = sorted(latencies)
        p50_idx = len(latencies_sorted) // 2
        p95_idx = int(len(latencies_sorted) * 0.95)
        p50_latency = latencies_sorted[p50_idx]
        p95_latency = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]
    else:
        p50_latency = p95_latency = 0.0

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_doc = {
        "model_label": model_label,
        "model_class": type(model).__name__,
        "n_examples": agg.n_examples,
        "aggregate": agg.to_dict(),
        "latency_p50_seconds": round(p50_latency, 3),
        "latency_p95_seconds": round(p95_latency, 3),
        "per_example": [s.to_dict() for s in example_scores],
        "eval_path": str(eval_path),
    }

    output_path.write_text(json.dumps(results_doc, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    log.info("=" * 65)
    log.info("EVAL RESULTS — %s", model_label.upper())
    log.info("  Examples:              %d", agg.n_examples)
    log.info("  Schema validity:       %.1f%%", agg.schema_validity_rate * 100)
    log.info("  Full-record match:     %.1f%%", agg.full_record_exact_match_rate * 100)
    log.info("  Overall field acc:     %.1f%%", agg.overall_field_accuracy * 100)
    log.info("  Parse failures:        %d", agg.parse_failures)
    log.info("  p50 latency:           %.3fs", p50_latency)
    log.info("  p95 latency:           %.3fs", p95_latency)
    log.info("  Per-field accuracy:")
    for fname, acc in sorted(agg.field_accuracy_by_field.items(), key=lambda x: x[1]):
        log.info("    %-20s  %.1f%%", fname, acc * 100)
    log.info("  Results written to: %s", output_path)
    log.info("=" * 65)

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run eval against eval_locked.jsonl for one model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-type", choices=["teacher", "base", "finetuned"], required=True,
        help="Which model to evaluate.",
    )
    p.add_argument(
        "--openrouter-model", default=None,
        help=(
            "OpenRouter model slug for --model-type teacher. "
            "Defaults to OPENROUTER_MODEL env var, then deepseek/deepseek-chat-v4-flash."
        ),
    )
    p.add_argument(
        "--hf-model", default=None,
        help="HuggingFace model ID or local path for --model-type base or finetuned.",
    )
    p.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load HuggingFace model in 4-bit (for VRAM-constrained inference).",
    )
    p.add_argument(
        "--eval-path", type=Path, default=EVAL_LOCKED_PATH,
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path.  Defaults to evaluation/results/{model_type}_model_results.json",
    )
    p.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between inference calls.",
    )
    p.add_argument(
        "--no-checksum-verify", action="store_true",
        help="Skip eval set integrity check.  Do NOT use this in production runs.",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    # Build model
    if args.model_type == "teacher":
        model: ModelInterface = OpenRouterModel(model_name=args.openrouter_model)
        label = f"teacher ({model.model_name})"
    else:
        if not args.hf_model:
            raise ValueError("--hf-model is required for --model-type base and finetuned")
        model = HuggingFaceModel(
            model_name_or_path=args.hf_model,
            load_in_4bit=args.load_in_4bit,
        )
        label = args.model_type

    # Default output path
    output_path = args.output or (
        RESULTS_DIR / f"{args.model_type}_model_results.json"
    )

    run_eval(
        model=model,
        eval_path=args.eval_path,
        output_path=output_path,
        model_label=label,
        inter_request_delay=args.delay,
        verify_eval_checksum=not args.no_checksum_verify,
    )
