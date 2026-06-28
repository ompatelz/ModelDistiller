"""
Training data formatter for Forge.

Converts curated {document_text, ground_truth_json} pairs into the
instruction-tuning format expected by the student model.

Model-specific format: Qwen2.5-Instruct (ChatML)
-------------------------------------------------
Qwen2.5-Instruct models use the ChatML conversation format natively.
The correct format for a single-turn extraction task is:

    <|im_start|>system
    {system_prompt}
    <|im_end|>
    <|im_start|>user
    {document_text}
    <|im_end|>
    <|im_start|>assistant
    {ground_truth_json}
    <|im_end|>

When using Unsloth's SFTTrainer with `train_on_responses_only=True`, the
trainer automatically masks the system + user tokens so the loss is only
computed on the assistant (ground truth) tokens.  This is the standard
and current recommended approach for Qwen2.5 with Unsloth.

NOTE: This formatting is applied inside the Colab training notebook via
Unsloth's built-in `get_chat_template` / `apply_chat_template` utilities.
This script produces an intermediate JSONL in ShareGPT / "conversations" format
which Unsloth's dataset utilities expect — verify against current Unsloth docs
at training time.

Output format (one JSON object per line):
-----------------------------------------
    {
        "id": "gen-00042",
        "conversations": [
            {"from": "system",    "value": "<system prompt>"},
            {"from": "human",     "value": "<document text with extraction instruction>"},
            {"from": "gpt",       "value": "<ground truth JSON string>"}
        ],
        "difficulty": "medium",
        "scenario_id": "consulting_milestone_medium"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (shared between training formatter and eval runner)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a structured data extraction assistant. Your task is to extract invoice
and receipt information from plain-text documents and return the result as a
valid JSON object — nothing else.

Rules:
- Return ONLY the JSON object. No explanation, no markdown fences, no extra text.
- Use null for any field not present in the document.
- Normalize dates to YYYY-MM-DD format regardless of the source format.
- tax_rate must be a decimal fraction (0.08 for 8%, not 8.0).
- currency must be a 3-letter ISO 4217 code (USD, EUR, GBP, CAD, etc.).
- line_items must contain at least one entry.\
"""

USER_INSTRUCTION_PREFIX = """\
Extract structured invoice data from the following document:\
"""

# ---------------------------------------------------------------------------
# Formatting functions
# ---------------------------------------------------------------------------

def format_example(example: dict) -> dict | None:
    """
    Convert one raw curated example into the ShareGPT conversation format.

    Returns None if the example is malformed.
    """
    doc_text = example.get("document_text", "").strip()
    gt = example.get("ground_truth_json")

    if not doc_text:
        log.warning("[%s] Empty document_text — skipping", example.get("id"))
        return None
    if not gt:
        log.warning("[%s] Missing ground_truth_json — skipping", example.get("id"))
        return None

    # Ground truth must be formatted as a JSON string (compact, no extra whitespace)
    # This is what the model is trained to produce
    try:
        gt_json_str = json.dumps(gt, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        log.warning("[%s] Could not serialize ground_truth_json: %s", example.get("id"), exc)
        return None

    user_message = f"{USER_INSTRUCTION_PREFIX}\n\n{doc_text}"

    return {
        "id": example.get("id", ""),
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": user_message},
            {"from": "gpt",    "value": gt_json_str},
        ],
        "difficulty": example.get("difficulty", ""),
        "scenario_id": example.get("scenario_id", ""),
    }


def format_dataset(
    input_path: Path,
    output_path: Path,
) -> dict:
    """
    Read curated JSONL and write instruction-tuned JSONL in ShareGPT format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"n_input": 0, "n_output": 0, "n_skipped": 0}

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["n_input"] += 1
            formatted = format_example(example)

            if formatted is None:
                stats["n_skipped"] += 1
            else:
                fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                stats["n_output"] += 1

    log.info("Dataset formatting complete:")
    log.info("  Input:   %d", stats["n_input"])
    log.info("  Output:  %d", stats["n_output"])
    log.info("  Skipped: %d", stats["n_skipped"])
    log.info("  Written: %s", output_path)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Format curated data into instruction-tuning (ShareGPT) JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/train_formatted.jsonl"))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    args = _parse_args()
    format_dataset(args.input, args.output)
