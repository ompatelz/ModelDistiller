"""
Invoice Extraction — Local Inference Script
============================================
Runs your fine-tuned Qwen 2.5 3B LoRA model locally on Windows with CUDA.

Usage
-----
    # Interactive mode (paste invoice text, press Enter twice):
    py -3 infer.py

    # From a text file:
    py -3 infer.py --input path/to/invoice.txt

    # Pretty-print output:
    py -3 infer.py --input invoice.txt --pretty

    # Use a different LoRA path:
    py -3 infer.py --lora models/lora_model --input invoice.txt

Requirements (install once)
---------------------------
    pip install transformers peft accelerate bitsandbytes torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH   = "models/lora_model"
MAX_TOKENS  = 1024
TEMPERATURE = 0.1

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
# ─────────────────────────────────────────────────────────────────────────────


def load_model(lora_path: str):
    """Load base model + LoRA adapter. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading base model: {BASE_MODEL}")
    print("(This will download ~2GB on first run — subsequent runs are instant)\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Applying LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    print("Model ready!\n" + "─" * 60)
    return model, tokenizer


def extract(document_text: str, model, tokenizer) -> tuple[dict, float]:
    """
    Run invoice extraction on document_text.
    Returns (parsed_dict, latency_seconds).
    """
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Extract structured invoice data from the following document:\n\n{document_text}"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    t0 = time.monotonic()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.monotonic() - t0

    # Decode only the newly generated tokens
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Strip markdown fences if model adds them
    generated = re.sub(r"```(?:json)?", "", generated).strip()
    if generated.endswith("```"):
        generated = generated[:-3].strip()

    # Parse JSON
    try:
        result = json.loads(generated)
    except json.JSONDecodeError:
        start, end = generated.find("{"), generated.rfind("}")
        if start != -1 and end != -1:
            try:
                result = json.loads(generated[start:end+1])
            except json.JSONDecodeError:
                result = {"error": "Could not parse output", "raw": generated[:500]}
        else:
            result = {"error": "No JSON found in output", "raw": generated[:500]}

    return result, latency


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract structured invoice data using fine-tuned Qwen 2.5 3B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  type=Path, default=None,
                   help="Path to a .txt invoice file. Uses interactive mode if omitted.")
    p.add_argument("--lora",   default=LORA_PATH,
                   help="Path to the LoRA adapter folder.")
    p.add_argument("--pretty", action="store_true",
                   help="Pretty-print the output JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load model once
    model, tokenizer = load_model(args.lora)

    if args.input:
        # ── File mode ─────────────────────────────────────────────────────
        document_text = args.input.read_text(encoding="utf-8")
        result, latency = extract(document_text, model, tokenizer)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent, ensure_ascii=False))
        print(f"\n✓ Done in {latency:.2f}s", file=sys.stderr)

    else:
        # ── Interactive loop ───────────────────────────────────────────────
        print("Interactive mode — paste your invoice text below.")
        print("Press ENTER twice (blank line) to extract. Type 'quit' to exit.\n")

        while True:
            print("─" * 60)
            print("Paste invoice text (blank line to submit):")
            lines = []
            try:
                while True:
                    line = input()
                    if line.strip().lower() == "quit":
                        print("Goodbye!")
                        return
                    if line == "" and lines:
                        break
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return

            document_text = "\n".join(lines).strip()
            if not document_text:
                continue

            print("\nExtracting...")
            result, latency = extract(document_text, model, tokenizer)

            indent = 2 if args.pretty else None
            print("\n" + json.dumps(result, indent=indent, ensure_ascii=False))
            print(f"\n✓ Done in {latency:.2f}s\n")


if __name__ == "__main__":
    main()
