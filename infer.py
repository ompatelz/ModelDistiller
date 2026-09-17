"""
Invoice Extraction — Local & Multi-Engine CLI Inference Script
=============================================================
Runs invoice extraction using:
1. Fine-tuned Qwen 2.5 3B LoRA model locally on Windows/Linux with CUDA.
2. OpenRouter teacher API (DeepSeek/Claude) if requested.
3. Built-in Forge Neural Simulator (instant zero-dependency testing on any machine).

Usage
-----
    # Interactive mode (paste invoice text, press Enter twice):
    python infer.py

    # From a text file with pretty printing:
    python infer.py --input path/to/invoice.txt --pretty

    # Force simulator mode (offline, instant testing without GPU):
    python infer.py --simulator --input path/to/invoice.txt --pretty

    # Use OpenRouter teacher model:
    python infer.py --engine openrouter --input path/to/invoice.txt --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LORA_PATH   = os.environ.get("LORA_PATH", "models/lora_model" if Path("models/lora_model").exists() else "ompatelz/Forge-Qwen2.5-3B-Invoice-LoRA")
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


def load_model(lora_path: str):
    """Load base model + LoRA adapter. Returns (model, tokenizer) or raises Exception."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading base model: {BASE_MODEL}")
    print(f"Applying LoRA adapter from: {lora_path}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb_config = None
    if device == "cuda":
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
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    print("Model ready!\n" + "-" * 60)
    return model, tokenizer


def extract_with_lora(document_text: str, model, tokenizer) -> tuple[dict, float]:
    """Run invoice extraction on document_text using loaded LoRA model."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Extract structured invoice data from the following document:\n\n{document_text}"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

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

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    generated = re.sub(r"```(?:json)?", "", generated).strip().rstrip("`").strip()

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
        description="Extract structured invoice data using Forge Distilled Extractor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  type=Path, default=None,
                   help="Path to a .txt invoice file. Uses interactive mode if omitted.")
    p.add_argument("--lora",   default=LORA_PATH,
                   help="Path to the LoRA adapter folder.")
    p.add_argument("--engine", choices=["auto", "lora", "simulator", "openrouter"], default="auto",
                   help="Extraction engine backend.")
    p.add_argument("--simulator", action="store_true",
                   help="Shortcut for --engine simulator (offline, zero-GPU testing).")
    p.add_argument("--pretty", action="store_true",
                   help="Pretty-print the output JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine_choice = "simulator" if args.simulator else args.engine

    model = None
    tokenizer = None

    if engine_choice in ("lora", "auto") and not args.simulator:
        try:
            model, tokenizer = load_model(args.lora)
            print("✓ Loaded PyTorch LoRA model")
        except Exception as exc:
            if engine_choice == "lora":
                print(f"Error loading LoRA model: {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"[Notice] Local LoRA weights not found ({exc}). Using Forge Neural Simulator.\n")
            engine_choice = "simulator"

    def run_inference(doc: str) -> tuple[dict, float, str]:
        if model and tokenizer and engine_choice != "simulator":
            res, lat = extract_with_lora(doc, model, tokenizer)
            return res, lat, "LoRA Qwen2.5-3B"
        else:
            from serving.inference_engine import engine_manager
            res, valid, lat_ms, eng = engine_manager.extract(
                doc, engine=engine_choice
            )
            return res, lat_ms / 1000.0, eng

    if args.input:
        # File mode
        document_text = args.input.read_text(encoding="utf-8")
        result, latency, eng = run_inference(document_text)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent, ensure_ascii=False))
        print(f"\n✓ Extracted via {eng} in {latency:.2f}s", file=sys.stderr)

    else:
        # Interactive loop
        print("Interactive mode — paste your invoice text below.")
        print("Press ENTER twice (blank line) to extract. Type 'quit' to exit.\n")

        while True:
            print("-" * 60)
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
            result, latency, eng = run_inference(document_text)

            indent = 2 if args.pretty else None
            print("\n" + json.dumps(result, indent=indent, ensure_ascii=False))
            print(f"\n✓ Extracted via {eng} in {latency:.2f}s\n")


if __name__ == "__main__":
    main()
