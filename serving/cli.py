"""
CLI for the fine-tuned Forge invoice extraction model.

Reads a document from stdin or a file and outputs structured JSON.
Designed to be the runnable demo artifact that a stranger can use
after setting up Ollama with the exported GGUF model.

Usage
-----
    # From stdin:
    echo "INVOICE\\nAcme Corp..." | python -m serving.cli

    # From file:
    python -m serving.cli --input path/to/invoice.txt

    # Pretty-print output:
    python -m serving.cli --input invoice.txt --pretty

    # Using HuggingFace model directly (if Ollama not set up):
    python -m serving.cli --hf-model path/to/merged_model --input invoice.txt

Prerequisites
-------------
Either:
  a) Ollama installed + GGUF model loaded:
       ollama pull forge-invoice  (after running serving/export_gguf.py)
  b) HuggingFace merged model downloaded + GPU/CPU available
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema.extraction_schema import InvoiceExtraction


def _run_extraction(document_text: str, model) -> tuple[dict, float, bool]:
    """
    Run extraction and parse output.

    Returns (extraction_dict, latency_seconds, schema_valid).
    """
    t0 = time.monotonic()
    raw_output, _ = model.extract(document_text)
    latency = time.monotonic() - t0

    raw = raw_output.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        parsed = json.loads(raw)
        inv = InvoiceExtraction.model_validate(parsed)
        return inv.model_dump(), latency, True
    except Exception:
        try:
            return json.loads(raw), latency, False
        except Exception:
            return {"error": "Could not parse model output", "raw": raw_output[:500]}, latency, False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract structured invoice data from a text document.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", type=Path, default=None,
        help="Path to input text file.  Reads from stdin if not provided.",
    )
    p.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the output JSON (default: compact single line).",
    )
    p.add_argument(
        "--show-latency", action="store_true",
        help="Print latency to stderr after extraction.",
    )
    p.add_argument(
        "--ollama-model", default=os.environ.get("OLLAMA_MODEL", "forge-invoice"),
        help="Ollama model name for local inference.",
    )
    p.add_argument(
        "--hf-model", default=None,
        help="HuggingFace model path (overrides Ollama).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Read document
    if args.input:
        document_text = args.input.read_text(encoding="utf-8")
    else:
        document_text = sys.stdin.read()

    if not document_text.strip():
        print("Error: No document text provided.", file=sys.stderr)
        sys.exit(1)

    # Load model
    if args.hf_model:
        from evaluation.run_eval import HuggingFaceModel
        model = HuggingFaceModel(model_name_or_path=args.hf_model)
        model_label = f"hf/{Path(args.hf_model).name}"
    else:
        try:
            import requests

            class OllamaModel:
                def __init__(self, model_name: str) -> None:
                    self.model_name = model_name
                    self._base_url = "http://localhost:11434"

                def extract(self, document_text: str) -> tuple[str, float]:
                    from training.format_dataset import SYSTEM_PROMPT, USER_INSTRUCTION_PREFIX
                    prompt = f"{USER_INSTRUCTION_PREFIX}\n\n{document_text}"
                    t0 = time.monotonic()
                    response = requests.post(
                        f"{self._base_url}/api/chat",
                        json={
                            "model": self.model_name,
                            "messages": [
                                {"role": "system",    "content": SYSTEM_PROMPT},
                                {"role": "user",      "content": prompt},
                            ],
                            "stream": False,
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    raw = response.json()["message"]["content"]
                    return raw, time.monotonic() - t0

            model = OllamaModel(args.ollama_model)
            model_label = f"ollama/{args.ollama_model}"
        except Exception as exc:
            print(
                f"Error: Could not connect to Ollama model '{args.ollama_model}'.\n"
                f"  Make sure Ollama is running and the model is loaded.\n"
                f"  Or use --hf-model path/to/model for direct HuggingFace inference.\n"
                f"  Details: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Run extraction
    extraction, latency, schema_valid = _run_extraction(document_text, model)

    # Output
    indent = 2 if args.pretty else None
    print(json.dumps(extraction, indent=indent, ensure_ascii=False))

    if args.show_latency:
        validity_str = "✓ schema valid" if schema_valid else "✗ schema invalid"
        print(
            f"\n[{model_label}]  {latency:.3f}s  {validity_str}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
