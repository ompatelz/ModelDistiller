"""
GGUF export script for Forge.

Exports the merged fine-tuned model to GGUF format for use with Ollama.

IMPORTANT — confirm current Unsloth export API before running
--------------------------------------------------------------
The GGUF export path in Unsloth has changed across versions.  The method below
reflects the pattern as of Unsloth v2024.x, but MUST be verified against the
current Unsloth documentation before running.  Look for:
  - The current save_pretrained_gguf() signature
  - Whether llama.cpp is bundled or needs to be installed separately
  - The current list of supported quantization types (q4_k_m is standard)

This script is designed to run in the SAME Colab environment where training
happened (the model is already loaded in memory), or by loading the merged
checkpoint fresh.

Usage (inside Colab, after training)
-------------------------------------
    # Typically called at the end of train_colab.ipynb:
    from serving.export_gguf import export_to_gguf
    export_to_gguf(model, tokenizer, output_path="forge-invoice")

Usage (standalone, from a saved merged checkpoint)
---------------------------------------------------
    python -m serving.export_gguf \\
        --model-path training/merged_model \\
        --output-name forge-invoice \\
        --quantization q4_k_m
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export via Unsloth (recommended path — run inside Colab after training)
# ---------------------------------------------------------------------------

def export_to_gguf_via_unsloth(
    model,          # FastLanguageModel or FastModel instance from Unsloth
    tokenizer,
    output_name: str = "forge-invoice",
    quantization: str = "q4_k_m",
) -> None:
    """
    Export using Unsloth's built-in GGUF export utility.

    CONFIRM-AT-TRAINING-TIME: verify the current save_pretrained_gguf API.
    The method signature and supported quantization types change with Unsloth
    versions.  Check: https://docs.unsloth.ai/

    Parameters
    ----------
    model          : The Unsloth model object (still in memory after training).
    tokenizer      : The Unsloth tokenizer object.
    output_name    : Ollama model name / directory prefix.
    quantization   : GGUF quantization type.
                     Common options: q4_k_m (good balance), q8_0 (higher quality),
                     q2_k (smallest, lower quality), f16 (no quantization).
    """
    log.info("Exporting to GGUF via Unsloth (quantization=%s) ...", quantization)
    log.info("NOTE: Verify save_pretrained_gguf signature against current Unsloth docs")

    # CONFIRM-AT-TRAINING-TIME: this call signature may have changed
    model.save_pretrained_gguf(
        output_name,
        tokenizer,
        quantization_method=quantization,
    )

    log.info("GGUF export complete: %s.gguf (or similar)", output_name)
    log.info("To load in Ollama, run:")
    log.info("  ollama create forge-invoice -f Modelfile")
    log.info("  (create a Modelfile that points to the .gguf file)")


# ---------------------------------------------------------------------------
# Modelfile generator for Ollama
# ---------------------------------------------------------------------------

def generate_ollama_modelfile(
    gguf_path: str,
    model_name: str = "forge-invoice",
    output_path: Path = Path("serving/Modelfile"),
) -> None:
    """
    Generate an Ollama Modelfile for the exported GGUF.

    After generating, run:
        ollama create forge-invoice -f serving/Modelfile
        ollama run forge-invoice
    """
    from training.format_dataset import SYSTEM_PROMPT

    modelfile_content = f"""\
FROM {gguf_path}

SYSTEM \"\"\"
{SYSTEM_PROMPT}
\"\"\"

PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER stop "<|im_end|>"
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(modelfile_content, encoding="utf-8")
    log.info("Modelfile written to %s", output_path)
    log.info("Next steps:")
    log.info("  1. ollama create %s -f %s", model_name, output_path)
    log.info("  2. ollama run %s", model_name)
    log.info("  3. python -m serving.cli --ollama-model %s --input invoice.txt", model_name)


# ---------------------------------------------------------------------------
# CLI (for standalone use)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export merged fine-tuned model to GGUF for Ollama. "
            "Typically run inside Colab after training; see docstring."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path",   type=str,  default="training/merged_model")
    p.add_argument("--output-name",  type=str,  default="forge-invoice")
    p.add_argument("--quantization", type=str,  default="q4_k_m")
    p.add_argument(
        "--generate-modelfile-only", action="store_true",
        help="Only generate the Ollama Modelfile (skip the export step).",
    )
    p.add_argument("--gguf-path", type=str, default=None,
                   help="Path to existing .gguf file (for --generate-modelfile-only).")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    args = _parse_args()

    if args.generate_modelfile_only:
        gguf_path = args.gguf_path or f"{args.output_name}.gguf"
        generate_ollama_modelfile(gguf_path=gguf_path, model_name=args.output_name)
    else:
        log.info(
            "Standalone GGUF export requires the model to be loaded in memory.\n"
            "Call export_to_gguf_via_unsloth() from inside the Colab notebook.\n"
            "See serving/export_gguf.py docstring for details."
        )
