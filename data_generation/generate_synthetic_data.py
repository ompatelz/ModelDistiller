"""
Synthetic data generation script for Forge.

Calls the OpenRouter API (OpenAI-compatible) to produce
{document_text, ground_truth_json} pairs using the 20 scenario configurations
defined in data_generation/prompts.py.

Usage
-----
    python -m data_generation.generate_synthetic_data \\
        --n-examples 1000 \\
        --output-dir data/raw \\
        --model deepseek/deepseek-chat-v4-flash \\
        --seed 42

Output
------
Each run appends to (or creates) ``data/raw/raw_generated.jsonl``.  Each line
is a JSON object with the following fields:

    {
        "id":               str,   # unique example ID, e.g. "gen-0042"
        "scenario_id":      str,   # which scenario config produced this
        "difficulty":       str,   # easy | medium | hard
        "document_text":    str,   # the synthetic invoice document
        "ground_truth_json": dict, # the extraction — conforms to InvoiceExtraction
        "model":            str,   # which Claude model generated this
        "generated_at":     str,   # ISO 8601 timestamp
        "generation_ms":    int    # wall-clock time for this API call
    }

A brief summary JSON is also written to ``data/raw/generation_run_{timestamp}.json``
with counts, cost estimates, and any parse failures logged.

Cost note
---------
DeepSeek V4 Flash via OpenRouter is used for both generation and teacher eval.
OpenRouter uses the OpenAI-compatible API format — the ``openai`` Python SDK
points at ``https://openrouter.ai/api/v1`` with your ``OPENROUTER_API_KEY``.

Required env vars
-----------------
    OPENROUTER_API_KEY   — your OpenRouter key (https://openrouter.ai/keys)
    OPENROUTER_MODEL     — model slug, e.g. deepseek/deepseek-chat-v4-flash
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "openai package is required for OpenRouter generation.\n"
        "Install with: pip install openai"
    ) from exc
from pydantic import ValidationError

# Ensure project root is on the path when run as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generation.prompts import (
    SCENARIO_CONFIGS,
    ScenarioConfig,
    build_generation_prompt,
    sample_scenario,
)
from schema.extraction_schema import InvoiceExtraction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _call_openrouter(
    client: "OpenAI",
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> tuple[str, int]:
    """
    Call the OpenRouter API and return (raw_text_response, input_tokens_used).

    OpenRouter is OpenAI-compatible — uses the same chat completions endpoint.
    Retries on transient errors with exponential back-off.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            tokens = response.usage.prompt_tokens if response.usage else 0
            return text, tokens
        except Exception as exc:
            # openai SDK raises openai.RateLimitError, openai.APIError, etc.
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt * 3
            log.warning("API error (attempt %d/%d) — sleeping %ds: %s", attempt + 1, max_retries, wait, exc)
            time.sleep(wait)

    raise RuntimeError(f"All {max_retries} API attempts failed")


def _parse_response(raw_text: str) -> dict | None:
    """
    Parse Claude's response into a dict with 'document_text' and 'ground_truth_json'.

    Returns None if parsing fails (the caller logs and skips the example).
    Claude is prompted to return raw JSON, but occasionally adds prose — this
    function strips common wrappers before attempting to parse.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = inner.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None
    if "document_text" not in parsed or "ground_truth_json" not in parsed:
        return None
    if not isinstance(parsed["document_text"], str):
        return None
    if not isinstance(parsed["ground_truth_json"], dict):
        return None

    return parsed


def _validate_ground_truth(ground_truth: dict) -> bool:
    """
    Attempt to parse ground_truth into InvoiceExtraction.

    Returns True if valid, False otherwise.  Does NOT raise — validation errors
    are a normal output of the curation step; here we just pre-filter the most
    egregiously broken examples before saving raw output.
    """
    try:
        InvoiceExtraction.model_validate(ground_truth)
        return True
    except ValidationError:
        return False


def generate_example(
    client: "OpenAI",
    config: ScenarioConfig,
    example_id: str,
    model: str,
) -> dict | None:
    """
    Generate one {document_text, ground_truth_json} example.

    Returns a fully-populated dict ready to write as a JSONL line, or None if
    the generation/parse/validation failed after retries.
    """
    sys_p, usr_p = build_generation_prompt(config)

    t0 = time.monotonic()
    try:
        raw_text, input_tokens = _call_openrouter(client, sys_p, usr_p, model)
    except Exception as exc:
        log.error("[%s] API call failed: %s", example_id, exc)
        return None
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    parsed = _parse_response(raw_text)
    if parsed is None:
        log.warning("[%s] Failed to parse JSON from response (scenario=%s)", example_id, config.scenario_id)
        return None

    is_valid = _validate_ground_truth(parsed["ground_truth_json"])
    if not is_valid:
        log.debug(
            "[%s] ground_truth_json failed schema validation (scenario=%s) — "
            "saving anyway; curation step will filter",
            example_id,
            config.scenario_id,
        )
        # We save it anyway — the curation pipeline will formally validate
        # and report attrition; pre-filtering here would hide that attrition count.

    return {
        "id": example_id,
        "scenario_id": config.scenario_id,
        "difficulty": config.difficulty,
        "document_text": parsed["document_text"],
        "ground_truth_json": parsed["ground_truth_json"],
        "schema_valid_at_generation": is_valid,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_ms": elapsed_ms,
        "input_tokens": input_tokens,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(
    n_examples: int,
    output_dir: Path,
    model: str,
    seed: int,
    inter_request_delay: float = 0.5,
) -> dict:
    """
    Generate ``n_examples`` invoice documents and write them to
    ``output_dir/raw_generated.jsonl``.

    Returns a summary dict with generation statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raw_generated.jsonl"

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set.  Export it before running:\n"
            "  export OPENROUTER_API_KEY=sk-or-v1-..."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    rng = random.Random(seed)

    log.info("Starting generation: %d examples, model=%s, seed=%d", n_examples, model, seed)
    log.info("Output: %s", output_path)

    stats = {
        "n_requested": n_examples,
        "n_generated": 0,
        "n_parse_failures": 0,
        "n_schema_invalid_at_generation": 0,
        "total_input_tokens": 0,
        "by_scenario": {},
        "by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    with output_path.open("a", encoding="utf-8") as fout:
        for i in range(n_examples):
            example_id = f"gen-{i:05d}"
            config = sample_scenario(rng)

            result = generate_example(client, config, example_id, model)

            if result is None:
                stats["n_parse_failures"] += 1
                log.warning("[%s] Skipped (parse/API failure)", example_id)
            else:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()  # Flush after each write — generation is slow, don't lose progress
                stats["n_generated"] += 1
                stats["total_input_tokens"] += result.get("input_tokens", 0)
                if not result["schema_valid_at_generation"]:
                    stats["n_schema_invalid_at_generation"] += 1
                stats["by_scenario"][config.scenario_id] = (
                    stats["by_scenario"].get(config.scenario_id, 0) + 1
                )
                stats["by_difficulty"][config.difficulty] += 1

                if (i + 1) % 50 == 0:
                    log.info(
                        "Progress: %d/%d generated, %d failures",
                        stats["n_generated"],
                        n_examples,
                        stats["n_parse_failures"],
                    )

            if i < n_examples - 1:
                time.sleep(inter_request_delay)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["success_rate"] = stats["n_generated"] / max(n_examples, 1)

    # Write run summary
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    summary_path = output_dir / f"generation_run_{timestamp}.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    log.info("=" * 60)
    log.info("Generation complete:")
    log.info("  Requested:       %d", stats["n_requested"])
    log.info("  Generated:       %d", stats["n_generated"])
    log.info("  Parse failures:  %d", stats["n_parse_failures"])
    log.info("  Success rate:    %.1f%%", stats["success_rate"] * 100)
    log.info("  Input tokens:    %d", stats["total_input_tokens"])
    log.info("  Summary saved:   %s", summary_path)
    log.info("=" * 60)

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic invoice documents via Claude API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-examples", type=int, default=1000,
        help="Number of examples to generate (target 800–1500 per PRD).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw"),
        help="Directory to write raw_generated.jsonl and run summary.",
    )
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"),
        help=(
            "OpenRouter model slug for generation. "
            "Defaults to OPENROUTER_MODEL env var, then deepseek/deepseek-chat-v4-flash. "
            "Find slugs at: https://openrouter.ai/models"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible scenario sampling.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to sleep between API requests (rate limit safety).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load .env from the repo root so OPENROUTER_API_KEY is available
    # without needing to manually export it in the shell.
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(Path(__file__).parent.parent / ".env")
    except ImportError:
        pass  # dotenv optional — rely on shell env if not installed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    generate_dataset(
        n_examples=args.n_examples,
        output_dir=args.output_dir,
        model=args.model,
        seed=args.seed,
        inter_request_delay=args.delay,
    )
