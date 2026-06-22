"""
Schema validation step for the Forge curation pipeline.

Reads raw_generated.jsonl and reports which examples pass / fail Pydantic
validation against InvoiceExtraction.  Failing examples are written to a
separate file for manual inspection rather than silently dropped.

Usage
-----
    python -m data_curation.validate_schema \\
        --input data/raw/raw_generated.jsonl \\
        --output data/curated/schema_valid.jsonl \\
        --failures data/curated/schema_failures.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema.extraction_schema import InvoiceExtraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_example(example: dict) -> tuple[bool, str | None]:
    """
    Validate the ground_truth_json field of a raw example.

    Returns
    -------
    (is_valid, error_message)
        is_valid      : True if ground_truth_json parses into InvoiceExtraction.
        error_message : Human-readable error summary (None if valid).
    """
    raw_gt = example.get("ground_truth_json")
    if raw_gt is None:
        return False, "Missing 'ground_truth_json' key"
    if not isinstance(raw_gt, dict):
        return False, f"ground_truth_json is not a dict, got {type(raw_gt).__name__}"

    try:
        InvoiceExtraction.model_validate(raw_gt)
        return True, None
    except ValidationError as exc:
        # Summarise the first 3 errors to keep logs readable
        errors = exc.errors()[:3]
        summary = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in errors
        )
        if len(exc.errors()) > 3:
            summary += f" (+ {len(exc.errors()) - 3} more)"
        return False, summary


def validate_schema(
    input_path: Path,
    output_path: Path,
    failures_path: Path,
) -> dict:
    """
    Read all examples from input_path, validate each, and write two output files.

    Parameters
    ----------
    input_path    : Raw generated JSONL.
    output_path   : JSONL of examples that passed validation.
    failures_path : JSONL of examples that failed, with an added 'validation_error' field.

    Returns
    -------
    dict with counts (n_total, n_valid, n_failed, by_scenario, by_difficulty).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict = {
        "n_total": 0,
        "n_valid": 0,
        "n_failed": 0,
        "by_scenario": {},
        "by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
        "failure_reasons": {},
    }

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
        failures_path.open("w", encoding="utf-8") as ffail,
    ):
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Line %d: JSON decode error — %s", line_no, exc)
                stats["n_failed"] += 1
                stats["n_total"] += 1
                continue

            stats["n_total"] += 1
            is_valid, error_msg = validate_example(example)

            scenario_id = example.get("scenario_id", "unknown")
            difficulty = example.get("difficulty", "unknown")

            if is_valid:
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats["n_valid"] += 1
                stats["by_scenario"][scenario_id] = stats["by_scenario"].get(scenario_id, 0) + 1
                if difficulty in stats["by_difficulty"]:
                    stats["by_difficulty"][difficulty] += 1
            else:
                failed_example = {**example, "validation_error": error_msg}
                ffail.write(json.dumps(failed_example, ensure_ascii=False) + "\n")
                stats["n_failed"] += 1
                # Track failure reason categories
                reason_key = error_msg.split(":")[0] if error_msg else "unknown"
                stats["failure_reasons"][reason_key] = (
                    stats["failure_reasons"].get(reason_key, 0) + 1
                )
                log.debug("[%s] Validation failed: %s", example.get("id", "?"), error_msg)

    pass_rate = stats["n_valid"] / max(stats["n_total"], 1)
    log.info("Schema validation complete:")
    log.info("  Total:      %d", stats["n_total"])
    log.info("  Valid:      %d  (%.1f%%)", stats["n_valid"], pass_rate * 100)
    log.info("  Failed:     %d", stats["n_failed"])
    log.info("  Valid →     %s", output_path)
    log.info("  Failures →  %s", failures_path)

    if stats["failure_reasons"]:
        log.info("  Top failure reasons:")
        for reason, count in sorted(
            stats["failure_reasons"].items(), key=lambda x: -x[1]
        )[:5]:
            log.info("    %-50s  %d", reason, count)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate ground_truth_json against InvoiceExtraction schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",    type=Path, default=Path("data/raw/raw_generated.jsonl"))
    p.add_argument("--output",   type=Path, default=Path("data/curated/schema_valid.jsonl"))
    p.add_argument("--failures", type=Path, default=Path("data/curated/schema_failures.jsonl"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validate_schema(args.input, args.output, args.failures)
