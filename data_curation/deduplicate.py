"""
Near-duplicate detection for the Forge curation pipeline.

Reads schema-validated examples and removes near-duplicates.

Strategy
--------
The goal is to detect *structurally* similar documents, not just documents
with the same vendor name.  A naive invoice template that was run 200 times
with different amounts will have very similar text structure even though the
numbers differ.

Approach: character-level shingle Jaccard similarity on *normalized* document
text.  Normalization strips all digits, collapses punctuation, and lowercases,
so that two invoices with the same structure but different amounts/dates are
correctly identified as near-duplicates.

Parameters
----------
SIMILARITY_THRESHOLD : float
    Jaccard similarity above which two examples are considered near-duplicates.
    Default 0.72 — tuned to catch structural repetition while allowing two
    invoices from the same industry / format to coexist if they're genuinely
    different in structure.

SHINGLE_SIZE : int
    Character n-gram size.  5 works well for invoice text (captures phrase-level
    structure without being too sensitive to individual word changes).

Complexity
----------
O(n²) pairwise comparison — fine for n ≤ 2000 examples.  For larger datasets
switch to MinHash LSH (datasketch library) but that adds a dependency we don't
need at this scale.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD: float = 0.72
SHINGLE_SIZE: int = 5


# ---------------------------------------------------------------------------
# Shingling and similarity helpers
# ---------------------------------------------------------------------------

_DIGIT_RE = re.compile(r"\d")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Normalize invoice text for structural comparison.

    1. Lowercase
    2. Remove all digits (amounts / dates become invisible)
    3. Remove punctuation
    4. Collapse whitespace
    """
    text = text.lower()
    text = _DIGIT_RE.sub("", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def shingle_set(text: str, k: int = SHINGLE_SIZE) -> frozenset[str]:
    """Return the set of character k-grams from ``text``."""
    if len(text) < k:
        return frozenset({text})
    return frozenset(text[i : i + k] for i in range(len(text) - k + 1))


def jaccard_similarity(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """Jaccard similarity between two shingle sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Deduplication logic
# ---------------------------------------------------------------------------

def deduplicate(
    examples: list[dict],
    threshold: float = SIMILARITY_THRESHOLD,
    shingle_size: int = SHINGLE_SIZE,
) -> tuple[list[dict], list[dict]]:
    """
    Remove near-duplicate examples from ``examples``.

    Returns
    -------
    (kept, removed)
        kept    : Examples to keep (one representative per cluster).
        removed : Duplicates, each with an added 'duplicate_of' field.

    Algorithm
    ---------
    1. Compute normalised shingle sets for all examples.
    2. O(n²) pairwise comparison: if Jaccard ≥ threshold, mark the later
       example as a duplicate of the earlier one.
    3. Return the non-duplicate set.
    """
    n = len(examples)
    log.info("Computing shingles for %d examples (k=%d) ...", n, shingle_size)

    shingles: list[frozenset[str]] = []
    for ex in examples:
        doc_text = ex.get("document_text", "")
        normalized = normalize_text(doc_text)
        shingles.append(shingle_set(normalized, k=shingle_size))

    log.info("Running pairwise similarity (O(n²) for n=%d) ...", n)

    is_duplicate = [False] * n
    duplicate_of = [""] * n

    for i in range(n):
        if is_duplicate[i]:
            continue
        for j in range(i + 1, n):
            if is_duplicate[j]:
                continue
            sim = jaccard_similarity(shingles[i], shingles[j])
            if sim >= threshold:
                is_duplicate[j] = True
                duplicate_of[j] = examples[i].get("id", str(i))
                log.debug(
                    "Near-duplicate: %s ≈ %s  (Jaccard=%.3f)",
                    examples[j].get("id", str(j)),
                    examples[i].get("id", str(i)),
                    sim,
                )

    kept = [ex for i, ex in enumerate(examples) if not is_duplicate[i]]
    removed = [
        {**ex, "duplicate_of": duplicate_of[i], "jaccard_threshold": threshold}
        for i, ex in enumerate(examples)
        if is_duplicate[i]
    ]

    return kept, removed


def run_deduplication(
    input_path: Path,
    output_path: Path,
    removed_path: Path,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Read JSONL from input_path, deduplicate, write kept and removed to separate files.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading examples from %s ...", input_path)
    examples: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Skipping malformed JSON line")

    if not examples:
        log.warning("No examples loaded from %s", input_path)
        return {"n_input": 0, "n_kept": 0, "n_removed": 0}

    kept, removed = deduplicate(examples, threshold=threshold)

    with output_path.open("w", encoding="utf-8") as fout:
        for ex in kept:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    removed_path.parent.mkdir(parents=True, exist_ok=True)
    with removed_path.open("w", encoding="utf-8") as fout:
        for ex in removed:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {
        "n_input": len(examples),
        "n_kept": len(kept),
        "n_removed": len(removed),
        "removal_rate": len(removed) / max(len(examples), 1),
        "threshold": threshold,
    }

    log.info("Deduplication complete:")
    log.info("  Input:   %d", stats["n_input"])
    log.info("  Kept:    %d  (%.1f%%)", stats["n_kept"], (1 - stats["removal_rate"]) * 100)
    log.info("  Removed: %d  (%.1f%%)", stats["n_removed"], stats["removal_rate"] * 100)
    log.info("  Kept  →  %s", output_path)
    log.info("  Removed → %s", removed_path)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove near-duplicate invoice documents (Jaccard shingle similarity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",     type=Path, default=Path("data/curated/schema_valid.jsonl"))
    p.add_argument("--output",    type=Path, default=Path("data/curated/deduped.jsonl"))
    p.add_argument("--removed",   type=Path, default=Path("data/curated/duplicates_removed.jsonl"))
    p.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                   help="Jaccard similarity threshold above which an example is a near-duplicate.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    args = _parse_args()
    run_deduplication(args.input, args.output, args.removed, args.threshold)
