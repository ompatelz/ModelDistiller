"""
Train / val / eval split for the Forge curation pipeline.

This is where the LOCKED EVAL SET is created.  Read the rules below carefully
before modifying this file.

Eval set rules (from PRD.md Section 6 and CONTEXT.md Section 5)
----------------------------------------------------------------
1.  eval_locked.jsonl is created ONCE by this script, before any training run.
2.  After creation, no example in this file is edited, removed, or "fixed"
    based on model performance.
3.  If a genuine labeling error is found later, it must be flagged to the user
    with the specific example and error — not silently patched.
4.  The eval lock is enforced mechanically: after writing eval_locked.jsonl,
    this script computes its SHA-256 hash and writes it to eval_locked.jsonl.sha256.
    The evaluation runner (evaluation/run_eval.py) verifies this hash before
    every eval run — if the file has been modified, the run aborts with an error.

Split strategy
--------------
Stratified by difficulty (easy / medium / hard) to ensure all three splits
have a representative mix.  The eval set is sampled first so that the
training set maximises training signal.

Default proportions (adjustable via CLI args):
  eval    : EVAL_FRACTION of total (min EVAL_MIN_EXAMPLES examples)
  val     : VAL_FRACTION of the remainder (used for training checkpointing)
  train   : the rest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default split parameters
# ---------------------------------------------------------------------------

EVAL_FRACTION: float = 0.12      # ~12% of curated set → eval
EVAL_MIN_EXAMPLES: int = 50      # Minimum eval examples (12% of ~900 curated ≈ 108, well above this)
VAL_FRACTION: float = 0.10       # 10% of remaining → val
RANDOM_SEED: int = 2024          # Fixed seed — reproducible, but separate from generation seed


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------

def _stratified_sample(
    examples: list[dict],
    n: int,
    rng: random.Random,
    difficulty_field: str = "difficulty",
) -> tuple[list[dict], list[dict]]:
    """
    Draw ``n`` examples from ``examples`` with stratification by difficulty.

    Returns (sampled, remainder).  Preserves difficulty distribution in the
    sampled set within ±1 example per stratum due to rounding.
    """
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_difficulty[ex.get(difficulty_field, "unknown")].append(ex)

    # Shuffle within each stratum
    for stratum in by_difficulty.values():
        rng.shuffle(stratum)

    # Compute per-stratum target counts (proportional)
    total = len(examples)
    sampled: list[dict] = []
    remainder: list[dict] = []

    for difficulty, stratum in by_difficulty.items():
        fraction = len(stratum) / max(total, 1)
        target = round(fraction * n)
        target = min(target, len(stratum))  # can't take more than available
        sampled.extend(stratum[:target])
        remainder.extend(stratum[target:])

    # If rounding left us short, top up from remainder (any difficulty)
    shortfall = n - len(sampled)
    if shortfall > 0:
        rng.shuffle(remainder)
        sampled.extend(remainder[:shortfall])
        remainder = remainder[shortfall:]

    # If we overshot, move extras back
    while len(sampled) > n and remainder is not None:
        remainder.append(sampled.pop())

    rng.shuffle(sampled)    # Final shuffle within the sample
    rng.shuffle(remainder)

    return sampled, remainder


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------

def _compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksum(data_path: Path) -> str:
    """Write SHA-256 of data_path to data_path.sha256 and return the digest."""
    digest = _compute_sha256(data_path)
    checksum_path = data_path.with_suffix(data_path.suffix + ".sha256")
    checksum_path.write_text(digest + "\n", encoding="utf-8")
    log.info("Eval set checksum written to %s", checksum_path)
    log.info("  SHA-256: %s", digest)
    return digest


def verify_checksum(data_path: Path) -> bool:
    """
    Verify data_path's SHA-256 matches the stored .sha256 file.

    Returns True if valid, False if tampered.  Called by run_eval.py before
    any eval run to enforce the eval-set integrity rule.
    """
    checksum_path = data_path.with_suffix(data_path.suffix + ".sha256")
    if not checksum_path.exists():
        log.error("No checksum file found at %s — was eval_locked.jsonl created by split_dataset.py?", checksum_path)
        return False
    stored = checksum_path.read_text(encoding="utf-8").strip()
    actual = _compute_sha256(data_path)
    if stored != actual:
        log.error(
            "EVAL SET INTEGRITY VIOLATION: eval_locked.jsonl has been modified!\n"
            "  Stored SHA-256:  %s\n"
            "  Actual SHA-256:  %s\n"
            "This is a serious issue — per PRD Section 6, the eval set must not\n"
            "be modified after locking.  If you believe there is a genuine labeling\n"
            "bug, flag it to the project owner with the specific example and error.",
            stored,
            actual,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------

def split_dataset(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    eval_path: Path,
    eval_fraction: float = EVAL_FRACTION,
    eval_min: int = EVAL_MIN_EXAMPLES,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Read quality-filtered examples, produce stratified train / val / eval splits.

    After writing eval_locked.jsonl, computes and saves its SHA-256 checksum.
    """
    rng = random.Random(seed)

    log.info("Loading examples from %s ...", input_path)
    examples: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    total = len(examples)
    log.info("Loaded %d examples", total)

    if total < eval_min:
        raise ValueError(
            f"Only {total} examples available — cannot create an eval set of "
            f"{eval_min} examples.  Generate more data or lower --eval-min."
        )

    # ------------------------------------------------------------------
    # 1. Eval split (sampled first, before any training touches it)
    # ------------------------------------------------------------------
    eval_target = max(eval_min, round(total * eval_fraction))
    eval_target = min(eval_target, total - eval_min)  # Don't eat all examples
    log.info("Eval target: %d examples (%.1f%% of %d)", eval_target, eval_target / total * 100, total)

    eval_examples, remainder = _stratified_sample(examples, eval_target, rng)

    # ------------------------------------------------------------------
    # 2. Val split
    # ------------------------------------------------------------------
    val_target = round(len(remainder) * val_fraction)
    val_target = max(val_target, 1)
    val_examples, train_examples = _stratified_sample(remainder, val_target, rng)

    # ------------------------------------------------------------------
    # 3. Write all three splits
    # ------------------------------------------------------------------
    for path, split_examples in [
        (train_path, train_examples),
        (val_path, val_examples),
        (eval_path, eval_examples),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ex in split_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("Wrote %d examples → %s", len(split_examples), path)

    # ------------------------------------------------------------------
    # 4. Lock the eval set with a SHA-256 checksum
    # ------------------------------------------------------------------
    digest = write_checksum(eval_path)

    # ------------------------------------------------------------------
    # 5. Print difficulty distribution for eval set (for user confirmation)
    # ------------------------------------------------------------------
    eval_by_diff: dict[str, int] = defaultdict(int)
    for ex in eval_examples:
        eval_by_diff[ex.get("difficulty", "unknown")] += 1

    log.info("=" * 60)
    log.info("EVAL SET LOCKED — do not modify eval_locked.jsonl")
    log.info("  Total examples in eval: %d", len(eval_examples))
    log.info("  Difficulty breakdown:")
    for diff in ("easy", "medium", "hard"):
        count = eval_by_diff.get(diff, 0)
        pct = count / max(len(eval_examples), 1) * 100
        log.info("    %-8s  %3d  (%.1f%%)", diff, count, pct)
    log.info("  SHA-256: %s", digest)
    log.info("=" * 60)

    stats = {
        "n_total": total,
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "n_eval": len(eval_examples),
        "eval_sha256": digest,
        "eval_difficulty": dict(eval_by_diff),
        "seed": seed,
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split curated data into train / val / eval_locked sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",         type=Path,  default=Path("data/curated/quality_filtered.jsonl"))
    p.add_argument("--train",         type=Path,  default=Path("data/train.jsonl"))
    p.add_argument("--val",           type=Path,  default=Path("data/val.jsonl"))
    p.add_argument("--eval",          type=Path,  default=Path("data/eval_locked.jsonl"))
    p.add_argument("--eval-fraction", type=float, default=EVAL_FRACTION)
    p.add_argument("--eval-min",      type=int,   default=EVAL_MIN_EXAMPLES)
    p.add_argument("--val-fraction",  type=float, default=VAL_FRACTION)
    p.add_argument("--seed",          type=int,   default=RANDOM_SEED)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    args = _parse_args()
    stats = split_dataset(
        input_path=args.input,
        train_path=args.train,
        val_path=args.val,
        eval_path=args.eval,
        eval_fraction=args.eval_fraction,
        eval_min=args.eval_min,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))
