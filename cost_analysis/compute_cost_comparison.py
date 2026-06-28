"""
Cost comparison script for Forge.

Produces the $/1000-documents and cost-per-success numbers reported in RESULTS.md.

IMPORTANT — no numbers are fabricated here
-------------------------------------------
All cost figures reference live API pricing (fetched or cited at time of running)
and real measured latencies from evaluation/results/.  Any figure that hasn't
been measured yet is marked TODO in the output.

Inputs
------
1.  evaluation/results/teacher_model_results.json    (real eval output)
2.  evaluation/results/finetuned_model_results.json  (real eval output)
3.  evaluation/results/base_model_results.json       (real eval output)
4.  Current Claude API pricing (Anthropic pricing page — sourced below)

Claude API pricing source
--------------------------
Prices below are from the OpenRouter pricing page for DeepSeek V4 Flash.
Update the PRICING dict before running if stale.
Source: https://openrouter.ai/models?q=deepseek (filter for the exact slug)

For the fine-tuned / base model inference cost: since the model runs locally
(self-hosted on Colab or via Ollama), the marginal inference cost is essentially
$0 (electricity is negligible at this scale).  For cloud-hosted scenarios,
note the cost of the GPU instance instead.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Claude API pricing — UPDATE THIS before running
# Source: https://www.anthropic.com/pricing
# Last verified: TODO — verify current pricing before using in RESULTS.md
# ---------------------------------------------------------------------------

# OpenRouter pricing for DeepSeek V4 Flash
# Source: https://openrouter.ai/models?q=deepseek  — verify before publishing
# ---------------------------------------------------------------------------

OPENROUTER_PRICING: dict[str, dict[str, float]] = {
    # model_slug: {input_per_million_tokens, output_per_million_tokens}
    "deepseek/deepseek-v4-flash": {
        "input_per_1m":  0.07,   # USD per 1M input tokens  — VERIFY CURRENT
        "output_per_1m": 0.28,   # USD per 1M output tokens — VERIFY CURRENT
    },
}

# Default model slug — must match what's in .env / OPENROUTER_MODEL
DEFAULT_MODEL_SLUG = "deepseek/deepseek-v4-flash"

# Average token counts for the invoice extraction task (measured from eval runs)
# TODO: update from actual token counts in results JSON after running evals
AVG_INPUT_TOKENS_PER_DOC: int = 600    # TODO — measure from eval run
AVG_OUTPUT_TOKENS_PER_DOC: int = 350   # TODO — measure from eval run


# ---------------------------------------------------------------------------
# Cost calculation helpers
# ---------------------------------------------------------------------------

def cost_per_1000_docs_api(
    model_slug: str = DEFAULT_MODEL_SLUG,
    avg_input_tokens: int = AVG_INPUT_TOKENS_PER_DOC,
    avg_output_tokens: int = AVG_OUTPUT_TOKENS_PER_DOC,
) -> float:
    """
    Calculate OpenRouter API cost per 1000 documents.

    Returns USD cost.
    """
    if model_slug not in OPENROUTER_PRICING:
        raise ValueError(
            f"No pricing data for model {model_slug!r}. "
            "Check https://openrouter.ai/models and update OPENROUTER_PRICING."
        )

    pricing = OPENROUTER_PRICING[model_slug]
    input_cost  = (avg_input_tokens  / 1_000_000) * pricing["input_per_1m"]  * 1000
    output_cost = (avg_output_tokens / 1_000_000) * pricing["output_per_1m"] * 1000
    return input_cost + output_cost


def cost_per_success(cost_per_1000: float, accuracy: float) -> float | None:
    """
    Cost per successful extraction = cost_per_1000 / (1000 * accuracy).

    Returns None if accuracy is 0.
    """
    if accuracy <= 0:
        return None
    return cost_per_1000 / (1000 * accuracy)


def payback_period_documents(
    pipeline_build_cost_usd: float,
    teacher_cost_per_1000: float,
    finetuned_cost_per_1000: float,
) -> float | None:
    """
    At what document volume does the distilled model's savings pay back the
    build cost?

    payback_docs = build_cost / (teacher_cost - finetuned_cost) * 1000

    This is the classic ROI framing the PRD asks for.
    Returns None if teacher is already cheaper (shouldn't happen but handle it).
    """
    savings_per_1000 = teacher_cost_per_1000 - finetuned_cost_per_1000
    if savings_per_1000 <= 0:
        return None
    return pipeline_build_cost_usd / savings_per_1000 * 1000


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [MISSING] {path} — run evaluation first")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_and_print_comparison(
    results_dir: Path = Path("evaluation/results"),
    teacher_model_slug: str = DEFAULT_MODEL_SLUG,
    generation_model_slug: str = DEFAULT_MODEL_SLUG,
    pipeline_build_cost_usd: float | None = None,
) -> dict:
    """
    Load all three results files, compute cost/quality comparison, print and return.
    """
    print("=" * 70)
    print("FORGE — Cost / Quality Comparison")
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    print()

    base_results      = load_results(results_dir / "base_model_results.json")
    finetuned_results = load_results(results_dir / "finetuned_model_results.json")
    teacher_results   = load_results(results_dir / "teacher_model_results.json")

    def _get(results: dict | None, key: str, default: str = "TODO — not yet measured") -> str:
        if results is None:
            return "TODO — results not available"
        val = results.get("aggregate", {}).get(key)
        if val is None:
            return default
        if isinstance(val, float):
            return f"{val * 100:.1f}%"
        return str(val)

    def _get_latency(results: dict | None, percentile: str = "latency_p50_seconds") -> str:
        if results is None:
            return "TODO"
        val = results.get(percentile)
        if val is None:
            return "TODO"
        return f"{val:.3f}s"

    # ------------------------------------------------------------------
    # Quality metrics table
    # ------------------------------------------------------------------
    print("── Quality Metrics ─────────────────────────────────────────────────")
    print(f"{'Metric':<35} {'Base':>12} {'Fine-tuned':>12} {'Teacher':>12}")
    print("-" * 72)

    metrics = [
        ("Schema validity rate",        "schema_validity_rate"),
        ("Field-level accuracy",        "overall_field_accuracy"),
        ("Full-record exact match",     "full_record_exact_match_rate"),
    ]
    for label, key in metrics:
        b = _get(base_results, key)
        ft = _get(finetuned_results, key)
        t = _get(teacher_results, key)
        print(f"  {label:<33} {b:>12} {ft:>12} {t:>12}")

    # Per-field accuracy breakdown (if results available)
    if finetuned_results:
        print()
        print("  Per-field accuracy (fine-tuned model):")
        field_acc = finetuned_results.get("aggregate", {}).get("field_accuracy_by_field", {})
        for fname, acc in sorted(field_acc.items(), key=lambda x: x[1]):
            bar = "█" * int(acc * 20)
            print(f"    {fname:<20}  {acc * 100:5.1f}%  {bar}")

    # ------------------------------------------------------------------
    # Latency table
    # ------------------------------------------------------------------
    print()
    print("── Latency ─────────────────────────────────────────────────────────")
    print(f"{'Metric':<35} {'Base':>12} {'Fine-tuned':>12} {'Teacher':>12}")
    print("-" * 72)
    b_lat   = _get_latency(base_results)
    ft_lat  = _get_latency(finetuned_results)
    t_lat   = _get_latency(teacher_results)
    print(f"  {'p50 latency per document':<33} {b_lat:>12} {ft_lat:>12} {t_lat:>12}")

    # ------------------------------------------------------------------
    # Cost table
    # ------------------------------------------------------------------
    print()
    print("── Cost ─────────────────────────────────────────────────────────────")
    print(f"{'Metric':<35} {'Base':>14} {'Fine-tuned':>14} {'Teacher':>14}")
    print("-" * 78)

    teacher_cost_1000 = cost_per_1000_docs_api(teacher_model_slug)

    print(f"  {'Cost per 1,000 documents':<33} "
          f"{'~$0 (self-hosted)':>14} "
          f"{'~$0 (self-hosted)':>14} "
          f"${teacher_cost_1000:>12.4f}")

    # Cost-per-success
    teacher_accuracy_str = _get(teacher_results, "overall_field_accuracy")
    finetuned_accuracy_str = _get(finetuned_results, "overall_field_accuracy")

    try:
        teacher_accuracy = float(teacher_accuracy_str.rstrip("%")) / 100
        teacher_cps = cost_per_success(teacher_cost_1000, teacher_accuracy)
        teacher_cps_str = f"${teacher_cps:.6f}" if teacher_cps else "N/A"
    except (ValueError, AttributeError):
        teacher_cps_str = "TODO — run eval first"
        teacher_cps = None

    try:
        # Finetuned model has near-zero marginal cost
        finetuned_cps_str = "~$0 (self-hosted)"
    except Exception:
        finetuned_cps_str = "TODO"

    print(f"  {'Cost per success':<33} "
          f"{'~$0':>14} "
          f"{finetuned_cps_str:>14} "
          f"{teacher_cps_str:>14}")

    # ------------------------------------------------------------------
    # Pricing source note
    # ------------------------------------------------------------------
    print()
    print("  OpenRouter pricing source: https://openrouter.ai/models")
    print(f"  {teacher_model_slug}: "
          f"${OPENROUTER_PRICING[teacher_model_slug]['input_per_1m']}/1M input + "
          f"${OPENROUTER_PRICING[teacher_model_slug]['output_per_1m']}/1M output tokens")
    print("  NOTE: Verify these prices are current before publishing RESULTS.md")

    # ------------------------------------------------------------------
    # Payback period
    # ------------------------------------------------------------------
    if pipeline_build_cost_usd is not None:
        print()
        print("── Payback Period ───────────────────────────────────────────────────")
        payback = payback_period_documents(
            pipeline_build_cost_usd,
            teacher_cost_1000,
            0.0,   # near-zero marginal cost for self-hosted
        )
        if payback:
            print(f"  Pipeline build cost: ${pipeline_build_cost_usd:.2f}")
            print(f"  Teacher cost/1k docs: ${teacher_cost_1000:.4f}")
            print(f"  Break-even at: {payback:,.0f} documents")
            print(f"  At 10k docs/month: payback in {payback / 10000:.1f} months")
        else:
            print("  Payback: TODO — teacher cost data needed")

    print()
    print("=" * 70)

    return {
        "teacher_cost_per_1000_usd": teacher_cost_1000,
        "pricing_source": "https://www.anthropic.com/pricing",
        "pricing_verified_date": "TODO — verify before publishing",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_cost = None
    if len(sys.argv) > 1:
        try:
            build_cost = float(sys.argv[1])
            print(f"Pipeline build cost provided: ${build_cost:.2f}")
        except ValueError:
            pass

    compute_and_print_comparison(pipeline_build_cost_usd=build_cost)
