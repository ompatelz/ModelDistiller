# Forge Results

This file records what is evidenced by committed repository artifacts. Model-quality numbers should only be added here after the corresponding `evaluation/results/*.json` files are generated and committed or otherwise published with a reproducible run link.

## Dataset Evidence

| Metric | Value | Source |
| --- | ---: | --- |
| Generation requests | 1,000 | `data/raw/generation_run_20260621T195959.json` |
| Successfully generated examples | 926 | `data/raw/generation_run_20260621T195959.json` |
| Parse/API failures | 74 | `data/raw/generation_run_20260621T195959.json` |
| Schema-invalid at generation time | 145 | `data/raw/generation_run_20260621T195959.json` |
| Training examples | 600 | `data/train.jsonl` |
| Validation examples | 67 | `data/val.jsonl` |
| Locked eval examples | 91 | `data/eval_locked.jsonl` |

## Locked Eval Set

`data/eval_locked.jsonl` is protected by a SHA-256 checksum:

```text
0ec6e5175885a61467757403a6fe0f9207b5378b175f0563b08871c5beba9264
```

The evaluation runner verifies this checksum before running unless explicitly invoked with `--no-checksum-verify`.

## Quality Metrics

| Metric | Base Qwen2.5 | Fine-tuned Qwen2.5 | Teacher |
| --- | ---: | ---: | ---: |
| Schema validity rate | Pending | Pending | Pending |
| Field-level accuracy | Pending | Pending | Pending |
| Full-record exact-match rate | Pending | Pending | Pending |
| p50 latency per document | Pending | Pending | Pending |
| p95 latency per document | Pending | Pending | Pending |

The code to produce these metrics exists in `evaluation/run_eval.py` and `evaluation/scoring.py`, but the resulting JSON artifacts are not currently committed.

## Cost Metrics

Cost comparison is implemented in `cost_analysis/compute_cost_comparison.py`, but it depends on current OpenRouter pricing and measured token/latency outputs from eval runs. Do not publish cost-per-success or payback-period numbers until those inputs are refreshed.

## Current Limitations

- The eval set is synthetic, not real production invoices.
- The committed repo does not include trained LoRA adapter weights.
- The committed repo does not include `evaluation/results/base_model_results.json`, `evaluation/results/finetuned_model_results.json`, or `evaluation/results/teacher_model_results.json`.
- `training/run_log.md` still needs to be filled from the actual training run.
- Any headline benchmark should be treated as unverified until the corresponding artifacts are attached.
