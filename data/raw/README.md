# data/raw/ — raw generated output from generate_synthetic_data.py

# What goes here
# --------------
# data/raw/raw_generated.jsonl  — all generated examples (gitignored due to size)
# data/raw/generation_run_*.json — per-run summary with counts and cost

# What's committed to git
# -----------------------
# data/raw/sample_10.jsonl — 10 random examples for spot-checking
#   (created manually by: shuf -n 10 raw_generated.jsonl > sample_10.jsonl)

# NOTE: raw_generated.jsonl is gitignored to avoid committing 10+ MB of data.
# The curated final splits (data/train.jsonl, data/val.jsonl, data/eval_locked.jsonl)
# ARE committed to git because they are the reproducible artifacts of the pipeline.
