# CONTEXT.md — Engineering context for Forge

Reference material for the coding agent implementing this project. Read alongside PRD.md before writing code — PRD.md is the *what and why*, this is the *how, where, and with what*.

---

## 1. Repository structure (target layout)

```
forge/
├── README.md                        # primary portfolio artifact
├── RESULTS.md                       # base vs fine-tuned vs teacher comparison
├── PRD.md                           # this project's own PRD, kept in-repo
├── pyproject.toml / requirements.txt
├── .env.example
│
├── schema/
│   └── extraction_schema.py         # the fixed JSON schema (pydantic model) — defined ONCE, used everywhere
│
├── data_generation/
│   ├── __init__.py
│   ├── prompts.py                   # generation prompt templates (diverse scenarios, difficulty, edge cases)
│   ├── generate_synthetic_data.py   # calls Claude API, produces {document_text, ground_truth_json} pairs
│   └── raw/                         # raw generated output before curation (gitignore the bulk, keep a sample)
│
├── data_curation/
│   ├── __init__.py
│   ├── validate_schema.py           # checks ground_truth_json conforms to schema/extraction_schema.py
│   ├── deduplicate.py               # near-duplicate document detection
│   ├── quality_filter.py            # removes degenerate/garbled examples
│   └── split_dataset.py             # produces train/val/eval splits — THIS is where the eval lock happens
│
├── data/
│   ├── train.jsonl                  # final curated training set
│   ├── val.jsonl                    # validation set (used during training for checkpointing decisions)
│   └── eval_locked.jsonl            # LOCKED eval set — see CONTEXT.md Section 5 for handling rules
│
├── training/
│   ├── train_colab.ipynb            # the actual Colab-runnable training notebook
│   ├── format_dataset.py            # converts curated data into the model's instruction-tuning format
│   └── config.py                    # all hyperparameters in one place (LoRA rank, learning rate, epochs, etc.)
│
├── evaluation/
│   ├── __init__.py
│   ├── run_eval.py                  # runs a given model (base/fine-tuned/teacher) against eval_locked.jsonl
│   ├── scoring.py                   # field-level accuracy, full-record exact-match, schema-validity scoring
│   └── results/
│       ├── base_model_results.json
│       ├── finetuned_model_results.json
│       └── teacher_model_results.json
│
├── cost_analysis/
│   └── compute_cost_comparison.py   # produces the $/1000 docs and cost-per-success numbers in RESULTS.md
│
├── serving/
│   ├── export_gguf.py               # exports merged fine-tuned model to GGUF for Ollama
│   ├── api.py                       # minimal FastAPI wrapper for live demo (optional but recommended)
│   └── cli.py                       # simple CLI: feed a document, get structured JSON back
│
└── tests/
    ├── test_schema_validation.py
    └── test_scoring.py
```

---

## 2. Environment & dependencies

- **Language**: Python 3.11+ (Colab default is compatible; confirm at implementation time)
- **Training environment**: Google Colab, free tier, T4 GPU runtime — confirm `Runtime → Change runtime type → GPU → T4` is selected before any training cell runs
- **Core libraries** (categories below — confirm exact current versions before implementation; this ecosystem moves fast):
  - Fine-tuning: `unsloth` (handles `bitsandbytes`, `transformers`, `peft`, `trl` integration — install via the Colab-specific extras as currently documented by Unsloth, not a generic pip install)
  - Training loop: `trl`'s `SFTTrainer`
  - Quantization: `bitsandbytes` (comes via Unsloth)
  - Schema validation: `pydantic` (v2)
  - Synthetic data generation: `anthropic` Python SDK
  - Evaluation/scoring: standard library + `pydantic` for schema checks; avoid adding heavy eval frameworks for what's fundamentally a deterministic field-comparison task
  - Export: GGUF conversion path as currently documented by Unsloth (this has changed across versions — confirm current method, don't assume an old tutorial's exact command)
  - Local serving: `ollama` (for running the exported GGUF model) and/or `fastapi` for a minimal API wrapper

- **Secrets**: `ANTHROPIC_API_KEY` required for synthetic data generation and teacher-model evaluation. No other paid dependency should be required — if Colab's free T4 proves insufficient, flag it (per PRD.md Section 10), don't silently introduce a paid GPU dependency.

---

## 3. Task and schema selection

- **To be selected by the user before implementation begins** (PRD.md Section 3 has the candidate options and a recommended default). Coding agent should present the options with the one-line tradeoffs already in the PRD and let the user confirm or override the recommended default (invoices/receipts) before writing the schema.
- Once chosen, `schema/extraction_schema.py` is defined ONCE as a pydantic model and treated as a contract — every other module (data generation prompts, curation validation, training data formatting, evaluation scoring) imports from this single source of truth. Do not let field definitions drift or get redefined ad hoc in multiple places.

---

## 4. Engineering conventions

- **The schema is law.** Synthetic data generation prompts should explicitly instruct the teacher model to produce output conforming to the schema; curation validates against it; evaluation scores against it. If the schema needs to change mid-project, that's a flag-to-user moment (it likely means regenerating data), not a quiet patch in one file.
- **Synthetic data generation must produce real diversity, not 1,000 near-identical documents.** The prompt templates in `data_generation/prompts.py` should explicitly vary: document format/layout, difficulty (clean vs. messy), edge cases (missing fields, ambiguous values, unusual currency/date formats), and length. Generating in batches with varied prompt parameters (not one static prompt looped 1,000 times) is required, not optional — a dataset of near-duplicates will produce a model that looks good on paper and fails immediately on real documents.
- **Curation is a real step with visible output, not a rubber stamp.** `data_curation/` should produce a short report (counts: generated → passed schema validation → passed dedup → passed quality filter → final count) so the README can state "generated 1,200 examples, curated down to 850" with actual numbers, not vague language.
- **The eval lock is enforced by file organization, not just discipline.** `data/eval_locked.jsonl` should be created once by `split_dataset.py` and from that point forward treated as read-only. Consider a checksum or git-tag-it-and-don't-touch-it convention so it's obvious if it's been modified later in the project.
- **Every training run's hyperparameters are recorded**, not just the final one — even failed/abandoned runs are worth a line in a `training/run_log.md` (model size, LoRA rank, learning rate, epochs, final eval score) so the README's "what I tried" story is real, not reconstructed from memory afterward.
- **Evaluation scoring code is shared across all three systems** (base, fine-tuned, teacher) — the same `scoring.py` functions must be called for each, with only the model-inference call differing. This is what makes the comparison table legitimate; do not write three slightly different scoring paths.
- **Type hints everywhere.** Same standard as the rest of this project series.

---

## 5. Handling the locked eval set (critical — read before touching `data/eval_locked.jsonl`)

1. `eval_locked.jsonl` is created exactly once, by `data_curation/split_dataset.py`, before any training run begins.
2. After creation, no example in this file should be edited, removed, or "fixed" based on how any model performs on it.
3. **Exception**: if, during eval review, an example is found to have a genuinely incorrect ground-truth label (a real curation bug, not just a hard example) — this must be flagged explicitly to the user with the specific example and the specific error, and only corrected with the user's explicit confirmation. Document any such correction in `RESULTS.md` with a one-line note ("corrected ground truth for eval example #47, which had a transcription error in the tax field, confirmed before training began" or similar) so the eval set's integrity is auditable, not just asserted.
4. If the fine-tuned model scores worse than hoped, the response is to improve the training data or training process — never to adjust the eval set to make the model look better. If this happens, document it honestly; PRD.md Section 6 covers why this matters for the project's credibility.

---

## 6. What "done" looks like for each phase

### Phase 1 — Schema + task lock
Done when: the JSON schema for the chosen extraction task is finalized as a pydantic model, with the user having confirmed the task choice and schema fields.

### Phase 2 — Synthetic data generation
Done when: `data_generation/generate_synthetic_data.py` has produced 800-1,500 raw `{document_text, ground_truth_json}` pairs with visibly varied formatting/difficulty (manually spot-check 15-20 random examples to confirm they're not near-duplicates of each other).

### Phase 3 — Curation
Done when: the curation report (Section 4) shows real attrition numbers, `eval_locked.jsonl` is created and contains 100+ examples, and `train.jsonl`/`val.jsonl` are finalized — and the user has been shown the eval set composition (difficulty mix, edge case count) for confirmation before training begins.

### Phase 4 — Training
Done when: a QLoRA fine-tuning run completes successfully on Colab's free T4 tier, producing a merged model checkpoint, with the run's hyperparameters and wall-clock training time recorded.

### Phase 5 — Evaluation
Done when: `evaluation/run_eval.py` has been run against the base model, the fine-tuned model, and the teacher model, all against the same `eval_locked.jsonl`, producing the three results files in `evaluation/results/` with real scores — no placeholders.

### Phase 6 — Cost analysis + packaging
Done when: `cost_analysis/compute_cost_comparison.py` produces real $/1000-docs and cost-per-success numbers (using actual current API pricing for the teacher model, sourced and cited, not guessed), and the fine-tuned model has been exported to a runnable format (GGUF/Ollama) with a working demo (CLI or minimal API) that a stranger could run.

---

## 7. Things to flag back to the user, not decide silently

- Which extraction task and schema to use (PRD.md Section 3)
- Any correction to the locked eval set (Section 5, exception case)
- If free-tier Colab compute proves insufficient for the chosen model size
- Any deviation from the architecture in PRD.md Section 5
- If the fine-tuned model's results come back surprisingly low — report it, propose a real fix, let the user decide whether to iterate further or document the gap honestly as-is

---

## 8. Tone for the README (final deliverable)

- Lead with the architecture diagram, the one-line pitch, and the comparison table (base vs. fine-tuned vs. teacher) — this table is the entire point, don't bury it.
- State real numbers throughout: examples generated, examples after curation, training time, total cost incurred to build the project.
- Include the cost-per-success and payback-period framing explicitly — this is the language that makes a non-AI-native startup interviewer immediately understand the value of the project.
- Include a short demo (video/GIF) of a real document going in and structured JSON coming out.
- Include an honest limitations section — e.g. "the eval set is synthetic, not real-world documents; a production version would need real-world validation," or specific field types where the gap to the teacher model remains largest. This reads as engineering maturity, not weakness.
