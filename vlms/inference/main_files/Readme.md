# VLM Inference — Main Scripts

Runs vision-language models on the **GUIC** multiple-choice dataset (5 image variants per
question: `notext`, `correct_answer`, `misleading_groundable`, `misleading_ungroundable`,
`irrelevant_word`) and analyzes how accuracy changes across variants and models.

Two stages:

1. **Inference** — `infere_vlms.py` runs one model against all 5 variants of every
   question and saves one results JSON per variant (+ a summary JSON).
2. **Analysis** — `analyze_vlms_results.py` reads every result file produced by stage 1
   across all models/variants, computes accuracy (with confidence intervals) and
   baseline-relative accuracy gaps, and writes CSVs, a text report, and comparison plots.

`run_inference.sh` is a SLURM wrapper that runs stage 1 for all four supported models
back to back.

---

## 0. Setup

From `Reading_Between_Pixels/vlms/`:

```bash
pip install -r requirements.txt
pip install pandas seaborn   # needed by analyze_vlms_results.py, not in requirements.txt

# Model-specific extras:
pip install qwen-vl-utils torchvision   # Qwen-VL
pip install -U transformers             # LLaVA-NeXT
pip install torchvision einops timm     # InternVL
```

Both scripts pull the dataset from Hugging Face (`AHAAM/GUIC` by default) via
`get_or_download_hf_dataset`, caching it to disk on first run so subsequent runs are fast.

---

## 1. `infere_vlms.py` — Stage 1, run a model over all variants

For a single model, evaluates **all 5 image variants** in one invocation (no `--variant`
flag — the variant loop is hardcoded in `main()`). For each variant it:

1. Builds the MCQ question list from the HF dataset: for every sample, the 4 candidate
   texts (`correct_answer`, `misleading_groundable`, `misleading_ungroundable`,
   `irrelevant_word`) are shuffled into options A–D deterministically per
   `question_id` + `--seed` (`shuffle_options=True` is hardcoded on), and the correct
   letter is recorded based on where `correct_answer` landed.
2. Formats the MCQ prompt and generates a response with the model (greedy, up to
   `--max_tokens` new tokens).
3. Extracts the predicted letter from the response text with a regex-based parser
   (looks for `ANSWER: X`, a bare letter, etc., falling back to `UNKNOWN`).
4. Saves per-example results (question, options, correct/predicted letter, full response,
   correctness) to a variant-suffixed JSON, and appends per-example generation logits
   (top-10 tokens per step) under `logits_debug/`.
5. After all variants, writes a `*_summary.json` with sample counts and accuracy per
   variant.

**Supported `--model_type` values** (import-gated — a model is only registered if its
package is installed):

| `--model_type` | Default `--model_id` | Extra install |
|---|---|---|
| `llava` | `llava-hf/llava-1.5-7b-hf` | — |
| `qwen-vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen-vl-utils`, `torchvision` |
| `llava-next` | `llava-hf/llava-v1.6-mistral-7b-hf` | `transformers>=latest` |
| `internvl` | `OpenGVLab/InternVL3_5-8B` | `torchvision`, `einops`, `timm` |

**Run directly:**
```bash
python infere_vlms.py \
  --model_type qwen-vl \
  --hf_dataset AHAAM/GUIC \
  --hf_cache_dir ./hf_cache_GUIC \
  --output ./results_GUIC/qwen-vl_results.json
```

**Key arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `--model_type` | `llava` | which evaluator to use (table above) |
| `--model_id` | model's registry default | override the HF checkpoint |
| `--hf_dataset` | *(required)* | HF dataset repo id, e.g. `AHAAM/GUIC` |
| `--hf_cache_dir` | `./hf_dataset_GUIC` | local dataset cache dir |
| `--output` | `results.json` | base output path — **must be named `<model>_results.json`** (see naming note below) |
| `--batch_size` | `4` | see caveat below — does **not** batch the forward pass |
| `--max_tokens` | `50` | max new tokens generated per answer |
| `--device` | `cuda` | `cuda`, `cpu`, or `auto` |
| `--list_models` | off | print which model types are importable and exit |

⚠️ **`--batch_size` doesn't batch GPU calls.** `process_batch` still calls
`process_single` once per image inside the batch loop — the argument only controls how
many items are grouped per outer-loop iteration/tqdm update, not actual batched
inference. Don't expect a speedup from raising it.

⚠️ **Output filename convention:** each variant's results are saved via
`safe_suffix(--output, variant)`, i.e. `<stem>_<variant><ext>`. `analyze_vlms_results.py`
later parses `<model>_results_<variant>.json` back apart on the literal substring
`"_results_"`, so `--output` must already be `<model_name>_results.json` (as
`run_inference.sh` does) for the analysis stage to correctly recover the model name.

**Output files** (per invocation, in the directory of `--output`):
- `<stem>_<variant>.json` — one per variant, list of per-question result dicts:
  `image_id`, `question`, `options`, `correct_answer`, `predicted_answer`,
  `full_response`, `is_correct`
- `<stem>_summary.json` — `[{variant, num_samples, accuracy_percent, output_file}, ...]`
- `logits_debug/logits_<variant>_<model_id>_<qid>.json` — per-step top-10 token logits
  for every generated example (created as a side effect of every `process_single` call)

---

## 2. `analyze_vlms_results.py` — Stage 2, aggregate & compare

Scans a results directory for every `<model>_results_<variant>.json` file produced by
stage 1 (across as many models as you've run), joins each result back to the HF dataset by
`question_id`/`image_id`, and computes:

- **Accuracy per (model, variant)** with a Wilson score confidence interval
  (`calculate_accuracy` / `wilson_ci`).
- **Paired accuracy gap vs. a baseline variant** (`notext` by default): for each model,
  restricts to question ids present in both the baseline and the variant, takes the
  per-example correctness difference, and bootstraps a CI on the mean gap
  (`create_accuracy_gap_table` / `bootstrap_mean_ci`, 10000 resamples).
- A grouped bar chart per category comparing all variants against the `notext` baseline
  for each model (`plot_category_baseline_comparison` — note: with this dataset there is
  a single implicit category, `"GUIC"`, since `calculate_accuracy` hardcodes
  `category = "GUIC"`).

**Run:**
```bash
python analyze_vlms_results.py \
  --results_dir ./results_GUIC \
  --hf_dataset AHAAM/GUIC \
  --hf_cache_dir ./hf_cache_GUIC \
  --output ./benchmarking_GUIC
```

**Key arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `-r`, `--results_dir` | `./results_GUIC` | directory of `<model>_results_<variant>.json` files from stage 1 |
| `--hf_dataset` | `AHAAM/GUIC` | dataset repo id (for `question`/`caption` lookup only — ground truth comes from the result files themselves) |
| `--hf_cache_dir` | `./hf_cache_GUIC/` | local dataset cache dir |
| `-o`, `--output` | `./benchmarking_GUIC` | output directory for all generated files |

**Outputs**, written under `--output`:
- `overall_accuracy.csv` — accuracy + CI per (model, variant)
- `category_accuracy.csv` — same, broken out by category (currently always `"GUIC"`)
- `accuracy_gap.csv` — paired accuracy gap vs. `notext`, per (model, variant), with
  improved/degraded/unchanged counts
- `analysis_report.txt` — human-readable summary: best/worst performer, per-model and
  per-variant averages, full per-category tables
- `detailed_analysis.json` — the full nested analysis dict (overall + per-category stats,
  first 20 incorrect examples, full per-example outcomes) for every (model, variant)
- `plots/category_baseline_comparison.png` (and a `.pdf` written to the same filename —
  see note below) — grouped bar chart per category, one bar per variant per model,
  annotated with % and Δ vs. `notext`

⚠️ `plot_category_baseline_comparison` builds both a `.pdf` and a `.png` output path but
only saves the `.png` (the second `os.path.join` call overwrites the first `output_path`
variable) — the PDF is not actually written despite what the variable name suggests.

---

## 3. `run_inference.sh` — SLURM job wrapper for stage 1

SBATCH script (1 GPU, 60G RAM, 16 CPUs, 24h) that loops stage 1 over all four supported
models — `qwen-vl`, `llava-next`, `llava`, `internvl` — writing each model's 5 variant
result files (+ summary) to `./results_GUIC/`, and caching the dataset once to
`./hf_cache_GUIC/`.

**Before submitting:** update the `cd` path near the top if your checkout isn't at
`/nfs-stor/ali.mekky/reading_between_pixels/...`, and the `conda activate text_in_image`
line if you use a different environment name.

**Submit:**
```bash
sbatch run_inference.sh
```

Logs land in `jobs_logs/<job-name>_<job-id>.{out,err}`. Each model's output file is named
`${model}_results.json`, so the per-variant files come out as
`qwen-vl_results_notext.json`, `llava-next_results_correct_answer.json`, etc. — matching
the naming convention `analyze_vlms_results.py` expects.

---

## `old_files/`

The sibling `old_files/` directory (`detect_flip_layer.py`, `detect_flips_trace.sh`,
`infere_manual_packing.py`, `validate_dataset.py`) holds earlier/exploratory scripts kept
for reference — not covered here since they aren't part of the current inference →
analysis pipeline.
