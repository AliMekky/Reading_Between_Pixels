# Open-Ended Generation Evaluation

This folder contains the experiment-specific launchers, semantic answer evaluation, statistics, and plots. Model loading and generation reuse `../inference/main_files/infere_vlms.py`.

Progress is tracked in [`MILESTONES_CHECKLIST.md`](MILESTONES_CHECKLIST.md).

## Scope

The current complete evaluation uses six models: LLaVA-1.5-7B, LLaVA-NeXT-7B,
Qwen2.5-VL-7B, InternVL3.5-8B, Qwen3-VL-2B, and Qwen3-VL-8B. Qwen3-VL-32B is
deferred and excluded from current aggregates.

## Generation smoke test

Run the CPU-only input validation first:

```bash
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
python main_files/validate_generation_inputs.py
```

Expected: 474/474 aligned questions, five conditions, nonempty references, stable option mappings, and no MCQ content in the open-ended prompt. Known duplicate distractors are logged as audit metadata and do not remove questions.

Submit one batch at a time because the CSCC QoS permits only two submitted jobs:

```bash
cd /nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/open_ended_evaluation
bash main_files/submit_open_ended_smoke.sh 0-1
bash main_files/submit_open_ended_smoke.sh 2-3
bash main_files/submit_open_ended_smoke.sh 4-5
bash main_files/submit_open_ended_smoke.sh 6
```

Wait for each batch to finish before submitting the next. The indices correspond
to LLaVA-1.5, LLaVA-NeXT, Qwen2.5-VL, InternVL3.5, then Qwen3-VL 2B, 8B, and
32B. Each model generates two questions under all five conditions, for 70 total
generations across the four batches. The 32B submission requests two 40-GB GPUs
because its FP16 weights do not fit on one GPU.

Expected validation:

- Seven model tasks complete across four submissions.
- Each task prints `evaluation_format=open_ended`, `max_tokens=32`, and five conditions.
- Every condition loads 2/2 questions.
- No prompt contains `Options:` or the MCQ answer-letter instruction.
- Each task saves 10 successful JSONL records and a summary file.
- Raw responses, output token IDs, termination reasons, image hashes, model IDs, and resolved revisions are saved.

## Full open-ended generation

After all smoke tests pass, submit one batch at a time and wait for it to finish:

```bash
bash main_files/submit_open_ended_full.sh 0-1
bash main_files/submit_open_ended_full.sh 2-3
bash main_files/submit_open_ended_full.sh 4-5
bash main_files/submit_open_ended_full.sh 6
```

Each model writes 474 records for each of the five conditions. The launcher then
runs `validate_generation_outputs.py`; success requires 2,370 aligned records,
zero failed questions, no MCQ prompt leakage, and complete audit metadata. Runs
are JSONL-checkpointed and may be resubmitted safely after a timeout or failure.

## Deterministic answer evaluation

Run normalized exact matching and the fixed 0.90 edit-similarity stage on every
completed model:

```bash
bash main_files/run_deterministic_evaluation.sh
```

Outputs are written under `outputs/deterministic_evaluation/<model>/`. Each
classified record retains the raw generation, normalized response and references,
all four edit similarities, resolution stage, and category. Only unresolved cases
are written to `unresolved_for_judges.jsonl`; this command makes no API calls.

## Semantic judges and final statistics

API credentials are read from `.secrets/open_ended_eval.env`; the scripts never
store the keys in experiment outputs. Prepare and submit the two counterbalanced
Batch API files with:

```bash
bash main_files/prepare_and_submit_judges.sh
```

Check both jobs and download their outputs when ready:

```bash
set -a
source /nfs-stor/ali.mekky/.secrets/open_ended_eval.env
set +a
/home/ali.mekky/.conda/envs/text_in_image/bin/python main_files/check_judge_batches.py \
  --batch_dir outputs/judge_batches_six_models
```

After both result files are complete, validate every judgment, merge the two
decisions, and compute paired six-model statistics:

```bash
bash main_files/check_and_finalize_judges.sh
```

Expected: 5,325 valid judgments from each provider, 14,220 final response
classifications, and model-level plus unweighted six-model statistics. A judge
disagreement is retained as `ambiguous`; missing or malformed output stops the
merge instead of being silently accepted.

## Six-model results

The completed analysis and interpretation are in
[`open_ended_evaluation_results_six_models.md`](open_ended_evaluation_results_six_models.md).
Numerical outputs are under `outputs/statistics_six_models/` and
`outputs/tables_six_models/`; figures and their separate interpretations are under
`outputs/figures_six_models/`. Regenerate all completed plots with:

```bash
MPLCONFIGDIR=/tmp/open_ended_mpl bash plotting_scripts/run_all_plots.sh
```

The Qwen3-specific behavioral results are connected to the causal experiments
in [`../qwen3_vl_generation_causal/qwen3_vl_results_report.md`](../qwen3_vl_generation_causal/qwen3_vl_results_report.md).
