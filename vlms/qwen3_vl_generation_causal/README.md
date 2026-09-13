# Qwen3-VL Generation Causal Experiment

This folder implements
[`qwen3_vl_generation_causal_experiment_profile.md`](../../qwen3_vl_generation_causal_experiment_profile.md).
It is independent of the earlier MCQ causal experiments but reuses their
validated dataset, sample manifest, and spatial mapping utilities.

The connected behavioral, activation-patching, and attention-intervention
results are documented in
[`qwen3_vl_results_report.md`](qwen3_vl_results_report.md).

## Step 1: validate multi-token scoring

Run the CPU unit test:

```bash
cd main_files
python test_sequence_scoring.py
```

Submit the one-question, two-model GPU smoke test:

```bash
sbatch run_step1_scoring_smoke.sh
```

Expected:

- two Slurm tasks, one for Qwen3-VL-2B and one for Qwen3-VL-8B;
- four reference answers and the freely generated answer receive finite
  full-sequence scores on both no-text and overlay images;
- prompt prefixes are identical for every candidate;
- saved per-token log probabilities reproduce each saved mean;
- current deterministic generation matches the previously saved generation
  when the checkpoint revision and input hash match;
- each task ends with `[PASS] multi-token scoring validation complete`.

Outputs are written under `outputs/step1_scoring_smoke/` and logs under
`logs/`.

Status: completed successfully as Slurm array job `218340`. See
[`results_discussion.md`](results_discussion.md) for the validated numbers and
their limited interpretation.

## Step 2: validate mapping and DeepStack hooks

```bash
sbatch main_files/run_step2_spatial_deepstack_smoke.sh
```

Expected: paired spatial positions align; the three ViT DeepStack features are
observed after language layers 0--2 and therefore at `resid_pre` of language
layers 1--3; no-op error is at most `1e-3`; restoration at layer 3 changes only
the mapped text-token positions.

Status: the original thresholded run `219943` was superseded after manual
inspection. The final positive-overlap run `220272` passed for both sizes.

## Step 3: multi-layer and DeepStack-aware patching

```bash
sbatch main_files/run_step3_multilayer_patch.sh
```

This one-question gate tests restoration and insertion for text, three random
controls, and all image tokens at eight representative layers. It separately
replaces the initial visual embedding plus all three DeepStack additions.

Status: completed successfully as Slurm array job `220321`; each model saved
all 90 expected records and passed the no-op and patch-integrity checks.

## Step 4: 40-question all-layer discovery screen

```bash
sbatch main_files/run_step4_discovery.sh
```

This outcome-blind seed-42 sample runs all 28 Qwen3-VL-2B decoder layers for
all four overlays. Two GPU tasks process one 20-question shard each and run the
four overlays sequentially. Every question is checkpointed separately.

Expected: 40 questions per overlay, 290 records per question, 46,400 records
overall, exact patch integrity, and four final `[PASS]` lines in each task log.

Status: completed successfully as two-task Slurm array job `220952`. Both
shards completed all four overlays, producing the expected 46,400 records.

## Step 5: aggregate and freeze the confirmation window

```bash
sbatch main_files/run_step5_analysis.sh
```

The CPU analysis audits all records, saves question-level and layer-level
tables, scores every six-layer contiguous window using the preregistered shared
text-minus-random statistic, freezes the normalized 2B/8B windows, and creates
the discovery profile figure.

Status: completed locally. The frozen window is layers 3--8 for 2B and layers
4--11 was the initial proportional 8B mapping. Because cross-scale alignment
cannot be assumed, the mapped 8B window is provisional pending Step 5b.

## Step 5b: independent 8B all-layer discovery

```bash
sbatch main_files/run_step5b_8b_discovery.sh
```

This repeats discovery on the identical outcome-blind 40 questions but scans
all 36 Qwen3-VL-8B layers. Two GPU tasks process 20-question shards across all
four overlays. Expected output: 59,200 records. The 8B window will be selected
independently with an eight-layer width, matching approximately 20% depth.

Status: submitted as two-task Slurm array job `222464`.

Completed successfully. All 59,200 records passed the numerical audit. The
independently selected 8B primary window is layers 5--12. The final
confirmation also uses four equal-width quartile windows per model, frozen in
`outputs/confirmation_windows.json`, so every decoder layer is represented.

## Step 6a: 8B full-305 all-layer main experiment

```bash
sbatch main_files/run_step6_8b_full_all_layers.sh
```

This is the main 8B activation-patching run: all 305 shared questions, four
overlay conditions, all 36 layers, both directions, text, three matched-random
controls, all image tokens, and the DeepStack source bundle. Two resumable
shards produce 451,400 records in total. The observed discovery runtime implies
approximately 32 GPU-hours.

Status: completed successfully as two-task Slurm array job `224908`. The audit
verified all 1,220 question-condition files and all 451,400 expected records,
with zero no-op or patch-integrity error.

Aggregate and create the three independent figures:

```bash
python main_files/analyze_step6_8b_full_layers.py
bash plotting_scripts/run_step6_8b_plots.sh
```

Each figure has its own Python file under `plotting_scripts/`; numerical tables,
figures, and short interpretation files are under
`outputs/step6_8b_analysis/`.
