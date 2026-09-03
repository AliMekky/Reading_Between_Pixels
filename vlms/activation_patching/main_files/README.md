# Activation Patching Experiment

This directory contains the complete activation-patching pipeline, beginning
with the one-sample validation and extending to the shared-subset layer and
window analyses. The experiment tests whether visual representations at the
annotated overlay-text location causally affect the model's answer.

## Current Status

- Model: `llava-hf/llava-v1.6-mistral-7b-hf`, pinned revision
  `2424fdd47412fccc66d91719126b420e9fbd7065`.
- Dataset: cleaned `AHAAM/GUIC`, pinned revision
  `27b45899d1154ef1f08ce5c40d45d2468e4ea3e2`.
- Primary sample: the same 305 question IDs used by the attention and
  integrated-gradients analyses.
- Completed full 32-layer conditions: grounded misleading, ungrounded
  misleading, and irrelevant option text.
- Each completed condition contains 136,576 interventions; together they
  contain 409,728 validated interventions.
- Primary controls: three text-size- and base/mosaic-matched random regions.
  Object-region controls and the all-image upper bound are also retained.
- Correct-answer overlay: unfinished and excluded from completed-condition
  claims. Its current checkpoint has `status: in_progress` and must be resumed
  and validated before it is reported.
- Main report:
  [`../../../activation_patching_full_report.md`](../../../activation_patching_full_report.md).
- Result discussion:
  [`../../../activation_patching_results_discussion.md`](../../../activation_patching_results_discussion.md).

## Mechanistic Continuation: Attention-Path Intervention

Activation patching establishes **whether** representations at the text region
causally affect the answer. The follow-up attention-path intervention tests
**where that information is sent** and **how it combines with object and
linguistic evidence**. It is Stage 2 of the same mechanistic investigation, not
an independent experiment.

The continuation preserves the same pinned model, cleaned dataset, 305
questions, prompts, shuffled option order, visual-token map, region controls,
and correct-versus-condition-specific misleading logit margin. Its independent
implementation is located at:

```text
../../attention_path_intervention/
```

Current Stage 2 validation status:

1. Static architecture and Q/P/D/T/C/G/R token inspection passed.
2. Synthetic FP32/FP16 directed-mask validation passed.
3. One real sample at layer 3 passed exact no-op, edge-zeroing, normalization,
   and saved-logit checks. The diagnostic `T -> D` margin effect was zero for
   that one layer/sample; route and window comparisons are therefore required
   before drawing a mechanistic conclusion.

See the approved
[`../../../attention_path_intervention_experiment_profile.md`](../../../attention_path_intervention_experiment_profile.md)
and the live
[`../../attention_path_intervention/main_files/IMPLEMENTATION_CHECKLIST.md`](../../attention_path_intervention/main_files/IMPLEMENTATION_CHECKLIST.md).

## Which Files Should Be Used?

### Main experiment files

These are the authoritative files for the final 305-question experiment.

| File | Role |
|---|---|
| `activation_patch_window_confirmation.py` | Main inference engine. It loads the pinned model and cleaned dataset, constructs visual regions, captures donor activations, patches single layers or simultaneous layer windows, validates every intervention, and checkpoints raw JSON records. |
| `activation_patch_final_selection_shared_305_three_conditions.json` | Frozen 305-question manifest for grounded misleading, ungrounded misleading, and irrelevant overlays. It also records dataset provenance and region coverage. |
| `run_activation_patch_shared_layer_sweep.sh` | Primary SLURM array launcher. Its three tasks run all 32 layers separately for the three completed overlay conditions. This is the launcher to use to reproduce or resume the main layer-wise experiment. |
| `activation_patch_confirmation_selection_shared_305.json` | Frozen manifest used by the simultaneous early/middle/late window analysis. |
| `run_activation_patch_window_confirmation.sh` | Runs the three predefined six-layer windows on the shared 305 questions. This is the confirmatory window-level analysis accompanying the main layer sweep. |
| `../plotting_scripts/prepare_plot_data.py` | Converts the three large validated raw reports into compact CSV tables and computes the paired statistics used by the final plots. |
| `../plotting_scripts/run_all_plots.sh` | Authoritative launcher for the aggregate paper figures. It prepares plot data and runs each independently editable figure script. |
| `../plotting_scripts/plot_01_layerwise_effects.py` | Main layer-wise text, matched-random, and all-image effect figure. |
| `../plotting_scripts/plot_02_condition_comparison.py` | Direct comparison of grounded, ungrounded, and irrelevant text effects. |
| `../plotting_scripts/plot_03_prediction_transitions.py` | Recovery and condition-specific target-flip rates. |
| `../plotting_scripts/plot_04_object_controls.py` | Appendix figure for clean correct-object, grounded-object, and random controls. |
| `../plotting_scripts/plot_style.py` | Shared paper typography, colors, labels, saving, and interpretation helpers. |

The primary execution sequence is:

```bash
# Three SLURM array tasks, with at most two running concurrently.
sbatch run_activation_patch_shared_layer_sweep.sh

# Confirmatory simultaneous layer-window analysis.
sbatch run_activation_patch_window_confirmation.sh

# Final aggregate tables, figures, and interpretation files; no GPU needed.
cd ../plotting_scripts
bash run_all_plots.sh
```

Raw outputs from the main experiment are the three files named
`activation_patch_shared_305_all_layers_<condition>.json` under
`../layer_sweep_shared_305_outputs/`. These raw records must be retained;
figures can be regenerated from them without rerunning inference.

### Behavioral subgroup analysis

These files provide the secondary Fooled-versus-Robust analysis. They do not
replace the all-305-sample primary result.

| File | Role |
|---|---|
| `../plotting_scripts/prepare_behavior_breakdown.py` | Reconstructs Fooled and Robust membership from paired no-text/overlay predictions and produces compact subgroup tables. |
| `../plotting_scripts/behavior_breakdown_plot.py` | Shared plotting implementation for a single condition. |
| `../plotting_scripts/plot_05_ungrounded_breakdown.py` | Ungrounded Fooled-versus-Robust figure. |
| `../plotting_scripts/plot_06_grounded_breakdown.py` | Grounded Fooled-versus-Robust figure. |
| `../plotting_scripts/plot_07_irrelevant_breakdown.py` | Irrelevant Fooled-versus-Robust figure. |
| `../plotting_scripts/run_behavior_breakdown.sh` | Runs the subgroup table builder and all three subgroup plots. |
| `../plotting_scripts/run_plots_without_irrelevant.sh` | Regenerates the aggregate figures with only grounded and ungrounded conditions, using already prepared CSV data. |

### Correct-answer overlay extension

The correct-answer overlay is separate from the completed three-condition
result because it uses a different frozen margin comparator.

| File | Role |
|---|---|
| `activation_patch_final_selection_shared_305_correct_answer.json` | Frozen shared-subset manifest for the correct-answer overlay. |
| `run_activation_patch_correct_overlay_debug.sh` | One-question metric gate confirming that the strongest incorrect no-text option is frozen correctly. |
| `run_activation_patch_correct_overlay.sh` | Full 305-question, 32-layer correct-overlay launcher. Its current checkpoint is incomplete and must be resumed before use. |

### Validation and debugging files

These files established that the implementation was correct. They should be
kept for auditability, but their small or pilot results are not the main
dataset-scale evidence.

| File | Role |
|---|---|
| `activation_patch_llava_next_debug.py` | One sample, one layer, text region, restoration and insertion. Validates answer tokens, paired inputs, region mapping, no-op reproduction, and activation-copy integrity. |
| `run_activation_patch_debug.sh` | SLURM launcher for the first real-GPU validation. |
| `activation_patch_layer_sweep.py` | One-sample sweep over all 32 layers using text and one matched-random region. |
| `run_activation_patch_layer_sweep.sh` | Runs and plots the one-sample all-layer diagnostic. |
| `plot_activation_patch_layer_sweep.py` | Diagnostic plot for that one-sample sweep. |
| `activation_patch_control_pilot.py` | Reusable pilot engine for multiple samples, regions, directions, and layers. It was used for the ten-sample and discovery stages. |
| `run_activation_patch_control_pilot.sh` | Ten-sample, five-layer control pilot launcher. |
| `plot_activation_patch_control_pilot.py` | Pilot/discovery summaries and plots. |
| `activation_patch_discovery_50_selection.json` | Frozen 50-question discovery set selected without inspecting intervention outcomes. |
| `run_activation_patch_discovery_50.sh` | All-layer discovery run used to inspect the layer profile before confirmation. |
| `run_activation_patch_window_confirmation_debug.sh` | One-sample validation of simultaneous six-layer window patching. |
| `run_activation_patch_shared_layer_sweep_debug.sh` | One-sample regression test covering all three conditions, 32 layers, and all region controls. |
| `IMPLEMENTATION_CHECKLIST.md` | Audit trail showing which validations passed at each implementation milestone. |

### Dataset preparation and provenance utilities

These scripts created and verified the cleaned dataset. They are not activation
intervention scripts.

| File | Role |
|---|---|
| `create_clean_overlay_preview.py` | Creates one pixel-exact cleaned overlay for manual inspection and reports inside/outside-box differences. |
| `build_and_push_cleaned_guic.py` | Constructs every `cleaned_image` from the no-text image plus the exact annotated overlay box; optionally pushes the dataset. |
| `validate_remote_cleaned_guic.py` | Downloads the pinned remote revision and verifies that all overlay pairs are unchanged outside their text boxes. |
| `prepare_activation_patch_confirmation_selection.py` | Reconstructs and validates the shared attention/IG question set and region coverage. The frozen manifests should normally be reused instead of regenerated. |

### Superseded plotting and merge helpers

`merge_activation_patch_layer_sweeps.py`,
`plot_activation_patch_shared_layer_sweep.py`, and
`finalize_activation_patch_shared_layer_sweep.sh` produced an earlier combined
summary. `plot_activation_patch_window_confirmation.py` produces the dedicated
window plot. They remain useful for reproducing earlier diagnostics, but the
independent scripts under `../plotting_scripts/` are the authoritative source
for the final layer-wise paper figures and interpretations.

## Initial Debug Milestone

The first implementation milestone uses one LLaVA-NeXT sample, one decoder
layer, the annotated text region, and both restoration and insertion
directions.

## Files

- `activation_patch_llava_next_debug.py`: experiment and validation logic.
- `run_activation_patch_debug.sh`: four-hour, one-GPU SLURM launcher.

## Run

From this directory:

```bash
QUESTION_ID=14412508 \
VARIANT=misleading_groundable \
LAYER=15 \
sbatch run_activation_patch_debug.sh
```

The environment variables are optional; the values above are the defaults.
The SLURM output paths and runtime working directory use the absolute project
path. This is necessary because SLURM executes a temporary spool copy of the
launcher, making `BASH_SOURCE[0]` unsuitable for locating the repository. If
the checkout is intentionally moved, update the two `#SBATCH` log paths and set
`ACTIVATION_PATCH_ROOT` to the new `vlms/activation_patching` directory.

For a direct interactive run:

```bash
python -u activation_patch_llava_next_debug.py \
  --question_id 14412508 \
  --variant misleading_groundable \
  --layer 15 \
  --shuffle_options \
  --streams base,mosaic \
  --out_dir ../debug_outputs
```

## Expected Validation Output

A valid run prints the following sections in order:

1. `[CONFIG]`: model, dataset, variant, layer, streams, device, and precision.
2. `[MODEL]` followed by `[PASS]`: decoder path and valid layer count.
3. `[TOKENS]` followed by `[PASS]`: correct and misleading answer letters decode from validated single tokens.
4. `[PIXELS]`: the cleaned overlay must exactly match the no-text image outside
   the annotated box and the original overlay inside it. The run stops if either
   comparison has any mismatched pixels.
5. `[ALIGN]` followed by `[PASS]`: identical prompts and one-to-one packed visual-token alignment.
6. `[REGION]` followed by `[PASS]`: the text box maps to at least one valid image-token position.
7. `[NOOP]` followed by `[PASS]`: capture hooks reproduce normal logits within the configured tolerance and preserve the prediction.
8. `[PATCH]` followed by `[PASS]`: patched values equal donor values, unpatched values are directly unchanged, and donor/recipient activations differ before patching.
9. `[RESULT]`: before/after margins, oriented effect, predictions, and outcome for restoration and insertion.
10. `[SAVE]` and `[COMPLETE]`: the JSON report was saved and every milestone check passed.

Any failed invariant prints `[FAIL]` and terminates with a nonzero exit code. A failed run must not be treated as an experimental result.

## Structured Output

The report is saved to:

```text
../debug_outputs/<question_id>/<variant>/layer_<NN>_text_region_cleaned_image_debug.json
```

It contains the pinned dataset provenance, configuration, answer-token audit,
input alignment, exact cleaned-image pixel validation, complete mapped token
indices, no-op results, baseline logits, intervention effects, patch-integrity
measurements, and checklist status.

Schema version 2 additionally records all A-D logits, probabilities normalized
over the four choices, complete option rankings, correct and misleading ranks,
and the top-prediction outcome before and after every intervention.

The implementation supports both the older `num_logits_to_keep` name and the
newer `logits_to_keep` name used by Transformers 4.57. This avoids allocating
logits for the full multimodal sequence.

## Scope

This milestone does not yet implement object controls, matched random regions, all-image-token patching, full layer sweeps, window patching, dataset-scale checkpointing, or plotting. Those are added only after this debug milestone passes on actual GPU runs.

See `IMPLEMENTATION_CHECKLIST.md` for the current audit status. Static and
lightweight tests are not a substitute for the first real GPU validation run.

## Ten-Sample Region-Control Pilot

After the all-layer one-sample diagnostic, run the predefined control pilot:

```bash
sbatch run_activation_patch_control_pilot.sh
```

It evaluates 10 deterministically selected samples at layers `0, 8, 15, 23,
31`, for both misleading conditions and both patch directions. Each setting
patches the text region, correct-object region, grounded-object region, three
separately selected matched-random regions, and all packed image tokens. This
produces exactly 1,400 intervention records.

The three random controls each match the text region's base/mosaic token count
and are disjoint from the annotated text and object regions and from one
another. They are averaged within each sample before plotting. The all-image
control patches the complete packed visual sequence, including packing newline
tokens, and serves only as an approximate upper bound.

Expected outputs:

```text
../control_pilot_outputs/control_pilot_10_samples_5_layers.json
../control_pilot_outputs/control_pilot_10_samples_5_layers_summary.csv
../control_pilot_outputs/control_pilot_10_samples_5_layers.png
```

The JSON checkpoints after every complete sample/condition/layer block and can
resume safely. The plot contains four panels (grounded/ungrounded by
restoration/insertion), bootstrap intervals across the 10 samples, and five
curves: text, correct object, grounded object, the within-sample mean of the
three random controls, and all image tokens. It is a diagnostic pilot, not the
final dataset-scale statistical result.

## Fifty-Sample All-Layer Discovery

The next milestone uses 50 locked samples that exclude every question used by
the one-sample diagnostics and ten-sample pilot. Selection used only annotation
and packed-token geometry; no model prediction, logit, or patching result was
inspected. The exact IDs and dataset rows are recorded in
`activation_patch_discovery_50_selection.json`.

Submit with:

```bash
sbatch run_activation_patch_discovery_50.sh
```

This evaluates all 32 decoder layers and the same seven region instances,
producing exactly 44,800 interventions. It checkpoints every 64 complete
sample/condition/layer blocks to limit repeated writes of the large aggregate
JSON while retaining safe resume behavior.

Expected outputs:

```text
../discovery_50_outputs/activation_patch_discovery_50_all_layers.json
../discovery_50_outputs/activation_patch_discovery_50_all_layers_summary.csv
../discovery_50_outputs/activation_patch_discovery_50_all_layers.png
```

The discovery result is used to lock equal-width early, middle, and late
windows. Those windows are then evaluated on the exact 305-question subset
shared with attention and integrated gradients. Forty-four discovery questions
are in that subset, so the final run is not described as strictly held out.

## Shared-Subset Layer-Window Analysis

The locked manifest contains exactly the 305 question IDs present in both the
prior LLaVA-NeXT attention and integrated-gradients outputs. No sample is
removed because an object control is missing or overlaps text tokens. Primary
text, matched-random, and all-image results therefore have 305/305 coverage.
Optional clean object controls report their own sample counts.

Run the required one-sample GPU gate first:

```bash
sbatch run_activation_patch_window_confirmation_debug.sh
```

After its 84/84 interventions pass, submit the full experiment:

```bash
sbatch run_activation_patch_window_confirmation.sh
```

The full run simultaneously patches every layer in the early `0–5`, middle
`10–15`, or late `26–31` window. It produces 25,608 intervention records, a
60-row numerical summary, a four-row paired-comparison table, and one
four-panel grouped-bar plot.

## Final Shared-Subset Plotting

The final 32-layer run is plotted by independent scripts in
`../plotting_scripts`. Regenerate all compact statistics, PDF/PNG figures, and
their interpretation files with:

```bash
cd ../plotting_scripts
bash run_all_plots.sh
```

Outputs are saved under:

```text
../layer_sweep_shared_305_outputs/plot_data/
../layer_sweep_shared_305_outputs/plots/
```

After the first run, an individual figure can be changed and regenerated from
the compact CSV files without loading the raw reports or using a GPU. Each
figure script writes a same-named `_interpretation.md` file based on the
observed numerical results.

Generate the secondary Fooled-versus-Robust breakdown for grounded,
ungrounded, and irrelevant overlays with:

```bash
cd ../plotting_scripts
bash run_behavior_breakdown.sh
```

These subgroup figures are saved under
`../layer_sweep_shared_305_outputs/fooled_vs_robust_plots/`. The aggregate
estimate over all 305 samples remains the primary result.

## Correct-Answer Overlay Extension

The correct-overlay condition uses the same 305 questions and spatial controls.
Its margin is correct minus the strongest incorrect no-text option, frozen per
sample before patching. Run the metric gate and then the full sweep with:

```bash
sbatch run_activation_patch_correct_overlay_debug.sh
sbatch --dependency=afterok:<debug_job_id> run_activation_patch_correct_overlay.sh
```
