# Activation Patching

This experiment tests whether representations at the overlay-text location
causally change LLaVA-NeXT's answer. It uses the cleaned GUIC images and the
same 305 questions as the attention and integrated-gradients analyses.

## 1. Main Experiment and Reproduction

### Fixed setup

- Model: `llava-hf/llava-v1.6-mistral-7b-hf`
- Model revision: `2424fdd47412fccc66d91719126b420e9fbd7065`
- Dataset revision: `27b45899d1154ef1f08ce5c40d45d2468e4ea3e2`
- Conditions: grounded misleading, ungrounded misleading, irrelevant option,
  and correct-answer text
- Primary metric: correct-minus-condition-specific option margin; the correct
  overlay uses the strongest incorrect no-text option as its fixed comparator
- Primary controls: three random regions matched to the text region in token
  count and base/mosaic composition

### Authoritative files

| File | Purpose |
|---|---|
| `activation_patch_window_confirmation.py` | Main inference engine for single-layer and simultaneous layer-window patching. It also performs runtime validation and checkpointing. |
| `activation_patch_final_selection_shared_305_three_conditions.json` | Frozen 305-question manifest for the three overlay conditions. |
| `run_activation_patch_shared_layer_sweep.sh` | Main SLURM array launcher: three conditions × 32 layers. |
| `run_activation_patch_correct_overlay.sh` | Main 32-layer launcher for the correct-answer positive control. |
| `activation_patch_confirmation_selection_shared_305.json` | Frozen manifest for the layer-window analysis. |
| `run_activation_patch_window_confirmation.sh` | Confirmatory early/middle/late layer-window launcher. |
| `../plotting_scripts/prepare_plot_data.py` | Creates compact statistics and validation tables from the four raw reports. |
| `../plotting_scripts/run_all_plots.sh` | Generates all aggregate paper plots and interpretation files. |
| `../plotting_scripts/run_full_analysis_all_conditions.sh` | Validates all four raw reports and generates the final four-condition tables and plots. |
| `../plotting_scripts/run_behavior_breakdown.sh` | Generates the secondary Fooled-versus-Robust analysis. |

### Run the main 32-layer experiment

From `vlms/activation_patching/main_files`:

```bash
sbatch run_activation_patch_shared_layer_sweep.sh
sbatch run_activation_patch_correct_overlay.sh
```

The first launcher defines three array tasks with at most two running at once;
the second runs the correct-answer control. Each checkpoints its report and can
resume from existing output.

Expected raw outputs under `../layer_sweep_shared_305_outputs/`:

```text
activation_patch_shared_305_all_layers_misleading_groundable.json
activation_patch_shared_305_all_layers_misleading_ungroundable.json
activation_patch_shared_305_all_layers_irrelevant_word.json
activation_patch_shared_305_all_layers_correct_answer.json
```

A complete condition contains 136,576 intervention records. The four
conditions contain 546,304 records in total.

### Run the confirmatory layer-window analysis

```bash
sbatch run_activation_patch_window_confirmation.sh
```

This patches the six layers in each predefined window simultaneously:
`0–5`, `10–15`, and `26–31`.

### Regenerate final plots and tables

No GPU is needed:

```bash
cd ../plotting_scripts
bash run_full_analysis_all_conditions.sh
```

This validates and produces four-condition layer-wise effects, condition
comparisons, prediction transitions, and appendix object-control plots. To
regenerate the secondary three-condition Fooled-versus-Robust analysis:

```bash
bash run_behavior_breakdown.sh
```

The raw JSON reports should be preserved because all figures and statistics
can be regenerated from them without rerunning inference.

## 2. Debugging and Validation Runs

These runs validate the implementation; they are not the final statistical
evidence.

### A. One sample and one layer

Files: `run_activation_patch_debug.sh` and
`activation_patch_llava_next_debug.py`.

```bash
QUESTION_ID=14412508 \
VARIANT=misleading_groundable \
LAYER=15 \
sbatch run_activation_patch_debug.sh
```

Checks prompt alignment, answer tokens, cleaned pixels, visual-token mapping,
no-op reproduction, activation copying, and restoration/insertion effects.
Success ends with `[COMPLETE] All milestone validation checks passed`.

### B. One-sample 32-layer sweep

Files: `run_activation_patch_layer_sweep.sh`,
`activation_patch_layer_sweep.py`, and
`plot_activation_patch_layer_sweep.py`.

```bash
QUESTION_ID=14412508 sbatch run_activation_patch_layer_sweep.sh
```

Tests text and matched-random patching at every layer for both misleading
conditions and directions. Expected total: 256 interventions.

### C. Ten-sample region-control pilot

Files: `run_activation_patch_control_pilot.sh`,
`activation_patch_control_pilot.py`, and
`plot_activation_patch_control_pilot.py`.

```bash
sbatch run_activation_patch_control_pilot.sh
```

Tests five layers, two misleading conditions, both directions, object
controls, three matched-random controls, and all-image patching. Expected
total: 1,400 interventions.

### D. Fifty-sample layer discovery

Files: `run_activation_patch_discovery_50.sh`,
`activation_patch_control_pilot.py`,
`plot_activation_patch_control_pilot.py`, and
`activation_patch_discovery_50_selection.json`.

```bash
sbatch run_activation_patch_discovery_50.sh
```

Runs all 32 layers on a frozen 50-question diagnostic set. Expected total:
44,800 interventions. This run was used for layer-profile discovery, not as
the primary result.

### E. One-sample layer-window validation

Files: `run_activation_patch_window_confirmation_debug.sh` and the main engine
`activation_patch_window_confirmation.py`.

```bash
sbatch run_activation_patch_window_confirmation_debug.sh
```

Validates simultaneous six-layer hooks for all three windows. Expected total:
84 interventions.

### F. Full-path regression test

Files: `run_activation_patch_shared_layer_sweep_debug.sh` and the main engine
`activation_patch_window_confirmation.py`.

```bash
sbatch run_activation_patch_shared_layer_sweep_debug.sh
```

Runs one question across the three conditions, all 32 layers, and all region
controls. Use this after changing the main inference engine.

### G. Correct-overlay metric check

Files: `run_activation_patch_correct_overlay_debug.sh`,
`activation_patch_final_selection_shared_305_correct_answer.json`, and the main
engine `activation_patch_window_confirmation.py`.

```bash
sbatch run_activation_patch_correct_overlay_debug.sh
```

Checks that the correct-overlay experiment freezes the strongest incorrect
no-text option as its margin comparator. Expected total: 448 interventions.

### Supporting validation files

- `IMPLEMENTATION_CHECKLIST.md`: validation history for every development gate.
- `create_clean_overlay_preview.py`: creates one cleaned image for manual pixel inspection.
- `build_and_push_cleaned_guic.py`: constructs the cleaned overlay fields.
- `validate_remote_cleaned_guic.py`: verifies the pinned remote dataset pixel by pixel.
- `prepare_activation_patch_confirmation_selection.py`: reconstructs and validates the shared 305-question manifest.

Older merge and plotting helpers remain in this folder for reproducibility,
but final paper figures should be generated with the independent scripts under
`../plotting_scripts/`.
