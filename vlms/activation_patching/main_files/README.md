# Activation Patching: Debug Milestone

This directory contains the first implementation milestone for the activation-patching experiment: one LLaVA-NeXT sample, one decoder layer, the annotated text region, and both restoration and insertion directions.

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

The implementation supports both the older `num_logits_to_keep` name and the
newer `logits_to_keep` name used by Transformers 4.57. This avoids allocating
logits for the full multimodal sequence.

## Scope

This milestone does not yet implement object controls, matched random regions, all-image-token patching, full layer sweeps, window patching, dataset-scale checkpointing, or plotting. Those are added only after this debug milestone passes on actual GPU runs.

See `IMPLEMENTATION_CHECKLIST.md` for the current audit status. Static and
lightweight tests are not a substitute for the first real GPU validation run.
