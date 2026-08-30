# Activation-Patching Implementation Checklist

This file tracks implementation progress against Section 2.10 of
`../../../activation_patching_experiment_profile.md`. A check is marked as
runtime-validated only after it passes with the real LLaVA-NeXT model on a GPU.

## Milestone 1: One Sample, One Layer, Text Region

| Profile check | Implemented | Lightweight test | Real GPU validation |
|---|---:|---:|---:|
| A. Configuration validation | Yes | Yes | Passed: jobs 182039 and 191212 |
| B. Answer-token validation | Yes | Yes | Passed: jobs 182039 and 191212 |
| C. Paired-input alignment | Yes | Yes | Passed: jobs 182039 and 191212 |
| D. Text-region mapping | Yes | Yes | Passed: jobs 182039 and 191212 |
| E. No-op baseline reproduction | Yes | Toy model only | Passed: jobs 182039 and 191212 |
| F. Patch-integrity validation | Yes | Toy model only | Passed: jobs 182039 and 191212 |
| G. Intervention-result validation | Yes | Toy model only | Passed: jobs 182039 and 191212 |
| H. Dataset-scale progress and completion | Not in this milestone | Not applicable | Not applicable |

### Lightweight Test Evidence

- Python compilation: passed.
- SLURM launcher syntax: passed with `bash -n`.
- Active environment import and CLI parsing: passed under `text_in_image` with
  Transformers 4.57.1.
- GUIC question `14412508`: processor produced 2,340 image placeholders and
  the independent packed-token map produced 2,340 entries.
- The grounded text box mapped to 31 valid base/mosaic visual tokens at the
  configured 0.25 token-overlap threshold.
- A, B, C, and D mapped to four distinct single token IDs and decoded back to
  their intended letters.
- Deterministic toy hook: no-op maximum logit error was 0; two patched positions
  exactly matched the donor; direct change at unpatched positions was 0.

### Required Gate Before Milestone 2

Run the real one-sample GPU job and require the final line:

```text
[COMPLETE] All milestone validation checks passed
```

Then inspect the saved JSON and update the final column above. Do not begin the
layer sweep or add controls until all A-G checks pass on the real model.

### Real-Model Result: Job 182039

- All A-G runtime checks passed and the structured JSON was saved.
- Text-region restoration effect at layer 15: `+0.1875` margin units.
- Text-region insertion effect at layer 15: `+0.046875` margin units.
- Neither intervention changed the constrained prediction for this sample.
- The sample was wrong on both no-text and grounded-overlay inputs, predicting
  the ungrounded option in both cases; it is a mechanical validation sample,
  not evidence of a recovery or misleading flip.

### Paired-Image Caveat Found During Audit

For question `14412508`, the no-text and grounded-overlay images differ beyond
the annotated text box. Outside the box, mean absolute RGB-channel difference
is `4.33/255`, median difference is `2.67/255`, and 95th-percentile difference
is `13/255`; 97.3% of outside pixels have a nonzero difference. These may be
small encoding or generation changes, but the pair is not pixel-identical
outside the text box.

Before the dataset-scale experiment, quantify this diagnostic for all pairs and
use matched random-region controls. Claims must describe the donor/recipient as
the dataset's no-text and overlay images, not as pixel-identical images that
differ only inside the annotated text box.

The dataset-wide audit is now complete and recorded in
`../paired_image_pixel_audit.md`: all 1,896 pairs have outside-box changes, and
expanding the excluded box by up to 40 pixels does not reduce the mean outside
difference. Clean-pair behavioral validation remains required before scaling.

### Cleaned-Image Real-Model Validation: Job 191212

- The run used `cleaned_image` from pinned dataset commit
  `27b45899d1154ef1f08ce5c40d45d2468e4ea3e2`.
- All A-G runtime checks passed and the separate structured report was saved as
  `../debug_outputs/14412508/misleading_groundable/layer_15_text_region_cleaned_image_debug.json`.
- The cleaned pair changed 1,536 pixels inside the dataset text box and zero
  pixels outside it. The cleaned box exactly matched the original overlay box.
- The no-text logit margin remained `1.890625` in both the original and cleaned
  runs.
- The overlay margin changed from `0.609375` to `0.781250` after removing the
  outside-box artifacts.
- Restoration effect increased from `+0.187500` to `+0.265625`; insertion
  effect remained `+0.046875`.
- Neither intervention changed the constrained prediction. This remains a
  mechanical validation sample and is not evidence for an aggregate effect.

The diffusion-artifact gate is therefore resolved for the cleaned dataset.
Dataset-scale activation patching must use `cleaned_image`, retain the pinned
dataset revision, and enforce zero outside-box changes before processing each
sample.
