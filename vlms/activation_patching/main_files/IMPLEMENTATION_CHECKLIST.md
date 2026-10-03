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

### Expanded Prediction Audit: Job 191588

- Output schema version 2 was validated on the real model.
- Every baseline and patched run now saves all A-D logits, probabilities
  normalized over A-D, the full ranking, per-option ranks, and the constrained
  top prediction.
- Intervention records retain the complete recipient and patched summaries.
- Outcome classification distinguishes recovery, condition-specific misleading
  flip, other-option flip, and no prediction change. A move from any unrelated
  option to the relevant misleading option is counted as a misleading flip.
- For the validation sample, the ranking remained `B > C > D > A`; restoration
  and insertion changed the targeted margin without changing either relevant
  option's rank or the top prediction.

## Milestone 3: Ten-Sample, Five-Layer Region-Control Pilot

| Check | Implemented | Lightweight test | Real GPU validation |
|---|---:|---:|---:|
| Deterministic selection before inference | Yes | Passed: 10 locked IDs | Passed: job 191659 |
| Cleaned-image zero outside-box validation | Yes | Passed in prior dataset audit | Passed: job 191659 |
| Text and two annotated object mappings | Yes | Passed: 20 condition maps | Passed: job 191659 |
| Three independent matched-random controls | Yes | Passed: 60 controls | Passed: job 191659 |
| Exact random/text base-mosaic composition | Yes | Passed: 60 controls | Passed: job 191659 |
| Random controls avoid text, objects, and each other | Yes | Passed: 60 controls | Passed: job 191659 |
| Complete packed-image upper-bound intervention | Yes | Passed: mapping/count checks | Passed: job 191659 |
| Layers `0, 8, 15, 23, 31` at `resid_pre` | Yes | Yes | Passed: job 191659 |
| Restoration and insertion | Yes | Yes | Passed: job 191659 |
| Schema-v2 A-D logits, ranks, and outcomes | Yes | Yes | Passed: job 191659 |
| No-op and patch-integrity gates | Yes | Toy/previous milestone | Passed: job 191659 |
| Atomic checkpointing and exact record count | Yes | Yes | Passed: 1,400/1,400 |
| Four-panel pilot plot and CSV summary | Yes | Passed: synthetic aggregation test | Passed: job 191659 |

The real-model gate is the final log line:

```text
[COMPLETE] All 1400 control-pilot interventions passed validation
```

The subsequent plotting step must then report 100 summary rows: 2 variants ×
2 directions × 5 layers × 5 displayed region curves.

### Real-Model Result: Job 191659

- All 1,400/1,400 interventions passed validation for 10/10 samples.
- No-op errors, patch-integrity checks, region composition checks, and cleaned
  outside-box checks passed.
- The run saved the schema-v2 JSON, a 100-row summary CSV, and the four-panel
  diagnostic plot.
- Text-region effects were strong at layers 0 and 8, smaller at layer 15, and
  approximately zero at layers 23 and 31. This is a pilot observation, not a
  final layer-localization claim.

## Milestone 4: Independent 50-Sample All-Layer Discovery

| Check | Implemented | Lightweight test | Real GPU validation |
|---|---:|---:|---:|
| Prior diagnostic samples excluded | Yes | Passed: 11 IDs disjoint | Passed: job 191801 |
| Fifty IDs locked before inference | Yes | Passed: 50 unique IDs | Passed: job 191801 |
| Selection manifest pinned to dataset revision | Yes | Passed | Passed: job 191801 |
| All decoder layers `0–31` | Yes | Yes | Passed: job 191801 |
| Same seven region instances | Yes | Passed: 100 condition maps | Passed: job 191801 |
| Exact expected count: 44,800 | Yes | Yes | Passed: 44,800/44,800 |
| Bounded checkpoint write frequency | Yes | Passed: every 64 blocks | Passed: job 191801 |
| Safe checkpoint resume | Yes | Previous milestone | Passed: job 191801 |
| Four-panel, 640-row discovery summary | Yes | Synthetic aggregation test | Passed: job 191801 |

The real-model gate is:

```text
[COMPLETE] All 44800 control-pilot interventions passed validation
```

The final plotting step must report 640 rows: 2 variants × 2 directions × 32
layers × 5 displayed region curves.

### Real-Model Result: Job 191801

- Completed 44,800/44,800 interventions for 50/50 samples with zero failures.
- Saved the 640-row summary and four-panel all-layer discovery plot.
- The independent discovery confirmed a strong plateau at layers 0–6, a
  decline through layers 7–16, and approximately zero visual-position effects
  after layer 17.

## Milestone 5: Shared-Subset Six-Layer Window Analysis

| Check | Implemented | Lightweight test | Real GPU validation |
|---|---:|---:|---:|
| Exact prior attention/IG ID subset used | Yes | Passed: both sets identical | Passed: job 196670 |
| Shared selection locked without inference | Yes | Passed | Passed: job 196670 |
| Primary coverage and control availability saved | Yes | Passed: 305/305 selected | Passed: 305/305 |
| Geometry used only for control validity, not sample exclusion | Yes | Passed: zero primary exclusions | Passed: zero primary exclusions |
| Early `0–5`, middle `10–15`, late `26–31` | Yes | Passed | Passed: job 196670 |
| Simultaneous six-layer capture and patching | Yes | Toy model passed | Passed: corrected debug job 196571 |
| Per-layer patch-integrity audit | Yes | Toy model passed | Passed: corrected debug job 196571 |
| Expected 25,608 full-run records | Yes | Passed | Passed: 25,608/25,608 |
| Five-sample atomic checkpoint interval | Yes | Code inspection | Passed |
| 60-row summary and four paired comparisons | Yes | Passed on debug report | Passed |

The corrected one-sample real-model gate completed 84/84 interventions in job
`196571`; the shared-subset job is ready to submit.

### Real-Model Smoke Test: Job 196484

- Completed 84/84 interventions for one sample under the earlier control path.
- Both 18-layer no-op captures reproduced baseline logits exactly.
- Every six-layer intervention passed per-layer donor-copy and unpatched-state
  integrity checks.
- The debug report produced the expected 60-row summary, four-row paired table,
  and grouped-bar plot.
- This validates simultaneous hooks, but the revised random-control and
  shared-subset path still requires its own smoke test.
- The old 263-sample job `196495` was cancelled because its question IDs did
  not match the prior attention/IG subset. A new smoke test is required for the
  corrected 305-sample path.

## Milestone 6: Full 32-Layer Shared-Subset Sweep

| Check | Implemented | Lightweight test | Real GPU validation |
|---|---:|---:|---:|
| Exact 305-question manifest reused | Yes | Passed | Passed: job 199981 |
| All 32 layers patched individually | Yes | Static check passed | Passed: job 199981 |
| Text, two objects, three random controls, and all-image regions | Yes | Static check passed | Passed: job 199981 |
| Grounded, ungrounded, and irrelevant-option conditions | Yes | Manifest passed: 305 each | Passed: 305 each |
| Expected 409,728 combined full-run records | Yes | Accounting passed | Passed: 409,728/409,728 |
| Three 136,576-record condition tasks and deterministic merge | Yes | Accounting passed | Passed: zero failures |
| Expected 1,344 one-sample records | Yes | Accounting passed | Passed: 1,344/1,344, job 197863 |
| Independent paper-style plotting scripts and one runner | Yes | Passed | Passed on final outputs |
| Main layer plot, object zoom, condition plot, and 960-row summary | Yes | Imports compile | Passed on final outputs |
| 64-row paired-comparison table with layer-wise FDR correction | Yes | Imports compile | Passed on final outputs |
| 192-row conditional prediction table | Yes | Imports compile | Passed on final outputs |
| Zero-difference object/random controls recorded as valid nulls | Yes | Passed on qid 05334777 | Passed: 1,344/1,344, job 199960 |
| Text and all-image patches still require nonzero donor difference | Yes | Code inspection | Passed: job 199960 |

## Milestone 7: Correct-Answer Overlay Sweep

| Check | Implemented | Validation status |
|---|---:|---:|
| Exact shared 305-question subset | Yes | Manifest passed: 305/305 |
| Strongest incorrect no-text comparator frozen per sample | Yes | Pure metric test passed |
| Positive insertion means increased correct margin | Yes | Pure metric test passed |
| Positive restoration means removal of correct evidence | Yes | Pure metric test passed |
| All 32 layers and seven region instances | Yes | Static accounting: 136,576 |
| One-sample real-GPU metric gate | Yes | Passed: 448/448, job 203051 |
| Full sweep after successful gate | Yes | Running: job 203055 |
