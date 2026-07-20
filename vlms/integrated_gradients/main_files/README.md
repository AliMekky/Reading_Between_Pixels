# Integrated Gradients — Main Scripts

Pipeline for computing **token-space Integrated Gradients (IG)** attributions over VLM
image tokens on the GUIC MCQ dataset, then testing whether the model's attribution mass
concentrates on the *correct object*, the *misleading object*, or the *overlaid text*
region of the image.

Three stages:

1. **Extraction** — `updated_ig.py` (LLaVA-NeXT) / `ig_qwen.py` (Qwen2.5-VL) run the
   model once per question to pick an answer letter, then compute IG attributions of
   that answer over the packed image-token embeddings (via Captum), mapping each score
   back to a pixel bbox and caching everything to one `.npz` per question.
2. **Region detection** — `ig_regions.py` loads one question's IG `.npz`, builds three
   attribution "targets" (the correct-answer object, the misleading object, and the
   chosen variant's overlaid-text bbox), and for each one dynamically decides whether the
   surrounding attribution is net-positive or net-negative and grows a same-signed
   connected region around it, reporting region size/mass/IoU-vs-annotation.
   `run_ig_for_all.py` batches this over every (variant, qid) pair found under an
   extraction output root.
3. **Plotting** — `plot_ig.py` renders a publication-quality 2-row × 5-column figure
   (BASE/MOSAIC encoding × the 5 variants) with signed IG overlays for one question.

`run_ig.sh`, `run_ig_qwen.sh`, and `ig_regions_detection.sh` are SLURM wrappers for
stages 1 and 2.

---

## 0. Setup

From `Reading_Between_Pixels/vlms/`:

```bash
pip install -r requirements.txt   # includes captum, matplotlib, scipy
pip install qwen-vl-utils         # only needed for ig_qwen.py
```

All scripts pull the **GUIC** dataset (`AHAAM/GUIC`) via a `get_or_download_hf_dataset`
helper duplicated in each file, caching it to `--hf_cache_dir` on first run. Most default
to `--ids_file ../inference/no_overlap_question_ids.txt`, so run from inside
`main_files/` (or pass the flag explicitly).

---

## 1. `updated_ig.py` — Stage 1, LLaVA-NeXT token-space IG

Per question: builds the MCQ prompt, greedily predicts an answer letter, then runs
**Integrated Gradients** (Captum) over the model's *packed image-token embeddings*
(obtained via `model.get_image_features(...)`) with a mid-grey blank image as baseline,
targeting either:
- `teacher_forced` — the actual last input token when the answer letter is appended to
  the conversation, or
- `prefill_next_token` (default) — the log-probability of the answer-letter token at the
  next-token position right after the prompt.

It runs an IG completeness sanity check (`sum(attributions)` vs. `f(x) − f(baseline)`,
warns if relative error > 15%) — **printed only, not saved** to any output file. It then
maps the resulting per-image-token scores back to `(base_patch, mosaic_patch)` pixel grids
using the same anyres tiling/unpadding logic as `attention_llava_next.py` in the sibling
`attention_weights/` folder, and saves diverging (signed) overlays + the raw scores.

**Run directly:**
```bash
python updated_ig.py \
  --variant misleading_groundable \
  --shuffle_options \
  --out_dir ./llava-next_ig_token_outputs/misleading_groundable \
  --viz_signed --save_grids --block_overlay \
  --start 0 --end 500
```

**Key arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `--model_id` | `llava-hf/llava-v1.6-mistral-7b-hf` | HF model id |
| `--hf_dataset` | `AHAAM/GUIC` | dataset repo id |
| `--hf_cache_dir` | `./hf_dataset_GUIC` | local dataset cache |
| `--split` | `test` | dataset split |
| `--ids_file` | `../inference/no_overlap_question_ids.txt` | qid whitelist |
| `--variant` | `notext` | image variant |
| `--shuffle_options` | off | shuffle A/B/C/D deterministically per qid+seed |
| `--seed` | `42` | shuffle seed |
| `--mode` | `prefill_next_token` | `teacher_forced` or `prefill_next_token` (see above) |
| `--steps` | `256` | Captum IG integration steps |
| `--kinds` | `mosaic_patch,base_patch` | **parsed but never used — has no effect** |
| `--out_dir` | `./ig_token_outputs` | output root |
| `--max_samples` | `0` (no cap) | stop after N newly processed samples |
| `--device` | `cuda` | `cuda` or `cpu` (silently falls back to CPU if CUDA unavailable) |
| `--viz_signed` | off | **parsed but never used** — overlays are always signed regardless |
| `--viz_clamp` | `0.0` | if > 0, clip the signed grid to `[-x, x]` before normalizing |
| `--viz_diverging` | `RdBu_r` | matplotlib diverging colormap for overlays |
| `--save_grids` | off | also store the raw `base_grid_signed`/`mosaic_grid_signed` 2D arrays in the npz |
| `--debug_permutation` | off | also save overlays computed from a randomly shuffled copy of the scores, as a visual null check |
| `--block_overlay` | off | nearest/pixelated overlay instead of smooth bilinear |
| `--predicted` | off | attribute the *predicted* letter instead of the correct one (help text says "prefill_next_token mode only" but this isn't actually enforced — it applies under `teacher_forced` too) |
| `--region_bbox` | `""` | optional `"y0,x0,y1,x1"` pixel ROI; if set, also runs a permutation significance test of that region's attribution mass and saves `region_stats.json` |
| `--start` | `0` | dataset start index |
| `--end` | `500` | dataset end index (exclusive) — ⚠️ **default is `500`, not `0`**, even though the help text describes `0` as "no limit." Omitting `--end` silently scans only the first 500 dataset rows. |

⚠️ **Resume-skip is broken.** `main()` lists `out_dir`'s immediate subdirectories and
skips a qid if it appears there — but those immediate subdirectories are variant names
(e.g. `notext`), never qids, so the check never actually matches and reruns always
reprocess (and overwrite) existing output.

⚠️ **The `.npz` is written twice per sample**, and the second write fully overwrites the
first with more keys — harmless but means the first `np.savez_compressed` call is wasted
work.

---

## 2. `ig_qwen.py` — Stage 1, Qwen2.5-VL token-space IG

Same idea as `updated_ig.py` for `Qwen/Qwen2.5-VL-7B-Instruct`, adapted to Qwen's simpler
token layout: image patches merged into a single grid of `merge_size × merge_size` blocks
(no base/mosaic split), located via `<|image_pad|>` tokens. Structurally simpler and
cleaner than `updated_ig.py` — no known dead-flag or double-write issues.

**Run directly:**
```bash
python ig_qwen.py \
  --variant misleading_groundable \
  --shuffle_options \
  --out_dir ./qwen-vl_ig_token_outputs/misleading_groundable \
  --save_grids --block_overlay --predicted \
  --start 0 --end 500
```

**Key arguments** — same as `updated_ig.py` (`--model_id` defaults to
`Qwen/Qwen2.5-VL-7B-Instruct`, no `--kinds`/`--viz_signed`/`--viz_clamp`/`--viz_diverging`/
`--debug_permutation`/`--region_bbox` flags), plus:

| Argument | Default | Meaning |
|---|---|---|
| `--top_k` | `50` | how many top-scoring tokens `draw_topk_token_boxes` outlines (only used when `--block_overlay` is **not** set) |
| `--predicted` | off | attribute the predicted letter instead of the correct one (works in either `--mode`) |
| `--end` | `0` | (no-limit sentinel actually works correctly here, unlike `updated_ig.py`) |

No resume-skip logic at all in this script — every run reprocesses every matching qid
from scratch (the loop only checks the qid whitelist and `--start`/`--end`, not existing
output directories).

---

## Output format for stages 1 (`ig_{mode}.npz`)

Both extraction scripts write to `out_dir/{variant}/{qid}/`:

| File | Contents |
|---|---|
| `ig_{mode}.npz` | `token_scores` (float32, one signed IG score per packed image token), `meta` (JSON: mode, answer_letter, target_token_id, seq_len, logit_pos, n_img_tokens, embed_dim, steps), `mapping_summary` (JSON, tiling/grid geometry), `mapping_tokens` (JSON, per-token `{token_idx, kind, row, col, bbox}`), and — if `--save_grids` — `base_grid_signed`+`mosaic_grid_signed` (LLaVA) or `qwen_grid_signed` (Qwen) |
| `overlay_base_{mode}.png`, `overlay_mosaic_{mode}.png` (LLaVA) / `overlay_grid_{mode}.png` or `overlay_boxes_{mode}.png` (Qwen) | signed diverging overlay(s) of the attribution grid(s) on the original image |
| `run_info.json` | qid, variant, mode, predicted/correct letter, correctness, output paths, `meta` |
| `region_stats.json` (LLaVA only, if `--region_bbox` set) | ROI attribution mass + permutation-test significance |
| `out_dir/mismatched_qids.txt` | qids where `token_scores` length didn't match the expected packed-token count |

Because the shell wrappers pass `--out_dir <root>/<variant>` and the scripts themselves
also nest by `<variant>` internally, the on-disk layout ends up **double-nested**:
`<root>/<variant>/<variant>/<qid>/ig_<mode>.npz` — this is the layout `ig_regions.py`,
`run_ig_for_all.py`, and `plot_ig.py` all expect when reading these caches back.

---

## 3. `run_ig.sh` / `run_ig_qwen.sh` — SLURM wrappers for stage 1

SBATCH scripts (1 GPU, 60G RAM, 16 CPUs, 24h) that loop stage 1 over all five variants
for one model:

```bash
sbatch run_ig.sh        # updated_ig.py, LLaVA-NeXT
sbatch run_ig_qwen.sh   # ig_qwen.py, Qwen2.5-VL
```

Both hardcode `start`/`end` (LLaVA: `400`/`500`; Qwen: `0`/`500`) and a `model`/`output_dir`
pair near the top — edit those, and the `cd`/`conda activate` lines, before submitting.
`run_ig.sh` additionally redirects each variant's stdout to `logs/ig_${variant}_${start}_${end}.log`
(make sure `logs/` exists, or add `mkdir -p logs` — unlike `jobs_logs/` for SBATCH's own
output, this directory isn't created automatically).

---

## 4. `ig_regions.py` — Stage 2, per-question 3-region attribution scoring

⚠️ **Half the file is dead code.** Lines 1–1086 are an entire earlier draft, commented
out line-by-line (not a docstring — every line prefixed with `#`); only the second copy
starting at the second `#!/usr/bin/env python3` (line 1087 onward) is live. Ignore the
first half when reading or editing this file.

For **one** `(question_id, variant, npz)` triple, builds three attribution targets from
the GUIC annotations: `correct_object` (the correct answer's grounded object bbox),
`misleading_object` (the misleading-groundable object's bbox), and `{variant}_text` (the
chosen variant's overlaid-text bbox). For each target it:

1. Sums the signed IG grid inside the annotation bbox to dynamically choose whether this
   target is a **positive** or **negative** attribution region.
2. Builds a Gaussian-smoothed, sign-masked score map; finds the top-K globally strongest
   same-sign cells (`--topk_global`) and keeps the ones inside an expanded search box
   around the annotation (falling back to the single strongest in-box cell if none of the
   global top-K land there).
3. Grows an 8-connected same-sign region from each surviving seed (flood-fill, gated by
   `--alpha`/`--edge_tau`/`--floor_percentile`), unions them, dilates
   (`--dilate_iters`) and fills holes.
4. Computes region size/sum/mean/max, bbox-IoU and mask-IoU vs. the annotation, connected
   components, and saves per-target overlay PNGs + a masks `.npz`.

All three targets' results are combined into one `report.json` per `(qid, variant)`.

**Run directly** (only makes sense for a `.npz` you already have from stage 1):
```bash
python ig_regions.py \
  --question_id 06199707 \
  --npz ./llava-next_ig_token_outputs/misleading_groundable/misleading_groundable/06199707/ig_prefill_next_token.npz \
  --variant misleading_groundable \
  --out_dir ./one_question_three_regions
```

**Key arguments** (only `--variant` has help text in the script — defaults below are exact):

| Argument | Default | Meaning |
|---|---|---|
| `--question_id` | `"06199707"` | which GUIC question to analyze |
| `--npz` | *(hardcoded machine-specific absolute path — always override this)* | path to the stage-1 `ig_{mode}.npz` |
| `--variant` | `misleading_groundable` | which variant's text bbox is the 3rd target |
| `--hf_dataset` | `AHAAM/GUIC` | dataset repo id |
| `--split` | `test` | dataset split |
| `--hf_cache_dir` | `./hf_dataset_GUIC` | local dataset cache |
| `--dataset_is_disk` | off | if set, `--hf_dataset` is treated as a local `load_from_disk` path instead of a Hub id (`--split`/`--hf_cache_dir` are then ignored) |
| `--grid_source` | `base` | `base` or `mosaic` (LLaVA only — ignored if the npz has a Qwen `qwen_grid_signed` array, which always wins) |
| `--smooth_sigma` | `0.7` | Gaussian smoothing sigma on the score map |
| `--alpha` | `0.70` | region-growth threshold as a fraction of the seed's peak score |
| `--edge_tau` | `0.08` | ⚠️ **accepted but has no effect** — internally recomputed from the data (90th-percentile of neighbor differences) before use, and the recomputed value is what actually drives region growth (though the original `--edge_tau` value is still echoed into `report.json`, misleadingly implying it was honored) |
| `--floor_percentile` | `65.0` | percentile of positive score-map values used as a growth-threshold floor |
| `--min_region_size` | `3` | minimum cells for a grown region to count as a valid result |
| `--expand_ratio` | `0.20` | fractional expansion of the annotation bbox for seed search |
| `--expand_min_pixels` | `10` | minimum pixel expansion (floor on `--expand_ratio`) |
| `--dilate_iters` | `1` | binary dilation iterations on the unioned region mask |
| `--topk_global` | `20` (via `run_ig_for_all.py`'s CLI default) / `50` (this script's own internal default) | number of globally top-scoring seed candidates to consider — see the tuning note below |
| `--out_dir` | `./one_question_three_regions_dynamic_tau` | output root |

⚠️ **`--topk_global` needs tuning per model.** A comment in `run_ig_for_all.py` next to
its own `--topk_global` flag notes *"50 for llava next and 20 for qwen vl"* — the driver
script's CLI default (`20`) is tuned for Qwen, not LLaVA-NeXT; pass `--topk_global 50`
explicitly when running against LLaVA-NeXT outputs.

**Outputs**, under `out_dir/<question_id>/<variant>/`: `{name}_overlay.png`,
`{name}_grid_debug.png`, `{name}_components_overlay.png`, `{name}_masks.npz` per target
(only for targets where a region was found), `all_three_regions_on_base.png` (all three
overlaid on the `notext` image), and `report.json` (per-target stats + the params used).

---

## 5. `run_ig_for_all.py` — Stage 2 batch driver

Finds every `ig_{mode}.npz` under `--ig_root` (`rglob`, so it works with the
double-nested `<variant>/<variant>/<qid>/` layout stage 1 produces), filters to the
requested `--variants` and optionally an `--ids_file` whitelist, and shells out to
`ig_regions.py` once per `(variant, qid)` as a subprocess, forwarding all the region-growth
hyperparameters listed above.

**Run:**
```bash
python run_ig_for_all.py \
  --region_script ./ig_regions.py \
  --ig_root ./llava-next_ig_token_outputs \
  --out_dir ./one_question_three_regions_mask_based_strict_sign \
  --ids_file ../inference/no_overlap_question_ids.txt \
  --topk_global 50 \
  --skip_existing
```

**Key arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `--region_script` | *(required)* | path to `ig_regions.py` |
| `--ig_root` | *(required)* | root to search for `ig_<mode>.npz` files |
| `--out_dir` | *(required)* | output root, forwarded to each `ig_regions.py` call |
| `--mode` | `prefill_next_token` | `teacher_forced` or `prefill_next_token` — selects which `ig_<mode>.npz` filename to look for |
| `--variants` | all 5, comma-separated | which variants to include |
| `--ids_file` | `""` (no filter) | optional qid whitelist |
| `--hf_dataset`, `--split`, `--hf_cache_dir` | `AHAAM/GUIC`, `test`, `./hf_dataset_GUIC` | forwarded to each subprocess |
| `--grid_source` ... `--topk_global` | same defaults as `ig_regions.py` (`--topk_global` default here is `20`) | forwarded to each subprocess |
| `--skip_existing` | off | skip jobs whose `out_dir/<qid>/<variant>/report.json` already exists (this check, unlike stage 1's, actually works correctly) |
| `--dry_run` | off | print the commands without running them |
| `--stop_on_error` | off | abort on first subprocess failure instead of continuing |

Prints a running `[i/N] variant=... qid=...` progress line and a final
`success=X failed=Y total=N` summary.

---

## 6. `ig_regions_detection.sh` — SLURM wrapper for stage 2

CPU-only SBATCH script (no GPU needed — region growing is pure NumPy/SciPy) that runs
`run_ig_for_all.py` once against a hardcoded `--ig_root`/`--out_dir` pair. Edit the `cd`
path and the `--ig_root`/`--out_dir` arguments (and add `--topk_global 50` if targeting
LLaVA-NeXT, per the tuning note above) before submitting.

```bash
sbatch ig_regions_detection.sh
```

---

## 7. `plot_ig.py` — Stage 3, publication figure

For one `--ques_id`, renders a 2-row (BASE / MOSAIC encoding) × 5-column (one per
variant) figure: each cell overlays that variant's signed IG grid on its image, with the
text bbox, correct-object bbox, and misleading-object bbox drawn on top, plus a shared
diverging colorbar and a question/options header panel.

⚠️ **The NPZ input root is hardcoded, not driven by `--out_dir`.** Inside the per-variant
loop, `load_npz(...)` is called with the literal string
`"./llava-next_ig_token_outputs_correct_answer_token"` — `--out_dir` only controls where
the *rendered figure* is saved, not where the `.npz` inputs are read from. Edit that
string in the source before running against a different stage-1 output directory.

**Run:**
```bash
python plot_ig.py \
  --ques_id 122313 \
  --mode prefill_next_token \
  --out_dir ./ig_plots \
  --show_stats
```

**Key arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `--ques_id` | `"122313"` | which question to render |
| `--out_dir` | `./ig_plots` | where to save the figure (default filename: `grid_{ques_id}_{mode}_2x5.png`) |
| `--mode` | `prefill_next_token` | which `ig_{mode}.npz` to load |
| `--hf_dataset`, `--split` | `AHAAM/GUIC`, `test` | dataset for question text/options/bboxes |
| `--unsigned` | off | use an unsigned magnitude colormap (`hot`) instead of the signed diverging one |
| `--cmap` | `RdBu_r` | colormap when signed |
| `--clip_p0`, `--clip_p1` | `5.0`, `95.0` | percentile clipping band for normalization |
| `--smooth_sigma_grid` | `1.0` | Gaussian smoothing applied to the upsampled overlay |
| `--base_alpha`, `--mask_thr`, `--alpha_gamma` | `0.9`, `0.12`, `1.7` | overlay transparency tuning (masks out near-zero cells, gamma-shapes the rest) |
| `--bbox_linewidth` | `3.0` | annotation box line width |
| `--wrap_width` | `140` | question text wrap width |
| `--show_stats` | off | annotate each cell with mean\|IG\| (overall + inside the text bbox) |
| `--responses` | `""` | optional path to a JSON `{variant: model_response_string}` to annotate each column |
| `--save_path` | `""` | override the output path entirely |
| `--dpi` | `200` | output resolution |

---

## `old_files/`

The sibling `old_files/` directory (`integrated_gradients.py`) holds an earlier,
single-file version of this pipeline kept for reference — not covered here.
