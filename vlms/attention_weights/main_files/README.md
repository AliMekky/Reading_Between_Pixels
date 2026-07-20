# Attention Weights — Main Scripts

Pipeline for extracting and analyzing generation-step attention from VLMs (LLaVA-NeXT,
Qwen2.5-VL) on the GUTIC multiple-choice dataset, to see how much attention the model
puts on the text region vs. the correct/misleading object regions of the image when it
answers.

The pipeline has three stages that run in order:

1. **Extraction** — `attention_llava_next.py` / `attention_qwen.py` run the model on each
   question and cache the attention paid to every image token when generating the
   answer letter, saved as one `.npz` per question.
2. **Per-subset analysis** — `analyze_failure_cases.py` turns those `.npz` caches into
   layer-wise "attention density ratio" curves for a chosen subset of questions
   (e.g. only the ones the model got fooled on) and saves plots + a JSON report.
3. **Paper figures** — `generalization_plots.py` is a fixed-panel-layout version of
   stage 2 used to produce specific multi-panel comparison figures for the writeup.

The two `run_attention_*.sh` files are SLURM batch wrappers around stage 1.

---

## 0. Setup

From `Reading_Between_Pixels/vlms/`:

```bash
pip install -r requirements.txt
# Qwen additionally needs:
pip install qwen-vl-utils
```

Both extraction scripts expect the **T** dataset (`AHAAM/T` on Hugging Face) and a
question-id whitelist file. By default they look for:

- `../integrated_gradients/hf_dataset_T` — local cache dir for the HF dataset (downloaded
  automatically on first run if missing)
- `../inference/no_overlap_question_ids.txt` — newline-separated list of question IDs to
  process (only qids in this file are kept)

Run all commands below **from inside `main_files/`** so the relative default paths resolve
correctly, or override `--hf_cache_dir` / `--qid_file` explicitly.

⚠️ **Known mismatch:** `run_attention_qwen.sh` and `run_attention_llava_next.sh` invoke
`attention_output_prompt_qwen.py` and `attention_output_prompt.py` respectively — those
filenames don't exist in this folder. Edit the `python -u ...` line in each script to point
at `attention_qwen.py` / `attention_llava_next.py` before submitting the job.

---

## 1. `attention_llava_next.py` — Stage 1, LLaVA-NeXT

Runs `llava-hf/llava-v1.6-mistral-7b-hf` (or any LLaVA-NeXT checkpoint) on each T
question, generates one greedy token, and caches the attention that token paid to every
prompt token, then maps the image-token subset of that attention back onto pixel bounding
boxes on the original image.

**What it does, step by step, per question:**
1. Builds the 4-option MCQ prompt (optionally shuffling options deterministically per qid+seed).
2. Formats it with `processor.apply_chat_template(...)` and processes the chosen image
   variant (see "Variants" below).
3. Manual 2-step forward pass: a prefill pass over the prompt (`use_cache=True`,
   `output_attentions=False`) to get past KV cache + the greedy next-token id, then a single
   decode step for that token with `output_attentions=True`.
4. Averages that decode-step attention over heads, keeps only the slice pointing back at
   prompt tokens → `attn` array of shape `(num_layers, prompt_seq_len)`.
5. Locates the image placeholder token positions in `input_ids` and builds a full mapping
   from each *packed* image token → `(row, col, kind, bbox)` in original-image pixel
   coordinates. LLaVA-NeXT's anyres processing packs a downscaled "base" view plus a grid
   of high-res "mosaic" tiles (with a newline token ending each mosaic row) — this mapping
   models that layout exactly, including the "unpad" cropping logic the model itself uses.
6. Saves everything to one compressed `.npz` per question (see "Output format" below).
   Unlike an earlier version of this script, a mapping-count mismatch does **not** skip the
   sample — it's saved anyway with `meta["mapping_mismatch"]=True` and the qid is appended
   to `mismatched_qids.txt`.

**Run directly:**
```bash
python attention_llava_next.py \
  --variant misleading_groundable \
  --out_dir ./llava-next_attentions/misleading_groundable \
  --shuffle_options \
  --start 0 --end 500
```

**Key arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `--model_id` | `llava-hf/llava-v1.6-mistral-7b-hf` | HF model id |
| `--hf_dataset` | `AHAAM/T` | dataset repo id |
| `--hf_cache_dir` | `../integrated_gradients/hf_dataset_T` | local dataset cache |
| `--split` | `test` | dataset split |
| `--variant` | `misleading_groundable` | which image variant to feed (see below) |
| `--qid_file` | `../inference/no_overlap_question_ids.txt` | qid whitelist |
| `--out_dir` | `attn_cache_gen` | output root |
| `--max_samples` | `0` (no cap) | stop after N newly-saved samples |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--shuffle_options` | off | shuffle A/B/C/D order deterministically per qid |
| `--seed` | `42` | shuffle seed |
| `--debug_first_only` | off | process/save exactly one sample then exit |
| `--start`, `--end` | `0`, `500` | dataset index slice to process |

**Variants** (must match keys present in each T sample): `notext`,
`correct_answer`, `misleading_groundable`, `misleading_ungroundable`, `irrelevant_word`.
Each is a version of the image with a different piece of overlaid text (or none, for
`notext`); the MCQ options are always the same four candidate answers.

**Resuming:** the script skips a qid if `out_dir/<variant>/<qid>/` already exists, so a
crashed/killed run can just be restarted with the same command.

---

## 2. `attention_qwen.py` — Stage 1, Qwen2.5-VL

Same idea as above but for `Qwen/Qwen2.5-VL-7B-Instruct`, using **two model instances**:

- a **prediction model** (loaded exactly like the normal inference path, `device_map="auto"`,
  fp16) used only to pick the greedy next token — because Qwen's default attention backend
  reliably returns real logits but tends to return `None` for attentions on the decode step.
- a separate **attention model** (loaded with `attn_implementation="eager"`, fp32, single
  device) used only to redo the prefill + a decode step *forced* to the token chosen above,
  with `output_attentions=True`, so real attention tensors come back.

The rest of the pipeline mirrors the LLaVA script: average attention over heads for the
decode step, slice to prompt tokens, build a Qwen-specific image-token map (Qwen packs
image patches into a single `merged_patch` grid of `merge_size × merge_size` groups, no
base/mosaic split, no newline tokens), and save one `.npz` per question.

**Run directly:**
```bash
python attention_qwen.py \
  --variant misleading_groundable \
  --out_dir ./qwen-vl_attentions/misleading_groundable \
  --shuffle_options \
  --start 0 --end 500
```

**Key arguments** — same as `attention_llava_next.py` (`--model_id` defaults to
`Qwen/Qwen2.5-VL-7B-Instruct`), plus:

| Argument | Default | Meaning |
|---|---|---|
| `--device` | `cuda` | device for the **prediction** model |
| `--attn_device` | `cpu` | device for the **attention** model (kept separate/on CPU by default to avoid doubling GPU memory) |
| `--debug_topk` | `10` | how many top next-token candidates to print for debugging |

Same resume-by-skip-existing-qid-dir behavior as the LLaVA script.

---

## Output format (`gen_attn_gen_token.npz`)

Both extraction scripts write to `out_dir/<variant>/<qid>/gen_attn_gen_token.npz`
(and since the shell wrappers already pass `--out_dir .../<variant>`, the scripts' own
`variant` subfolder produces a **double-nested** path:
`out_dir/<variant>/<variant>/<qid>/gen_attn_gen_token.npz` — this is the layout the
downstream analysis scripts (`find_npz`) expect). Each `.npz` contains:

| Key | Shape / type | Meaning |
|---|---|---|
| `attn` | `(num_layers, prompt_seq_len)` float16 | per-layer, head-averaged attention from the generated token back to every prompt token |
| `meta` | JSON string | qid, variant, correct letter, predicted letter/token, option shuffle mapping, image size, mapping-mismatch flag, etc. |
| `packed_mapping_summary` | JSON string | grid/tiling geometry used to build the image-token map |
| `packed_mapping_tokens` | JSON string | list of `{token_idx, kind, row, col, bbox}` for every packed image token |
| `image_placeholder_positions` | int32 array | indices into `attn`'s prompt axis that correspond to image tokens, in `token_idx` order |
| `prompt_input_ids` | int32 array | the prompt's token ids |
| `attention_mask` | int32 array (optional) | prompt attention mask, if the processor returned one |

---

## 3. `run_attention_llava_next.sh` / `run_attention_qwen.sh` — SLURM job wrappers

SBATCH scripts (1 GPU, 60G RAM, 16 CPUs, 24h) that loop stage 1 over all five image
variants (`notext correct_answer misleading_groundable misleading_ungroundable
irrelevant_word`) for one model, writing each variant's output under
`./<model>_attentions/<variant>/`.

**Before submitting:** fix the python filename on the `python -u ...` line (see the
mismatch warning in §0), and update the `cd` path to your own checkout if it's not at
`/nfs-stor/ali.mekky/reading_between_pixels/...`.

**Submit:**
```bash
sbatch run_attention_llava_next.sh
sbatch run_attention_qwen.sh
```

Logs land in `jobs_logs/<job-name>_<job-id>.{out,err}`. Both scripts hardcode `start=0`
and `end=500` — edit those variables directly to change the slice.

---

## 4. `analyze_failure_cases.py` — Stage 2, per-subset density-ratio plots

Loads the `.npz` caches produced by stage 1 for one `variant` (plus its paired `notext`
run for the same qids) and computes, per transformer layer and per image "stream"
(`base`/`mosaic` for LLaVA, `merged` for Qwen):

```
density_ratio(region | stream) =
    (avg attention per image token that falls inside `region`, within `stream`)
    / (avg attention per image token in the whole `stream`)
```

for three regions: **text region** (the overlaid text's bbox), **correct object region**,
and **misleading object region** (both from the T sample's bounding-box annotations).
A layer's ratio is set to `NaN` if the stream received essentially no attention mass
(`< --min_denom`), so near-zero denominators don't blow up the ratio.

It restricts to a **subset** of questions selected by how the model's answer changed
between the `variant` run and the paired `notext` run:

| `--subset` | Meaning |
|---|---|
| `wrongly_consistent` (default) | wrong in both variant and notext, and the variant prediction specifically equals the `misleading_groundable` option |
| `fooled` | variant wrong AND notext correct AND the wrong answer matches this variant's misleading option (i.e. this variant's text is what caused the flip) |
| `robust` | correct in both variant and notext |
| `helped` | notext wrong → variant correct |
| `all` | keep every qid (no filtering) |

Questions not in the selected subset (or missing an `.npz`, mismatched layer counts, no
region annotations, etc.) are **not dropped from the loop** — they're kept as all-`NaN`
rows so the qid count in the report stays truthful even though they don't contribute to
the plotted mean/CI.

**Run:**
```bash
python analyze_failure_cases.py \
  --npz_root /path/to/llava-next_attentions \
  --variant misleading_groundable \
  --subset fooled \
  --out_dir plots_subset_noskip
```

**Key arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `--npz_root` | (hardcoded absolute path — override this) | root containing `<variant>/<variant>/<qid>/...npz` |
| `--variant` | `correct_answer` | one of `correct_answer`, `misleading_groundable`, `misleading_ungroundable`, `irrelevant_word` |
| `--qid_file` | `../inference/no_overlap_question_ids.txt` | qid whitelist; if empty, qids are taken from the variant's npz directory listing instead |
| `--out_dir` | `plots_subset_noskip` | plots + report land in `<out_dir>/<variant>/` |
| `--min_overlap_frac` | `0.25` | min fraction of a token's bbox that must overlap a region bbox to count as "in" that region |
| `--min_denom` | `1e-4` | minimum stream attention mass to treat a layer's ratio as valid |
| `--ci` | `95.0` | confidence interval percentile band on the plots |
| `--subset` | `wrongly_consistent` | see table above |

**Outputs**, written to `<out_dir>/<variant>/`, one triplet of PDFs per stream (`base`,
`mosaic`, or `merged`):

- `subset_only_<stream>_<subset>.pdf` — mean ± CI density ratio per region, variant run
- `subset_notext_only_<stream>_<subset>.pdf` — same, but for the paired notext run
- `subset_delta_vs_notext_<stream>_<subset>.pdf` — paired difference (variant − notext)
- `report_subset.json` — qid counts (total / selected / dropped by reason), settings used,
  and the list of output filenames

---

## 5. `generalization_plots.py` — Stage 3, fixed paper figures

Reuses the exact same density-ratio + subset logic as `analyze_failure_cases.py` (it's a
self-contained reimplementation, not an import) but instead of one ad-hoc plot per CLI
invocation, it hardcodes two specific multi-panel comparison figures used in the paper:

- **Fig A** (`figA_llava_misleading_groundable.{pdf,png}`) — 2×1 panel, LLaVA-NeXT,
  `misleading_groundable` variant: (a) `fooled` questions vs (b) `robust` questions, with a
  shaded "competition region" (layers 11–16) highlighted.
- **Fig BC** (`figBC_llava_changed_vs_stable.{pdf,png}`, currently commented out in
  `__main__`) — 2×4 grid, LLaVA-NeXT, one column per variant
  (`misleading_groundable`, `misleading_ungroundable`, `correct_answer`,
  `irrelevant_word`) × two rows (prediction changed vs. prediction stable), using the
  `fooled`/`helped`/`consistently_wrong` subset that makes sense per column.

**Before running:** edit the hardcoded path constants near the top of the file —
`LLAVA_NPZ_ROOT`, `QWEN_NPZ_ROOT`, `HF_DATASET`, `HF_CACHE_DIR`, `QID_FILE`, `OUTPUT_DIR` —
to match your setup. There are no CLI arguments.

**Run:**
```bash
python generalization_plots.py
```
This calls `make_fig_A(...)` only, by default (see `if __name__ == "__main__":` at the
bottom); uncomment the `make_fig_BC(...)` call to also produce Fig BC. Both `make_fig_A`
and `make_fig_BC` currently reference `LLAVA_NPZ_ROOT` even in the `PanelSpec`s labeled for
Qwen-style panels — update `npz_root=QWEN_NPZ_ROOT` there if you want a Qwen panel.

---

## `old_files/`

The sibling `old_files/` directory (not covered here) holds earlier iterations of this
pipeline kept for reference — see it directly if you need history on how the current
scripts evolved.
