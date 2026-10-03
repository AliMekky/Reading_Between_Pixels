# Scene Text in VLM Question Answering: Code and Data

Code and data for the anonymous submission *A Mechanistic Analysis of Scene Text in VLM Question Answering*.
It contains the GUTIC benchmark (474 questions; clean image plus four text overlays per question), the behavioral
evaluation of seven VLMs, the causal analyses on Qwen3-VL-8B, and the scripts that produce every figure and appendix
table from saved analysis outputs.

## Layout

| Path | Contents | Paper |
|---|---|---|
| `dataset/` | GUTIC as a HuggingFace `datasets` folder, split into 95 MB parts | Sec. 3 |
| `vlms/inference/` | Model loaders and generation for the seven VLMs | behavior |
| `vlms/open_ended_evaluation/` | Open-ended generation, exact/edit-distance matching, two-judge semantic evaluation | behavior |
| `vlms/format_replication_20260921/` | MCQ vs open-ended comparison (incl. Qwen3-VL-32B via API) | behavior, App. A10 |
| `vlms/qwen3_vl_generation_causal/main_files/` | Complete-answer scoring, spatial mapping, activation patching (all layers) | formation |
| `vlms/qwen3_vl_generation_causal/attention_intervention/` | Directed attention blocking between token groups | readout |
| `vlms/qwen3_vl_generation_causal/free_generation_intervention/` | Blocking late text→answer attention during generation; fooled vs robust | generation |
| `vlms/qwen3_vl_generation_causal/relevance_intervention/` | Starting-plausibility (question-only / gray / clean / overlay) analysis | App. A8–A9 |
| `vlms/qwen3_vl_generation_causal/paper_plots/` | All figure and table scripts; `appendix_tables/` holds the LaTeX tables | all figures |
| `vlms/activation_patching/`, `vlms/attention_path_intervention/` | Shared utilities (dataset loading, region mapping, attention masks) and the 305-question selection manifests | — |
| `vlms/*/outputs/` (selected CSVs) | Saved analysis summaries read by the plotting scripts | — |

## Setup

```bash
pip install -r vlms/requirements.txt

# Reassemble and extract the dataset (1.5 GB) into vlms/activation_patching/hf_dataset_GUIC_cleaned/
cd dataset && sha256sum -c SHA256SUMS && cd ..
cat dataset/GUTIC_dataset.tar.part_* | tar -xf -

export REPO_ROOT=$(pwd)                 # used by the shell/Slurm launchers
export API_ENV_FILE=/path/to/keys.env   # only for judging / API runs: OPENAI_API_KEY, GEMINI_API_KEY, QWEN_API_KEY
```

The dataset loads with `datasets.load_from_disk("vlms/activation_patching/hf_dataset_GUIC_cleaned/anonymous__GUIC")`.
Each row has the question, the clean image (`notext`), the segmentation image, and for each overlay condition
(`correct_answer`, `misleading_groundable`, `misleading_ungroundable`, `irrelevant_word`) the overlaid word, its text
box, the original overlay image, and the cleaned overlay (`cleaned_image`) that differs from the clean image only
inside the text box.

## Reproducing the figures and tables

No GPU or model inference is needed; the scripts read the saved summaries included in the repository.

```bash
cd vlms/qwen3_vl_generation_causal/paper_plots
./make_all.sh        # writes figures/, appendix_figures/ and appendix_tables/
```

## Re-running the experiments

The `*.sh` launchers are Slurm scripts. Submit them from the repository root (log paths are relative to it) after
setting `REPO_ROOT`, and add your cluster's partition/account directives. Model checkpoints are pinned to the
revisions recorded in the code. Each experiment folder has its own README with the exact stages, validation checks,
and expected record counts.
