# Qwen3-VL Generation Attention Intervention

This is Stage B of the generation causal experiment. It uses the open-ended
prompt and scores complete multi-token answers; no answer options are shown to
the model.

## Validation run

```bash
sbatch main_files/run_attention_smoke.sh
```

The run tests all six depth windows and all approved path families on one
paired no-text/overlay example. It must end with `[PASS]` and report zero
blocked attention probability, attention-row renormalization error at most
`2e-3`, no-op error at most `1e-3`, and exact record accounting.

Status: passed as job `229484` with 276/276 records, zero cached/no-op error,
zero blocked probability, and maximum row-sum error `0.000488`. The resumable
full-runner gate also passed as job `229501`.

## Full 305-question experiment

```bash
sbatch main_files/run_attention_full.sh
```

Two array tasks split the same 305 questions. Within each task, correct,
grounded misleading, ungrounded misleading, and irrelevant overlays run
sequentially. This layout respects the two-job QOS limit while preserving all
six windows and every approved pathway. Outputs are checkpointed under
`outputs/full/<condition>/shard_<0|1>/`.

Status: completed successfully as job `229504`. The audit verified 1,220
question-condition files and 323,472 records with zero validation failures.

Aggregate and reproduce the figures:

```bash
python main_files/analyze_attention_full.py
bash plotting_scripts/run_all_plots.sh
```

Question-level tables and statistical summaries are saved under
`outputs/analysis/`. Each independently editable figure and its interpretation
are under `outputs/plots/`.
