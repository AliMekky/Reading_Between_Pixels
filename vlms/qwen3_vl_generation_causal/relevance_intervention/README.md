# Relevance-constraint pilot

Question: does removing question-to-answer attention increase irrelevant-word adoption, and is that increase reduced by also removing text-to-answer attention?

## Fixed exploratory design

- Qwen3-VL-8B, pinned revision, eager float16, greedy generation, 32-token limit; no MCQ options.
- Twelve outcome-blind questions sampled from the shared 305 manifest with seed 271828; identical questions across all four overlays.
- Q means question tokens, T text-region visual tokens, R one same-size random visual region, A the current answer-prediction query. Spatial selection reuses positive-overlap mapping.
- Q blocking is tested separately in zero-based layers 18–23 and 30–35. The middle window follows the existing report's strong teacher-forced Q→A opposition. T and R blocking always use 30–35. These are limited middle/late exploratory tests, not a claim that other layers are irrelevant. No discovery or confirmatory inference from this pilot.
- Nine arms: baseline, T, R, Q-middle, Q-middle+T, Q-middle+R, Q-late, Q-late+T, Q-late+R.
- Every arm runs on overlay and paired no-text images, using that overlay's spatial mask. Total: 864 saved generations, plus 96 ordinary-forward baseline validation generations.
- This checks a route-level hypothesis, not whether the model explicitly represents relevance. R controls visual blocking, not a matched linguistic alternative to Q.

## Run and validation

Submit `sbatch run_pilot.sh` from this folder. Outputs are under `outputs/pilot_12`; logs are in the parent experiment's `logs` directory. Each generation is checkpointed. Configuration mismatch stops resume.

Expected checks: unchanged pixels outside the annotated box; identical paired input IDs and spatial mapping; disjoint question/image groups; equal-sized disjoint T/R; no-op generation identical to ordinary eager; blocked attention exactly zero; attention-row error at most 0.002. Any failed check stops the job, rather than silently dropping a question.

`sbatch run_pilot_cached.sh` runs the same design under `outputs/pilot_12_cached` with prompt-prefix KV reuse. Only queries predicting answers are intervened on, so earlier causal prompt states can be reused. The final prompt position is removed from the cached prefix and recomputed separately in each arm. Each arm receives an independent copy of the cache. All 72 generations for the first question are checked against uncached intervention generation, and all 96 baselines against ordinary uncached eager generation. Token mismatches stop the job. This is an execution optimization, not a different intervention or sample selection.

`python inspect_pilot.py outputs/pilot_12_cached` reports checkpoint progress and validation. Add `--details` to inspect raw answers. The first uncached submission was stopped for runtime; its partial checkpoints remain separate in `outputs/pilot_12`.

The console prints each raw answer and progress. Saved data include token IDs, source positions by layer, mapping, pixel checks and attention integrity.

## Read the results

`exact_counts.csv`: all arms, conditions and states; counts matching the overlay or correct reference, unmatched responses, and truncated generations. For correct overlays, correct and overlay counts coincide. Unmatched responses are not automatically errors; paraphrases require later semantic evaluation.

`exploratory_contrasts.csv`: let F be the fraction matching the overlay word. For arm X, D(X) = [F(overlay,X) − F(overlay,baseline)] − [F(no-text,X) − F(no-text,baseline)]. Positive D(Q) indicates increased overlay-specific adoption after question blocking.

Compare the incremental Q effect with T blocked, D(Q+T)−D(T), against Q with R blocked, D(Q+R)−D(R). Positive text-specific attenuation means removing text reduces the incremental Q effect more than removing random visual tokens does. Simply comparing Q+T against Q is insufficient: T blocking may lower adoption on its own.

An encouraging pattern would be positive D(Q) for irrelevant overlays, text-specific attenuation, and no predominant collapse into unmatched/truncated answers. A larger effect for irrelevant than misleading overlays would additionally support relevance selectivity, but is not required for the broader constraint hypothesis. Compare conditions on the same questions. Twelve questions cannot establish selectivity or rule it out: rare adoption and exact-match false negatives make a null result inconclusive. No paid judge calls or significance claims in this pilot. Any larger study must freeze windows, expand random controls, evaluate semantics, and test paired condition differences.

## Follow-up: scores beneath unchanged generations

`analyze_relevance_scores.py` audits all 864 pilot responses and reads the existing 305-question teacher-forced Q→A interventions across all six windows. It exports exact correct-minus-displayed margins, raw and paired-no-text-adjusted relative gains, blocked absolute scores, and exploratory paired condition contrasts under `outputs/relevance_score_followup/`. The three harmful overlay conditions are analyzed; correct-overlay pilot outcomes remain in the semantic audit.

The original attention files saved baseline margins but not their component absolute scores. Activation-patching baselines differ numerically and are not substituted. `sbatch score_pilot_baselines.sh` reproduces 144 eager-attention baseline scores on exactly the 12 pilot questions. Every recovered margin must match the original attention file within 0.001; cached/full and no-op checks are retained. Resume skips completed pairs.

After the scoring job completes, run:

```bash
python analyze_relevance_scores.py
python write_score_report.py
```

The final explanation, numerical tables, semantic-review provenance, and PNG/PDF figures are linked from `outputs/relevance_score_followup/report.md`. Absolute before/after component-score changes are exact only on the 12 reproduced questions. The full 305-question margin changes are exact from the original intervention records. Full-sample Q-minus-random component scores are explicitly labeled as comparisons against visual blocks, not unblocked absolute baselines or matched linguistic controls. No new paid judge calls or full-dataset generation run are performed by this follow-up.

## Question-conditioned baseline pilot

Submit `sbatch run_input_comparison.sh`, then run `python analyze_input_comparison.py` after completion. Outputs are under `outputs/input_comparison_12/`; the analyzer requires all 84 inputs. The same fixed 12 questions receive question-only, matched-size RGB-128 gray, clean-scene, and four overlay inputs. Each input scores all four candidate answers and generates one greedy answer: 336 scores and 84 generations. All inputs use the same modality-neutral answer instruction, so earlier image-specific-prompt scores are not substituted. No attention blocking is performed.

Vision features are reused within each visual input. All four scores and generated tokens are checked against native uncached execution for the first question's six visual states. Visual token layouts match across all six images for each question; candidate tokenization is checked across all seven states. Mean and summed candidate log probabilities, first-token probabilities, paired contrasts, generated-answer audit, and PNG/PDF figures are exported. Non-exact responses reuse the previous semantic audit where available; additional explicit review is saved separately in `semantic_review.json`. Unresolved labels remain visible.

This is an exploratory measurement of question-conditioned answer preferences and image-associated score changes, not isolation of a pure language prior or causal adjustment of Q→A effects. Gray images retain visual tokens but can trigger blank-image answers and are not assumed neutral or in-distribution.

## Full shared evaluation set

`sbatch run_input_comparison_full.sh` runs all 305 questions in the shared evaluation manifest, deterministically split across four GPU jobs. This is 2,135 generations and 8,540 candidate scores; the 12 pilot questions remain included. Outputs are checkpointed under `outputs/input_comparison_305/shard_00` through `shard_03`. The first question in each shard validates all six visual states against uncached execution. The final successful shard automatically runs `analyze_full_input_comparison.py` under a file lock; no extra Slurm submission is needed. Submission 269426 is the initial full run.

The full analyzer requires complete, disjoint coverage of the manifest. It exports image contributions relative to both question-only and gray baselines, each candidate's own-overlay contribution relative to the clean scene, all four-by-four overlay/candidate effects, candidate-versus-correct score margins in the same image, and direct paired differences between overlay conditions. Descriptive 95% bootstrap intervals resample whole questions (10,000 resamples, seed 271828); these are not multiplicity-adjusted significance tests. Summed log probabilities are exported as a sensitivity analysis. Generated-answer counts are explicitly normalized exact matches, with unmatched answers retained for semantic review rather than mislabeled as errors.

Results will be in `outputs/input_comparison_305/report.md` with CSVs and PNG/PDF figures. Rerun `python analyze_full_input_comparison.py` to regenerate the report after all shards finish. `analyze_full_input_comparison.sh` is an optional manual analysis submission script; the initial attempt to submit it as a dependency was rejected by the cluster's job-count limit, so automatic analysis instead runs within the final successful shard. The expanded analyzer was checked on the completed pilot and reproduces all four pilot overlay gains exactly.

After the full analysis, run `python analyze_input_distributions.py` to generate `distribution_report.md`, six ECDF panels, paired before/after margin scatter plots, quantiles, and margin-crossing counts. Both mean and summed log-probability summaries are saved. These are distributions across questions, not full-vocabulary model distributions.
