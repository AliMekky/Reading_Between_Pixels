# MCQ replication and dense Qwen3-VL-32B API evaluation

This run repeats MCQ on the six completed local models and adds the larger **dense** Qwen3-VL-32B-Instruct model in both open-ended and MCQ formats. The dataset is the existing **474-question behavioral evaluation**, not the 305-question causal subset. Each format uses all five image conditions, 2,370 generations per model/format.

## Models and scope

- LLaVA-1.5-7B, LLaVA-NeXT-7B, Qwen2.5-VL-7B, InternVL3.5-8B, Qwen3-VL-2B-Instruct, Qwen3-VL-8B-Instruct: fresh MCQ runs. Existing open-ended results remain available for later matched-format analysis.
- Qwen3-VL-32B-Instruct: API model ID `qwen3-vl-32b-instruct`, both formats, 4,740 full-run requests.
- Dense 32B API CPU submission: **271550**. The 20-request paired-format smoke test passed before submission.
- Local MCQ array submission: **271510**, two tasks running three models sequentially each, with a two-question/five-condition smoke test before every full run.

The initially selected `qwen3-vl-30b-a3b-instruct` is a different mixture-of-experts model. Its smoke test was interrupted when the intended dense 32B comparison was clarified. Its partial records are retained only for provenance in a separate directory; they are excluded from this study. No full 30B run was launched.

## Protocol

`run_generation.py` reuses the existing inference module's dataset loader, stable seed-42 option ordering, prompts, local model loaders, and generation implementation. MCQ asks for a letter with a 50-token cap, matching the existing MCQ default; open-ended asks for a short answer without options with a 32-token cap. Images use `cleaned_image` for overlays and unchanged `notext.image` for the clean baseline. The MCQ parser preserves the existing extraction rule and additionally stores a strict-letter field for parsing audits.

Local MCQ uses the existing audited generation method with an MCQ prompt, retaining only newly generated text and recording token IDs, image hash, and resolved model revision. API execution requests temperature zero; hosted generation need not be bitwise reproducible and provider preprocessing may differ from local inference. Record the provider model, request ID, token usage, stop reason, and fingerprint where returned. No revision or output token IDs are fabricated when the provider does not supply them. The API's returned model must match the requested model.

The API endpoint is `https://maas.qwencloudapi.com/compatible-mode/v1`. Credentials are loaded internally from `QWEN_API_KEY` in `<API_ENV_FILE>`; keys are never written to output records or logs. Images are transmitted losslessly as PNG. Transient API errors receive bounded retries; persistent failures stop rather than being silently counted as answers.

Sources: [QwenCloud 32B model](https://www.qwencloud.com/models/qwen3-vl-32b-instruct), [Qwen3-VL model family](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct).

## Launch and resume

Run `sbatch run_local.sh` for local MCQ. Run `sbatch run_api.sh` for dense 32B API smoke followed by full generation. Do not run two instances for the same model/output at once. Outputs are separated into `outputs/smoke/` and `outputs/full/`, then model and format/condition. Each answer is checkpointed. Resume requires matching configuration, prompts and image hashes; all expected questions must complete before `completion.json` is written.

This work generates responses; it does not label unreviewed open-ended paraphrases as errors or claim a completed seven-model semantic comparison. After generation, validate record coverage and image/option alignment, run the established deterministic and semantic evaluation stages, and compare each format's overlay effect against its own no-text baseline on paired questions. Keep API-versus-local deployment differences explicit in the interpretation.

## API resume after rate limiting

Job 271550 stopped on HTTP 429 `limit_requests` after 301 saved full-run answers. Job **271623** resumes those checkpoints. Requests are now spaced at least three seconds apart (at most 20 starts/minute), with up to eight attempts and 60–120-second rate-limit cooldowns; provider Retry-After is honored up to a bounded 300-second pause. Persistent errors still stop visibly. This changes transport pacing only, not model, prompt, decoding parameters, or sample selection.

## User-requested API pause

Job 271623 was cancelled at the user’s request to review cost before MCQ. No full MCQ requests had been saved; 2,256/2,370 full open-ended responses were saved. Do not resubmit the API job until the user authorizes resumption. Token usage is recorded in `api_usage_at_pause.json`; saved-response usage is not an invoice and may omit an in-flight request. Local MCQ jobs are unaffected.

## Open-ended-only authorization

The user authorized resuming open-ended generation only after the cost review. Job **272002**, submitted through `run_api_open_ended.sh`, passes `--only-format open_ended`, preserves the shared run configuration, and writes `completion_open_ended.json` when all 2,370 responses are saved. API MCQ remains paused and requires separate authorization.

## 32B open-ended evaluation

All 2,370 responses completed. Deterministic scoring resolves 1,397 exact matches; 973 cases use the established two-judge evaluation. Job **272091** waits for submitted batches, retries only invalid outputs with larger output limits, merges valid decisions, calculates paired metrics, and updates `evaluation_32b/results.md` and Section 17 of the main causal results report. Reports are explicitly provisional until semantic evaluation completes. The original six-model aggregates remain unchanged. API MCQ generation remains paused.

## Semantic judges paused by user

On 2026-09-22 the user requested cancellation of both judge APIs and a later retry. Worker 272091 was stopped; cancellation was requested for any active provider batch. Gemini had already completed. Saved generations and judge outputs are preserved. `evaluation_32b/PAUSED_BY_USER` blocks accidental finalizer resubmission. No retries are scheduled; renewed user authorization is needed to resume.

## Judge retries resumed

On 2026-09-22 the user authorized rerunning the remaining batches. Job **272641** runs provider retries independently: Gemini retries only its 798 malformed decisions and preserves 175 valid decisions. OpenAI waits for the cancelled original batch to release partial results, then retries only missing or invalid decisions. API answer generation, including MCQ, is not restarted. Final metrics and Markdown updates remain contingent on validated decisions from both judges.

## Dense 32B MCQ authorized

On 2026-09-22 the user authorized the full Qwen3-VL-32B MCQ API run. Job **272642** runs `run_api_mcq.sh` with `--only-format mcq`: 474 questions × five image conditions, 2,370 requests. It retains the three-second minimum request spacing and bounded rate-limit retries, and writes `completion_mcq.json` when complete. Open-ended generations are reused, not rerun. This supersedes earlier MCQ pause notes.

## Completed evaluation and format comparison

All seven MCQ runs (16,590 answers) and 32B open-ended generation are complete. Both judges now have 973 valid decisions: 1,397 deterministic matches plus 894 judge agreements and 79 disagreements retained as ambiguous cover all 2,370 32B answers. Two final Gemini retries used a constrained brief-reason enum to prevent repetitive output; semantic instructions and reference ordering were preserved. `retry_two_gemini.py` and raw outputs retain the audit trail.

Final 32B metrics are in `evaluation_32b/results.md`; the matched seven-model comparison is in `comparison/results.md`, mirrored in Sections 17–18 of the main Qwen3 results report. Run `write_32b_results.py` then `analyze_format_comparison.py` to regenerate. The original six-model aggregate tables remain unchanged. All 164 non-strict MCQ outputs have an unambiguous leading option letter consistent with the saved parser result; audit: `comparison/non_strict_mcq_audit.json`.

## Integrated main-paper results

The main report’s Section 3 now presents both formats for all seven models, with separate accuracy and adjusted-following tables, paired format amplification, and a discussion of prompt-option matching and initial question-conditioned plausibility. Sections 16–18 retain the detailed experiment and audit records. `integrate_main_behavior.py` documents the integration and produces the seven-model accuracy figure in `comparison/seven_model_accuracy.png` and `.pdf`. Tables are populated directly from `comparison/metrics.csv`.
