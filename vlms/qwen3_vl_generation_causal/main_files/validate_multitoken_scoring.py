#!/usr/bin/env python3
"""One-question Qwen3-VL validation of full-answer sequence scoring."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict

import torch
from datasets import load_from_disk
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from sequence_scoring import answer_margin, scored_token_logprobs, strongest_incorrect, summarize_answer_score


ROOT = Path("/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels")
CACHE = ROOT / "vlms/activation_patching/hf_dataset_GUIC_cleaned/AHAAM__GUIC"
SELECTION = ROOT / "vlms/activation_patching/main_files/activation_patch_confirmation_selection_shared_305.json"
PREVIOUS = ROOT / "vlms/open_ended_evaluation/outputs/full"
PROMPT_INSTRUCTION = "Answer the question using the image. Give only the short answer without explanation."
REVISIONS = {
    "Qwen/Qwen3-VL-2B-Instruct": "89644892e4d85e24eaac8bacfd4f463576704203",
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
}
VARIANTS = ("correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word")


def log(section: str, message: str) -> None:
    print(f"[{section}] {message}", flush=True)


def image_hash(image) -> str:
    return hashlib.sha256(f"{image.mode}:{image.size}".encode() + image.tobytes()).hexdigest()


def to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if key == "pixel_values":
                value = value.to(torch.float16)
        moved[key] = value
    return moved


def prepare(processor, image, prompt: str, suffix: str, device: str):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt},
    ]}]
    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    batch = processor(
        text=[formatted + suffix], images=images, videos=videos,
        padding=True, return_tensors="pt",
    )
    return to_device(dict(batch), device), formatted


@torch.inference_mode()
def score_answer(model, processor, image, prompt: str, answer: str, prompt_ids: torch.Tensor, device: str) -> Dict:
    batch, formatted = prepare(processor, image, prompt, answer, device)
    prompt_length = int(prompt_ids.shape[1])
    full_ids = batch["input_ids"]
    if full_ids.shape[1] <= prompt_length or not torch.equal(full_ids[:, :prompt_length], prompt_ids):
        raise AssertionError("candidate tokenization changed the fixed prompt prefix")
    answer_ids = full_ids[0, prompt_length:].tolist()
    outputs = model(**batch, use_cache=False)
    token_logprobs = scored_token_logprobs(outputs.logits, full_ids, prompt_length)
    summary = summarize_answer_score(
        answer,
        answer_ids,
        [processor.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in answer_ids],
        token_logprobs,
    )
    reproduced = sum(summary["token_logprobs"]) / summary["token_count"]
    if not math.isfinite(summary["mean_logprob"]) or abs(reproduced - summary["mean_logprob"]) > 1e-9:
        raise AssertionError("saved per-token values do not reproduce the mean score")
    summary["decoded_sequence"] = processor.tokenizer.decode(answer_ids, skip_special_tokens=False)
    summary["prompt_length"] = prompt_length
    summary["formatted_prompt"] = formatted
    return summary


@torch.inference_mode()
def generate(model, processor, prompt_batch: Dict[str, torch.Tensor], prompt_length: int) -> Dict:
    result = model.generate(**prompt_batch, max_new_tokens=32, do_sample=False, return_dict_in_generate=True)
    output_ids = [int(value) for value in result.sequences[0, prompt_length:].tolist()]
    return {
        "output_token_ids": output_ids,
        "raw_response": processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
    }


def previous_record(model_id: str, variant: str, qid: str):
    path = PREVIOUS / f"{model_id.replace('/', '__')}_{variant}.jsonl"
    if not path.exists():
        return None
    with path.open() as handle:
        return next((json.loads(line) for line in handle if json.loads(line).get("question_id") == qid), None)


def save_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, choices=tuple(REVISIONS))
    parser.add_argument("--variant", default="misleading_groundable", choices=VARIANTS)
    parser.add_argument("--question_id", default="14412508")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    revision = REVISIONS[args.model_id]
    selection = json.loads(SELECTION.read_text())
    selected = {str(row["question_id"]): int(row["dataset_index"]) for row in selection["selected_samples"]}
    if args.question_id not in selected or len(selected) != 305:
        raise AssertionError("question must belong to the exact 305-question shared manifest")
    sample = load_from_disk(str(CACHE))[selected[args.question_id]]
    prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion: {sample['question']}"
    if "Options:" in prompt or "A, B, C, or D" in prompt:
        raise AssertionError("MCQ options leaked into the open-ended prompt")

    log("CONFIG", f"model={args.model_id} revision={revision} dtype=float16 device={args.device}")
    log("CONFIG", f"qid={args.question_id} variant={args.variant} shared_sample=305 prompt={prompt!r}")
    log("EXPECTED", "prompt prefixes identical; answer tokens nonempty; finite per-token scores; saved means exactly reproduced")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, revision=revision, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, revision=revision)
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved != revision:
        raise AssertionError(f"resolved revision {resolved} != requested {revision}")
    layers = model.model.language_model.layers
    deepstack = list(model.config.vision_config.deepstack_visual_indexes)
    log("MODEL", f"resolved_revision={resolved} decoder_layers={len(layers)} deepstack_vision_indexes={deepstack}")

    images = {
        "no_text": sample["notext"]["image"].convert("RGB"),
        "overlay": sample[args.variant]["cleaned_image"].convert("RGB"),
    }
    references = {key: str(sample[key]["text"]) for key in VARIANTS}
    report = {
        "status": "complete", "model_id": args.model_id, "model_revision": revision,
        "question_id": args.question_id, "variant": args.variant, "question": sample["question"],
        "prompt": prompt, "decoder_layers": len(layers), "deepstack_vision_indexes": deepstack,
        "references": references, "states": {},
    }

    for state, image in images.items():
        prompt_batch, formatted = prepare(processor, image, prompt, "", args.device)
        prompt_length = int(prompt_batch["input_ids"].shape[1])
        image_tokens = int((prompt_batch["input_ids"] == model.config.image_token_id).sum())
        scores = {
            key: score_answer(model, processor, image, prompt, answer, prompt_batch["input_ids"], args.device)
            for key, answer in references.items()
        }
        generation = generate(model, processor, prompt_batch, prompt_length)
        generation_score = score_answer(
            model, processor, image, prompt, generation["raw_response"], prompt_batch["input_ids"], args.device
        )
        prior = previous_record(args.model_id, "notext" if state == "no_text" else args.variant, args.question_id)
        prior_check = {"available": prior is not None, "comparable": False}
        if prior is not None:
            comparable = prior.get("model_revision") == revision and prior.get("image_sha256") == image_hash(image)
            prior_check.update({
                "comparable": comparable,
                "revision_match": prior.get("model_revision") == revision,
                "image_hash_match": prior.get("image_sha256") == image_hash(image),
                "token_ids_match": generation["output_token_ids"] == prior.get("output_token_ids"),
                "response_match": generation["raw_response"] == prior.get("raw_response"),
                "previous_response": prior.get("raw_response"),
                "previous_token_ids": prior.get("output_token_ids"),
            })
            if comparable and not prior_check["token_ids_match"]:
                raise AssertionError("deterministic generation does not reproduce the prior comparable run")
        top_key = max(scores, key=lambda key: scores[key]["mean_logprob"])
        report["states"][state] = {
            "image_sha256": image_hash(image), "image_size": list(image.size),
            "prompt_length": prompt_length, "image_token_count": image_tokens,
            "formatted_prompt": formatted, "candidate_scores": scores,
            "top_reference_key": top_key, "generation": generation,
            "generation_score": generation_score, "previous_generation_check": prior_check,
        }
        log("INPUT", f"state={state} prompt_tokens={prompt_length} image_tokens={image_tokens} image_size={image.size}")
        for key, score in scores.items():
            log("SCORE", f"state={state} key={key} answer={score['answer']!r} ids={score['token_ids']} decoded={score['decoded_tokens']} mean={score['mean_logprob']:.6f}")
        log("GENERATION", f"state={state} response={generation['raw_response']!r} ids={generation['output_token_ids']} prior_check={prior_check}")

    comparison = args.variant
    if args.variant == "correct_answer":
        comparison = strongest_incorrect(report["states"]["no_text"]["candidate_scores"])
    for state in ("no_text", "overlay"):
        scores = report["states"][state]["candidate_scores"]
        margin = answer_margin(scores["correct_answer"], scores[comparison])
        report["states"][state]["comparison_key"] = comparison
        report["states"][state]["margin_correct_minus_comparison"] = margin
        log("MARGIN", f"state={state} correct_vs={comparison} margin={margin:.6f}")
    report["overlay_minus_no_text_margin"] = (
        report["states"]["overlay"]["margin_correct_minus_comparison"]
        - report["states"]["no_text"]["margin_correct_minus_comparison"]
    )
    save_json(args.output, report)
    log("OUTPUT", f"saved={args.output}")
    log("PASS", "multi-token scoring validation complete")


if __name__ == "__main__":
    main()
