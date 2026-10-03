#!/usr/bin/env python3
"""Small, testable utilities for scoring complete autoregressive answers."""

from typing import Dict, List, Sequence

import torch


def scored_token_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, answer_start: int
) -> List[float]:
    """Return log p(y_j | prompt, y_<j) for each answer token."""
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("logits/input_ids must align as [batch, sequence, ...]")
    if input_ids.shape[0] != 1 or not 1 <= answer_start < input_ids.shape[1]:
        raise ValueError("expected one sequence and at least one scored answer token")
    prediction_logits = logits[0, answer_start - 1 : -1].float()
    targets = input_ids[0, answer_start:]
    values = prediction_logits.log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
    return [float(value) for value in values.detach().cpu()]


def summarize_answer_score(
    answer: str,
    token_ids: Sequence[int],
    decoded_tokens: Sequence[str],
    token_logprobs: Sequence[float],
) -> Dict:
    if not token_ids or len(token_ids) != len(decoded_tokens) or len(token_ids) != len(token_logprobs):
        raise ValueError("answer tokens, decoded tokens, and log probabilities must be nonempty and aligned")
    values = [float(value) for value in token_logprobs]
    return {
        "answer": answer,
        "token_ids": [int(value) for value in token_ids],
        "decoded_tokens": list(decoded_tokens),
        "token_logprobs": values,
        "token_count": len(values),
        "mean_logprob": sum(values) / len(values),
        "sum_logprob": sum(values),
    }


def answer_margin(correct: Dict, comparison: Dict) -> float:
    return float(correct["mean_logprob"] - comparison["mean_logprob"])


def strongest_incorrect(scores: Dict[str, Dict], correct_key: str = "correct_answer") -> str:
    candidates = {key: value for key, value in scores.items() if key != correct_key}
    if not candidates:
        raise ValueError("at least one incorrect candidate is required")
    return max(candidates, key=lambda key: candidates[key]["mean_logprob"])
