"""Build an additive mask that blocks directed attention edges."""

from typing import Sequence

import torch


def add_directed_edge_block(
    base_mask: torch.Tensor,
    source_positions: Sequence[int],
    destination_positions: Sequence[int],
) -> torch.Tensor:
    """Return ``base_mask`` with destination-query -> source-key edges blocked."""
    if base_mask.ndim != 4:
        raise ValueError("base_mask must have shape [batch, heads-or-1, queries, keys]")
    query_count, key_count = base_mask.shape[-2:]
    sources = sorted(set(int(position) for position in source_positions))
    destinations = sorted(set(int(position) for position in destination_positions))
    if not sources or not destinations:
        raise ValueError("source and destination positions must be nonempty")
    if min(sources) < 0 or max(sources) >= key_count:
        raise IndexError("source position outside the key axis")
    if min(destinations) < 0 or max(destinations) >= query_count:
        raise IndexError("destination position outside the query axis")

    combined = base_mask.clone()
    blocked_value = torch.finfo(combined.dtype).min
    for destination in destinations:
        combined[..., destination, sources] = blocked_value
    return combined
