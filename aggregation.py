"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def _unique_layer_indices(n_layers: int, offsets: tuple[int, ...]) -> list[int]:
    """Return valid, de-duplicated layer indices for negative offsets."""
    indices: list[int] = []
    for offset in offsets:
        idx = n_layers + offset if offset < 0 else offset
        idx = max(0, min(n_layers - 1, idx))
        if idx not in indices:
            indices.append(idx)
    return indices


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    valid_mask = attention_mask.to(device=hidden_states.device).bool()
    valid_positions = valid_mask.nonzero(as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        raise ValueError("attention_mask must contain at least one real token")

    last_pos = int(valid_positions[-1].item())
    recent_positions = valid_positions[-64:]
    selected_layers = _unique_layer_indices(
        hidden_states.shape[0],
        offsets=(-8, -6, -4, -2, -1),
    )

    pooled_features: list[torch.Tensor] = []
    for layer_idx in selected_layers:
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)
        valid_tokens = layer[valid_positions]
        recent_tokens = layer[recent_positions]

        pooled_features.extend(
            [
                layer[last_pos],
                recent_tokens.mean(dim=0),
                valid_tokens.mean(dim=0),
                recent_tokens.std(dim=0, unbiased=False),
            ]
        )

    return torch.cat(pooled_features, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    valid_mask = attention_mask.to(device=hidden_states.device).bool()
    valid_positions = valid_mask.nonzero(as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        raise ValueError("attention_mask must contain at least one real token")

    selected_layers = _unique_layer_indices(
        hidden_states.shape[0],
        offsets=(-8, -6, -4, -2, -1),
    )
    layer_means = torch.stack(
        [hidden_states[idx, valid_positions].mean(dim=0) for idx in selected_layers]
    )
    layer_norms = torch.linalg.vector_norm(layer_means, dim=1)

    if layer_means.shape[0] > 1:
        cosines = torch.nn.functional.cosine_similarity(
            layer_means[:-1],
            layer_means[1:],
            dim=1,
        )
    else:
        cosines = torch.zeros(0, device=hidden_states.device)

    seq_len = torch.tensor(
        [float(valid_positions.numel()) / float(attention_mask.numel())],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    return torch.cat([layer_norms, cosines, seq_len], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
