"""Pure-PyTorch K-nearest neighbor search, replacing chamferdist."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KNNResult:
    """Result of KNN search.

    Attributes:
        dists: Distance to nearest neighbor, shape [B, N, K].
        idx: Index of nearest neighbor in target, shape [B, N, K].
    """

    dists: torch.Tensor
    idx: torch.Tensor


def knn_points(
    src: torch.Tensor, tgt: torch.Tensor, k: int = 1, chunk_size: int = 10000
) -> KNNResult:
    """Find K-nearest neighbors of src points in tgt point cloud.

    Uses chunked distance computation to bound memory usage.

    Args:
        src: Source points, shape [B, N, 3].
        tgt: Target points, shape [B, M, 3].
        k: Number of nearest neighbors. Default: 1.
        chunk_size: Process src points in chunks of this size.

    Returns:
        KNNResult with dists [B, N, K] and idx [B, N, K].
    """
    B, N, D = src.shape
    _, M, _ = tgt.shape

    dists_list = []
    idx_list = []

    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        src_chunk = src[:, start_idx:end_idx, :]  # [B, chunk, 3]

        # Compute squared distances: [B, chunk, M]
        diff = src_chunk.unsqueeze(2) - tgt.unsqueeze(1)  # [B, chunk, 1, 3] - [B, 1, M, 3] -> [B, chunk, M, 3]
        sq_dist = torch.sum(diff**2, dim=3)  # [B, chunk, M]

        # Get k-nearest
        if sq_dist.shape[-1] < k:
            # If target has fewer than k points, pad with zeros
            dists_chunk = sq_dist  # [B, chunk, M]
            idx_chunk = torch.arange(
                sq_dist.shape[-1], device=sq_dist.device, dtype=torch.int64
            )
            idx_chunk = idx_chunk.unsqueeze(0).unsqueeze(0).expand(
                B, end_idx - start_idx, -1
            )
        else:
            dists_chunk, idx_chunk = torch.topk(
                sq_dist, k=k, dim=2, largest=False
            )  # [B, chunk, k]

        dists_list.append(dists_chunk)
        idx_list.append(idx_chunk)

    # Concatenate chunks
    dists = torch.cat(dists_list, dim=1)  # [B, N, k]
    idx = torch.cat(idx_list, dim=1)  # [B, N, k]

    # Convert squared distances to actual distances
    dists = torch.sqrt(torch.clamp(dists, min=0.0))

    return KNNResult(dists=dists, idx=idx)
