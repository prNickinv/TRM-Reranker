"""Loss functions for reranking."""

import torch
import torch.nn.functional as F


def pairwise_margin_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """Pairwise margin ranking loss."""
    return F.relu(margin - (pos_scores - neg_scores)).mean()


def listnet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ListNet loss using cross-entropy."""
    prob_scores = F.softmax(scores, dim=-1)
    prob_labels = F.softmax(labels.float(), dim=-1)
    return -torch.sum(prob_labels * torch.log(prob_scores + 1e-10), dim=-1).mean()
