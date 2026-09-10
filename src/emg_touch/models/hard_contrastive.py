"""Trial-level hard supervised contrastive objective."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def trial_class_embeddings(features, labels, valid):
    """Create at most one embedding per class per trial."""
    embeddings, classes = [], []
    for row in range(features.shape[0]):
        for label in (0, 1):
            mask = valid[row] & (labels[row] == label)
            if mask.any():
                embeddings.append(features[row, mask].mean(0))
                classes.append(label)
    if not embeddings:
        return features.new_zeros((0, features.shape[-1])), labels.new_zeros((0,))
    return torch.stack(embeddings), torch.as_tensor(classes, device=labels.device)


def hard_trial_contrastive_loss(features, labels, valid, margin=.35):
    """Hardest-positive / closest-negative cosine triplet loss."""
    embeddings, classes = trial_class_embeddings(features, labels, valid)
    if len(embeddings) < 4 or any((classes == label).sum() < 2 for label in (0, 1)):
        return features.sum() * 0
    embeddings = F.normalize(embeddings, dim=-1)
    distance = 1 - embeddings @ embeddings.transpose(0, 1)
    eye = torch.eye(len(embeddings), dtype=torch.bool, device=features.device)
    same = classes[:, None].eq(classes[None, :]) & ~eye
    different = ~classes[:, None].eq(classes[None, :])
    hardest_positive = distance.masked_fill(~same, -torch.inf).max(1).values
    hardest_negative = distance.masked_fill(~different, torch.inf).min(1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()
