"""
Hybrid Contrastive Learning Branch for MANTA.

Paper Section 3.4, Eq. (11)-(13):
    "We introduce a hybrid contrastive learning branch that operates on
     three levels: supervised support-set contrastive (L^S_con),
     unsupervised query-set contrastive (L^Q_con), and combined
     support-query contrastive (L^SQ_con)."

The contrastive branch works in PARALLEL with the Mamba branch and
provides complementary learning signals. It operates DIRECTLY on the
temporal-pooled features without any additional projection head.

Loss Components:
    L^S_con  — Supervised contrastive on support set (labels available)
    L^Q_con  — Unsupervised contrastive on query set (pseudo-labels from episodic structure)
    L^SQ_con — Unsupervised contrastive on S ∪ Q (cross-set relationships)

    L_hc = L^S_con + L^Q_con + L^SQ_con  (Eq. 13)

InfoNCE Loss (Eq. 11-12):
    L_con(z, z⁺, {z⁻}) = -log[ exp(sim(z,z⁺)/τ) / (exp(sim(z,z⁺)/τ) + Σ exp(sim(z,z⁻)/τ)) ]

    where sim(·,·) = cosine similarity, τ = temperature (default 0.07)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    anchor: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE / NT-Xent contrastive loss for a single anchor.

    Implements Eq. (12):
        L = -log[ exp(sim(z, z⁺)/τ) / (exp(sim(z,z⁺)/τ) + Σ exp(sim(z,z⁻)/τ)) ]

    When multiple positives exist, averages the loss over all positive pairs.

    Args:
        anchor:    [D] — anchor feature vector
        positives: [P, D] — P positive samples (same class as anchor)
        negatives: [R, D] — R negative samples (different classes)
        temperature: Temperature scaling factor τ

    Returns:
        Scalar loss value
    """
    # L2 normalize for cosine similarity
    anchor = F.normalize(anchor, dim=-1)        # [D]
    positives = F.normalize(positives, dim=-1)  # [P, D]
    negatives = F.normalize(negatives, dim=-1)  # [R, D]

    # Compute similarities
    pos_sims = torch.matmul(positives, anchor) / temperature  # [P]
    neg_sims = torch.matmul(negatives, anchor) / temperature  # [R]

    # For each positive, compute InfoNCE loss
    losses = []
    for i in range(pos_sims.shape[0]):
        # Numerator: exp(sim(z, z⁺_i) / τ)
        # Denominator: exp(sim(z, z⁺_i) / τ) + Σ exp(sim(z, z⁻_j) / τ)
        all_sims = torch.cat([pos_sims[i:i+1], neg_sims], dim=0)  # [1+R]
        log_denominator = torch.logsumexp(all_sims, dim=0)
        loss_i = -(pos_sims[i] - log_denominator)
        losses.append(loss_i)

    return torch.stack(losses).mean()


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    L^S_con: Supervised contrastive loss for the support set.

    For each anchor sample:
        - Positives = other samples with the SAME label
        - Negatives = samples with DIFFERENT labels

    Args:
        features: [N*K, D] — flattened support features (temporal-pooled)
        labels:   [N*K] — integer class labels for each sample
        temperature: Temperature τ

    Returns:
        Scalar loss value (averaged over all valid anchors)
    """
    device = features.device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for i in range(len(features)):
        anchor = features[i]
        anchor_label = labels[i]

        # Positive mask: same label, exclude self
        pos_mask = (labels == anchor_label)
        pos_mask[i] = False

        # Negative mask: different label
        neg_mask = (labels != anchor_label)

        positives = features[pos_mask]
        negatives = features[neg_mask]

        # Skip if no valid positives or negatives
        if positives.shape[0] == 0 or negatives.shape[0] == 0:
            continue

        loss = info_nce_loss(anchor, positives, negatives, temperature)
        total_loss = total_loss + loss
        count += 1

    return total_loss / max(count, 1)


def unsupervised_contrastive_loss(
    features: torch.Tensor,
    n_way: int,
    k_shot: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    L^Q_con or L^SQ_con: Unsupervised contrastive loss.

    Assumes features are ordered as:
        [class_0_shot_0, class_0_shot_1, ..., class_0_shot_{K-1},
         class_1_shot_0, ..., class_{N-1}_shot_{K-1}]

    For each anchor x^n_k:
        - Positives: K-1 other samples from the same class position
        - Negatives: (N-1)*K samples from other class positions

    This is "unsupervised" because we don't use explicit labels — instead
    we rely on the episodic structure (samples at same class position
    are assumed to be the same class).

    Args:
        features: [N*K, D] — ordered features
        n_way: Number of classes N
        k_shot: Number of shots per class K
        temperature: Temperature τ

    Returns:
        Scalar loss value
    """
    device = features.device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for n in range(n_way):
        for k in range(k_shot):
            anchor_idx = n * k_shot + k
            anchor = features[anchor_idx]

            # Positives: other shots from the same class
            pos_indices = [n * k_shot + kk for kk in range(k_shot) if kk != k]
            if len(pos_indices) == 0:
                continue
            positives = features[pos_indices]

            # Negatives: all samples from other classes
            neg_indices = [
                nn * k_shot + kk
                for nn in range(n_way) if nn != n
                for kk in range(k_shot)
            ]
            if len(neg_indices) == 0:
                continue
            negatives = features[neg_indices]

            loss = info_nce_loss(anchor, positives, negatives, temperature)
            total_loss = total_loss + loss
            count += 1

    return total_loss / max(count, 1)


class HybridContrastiveLoss(nn.Module):
    """
    Hybrid Contrastive Learning Loss combining three components.

    Implements Eq. (13):
        L_hc = L^S_con + L^Q_con + L^SQ_con

    Components:
        L^S_con:  Supervised contrastive on support set (uses ground-truth labels)
        L^Q_con:  Unsupervised contrastive on query set (uses episodic structure)
        L^SQ_con: Unsupervised contrastive on combined S ∪ Q

    NOTE: No projection head is used — the paper operates directly on
    temporal-pooled features from the Matryoshka Mamba output.

    Args:
        temperature: Temperature scaling factor τ (default 0.07)
    """

    def __init__(self, temperature: float = 0.07, d_model: int = 2048):
        super().__init__()
        self.temperature = temperature
        # No projection head — paper operates directly on pooled features

    def forward(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> dict:
        """
        Compute hybrid contrastive loss.

        Operates directly on temporal-pooled features (no projection head).

        Args:
            support_features: [N, K, F, D] or [N*K, F, D] — support set features
            query_features:   [Q, F, D] or [Q, D] — query set features
            support_labels:   [N*K] — integer class labels for support
            n_way: Number of classes in episode
            k_shot: Number of shots per class

        Returns:
            dict with:
                'loss': Total L_hc = L_S + L_Q + L_SQ
                'L_S': Supervised contrastive loss value
                'L_Q': Unsupervised query contrastive loss value
                'L_SQ': Unsupervised combined contrastive loss value
        """
        # === Temporal pooling: [*, F, D] → [*, D] ===
        # Direct mean pooling over frames — no projection head per paper
        if support_features.dim() == 4:
            N, K, F, D = support_features.shape
            sup_pooled = support_features.mean(dim=2).view(N * K, D)  # [N*K, D]
        elif support_features.dim() == 3:
            sup_pooled = support_features.mean(dim=1)  # [N*K, D]
        else:
            sup_pooled = support_features  # Already [N*K, D]

        if query_features.dim() == 3:
            que_pooled = query_features.mean(dim=1)  # [Q, D]
        else:
            que_pooled = query_features  # Already [Q, D]

        # === L^S_con: Supervised contrastive on support set ===
        L_S = supervised_contrastive_loss(
            sup_pooled, support_labels, self.temperature
        )

        # === L^Q_con: Unsupervised contrastive on query set ===
        Q = que_pooled.shape[0]
        q_per_class = Q // n_way  # Number of queries per class
        L_Q = unsupervised_contrastive_loss(
            que_pooled, n_way, q_per_class, self.temperature
        )

        # === L^SQ_con: Unsupervised contrastive on S ∪ Q ===
        # Combine support and query, treating each class as having K + q_per_class samples
        # Order: [sup_class0, que_class0, sup_class1, que_class1, ...]
        combined_features = []
        for c in range(n_way):
            # Support samples for class c
            sup_c = sup_pooled[c * k_shot: (c + 1) * k_shot]  # [K, D]
            # Query samples for class c (assuming ordered by class)
            que_c = que_pooled[c * q_per_class: (c + 1) * q_per_class]  # [q_per_class, D]
            combined_features.append(torch.cat([sup_c, que_c], dim=0))

        sq_features = torch.cat(combined_features, dim=0)  # [N*(K+q_per_class), D]
        sq_shots = k_shot + q_per_class

        L_SQ = unsupervised_contrastive_loss(
            sq_features, n_way, sq_shots, self.temperature
        )

        # === Total hybrid contrastive loss (Eq. 13) ===
        L_hc = L_S + L_Q + L_SQ

        return {
            'loss': L_hc,
            'L_S': L_S.item() if L_S.requires_grad else L_S,
            'L_Q': L_Q.item() if L_Q.requires_grad else L_Q,
            'L_SQ': L_SQ.item() if L_SQ.requires_grad else L_SQ,
        }
