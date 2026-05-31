"""
MANTA — Main Framework: Matryoshka MAmba and CoNtrasTive LeArning.

Paper: "Manta: Enhancing Mamba for Few-Shot Action Recognition of Long Sub-Sequence"
       arXiv:2412.07481v7, AAAI 2025

Architecture Overview (Figure 2):
    ┌─────────────────────────────────────────────────────────────────┐
    │                        MANTA Framework                          │
    │                                                                 │
    │  Input: Support S = {(V^c_k, y^c)} and Query Q = {V^r}        │
    │                                                                 │
    │  ① Backbone ψ (frozen): Extract per-frame features              │
    │     S^c_k, Q^r ∈ ℝ^{F×D}                                       │
    │                                                                 │
    │  ② Matryoshka Mamba Branch:                                     │
    │     ├─ Inner Module (multi-scale, independent params)           │
    │     ├─ Outer Module (shared params, Conv2D weights)             │
    │     ├─ Build Prototypes P̂^c (Eq. 8)                            │
    │     └─ Cross Distance Calculation (Eq. 9-10) → L_ce            │
    │                                                                 │
    │  ③ Contrastive Branch (parallel):                               │
    │     └─ Hybrid Contrastive Loss (Eq. 11-13) → L_hc              │
    │                                                                 │
    │  ④ Total Loss (Eq. 14):                                         │
    │     L_total = λ × L_ce + L_hc                                   │
    │                                                                 │
    │  ⑤ Prediction: ŷ = argmin_c D(P^c, Q^r)                        │
    └─────────────────────────────────────────────────────────────────┘

Training:
    - Episodic training with N-way K-shot tasks
    - SGD optimizer with cosine annealing
    - λ = 4.0 (weight for cross-entropy loss)
    - τ = 0.07 (contrastive temperature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import get_backbone
from models.matryoshka_mamba import MatryoshkaMamba
from models.cross_distance import build_prototype, cross_distance_calculation
from models.contrastive import HybridContrastiveLoss


class Manta(nn.Module):
    """
    Complete MANTA framework for Few-Shot Action Recognition.

    Combines:
        1. Frozen backbone for feature extraction
        2. Matryoshka Mamba for multi-scale temporal modeling
        3. Cross Distance Calculation for classification
        4. Hybrid Contrastive Learning for auxiliary supervision

    Args:
        backbone_name: Feature extractor ('resnet50' | 'vitb')
        d_model: Feature dimension D (default 2048)
        scales: Multi-scale set O (default [1, 2, 4])
        d_state: Mamba-2 state space dimension (default 64)
        d_conv: Mamba-2 convolution width (default 4)
        expand: Mamba-2 expansion factor (default 2)
        temperature: Contrastive temperature τ (default 0.07)
        lambda_ce: Cross-entropy loss weight λ (default 4.0)
        pretrained_backbone: Use ImageNet pretrained weights (default True)
        freeze_backbone: Whether to freeze backbone parameters (default True, per paper)
    """

    def __init__(
        self,
        backbone_name: str = 'resnet50',
        d_model: int = 2048,
        scales: list = None,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        temperature: float = 0.07,
        lambda_ce: float = 4.0,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.lambda_ce = lambda_ce
        self.scales = scales if scales is not None else [1, 2, 4]

        # ① Feature Extraction Backbone (frozen by default)
        self.backbone = get_backbone(
            name=backbone_name,
            pretrained=pretrained_backbone,
            output_dim=d_model,
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.freeze_backbone = freeze_backbone

        # ② Matryoshka Mamba Branch
        self.matryoshka_mamba = MatryoshkaMamba(
            d_model=d_model,
            scales=self.scales,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # ③ Contrastive Branch
        self.contrastive_loss_fn = HybridContrastiveLoss(
            temperature=temperature,
            d_model=d_model,
        )

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame features from video using backbone.

        Args:
            videos: [B, F, C, H, W] — batch of videos with F frames each

        Returns:
            features: [B, F, D] — per-frame feature vectors
        """
        B, F, C, H, W = videos.shape

        # Reshape to process all frames as a batch
        frames = videos.reshape(B * F, C, H, W)  # [B*F, C, H, W]

        # Extract features (no gradient if backbone is frozen)
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone(frames)  # [B*F, D]
        else:
            feats = self.backbone(frames)  # [B*F, D]

        # Reshape back to [B, F, D]
        return feats.view(B, F, self.d_model)

    def forward(
        self,
        support_videos: torch.Tensor,
        query_videos: torch.Tensor,
        support_labels: torch.Tensor,
        query_labels: torch.Tensor = None,
        n_way: int = 5,
        k_shot: int = 1,
        mode: str = 'train',
    ) -> dict:
        """
        Full forward pass of MANTA.

        Args:
            support_videos: [N, K, F, C, H, W] — support set videos
            query_videos:   [Q, F, C, H, W] — query set videos
            support_labels: [N*K] — class labels for support (0 to N-1)
            query_labels:   [Q] — class labels for query (only needed in training)
            n_way: Number of classes N in the episode
            k_shot: Number of shots K per class
            mode: 'train' (compute losses) or 'test' (prediction only)

        Returns:
            dict containing:
                Training mode:
                    'loss': Total loss L_total = λ*L_ce + L_hc
                    'L_ce': Cross-entropy loss
                    'L_hc': Hybrid contrastive loss
                    'pred_labels': [Q] predicted class indices
                    'distances': [Q, N] distance matrix
                    'accuracy': Episode accuracy

                Test mode:
                    'pred_labels': [Q] predicted class indices
                    'distances': [Q, N] distance matrix
        """
        N, K, F, C, H, W = support_videos.shape
        Q = query_videos.shape[0]

        # ==================== ① Feature Extraction ====================
        # Support: [N*K, F, D]
        sup_flat = support_videos.reshape(N * K, F, C, H, W)
        sup_feats = self.extract_features(sup_flat)  # [N*K, F, D]

        # Query: [Q, F, D]
        que_feats = self.extract_features(query_videos)  # [Q, F, D]

        # ==================== ② Matryoshka Mamba Branch ====================
        # Apply Matryoshka Mamba to both support and query
        sup_enhanced = self.matryoshka_mamba(sup_feats)  # [N*K, F, D]
        que_enhanced = self.matryoshka_mamba(que_feats)  # [Q, F, D]

        # Reshape support for prototype construction: [N, K, F, D]
        sup_enhanced_4d = sup_enhanced.view(N, K, F, self.d_model)

        # Build class prototypes (Eq. 8): average over K shots
        prototypes = build_prototype(sup_enhanced_4d)  # [N, F, D]

        # Cross Distance Calculation (Eq. 9-10)
        distances = cross_distance_calculation(prototypes, que_enhanced)  # [Q, N]

        # Prediction: argmin distance = predicted class
        pred_labels = distances.argmin(dim=-1)  # [Q]

        # ==================== Test Mode: Return predictions only ====================
        if mode == 'test':
            return {
                'pred_labels': pred_labels,
                'distances': distances,
            }

        # ==================== ③ Training: Compute Losses ====================
        assert query_labels is not None, (
            "query_labels must be provided in training mode"
        )

        # --- L_ce: Cross-entropy loss from distance-based logits ---
        # Convert distances to logits (negate: smaller distance = higher logit)
        logits = -distances  # [Q, N]
        L_ce = F.cross_entropy(logits, query_labels)

        # --- L_hc: Hybrid Contrastive Loss ---
        contrastive_result = self.contrastive_loss_fn(
            support_features=sup_enhanced_4d,  # [N, K, F, D]
            query_features=que_enhanced,        # [Q, F, D]
            support_labels=support_labels,      # [N*K]
            n_way=n_way,
            k_shot=k_shot,
        )
        L_hc = contrastive_result['loss']

        # --- L_total (Eq. 14): λ × L_ce + L_hc ---
        L_total = self.lambda_ce * L_ce + L_hc

        # --- Compute episode accuracy ---
        accuracy = (pred_labels == query_labels).float().mean()

        return {
            'loss': L_total,
            'L_ce': L_ce,
            'L_hc': L_hc,
            'L_S': contrastive_result['L_S'],
            'L_Q': contrastive_result['L_Q'],
            'L_SQ': contrastive_result['L_SQ'],
            'pred_labels': pred_labels,
            'distances': distances,
            'accuracy': accuracy,
        }
