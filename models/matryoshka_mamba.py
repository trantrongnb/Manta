"""
Matryoshka Mamba — Multi-Scale Nested Mamba Structure.

Paper Section 3.2, Algorithm 1, Eq. (7):
    "Matryoshka Mamba processes sub-sequences at multiple temporal scales
     O = {o₁, o₂, ..., o_|O|} simultaneously, where each scale o defines
     the fragment length for the Inner Module."

The name "Matryoshka" comes from Russian nesting dolls — the multi-scale
structure is analogous to nested dolls of different sizes.

Algorithm 1 (Matryoshka Mamba):
    Input:  S^c_k ∈ ℝ^{F×D} — per-frame features from backbone
    Output: Ŝ^c_k ∈ ℝ^{F×D} — enhanced features

    For each scale o ∈ O = {1, 2, 4}:
        1. Divide S^c_k into F/o non-overlapping fragments of length o
        2. Apply Inner Module to each fragment → Ṡ^ck_i (local enhancement)
        3. Concatenate fragments → Ṡ^ck_fo (full enhanced sequence)
        4. Apply Outer Module: scale_out = wₒ ⊗ OM(Ṡ^ck_fo, S^c_k)

    Final output (Eq. 7):
        Ŝ^c_k = (1/|O|) × Σ_{o∈O} Ṡ^ck_fo

    Complexity: O(F × |O|) — linear in sequence length F

Design choices:
    - Each scale has its OWN Inner Module (independent parameters)
    - All scales SHARE a single Outer Module (shared parameters)
    - Default scales O = {1, 2, 4} (Table 3 ablation)
"""

import torch
import torch.nn as nn

from models.inner_module import InnerModule, apply_inner_module_to_sequence
from models.outer_module import OuterModule


class MatryoshkaMamba(nn.Module):
    """
    Multi-scale Matryoshka Mamba module.
    
    Processes input features at multiple temporal granularities and
    averages the results for robust temporal representation.
    
    Args:
        d_model: Feature dimension D (default 2048 for ResNet-50)
        scales: List of scale values O (default [1, 2, 4] per paper Table 3)
        d_state: Mamba-2 state space dimension (default 64)
        d_conv: Mamba-2 local convolution width (default 4)
        expand: Mamba-2 expansion factor (default 2)
    """

    def __init__(
        self,
        d_model: int = 2048,
        scales: list = None,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.scales = scales if scales is not None else [1, 2, 4]

        # Each scale has its OWN Inner Module (independent parameters per scale)
        self.inner_modules = nn.ModuleDict({
            str(o): InnerModule(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for o in self.scales
        })

        # Single SHARED Outer Module across all scales
        # (Paper uses one OM for all scales — efficient parameter sharing)
        self.outer_module = OuterModule(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale processing following Algorithm 1.

        Implements Eq. (7):
            Ŝ^c_k = (1/|O|) × Σ_{o∈O} [wₒ ⊗ OM(IM_o(S^c_k), S^c_k)]

        Args:
            features: [B, F, D] — per-frame features from backbone

        Returns:
            [B, F, D] — multi-scale enhanced features (average over valid scales)
        """
        B, F, D = features.shape
        scale_outputs = []

        for o in self.scales:
            # Skip scales that don't evenly divide F
            if F % o != 0:
                continue

            # Step 1-3: Inner Module — local fragment enhancement at scale o
            inner_module = self.inner_modules[str(o)]
            enhanced = apply_inner_module_to_sequence(features, inner_module, o)
            # enhanced: [B, F, D] — locally enhanced sequence

            # Step 4: Outer Module — temporal alignment + learnable weights
            scale_out, _weights = self.outer_module(enhanced, features)
            # scale_out: [B, F, D] — wₒ ⊗ OM(enhanced, original)

            scale_outputs.append(scale_out)

        # Fallback: if no valid scales, return original features
        if len(scale_outputs) == 0:
            return features

        # Eq. (7): Average all scale outputs
        stacked = torch.stack(scale_outputs, dim=0)  # [|O_valid|, B, F, D]
        return stacked.mean(dim=0)  # [B, F, D]

    def forward_with_details(self, features: torch.Tensor) -> dict:
        """
        Forward pass with detailed per-scale outputs for analysis/visualization.

        Args:
            features: [B, F, D] — per-frame features

        Returns:
            dict with:
                'output': [B, F, D] — final averaged output
                'scale_outputs': dict mapping scale → [B, F, D]
                'scale_weights': dict mapping scale → [B, F, D] (learnable weights wₒ)
        """
        B, F, D = features.shape
        scale_outputs = {}
        scale_weights = {}

        for o in self.scales:
            if F % o != 0:
                continue

            inner_module = self.inner_modules[str(o)]
            enhanced = apply_inner_module_to_sequence(features, inner_module, o)
            scale_out, weights = self.outer_module(enhanced, features)

            scale_outputs[o] = scale_out
            scale_weights[o] = weights

        if len(scale_outputs) == 0:
            return {
                'output': features,
                'scale_outputs': {},
                'scale_weights': {},
            }

        stacked = torch.stack(list(scale_outputs.values()), dim=0)
        output = stacked.mean(dim=0)

        return {
            'output': output,
            'scale_outputs': scale_outputs,
            'scale_weights': scale_weights,
        }
