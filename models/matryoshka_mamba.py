"""
Matryoshka Mamba — Multi-Scale Nested Mamba Structure.

Paper Section 3.2, Algorithm 1, Eq. (4)-(7):
    "Matryoshka Mamba processes sub-sequences at multiple temporal scales
     O = {o₁, o₂, ..., o_|O|} simultaneously, where each scale o defines
     the fragment length for the Inner Module."

The name "Matryoshka" comes from Russian nesting dolls — the multi-scale
structure is analogous to nested dolls of different sizes.

Algorithm 1 (Matryoshka Mamba):
    Input:  I ∈ ℝ^{F×D} — per-frame features from backbone
    Output: Î ∈ ℝ^{F×D} — enhanced features

    For each scale o ∈ O = {1, 2, 4}:
        1. Divide I into F/o non-overlapping fragments of length o
        2. For each fragment i:
           a. E ← IM(I[i*o : (i+1)*o]) + I[i*o : (i+1)*o]  (inner module + residual)
           b. If first fragment: Ĩ ← E
           c. Else: Ĩ ← OM(Concat(Ĩ, E, dim=0))  (outer module on growing sequence)
        3. Compute weights: w ← σ(CB(Ĩ ⊕ I))
        4. Scale output: ˚I ← w ⊗ Ĩ

    Final output (Eq. 7):
        Î = (1/|O|) × Σ_{o∈O} ˚I_o

    Complexity: O(F × |O|) — linear in sequence length F

Design choices:
    - Each scale has its OWN Inner Module (independent parameters)
    - All scales SHARE a single Outer Module (shared parameters)
    - Default scales O = {1, 2, 4} (Table 3 ablation)
"""

import torch
import torch.nn as nn

from models.inner_module import InnerModule
from models.outer_module import OuterModule


class MatryoshkaMamba(nn.Module):
    """
    Multi-scale Matryoshka Mamba module implementing Algorithm 1.
    
    Processes input features at multiple temporal granularities using
    an incremental approach: Inner Module enhances local fragments,
    then Outer Module progressively aligns them through bidirectional
    Mamba scanning.
    
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

    def _process_scale(
        self,
        features: torch.Tensor,
        scale_o: int,
    ) -> tuple:
        """
        Process a single scale following Algorithm 1 inner loop.
        
        Implements the incremental OM approach:
            For each fragment:
                E ← IM(fragment) + fragment  (residual)
                Ĩ ← OM(Concat(Ĩ, E, dim=0))  (incremental alignment)
            After loop:
                w ← σ(CB(Ĩ ⊕ I))
                ˚I ← w ⊗ Ĩ
        
        Args:
            features: [B, F, D] — input per-frame features
            scale_o: Fragment size o
        
        Returns:
            tuple of:
                scale_output: [B, F, D] — w ⊗ Ĩ
                weights: [B, F, D] — learnable weights
        """
        B, F, D = features.shape
        inner_module = self.inner_modules[str(scale_o)]
        num_fragments = F // scale_o

        # ======== Algorithm 1 Inner Loop (lines 5-13) ========
        I_tilde = None  # Ĩ — accumulates enhanced + OM-aligned sequence

        for i in range(num_fragments):
            # Extract fragment: I[i*o : (i+1)*o, :]
            start_idx = i * scale_o
            end_idx = (i + 1) * scale_o
            fragment = features[:, start_idx:end_idx, :]  # [B, o, D]

            # Line 6: E ← IM(fragment) ⊕ fragment (⊕ = addition, residual)
            # Note: InnerModule already includes residual connection internally
            E = inner_module(fragment)  # [B, o, D] — enhanced fragment with residual

            if I_tilde is None:
                # Line 8: First fragment — Ĩ ← E
                I_tilde = E  # [B, o, D]
            else:
                # Line 11: Ĩ ← OM(Concat(Ĩ, E, dim=0))
                # Concat along temporal dimension, then apply OM
                I_tilde = torch.cat([I_tilde, E], dim=1)  # [B, current_len + o, D]
                I_tilde = self.outer_module.mamba_scan(I_tilde)  # [B, current_len + o, D]

        # ======== Algorithm 1 After Loop (lines 14-15) ========
        # Line 14: w ← Sigmoid(CB(Ĩ ⊕ I))
        # Ĩ now has shape [B, F, D] (accumulated all fragments)
        w = self.outer_module.compute_weights(I_tilde, features)  # [B, F, D]

        # Line 15: ˚I ← w ⊗ Ĩ
        # Note: Ĩ has already been processed through OM in the loop
        scale_output = w * I_tilde  # [B, F, D]

        return scale_output, w

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale processing following Algorithm 1.

        Implements Eq. (7):
            Î = (1/|O|) × Σ_{o∈O} ˚I_o

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

            # Process this scale following Algorithm 1
            scale_out, _weights = self._process_scale(features, o)
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

            scale_out, weights = self._process_scale(features, o)
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
