"""
Inner Module — Local Fragment Enhancement via Bidirectional Mamba-2.

Paper Section 3.2, Eq. (3)-(4):
    "The Inner Module enhances local temporal features within each fragment
     of the sub-sequence using bidirectional Mamba-2 scanning."

Key design choices (from paper Table II ablation):
    - Two sub-branches: IM_Fw (forward) and IM_Bw (backward)
    - INDEPENDENT parameters between Fw and Bw (I-P configuration)
    - Each branch: LayerNorm → Mamba-2 → Linear
    - Output: concatenate Fw and Bw → project → residual connection

Algorithm:
    Given a fragment x ∈ ℝ^{o×D} (o = scale size):
        1. h_fw = Mamba2_Fw(LayerNorm(x))                    — forward scan
        2. h_bw = Mamba2_Bw(LayerNorm(flip(x)))              — backward scan
        3. h_bw = flip(h_bw)                                  — re-align temporal order
        4. output = Linear([h_fw; h_bw]) + x                  — concat, project, residual

    Eq. (4): Ṡ^ck_i = IM(S^ck_i) = W_o · [IM_Fw(S^ck_i); IM_Bw(S^ck_i)] + S^ck_i
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class InnerModuleBranch(nn.Module):
    """
    Single directional branch (forward OR backward) of the Inner Module.
    
    Architecture: LayerNorm → Mamba-2 → Linear
    
    Args:
        d_model: Feature dimension D
        d_state: SSM state dimension (default 64 for Mamba-2)
        d_conv: Local convolution width (default 4)
        expand: Expansion factor for inner dimension (default 2)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba2 = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one directional branch.

        Args:
            x: [B, L, D] — input sequence (L = fragment length o)

        Returns:
            [B, L, D] — enhanced features from this direction
        """
        x = self.norm(x)
        x = self.mamba2(x)
        return self.linear(x)


class InnerModule(nn.Module):
    """
    Complete Inner Module with bidirectional Mamba-2 branches.
    
    Key property: Fw and Bw branches have INDEPENDENT parameters (I-P).
    This is critical — paper ablation (Table II) shows I-P outperforms S-P for IM.
    
    Args:
        d_model: Feature dimension D (default 2048 for ResNet-50)
        d_state: SSM state dimension (default 64)
        d_conv: Local convolution width (default 4)
        expand: Expansion factor (default 2)
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        # Two INDEPENDENT branches — NOT sharing parameters
        self.fw_branch = InnerModuleBranch(d_model, d_state, d_conv, expand)
        self.bw_branch = InnerModuleBranch(d_model, d_state, d_conv, expand)

        # Projection from concatenated 2D → D
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional enhancement of a single fragment.

        Implements Eq. (4):
            Ṡ^ck_i = W_o · [IM_Fw(S^ck_i); IM_Bw(S^ck_i)] + S^ck_i

        Args:
            x: [B, o, D] — fragment features (o = scale/fragment length)

        Returns:
            [B, o, D] — enhanced fragment features with residual connection
        """
        # Forward branch: process in natural temporal order
        fw_out = self.fw_branch(x)  # [B, o, D]

        # Backward branch: flip → process → flip back to align
        bw_out = self.bw_branch(x.flip(dims=[1]))  # [B, o, D]
        bw_out = bw_out.flip(dims=[1])  # Re-align temporal order

        # Concatenate and project (Eq. 4)
        combined = torch.cat([fw_out, bw_out], dim=-1)  # [B, o, 2D]
        out = self.out_proj(combined)  # [B, o, D]

        # Residual connection
        return out + x  # [B, o, D]


def apply_inner_module_to_sequence(
    features: torch.Tensor,
    inner_module: InnerModule,
    scale_o: int,
) -> torch.Tensor:
    """
    Apply Inner Module to an entire sub-sequence by splitting into
    non-overlapping fragments of size o.

    Paper Algorithm 1, Step 3:
        "Divide S^c_k into fragments {S^ck_1, ..., S^ck_{F/o}} of length o"
        "Apply IM to each fragment independently"

    Args:
        features: [B, F, D] — full sub-sequence features
        inner_module: Initialized InnerModule instance
        scale_o: Fragment size (must divide F evenly)

    Returns:
        [B, F, D] — enhanced features (all fragments concatenated back)
    """
    B, F, D = features.shape
    assert F % scale_o == 0, (
        f"Sequence length F={F} must be divisible by scale_o={scale_o}. "
        f"Valid scales for F={F}: {[s for s in [1,2,4,8,16,32,64] if F % s == 0]}"
    )

    num_fragments = F // scale_o

    # Reshape into fragments: [B, num_fragments, o, D]
    fragments = features.view(B, num_fragments, scale_o, D)

    # Process each fragment through Inner Module
    # Reshape to [B * num_fragments, o, D] for batch processing
    fragments_flat = fragments.reshape(B * num_fragments, scale_o, D)
    enhanced_flat = inner_module(fragments_flat)  # [B * num_fragments, o, D]

    # Reshape back and concatenate along temporal dimension
    enhanced = enhanced_flat.view(B, num_fragments, scale_o, D)
    return enhanced.reshape(B, F, D)  # [B, F, D]
