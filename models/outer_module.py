"""
Outer Module — Implicit Temporal Alignment via Bidirectional Mamba-2.

Paper Section 3.2, Eq. (5)-(6):
    "The Outer Module performs implicit temporal alignment across the entire
     enhanced sub-sequence using bidirectional Mamba-2 with SHARED parameters
     and learnable scale weights from a Conv2D Block."

Key design choices (from paper Table II ablation):
    - Two sub-branches: OM_Fw (forward) and OM_Bw (backward)
    - SHARED parameters between Fw and Bw (S-P configuration)
    - Learnable weights wₒ computed via Conv2D Block (CB)
    - Output: wₒ ⊗ OM(enhanced_seq)  (element-wise multiplication)

Algorithm:
    Given enhanced sequence Ṡ ∈ ℝ^{F×D} from Inner Module:
        1. h_fw = Mamba2_shared(LayerNorm(Ṡ))                — forward scan
        2. h_bw = Mamba2_shared(LayerNorm(flip(Ṡ)))          — backward scan (SAME weights)
        3. h_bw = flip(h_bw)                                  — re-align
        4. om_out = W_o · [h_fw; h_bw]                        — concat and project
        5. wₒ = σ(CB([Ṡ^ck_fo ⊕ S^c_k]))                    — learnable weights via Conv2D Block
           NOTE: ⊕ = concatenation along CHANNEL dimension (not addition!)
        6. scale_out = wₒ ⊗ om_out                            — weighted output

    Eq. (5): wₒ = σ(CB([Ṡ^ck_fo ⊕ S^c_k]))   — ⊕ is CONCATENATION
    Eq. (6): Ṡ^ck_fo = wₒ ⊗ OM(Ṡ^ck_fo)
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class Conv2DBlock(nn.Module):
    """
    Conv2D Block (CB) for computing learnable scale weights wₒ.
    
    Architecture: Conv2d(1×1) → BN → ReLU → Conv2d(3×3) → BN → ReLU → Conv2d(1×1) → BN
    
    Input is the CONCATENATION of enhanced and original features along
    the channel dimension: [B, 2, F, D] (2 channels: enhanced + original).
    Output is projected back to [B, 1, F, D] for the weight map.
    
    Args:
        in_channels: Number of input channels (2 for concatenated enhanced⊕original)
        mid_channels: Intermediate channel count
    """

    def __init__(self, in_channels: int = 2, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = max(in_channels * 4, 16)

        self.conv_block = nn.Sequential(
            # 1×1 conv: channel mixing (fuse enhanced and original)
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3×3 conv: local spatial context (temporal-feature neighborhood)
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 1×1 conv: project to single channel (weight map)
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2, F, D] — concatenated enhanced⊕original features
               Channel 0: enhanced sequence Ṡ^ck_fo
               Channel 1: original sequence S^c_k

        Returns:
            [B, 1, F, D] — learned weight map (before sigmoid)
        """
        return self.conv_block(x)


class OuterModule(nn.Module):
    """
    Outer Module with bidirectional Mamba-2 (shared parameters) and
    learnable scale weights via Conv2D Block.
    
    Key property: Fw and Bw branches SHARE the same Mamba-2 instance (S-P).
    This is critical — paper ablation (Table II) shows S-P outperforms I-P for OM.
    
    Args:
        d_model: Feature dimension D (default 2048)
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
        self.d_model = d_model

        # Layer normalization before Mamba-2
        self.norm = nn.LayerNorm(d_model)

        # SHARED Mamba-2 instance for both forward and backward scans
        self.mamba2_shared = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Projection from concatenated 2D → D
        self.out_proj = nn.Linear(d_model * 2, d_model)

        # Conv2D Block for learnable weights wₒ (Eq. 5)
        # Input: 2 channels (enhanced ⊕ original concatenated along channel dim)
        # Output: 1 channel (weight map)
        self.conv_block = Conv2DBlock(in_channels=2)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        enhanced_seq: torch.Tensor,
        orig_seq: torch.Tensor,
    ) -> tuple:
        """
        Compute temporally-aligned output with learnable scale weights.

        Implements Eq. (5)-(6):
            wₒ = σ(CB([Ṡ^ck_fo ⊕ S^c_k]))    — ⊕ = concatenation along channel
            output = wₒ ⊗ OM(Ṡ)                — ⊗ = element-wise multiplication

        Args:
            enhanced_seq: [B, F, D] — output from Inner Module (Ṡ^ck_fo)
            orig_seq:     [B, F, D] — original features before Inner Module (S^c_k)

        Returns:
            tuple of:
                scale_output: [B, F, D] — wₒ ⊗ OM(·), the final scale output
                weights:      [B, F, D] — learnable weights wₒ for analysis/visualization
        """
        B, F, D = enhanced_seq.shape

        # === Bidirectional Mamba-2 with SHARED parameters ===
        normed = self.norm(enhanced_seq)

        # Forward scan
        fw_out = self.mamba2_shared(normed)  # [B, F, D]

        # Backward scan (same Mamba-2 instance, flipped input)
        bw_out = self.mamba2_shared(normed.flip(dims=[1]))  # [B, F, D]
        bw_out = bw_out.flip(dims=[1])  # Re-align temporal order

        # Concatenate and project
        om_out = self.out_proj(
            torch.cat([fw_out, bw_out], dim=-1)
        )  # [B, F, D]

        # === Learnable weights wₒ via Conv2D Block (Eq. 5) ===
        # CONCATENATION ⊕ along channel dimension (NOT addition!)
        # enhanced_seq → [B, 1, F, D], orig_seq → [B, 1, F, D]
        # concat → [B, 2, F, D]
        weight_input = torch.cat([
            enhanced_seq.unsqueeze(1),  # [B, 1, F, D] — channel 0: enhanced
            orig_seq.unsqueeze(1),      # [B, 1, F, D] — channel 1: original
        ], dim=1)  # [B, 2, F, D] — ⊕ concatenation

        w = self.conv_block(weight_input)  # [B, 1, F, D]
        w = self.sigmoid(w.squeeze(1))     # [B, F, D] — values in (0, 1)

        # === Scale output (Eq. 6): Ṡ^ck_fo = wₒ ⊗ OM(·) ===
        scale_out = w * om_out  # [B, F, D] — element-wise multiplication

        return scale_out, w
