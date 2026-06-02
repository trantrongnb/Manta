"""
Outer Module — Implicit Temporal Alignment via Bidirectional Mamba-2.

Paper Section 3.2, Eq. (5)-(6), Algorithm 1:
    "The Outer Module performs implicit temporal alignment across the entire
     enhanced sub-sequence using bidirectional Mamba-2 with SHARED parameters
     and learnable scale weights from a Conv2D Block."

Key design choices (from paper Table II ablation):
    - Two sub-branches: OM_Fw (forward) and OM_Bw (backward)
    - SHARED parameters between Fw and Bw (S-P configuration)
    - Learnable weights wₒ computed via Conv2D Block (CB)
    - Output: wₒ ⊗ OM(enhanced_seq)  (element-wise multiplication)

Algorithm 1 Structure:
    The OM (bidirectional Mamba scan) is used INSIDE the inner loop to
    incrementally process the growing sequence. The Conv2D Block and
    Sigmoid are applied AFTER the loop completes.

    Inside loop (line 11): Ĩ ← OM(Concat(Ĩ, E, dim=0))
    After loop (line 14-15):
        w ← Sigmoid(CB(Ĩ ⊕ I))
        ˚I ← w ⊗ Ĩ

    The OM component: OM(·) = Linear[OM_Fw(·) ⊕ OM_Bw(·)] ∈ R^{F×D}
    The CB component: w = σ(CB([enhanced ⊕ original]))

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
    
    This module exposes two sub-operations per Algorithm 1:
        1. mamba_scan(): Bidirectional Mamba-2 scan only (used inside inner loop)
        2. compute_weights(): Conv2D Block + Sigmoid for learnable weights (after loop)
        3. forward(): Full pipeline (scan + weights + multiply) for convenience
    
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

    def mamba_scan(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional Mamba-2 scan ONLY (no weights).
        
        This implements the OM(·) operation from the paper:
            OM(·) = Linear[OM_Fw(·) ⊕ OM_Bw(·)] ∈ R^{L×D}
        
        Used INSIDE the inner loop of Algorithm 1 (line 11):
            Ĩ ← OM(Concat(Ĩ, E, dim=0))
        
        Args:
            seq: [B, L, D] — input sequence of any length L
        
        Returns:
            [B, L, D] — bidirectionally scanned and projected output
        """
        normed = self.norm(seq)

        # Forward scan
        fw_out = self.mamba2_shared(normed)  # [B, L, D]

        # Backward scan (same Mamba-2 instance, flipped input)
        bw_out = self.mamba2_shared(normed.flip(dims=[1]))  # [B, L, D]
        bw_out = bw_out.flip(dims=[1])  # Re-align temporal order

        # Concatenate and project
        om_out = self.out_proj(
            torch.cat([fw_out, bw_out], dim=-1)
        )  # [B, L, D]

        return om_out

    def compute_weights(
        self,
        enhanced_seq: torch.Tensor,
        orig_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute learnable weights wₒ via Conv2D Block (Eq. 5).
        
        Used AFTER the inner loop of Algorithm 1 (line 14):
            w ← Sigmoid(CB(Ĩ ⊕ I))
        
        Args:
            enhanced_seq: [B, F, D] — enhanced sequence (Ĩ from Algorithm 1)
            orig_seq:     [B, F, D] — original features (I)
        
        Returns:
            [B, F, D] — learnable weights wₒ ∈ (0, 1)
        """
        # CONCATENATION ⊕ along channel dimension (NOT addition!)
        weight_input = torch.cat([
            enhanced_seq.unsqueeze(1),  # [B, 1, F, D] — channel 0: enhanced
            orig_seq.unsqueeze(1),      # [B, 1, F, D] — channel 1: original
        ], dim=1)  # [B, 2, F, D]

        w = self.conv_block(weight_input)  # [B, 1, F, D]
        w = self.sigmoid(w.squeeze(1))     # [B, F, D]

        return w

    def forward(
        self,
        enhanced_seq: torch.Tensor,
        orig_seq: torch.Tensor,
    ) -> tuple:
        """
        Full Outer Module pipeline: Mamba scan + weights + multiply.

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
        # Bidirectional Mamba scan
        om_out = self.mamba_scan(enhanced_seq)  # [B, F, D]

        # Learnable weights (Eq. 5)
        w = self.compute_weights(enhanced_seq, orig_seq)  # [B, F, D]

        # Scale output (Eq. 6): Ṡ^ck_fo = wₒ ⊗ OM(·)
        scale_out = w * om_out  # [B, F, D]

        return scale_out, w
