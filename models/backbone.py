"""
Backbone Feature Extractors for MANTA.

Paper Section 3.1 — Feature Extraction:
    "We employ a frozen pre-trained backbone ψ (e.g., ResNet-50 or ViT-B/16)
     to extract per-frame features from video sub-sequences."

Given a video V with F frames, the backbone produces:
    S = {s₁, s₂, ..., s_F} ∈ ℝ^{F×D}

where D is the feature dimension (2048 for ResNet-50, 768 for ViT-B/16).

Supported backbones:
    - ResNet-50 (ImageNet pretrained, D=2048)
    - ViT-B/16 (ImageNet pretrained, D=768)
"""

import os
import torch
import torch.nn as nn
import timm
from torchvision.models import resnet50, ResNet50_Weights


class BackboneResNet50(nn.Module):
    """
    ResNet-50 backbone pretrained on ImageNet.
    
    Removes the final avgpool + fc layers, keeps spatial feature maps,
    then applies adaptive average pooling to get per-frame feature vectors.
    
    Output dimension: D = 2048
    """

    def __init__(self, pretrained: bool = True, weights_path: str = None):
        super().__init__()
        if pretrained and weights_path and os.path.exists(weights_path):
            # Load from local file
            base = resnet50(weights=None)
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            base.load_state_dict(state_dict)
            print(f"  ✓ Loaded ResNet-50 weights from: {weights_path}")
        else:
            # Download from torchvision hub (cached after first download)
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = resnet50(weights=weights)

        # Remove avgpool and fc — keep conv feature extractor
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame features.

        Args:
            x: [B, C, H, W] — batch of video frames (C=3, H=W=224)

        Returns:
            [B, D] — per-frame feature vectors, D=2048
        """
        feat = self.features(x)   # [B, 2048, h, w] where h=w=7 for 224×224 input
        feat = self.pool(feat)     # [B, 2048, 1, 1]
        return feat.flatten(1)     # [B, 2048]


class BackboneViTB(nn.Module):
    """
    Vision Transformer B/16 backbone pretrained on ImageNet.
    
    Uses timm library for ViT-B/16 with patch size 16.
    Removes classification head, outputs CLS token features.
    
    Output dimension: D = 768
    """

    def __init__(self, pretrained: bool = True, weights_path: str = None):
        super().__init__()
        if pretrained and weights_path and os.path.exists(weights_path):
            # Load from local file
            self.model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0  # Remove classification head
            )
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"  ✓ Loaded ViT-B/16 weights from: {weights_path}")
        else:
            self.model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=pretrained,
                num_classes=0  # Remove classification head
            )
        self.output_dim = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame features via ViT CLS token.

        Args:
            x: [B, C, H, W] — batch of video frames (C=3, H=W=224)

        Returns:
            [B, D] — per-frame feature vectors, D=768
        """
        return self.model(x)  # [B, 768]


def get_backbone(name: str, pretrained: bool = True, output_dim: int = 2048,
                 weights_path: str = None) -> nn.Module:
    """
    Factory function to create backbone with optional projection layer.

    If the backbone's native output dimension differs from the requested
    output_dim, a linear projection layer is appended.

    Args:
        name: Backbone architecture name ('resnet50' | 'vitb')
        pretrained: Whether to use ImageNet pretrained weights
        output_dim: Desired output feature dimension
        weights_path: Optional path to local pretrained weights file

    Returns:
        nn.Module with .output_dim attribute indicating the output dimension
    """
    if name == 'resnet50':
        backbone = BackboneResNet50(pretrained=pretrained, weights_path=weights_path)
    elif name == 'vitb':
        backbone = BackboneViTB(pretrained=pretrained, weights_path=weights_path)
    else:
        raise ValueError(
            f"Unknown backbone: '{name}'. Supported: 'resnet50', 'vitb'"
        )

    # Add projection layer if dimensions don't match
    if backbone.output_dim != output_dim:
        original_forward = backbone.forward
        proj = nn.Linear(backbone.output_dim, output_dim)

        def forward_with_proj(x):
            feat = original_forward(x)
            return proj(feat)

        backbone.forward = forward_with_proj
        backbone.proj = proj
        backbone.output_dim = output_dim

    return backbone
