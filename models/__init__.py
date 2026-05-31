"""
MANTA Model Components
======================
Matryoshka MAmba and CoNtrasTive LeArning for Few-Shot Action Recognition.

Architecture (Figure 2 of paper):
    Input Video → Backbone → Matryoshka Mamba (Inner + Outer) → Cross Distance
                                                               ↘ Contrastive Branch

Modules:
    - backbone.py:          Feature extraction (ResNet-50, ViT-B/16)
    - inner_module.py:      Inner Module — local fragment enhancement (Mamba-2, independent params)
    - outer_module.py:      Outer Module — temporal alignment (Mamba-2, shared params + Conv2D Block)
    - matryoshka_mamba.py:  Multi-scale nested Mamba structure
    - cross_distance.py:    Cross Distance Calculation (4 directional distances)
    - contrastive.py:       Hybrid Contrastive Learning (L_S + L_Q + L_SQ)
    - manta.py:             Main framework integrating all components
"""

from models.backbone import get_backbone
from models.inner_module import InnerModule
from models.outer_module import OuterModule
from models.matryoshka_mamba import MatryoshkaMamba
from models.cross_distance import cross_distance_calculation, build_prototype
from models.contrastive import HybridContrastiveLoss
from models.manta import Manta

__all__ = [
    'get_backbone',
    'InnerModule',
    'OuterModule',
    'MatryoshkaMamba',
    'cross_distance_calculation',
    'build_prototype',
    'HybridContrastiveLoss',
    'Manta',
]
