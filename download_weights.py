#!/usr/bin/env python3
"""
Download pretrained backbone weights and save locally to pretrained/ directory.

Supports:
    - ResNet-50 (ImageNet pretrained) → pretrained/resnet50_imagenet.pth
    - ViT-B/16   (ImageNet pretrained) → pretrained/vitb16_imagenet.pth

Usage:
    python download_weights.py              # Download both
    python download_weights.py --resnet50   # Only ResNet-50
    python download_weights.py --vitb16     # Only ViT-B/16
"""

import os
import sys
import argparse

import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights


def download_resnet50(save_dir: str = 'pretrained') -> str:
    """Download ResNet-50 ImageNet pretrained weights."""
    print("[1/2] Downloading ResNet-50 (ImageNet-1K V2) weights...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    
    save_path = os.path.join(save_dir, 'resnet50_imagenet.pth')
    torch.save(model.state_dict(), save_path)
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  ✓ Saved to: {save_path} ({file_size_mb:.1f} MB)")
    return save_path


def download_vitb16(save_dir: str = 'pretrained') -> str:
    """Download ViT-B/16 ImageNet pretrained weights via timm."""
    print("[2/2] Downloading ViT-B/16 (ImageNet-1K) weights...")
    try:
        import timm
    except ImportError:
        print("  ⚠ timm not installed. Skipping ViT-B/16.")
        print("    Install with: pip install timm==0.9.16")
        return None
    
    # Create model to trigger weight download
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    save_path = os.path.join(save_dir, 'vitb16_imagenet.pth')
    torch.save(model.state_dict(), save_path)
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  ✓ Saved to: {save_path} ({file_size_mb:.1f} MB)")
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description='Download pretrained backbone weights for MANTA'
    )
    parser.add_argument(
        '--resnet50', action='store_true',
        help='Download only ResNet-50 weights'
    )
    parser.add_argument(
        '--vitb16', action='store_true',
        help='Download only ViT-B/16 weights'
    )
    parser.add_argument(
        '--output-dir', type=str, default='pretrained',
        help='Output directory for saved weights (default: pretrained/)'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # If no specific flag, download both
    download_all = not args.resnet50 and not args.vitb16

    if download_all or args.resnet50:
        download_resnet50(args.output_dir)
    else:
        print("[1/2] Skipping ResNet-50 (use --resnet50 to download)")

    if download_all or args.vitb16:
        download_vitb16(args.output_dir)
    else:
        print("[2/2] Skipping ViT-B/16 (use --vitb16 to download)")

    print(f"\n✅ Done! Weights saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == '__main__':
    main()