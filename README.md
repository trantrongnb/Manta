# MANTA: Enhancing Mamba for Few-Shot Action Recognition of Long Sub-Sequence

> **Paper**: arXiv:2412.07481v7 — AAAI 2025  
> **Authors**: Wenbo Huang et al.  
> **Reimplementation**: Clean, modular codebase following the paper methodology exactly.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MANTA Framework                               │
│                                                                         │
│  Input Video (F frames) → Backbone ψ (frozen) → S ∈ ℝ^{F×D}           │
│                                                                         │
│  ┌─── Mamba Branch ───────────────────────────────────────────────┐    │
│  │  For each scale o ∈ O = {1, 2, 4}:                             │    │
│  │    ① Inner Module (independent params, bidirectional Mamba-2)  │    │
│  │       → Local fragment enhancement                              │    │
│  │    ② Outer Module (shared params, Conv2D Block weights)         │    │
│  │       → Global temporal alignment                               │    │
│  │  Average all scales → Ŝ                                        │    │
│  │  Build Prototypes → Cross Distance Calculation → L_ce           │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─── Contrastive Branch (parallel) ─────────────────────────────┐    │
│  │  L^S_con (supervised, support) + L^Q_con (unsupervised, query) │    │
│  │  + L^SQ_con (unsupervised, S∪Q) → L_hc                        │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  L_total = λ × L_ce + L_hc    (λ=4.0)                                  │
│  Prediction: ŷ = argmin_c D(P^c, Q^r)                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Contributions

1. **Matryoshka Mamba**: Multi-scale nested structure with:
   - Inner Module (I-P: independent params) for local fragment enhancement
   - Outer Module (S-P: shared params) for global temporal alignment
   - Learnable scale weights via Conv2D Block

2. **Cross Distance Calculation**: 4-directional distance metric leveraging temporal order

3. **Hybrid Contrastive Learning**: Three-level contrastive loss (supervised + unsupervised)

## Project Structure

```
manta/
├── configs/                    # YAML configuration files
│   ├── kinetics_resnet50.yaml
│   ├── ssv2_resnet50.yaml
│   ├── ucf101_resnet50.yaml
│   ├── hmdb51_resnet50.yaml
│   └── kinetics_resnet50_f16.yaml
├── models/                     # Model components
│   ├── __init__.py
│   ├── backbone.py             # ResNet-50, ViT-B/16
│   ├── inner_module.py         # Inner Module (Mamba-2, I-P)
│   ├── outer_module.py         # Outer Module (Mamba-2, S-P + Conv2D)
│   ├── matryoshka_mamba.py     # Multi-scale integration
│   ├── cross_distance.py      # Cross Distance (4 directions)
│   ├── contrastive.py         # Hybrid Contrastive Loss
│   └── manta.py               # Main framework
├── datasets/                   # Data loading
│   ├── __init__.py
│   ├── video_dataset.py        # Video frame dataset
│   └── episode_sampler.py      # N-way K-shot episodic sampler
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── metrics.py              # Accuracy, confidence interval
│   └── logger.py               # TensorBoard, console logging
├── tools/                      # Preprocessing tools
│   ├── preprocess_videos.py    # Video → frames
│   └── create_splits.py       # Train/val/test splits
├── train.py                    # Training script
├── test.py                     # Evaluation script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

```bash
# Create environment
conda create -n manta python=3.10 -y
conda activate manta

# Install PyTorch
pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install Mamba-2 (core dependency)
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.2

# Install remaining dependencies
pip install -r requirements.txt
```

## Data Preparation

```bash
# 1. Preprocess videos to frames
python tools/preprocess_videos.py \
    --dataset kinetics \
    --video_dir data/kinetics/videos \
    --output_dir data/kinetics/frames

# 2. Create train/val/test splits
python tools/create_splits.py \
    --frame_dir data/kinetics/frames \
    --output_dir data/kinetics/splits \
    --dataset kinetics
```

## Training

```bash
# Kinetics 5-way 1-shot
python train.py --config configs/kinetics_resnet50.yaml

# SSv2 (temporal-sensitive, no horizontal flip)
python train.py --config configs/ssv2_resnet50.yaml

# Resume training
python train.py --config configs/kinetics_resnet50.yaml \
    --resume checkpoints/kinetics_rn50_5way1shot_f8/best_model.pth
```

## Evaluation

```bash
# Test with 10,000 episodes
python test.py \
    --config configs/kinetics_resnet50.yaml \
    --checkpoint checkpoints/kinetics_rn50_5way1shot_f8/best_model.pth \
    --num_tasks 10000
```

## Expected Results (Table 1 of paper)

| Dataset  | Setting | ResNet-50 |
|----------|---------|-----------|
| SSv2     | 1-shot  | 63.4%     |
| SSv2     | 5-shot  | 87.4%     |
| Kinetics | 1-shot  | 82.4%     |
| Kinetics | 5-shot  | 94.2%     |
| UCF101   | 1-shot  | 95.9%     |
| HMDB51   | 1-shot  | 86.8%     |

## Hyperparameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ         | 4.0   | CE loss weight |
| τ         | 0.07  | Contrastive temperature |
| O         | {1,2,4} | Scale set |
| d_state   | 64    | Mamba-2 state dim |
| LR        | 0.001 | Initial learning rate |
| Optimizer | SGD   | With momentum 0.9 |

## Citation

```bibtex
@inproceedings{huang2025manta,
  title={Manta: Enhancing Mamba for Few-Shot Action Recognition of Long Sub-Sequence},
  author={Huang, Wenbo and others},
  booktitle={AAAI},
  year={2025}
}
```


---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
