# MANTA Training Pipeline — Phân tích chi tiết

## 📋 Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           MANTA Framework (Paper Fig. 2)                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  Input: Support S = {(V^c_k, y^c)} và Query Q = {V^r}                                │
│                                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  ① Feature Extraction (Backbone ψ, frozen)                                   │    │
│  │     S = {s₁, s₂, ..., s_F} ∈ ℝ^{F×D}                                         │    │
│  │     Backbone: ResNet-50 (D=2048) / ViT-B (D=768) / VMamba-B                  │    │
│  │     ImageNet pretrained, frozen trong training                                │    │
│  │     Output: per-frame feature vectors [B, F, D]                               │    │
│  └──────────────────────┬──────────────────────────────────────────────────────┘    │
│                         │                                                             │
│  ┌──────────────────────▼──────────────────────────────────────────────────────┐    │
│  │  ② Matryoshka Mamba Branch (Algorithm 1 + Eq. 4-7)                          │    │
│  │                                                                               │    │
│  │  Multi-scale O = {1, 2, 4}:                                                   │    │
│  │  ┌─────────────────────────────────────────────────────┐                     │    │
│  │  │  Với mỗi scale o ∈ O:                                 │                     │    │
│  │  │  ┌─────────────────────────────────────────────────┐ │                     │    │
│  │  │  │  Inner Loop (Algorithm 1 lines 5-13):            │ │                     │    │
│  │  │  │  For mỗi fragment i:                             │ │                     │    │
│  │  │  │    E = IM(fragment) + fragment  (Eq. 4)          │ │                     │    │
│  │  │  │    Ĩ = OM(Concat(Ĩ, E))       (incremental)      │ │                     │    │
│  │  │  │  After loop:                                      │ │                     │    │
│  │  │  │    w = σ(CB(Ĩ ⊕ I))          (Eq. 5, learnable)  │ │                     │    │
│  │  │  │    ˚I = w ⊗ Ĩ                (Eq. 6)             │ │                     │    │
│  │  │  └─────────────────────────────────────────────────┘ │                     │    │
│  │  │  Output scale: ˚I_o ∈ ℝ^{F×D}                        │                     │    │
│  │  └─────────────────────────────────────────────────────┘                     │    │
│  │                                                                               │    │
│  │  Eq. (7): Output = (1/|O|) × Σ ˚I_o  (average over scales)                   │    │
│  │                                                                               │    │
│  │  IM design choices (paper Table II):                                          │    │
│  │    • Fw và Bw branches có INDEPENDENT params (I-P)                           │    │
│  │    • Architecture: LayerNorm → Mamba-2 → Linear                              │    │
│  │    • Residual connection: output = Linear([Fw; Bw]) + input                  │    │
│  │                                                                               │    │
│  │  OM design choices (paper Table II):                                          │    │
│  │    • Fw và Bw branches SHARE params (S-P)                                    │    │
│  │    • Conv2D Block: Conv1x1→BN→ReLU→Conv3x3→BN→ReLU→Conv1x1→BN               │    │
│  │    • Learnable weights: w = σ(CB([enhanced ⊕ original]))                     │    │
│  └──────────────────────┬──────────────────────────────────────────────────────┘    │
│                         │                                                             │
│  ┌──────────────────────▼──────────────────────────────────────────────────────┐    │
│  │  ③ Prototype Construction (Eq. 8)                                           │    │
│  │     P̂^c = (1/K) × Σ_{k=1}^K Ŝ^c_k                                           │    │
│  │     Output: [N, F, D] prototypes                                             │    │
│  └──────────────────────┬──────────────────────────────────────────────────────┘    │
│                         │                                                             │
│  ┌──────────────────────▼──────────────────────────────────────────────────────┐    │
│  │  ④ Cross Distance Calculation (Eq. 9-10)                                     │    │
│  │     D1 = ||P̂ - Q̂||     (forward-forward)                                     │    │
│  │     D2 = ||P̌ - Q̌||     (inverted-inverted, P̌ = -P̂)                         │    │
│  │     D3 = 1/||P̂ - Q̌||   (cross, reciprocal)                                  │    │
│  │     D4 = 1/||P̌ - Q̂||   (cross, reciprocal)                                  │    │
│  │     D = (D1 + D2 + D3 + D4) / 4                                              │    │
│  │     L_ce = CrossEntropy(-D, y^Q)                                             │    │
│  └──────────────────────┬──────────────────────────────────────────────────────┘    │
│                         │                                                             │
│  ┌──────────────────────▼──────────────────────────────────────────────────────┐    │
│  │  ⑤ Hybrid Contrastive Branch (Eq. 11-13)— PARALLEL                           │    │
│  │     Temporal pooling: mean(dim=1) trên temporal dimension                    │    │
│  │     L_S  = supervised contrastive loss trên support set                      │    │
│  │     L_Q  = unsupervised contrastive loss trên query set                      │    │
│  │     L_SQ = unsupervised contrastive loss trên S ∪ Q                          │    │
│  │     L_hc = L_S + L_Q + L_SQ  (Eq. 13)                                        │    │
│  │     InfoNCE (Eq. 12): -log[exp(sim(z,z⁺)/τ) / (exp(sim(z,z⁺)/τ) + Σ ...)]  │    │
│  └──────────────────────┬──────────────────────────────────────────────────────┘    │
│                         │                                                             │
│  ┌──────────────────────▼──────────────────────────────────────────────────────┐    │
│  │  ⑥ Total Loss (Eq. 14)                                                       │    │
│  │     L_total = λ × L_ce + L_hc    (λ = 4.0)                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 ĐỐI CHIẾU SOURCE CODE vs. PAPER — TỪNG BƯỚC

### ⚠️ VẤN ĐỀ QUAN TRỌNG: Algorithm 1 — Sliding Window

| | Paper (Algorithm 1 dòng 2444-2484) | Code (`matryoshka_mamba.py` dòng 124-141) |
|---|---|---|
| **Fragment selection** | `W = F - o + 1` → **Sliding window**, stride = 1 | `F // o` → **Non-overlapping**, stride = o |
| **Số fragments** | `W = F - o + 1` | `F / o` |
| **Vòng lặp** | `for i ∈ [0, W-1]` | `for i in range(F // o)` |
| **Fragment i** | `I[i : i+o, :]` | `features[:, i*o : (i+1)*o, :]` |

**Ví dụ với F=16, o=4:**
- Paper: 13 fragments — `[0:4], [1:5], [2:6], ..., [12:16]`
- Code: 4 fragments — `[0:4], [4:8], [8:12], [12:16]`

**Mâu thuẫn trong paper:** Section 3.2 mô tả *"non-overlapping fragments"* (giống code), nhưng Algorithm 1 dùng sliding window stride=1. Có khả năng **code đúng với ý đồ thuật toán thực tế**, còn Algorithm 1 trong paper là pseudo-code diễn giải chưa chính xác.

---

### ✅ PHẦN KHỚP CHÍNH XÁC 100%

#### 1. Feature Extraction (Eq. 1-2)

| Thành phần | Paper | Code | Trạng thái |
|---|---|---|---|
| Backbone | ResNet-50 / ViT-B / VMamba-B | `resnet50` / `vitb` | ✅ |
| Feature dimension | D=2048 | `d_model=2048` | ✅ |
| Backbone frozen | Có | `freeze_backbone=True` | ✅ |
| Frame sampling | Uniform temporal | `_sample_frame_indices()` với jitter (train) / center (test) | ✅ |
| Augmentation | Resize 256, RandomCrop 224, horizontal flip | `Resize(256)` + `RandomCrop(224)` + `RandomHorizontalFlip(0.5)` | ✅ |
| Normalization | ImageNet mean/std | `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` | ✅ |

#### 2. Inner Module (Eq. 4)

| Thành phần | Paper | Code | Trạng thái |
|---|---|---|---|
| Architecture | LayerNorm → Mamba-2 → Linear | `norm → mamba2 → linear` | ✅ |
| Directional branches | Fw + Bw | `fw_branch` + `bw_branch` | ✅ |
| Parameter strategy | **Independent** (I-P) per paper Table II | Independent `InnerModuleBranch` instances | ✅ |
| Bidirectional | Fw: forward scan; Bw: flip→scan→flip | `x` → `fw_branch(x)`, `x.flip(1)` → `bw_branch` → `flip(1)` | ✅ |
| Output projection | `Linear([Fw; Bw])` | `out_proj(torch.cat([fw, bw], dim=-1))` | ✅ |
| Residual | `+ S^ck_i` | `out + x` | ✅ |

#### 3. Outer Module (Eq. 5-6)

| Thành phần | Paper | Code | Trạng thái |
|---|---|---|---|
| Architecture | LayerNorm → Mamba-2 → Linear | `norm → mamba2 → out_proj` | ✅ |
| Parameter strategy | **Shared** (S-P) per paper Table II | Single `mamba2_shared` instance | ✅ |
| Learnable weights | `w = σ(CB([enhanced ⊕ original]))` | `compute_weights()` | ✅ |
| Conv2D Block | Conv1x1→BN→ReLU→Conv3x3→BN→ReLU→Conv1x1→BN | `nn.Sequential(Conv2d(2,mid,1), BN, ReLU, Conv2d(mid,mid,3), BN, ReLU, Conv2d(mid,1,1), BN)` | ✅ |
| ⊕ operation | Concatenation (channel dim) | `torch.cat([enh.unsq(1), orig.unsq(1)], dim=1)` | ✅ |

#### 4. Cross Distance Calculation (Eq. 9-10)

| Component | Công thức | Code | Trạng thái |
|---|---|---|---|
| D1 | `||P̂ - Q̂||` | `compute_temporal_l2_distance(P_hat, Q_hat)` | ✅ |
| D2 | `||P̌ - Q̌||` | `compute_temporal_l2_distance(P_check, Q_check)` | ✅ |
| D3 | `1/||P̂ - Q̌||` | `1.0 / (compute_temporal_l2_distance(P_hat, Q_check) + eps)` | ✅ |
| D4 | `1/||P̌ - Q̂||` | `1.0 / (compute_temporal_l2_distance(P_check, Q_hat) + eps)` | ✅ |
| Final | `(D1 + D2 + D3 + D4) / 4` | `(D1 + D2 + D3 + D4) / 4.0` | ✅ |
| Inversion | Feature-level: `P̌ = -P̂` | `invert_features(x) = -x` | ✅ |
| Norm | L2 (Frobenius): `sqrt(Σ(f,d) (a-b)²)` | `diff_sq.sum(dim=(-2,-1)).sqrt()` | ✅ |
| Prediction | `ŷ = argmin_c D(P^c, Q^r)` | `distances.argmin(dim=-1)` | ✅ |

#### 5. Prototype Construction (Eq. 8)

| Paper | Code | Trạng thái |
|---|---|---|
| `P̂^c = (1/K) × Σ_{k=1}^K Ŝ^ck_f` | `support_features.mean(dim=1)` | ✅ |

#### 6. Hybrid Contrastive Loss (Eq. 11-13)

| Component | Paper | Code | Trạng thái |
|---|---|---|---|
| InfoNCE base | `-log[exp(sim/τ) / (exp(sim/τ) + Σ exp(sim/τ))]` | `info_nce_loss()` | ✅ |
| Similarity | Cosine similarity | `F.normalize()` + `matmul` | ✅ |
| **L_S** (supervised) | Positives = same label | `supervised_contrastive_loss()` | ✅ |
| **L_Q** (unsupervised) | Positives = same class position | `unsupervised_contrastive_loss()` trên `que_pooled` | ✅ |
| **L_SQ** (unsupervised) | Positives = same class in S∪Q | `unsupervised_contrastive_loss()` trên `sq_features` | ✅ |
| L_hc | `= L_S + L_Q + L_SQ` | `L_hc = L_S + L_Q + L_SQ` | ✅ |
| Temperature τ | τ = 0.07 | `temperature=0.07` | ✅ |

#### 7. Training Objective (Eq. 14)

| Paper | Code | Trạng thái |
|---|---|---|
| `L_total = λ × L_ce + L_hc` | `L_total = self.lambda_ce * L_ce + L_hc` | ✅ |
| λ = 4.0 | `lambda_ce: 4.0` | ✅ |

#### 8. Multi-scale Averaging (Eq. 7)

| Paper | Code | Trạng thái |
|---|---|---|
| `Î = (1/|O|) × Σ_{o∈O} ˚I_o` | `stacked.mean(dim=0)` | ✅ |
| O = {1, 2, 4} | `scales: [1, 2, 4]` | ✅ |

---

### 🟡 KHÁC BIỆT NHỎ

#### 1. Temporal Pooling trong Contrastive Branch

- **Code:** `mean(dim=1)` — average pool trên frames
- **Paper:** Không nói rõ pooling strategy
- **Đánh giá:** Không phải lỗi — tất cả FSAR methods dùng mean pooling

#### 2. L_SQ Query Count

- **Paper Supplementary:** Giả định `Q = [qx¹₁...qx¹_K, ..., qx^N₁...qx^N_K]` (N×K samples, bằng support)
- **Code:** `q_per_class = Q // n_way` — linh hoạt hơn
- **Đánh giá:** Hợp lý vì query set có thể khác size

#### 3. IM/OM Input/Output Shapes

- **Paper Table XII:** IM input `[D, o]`, output `[F, D]` — confusing notation
- **Code:** input `[B, o, D]`, output `[B, o, D]` — đúng thực tế

---

## 📊 ĐÁNH GIÁ TỔNG THỂ

| Thành phần | Độ khớp | Ghi chú |
|---|---|---|
| Algorithm 1 (Matryoshka Mamba) | ⚠️ **~70%** | Sliding window (paper) vs non-overlapping (code) |
| Inner Module (I-P) | ✅ **100%** | |
| Outer Module (S-P) | ✅ **100%** | |
| Conv2D Block + Learnable Weights | ✅ **100%** | |
| Cross Distance (Eq. 9-10) | ✅ **100%** | |
| Prototype Construction (Eq. 8) | ✅ **100%** | |
| Hybrid Contrastive Loss (Eq. 11-13) | ✅ **95%** | L_SQ slightly flexible |
| Total Loss (Eq. 14, λ=4) | ✅ **100%** | |
| Backbone (ResNet-50 frozen) | ✅ **100%** | |
| Data Augmentation | ✅ **100%** | |
| Training Loop (SGD, cosine) | ✅ **100%** | |

**Kết luận: ~95% implementation khớp với paper.** Vấn đề chính là non-overlapping vs sliding window. Kết quả training 96.07% cho thấy implementation hoạt động hiệu quả.

---

## 🏗️ CẤU TRÚC THƯ MỤC VÀ TRAINING

### Các file chính

```
Manta/
├── train.py                    # Training script (episodic training)
├── test.py                     # Test script (10,000 episodes)
├── prepare_data.py             # Extract frames from videos
├── download_weights.py         # Download pretrained weights
├── configs/
│   └── lsa64_resnet50.yaml     # Configuration (LSA64, 5-way 1-shot, F=16)
├── models/
│   ├── manta.py                # Main Manta framework
│   ├── backbone.py             # ResNet-50 / ViT-B feature extractors
│   ├── matryoshka_mamba.py     # Algorithm 1 — multi-scale Matryoshka Mamba
│   ├── inner_module.py         # Inner Module — bidirectional Mamba-2 (I-P)
│   ├── outer_module.py         # Outer Module — shared Mamba-2 + Conv2D Block
│   ├── cross_distance.py       # Cross Distance Calculation (Eq. 9-10)
│   └── contrastive.py          # Hybrid Contrastive Loss (Eq. 11-13)
├── datasets/
│   ├── video_dataset.py        # VideoDataset — load frames, uniform sampling
│   └── episode_sampler.py      # EpisodicSampler — N-way K-shot episode construction
├── utils/
│   ├── metrics.py              # Accuracy, confidence interval
│   └── logger.py               # Console logging + TensorBoard
├── tools/
│   ├── create_splits.py        # Create train/val/test splits
│   └── preprocess_videos.py    # Frame extraction from videos
└── checkpoints/                # Model checkpoints (output)
```

### Data Pipeline

```
split.txt                        EpisodeDataset(DataLoader)
(video_path class)              ┌─────────────────┐
       │                        │ NumEpisodes ×    │
       ▼                        │ sample_episode() │
VideoDataset                    └──────┬──────────┘
┌──────────────────┐                   │
│ - Video frames    │                   ▼
│ - Frame sampling  │         support [N,K,F,C,H,W]
│ - Augmentation    │         query   [N*n_query,F,C,H,W]
│ - Normalization   │         sup_labels [N*K]
└──────────────────┘         que_labels [N*n_query]
```

### Training Loop (train.py)

```
for task_idx in range(num_tasks):
    1. Sample episode (EpisodicSampler)
    2. Forward pass (Manta)
       a. Backbone: extract features [B, F, D]
       b. Matryoshka Mamba: enhance features
       c. Build prototypes (Eq. 8)
       d. Cross Distance (Eq. 9-10) → L_ce
       e. Contrastive loss (Eq. 11-13) → L_hc
       f. L_total = λ × L_ce + L_hc (Eq. 14)
    3. Backward pass (only trainable params)
    4. Optimizer step (SGD) + scheduler step (cosine)
    5. Logging every 100 tasks
    6. Validation every 1,000 tasks
       → Save best model if val_acc improves
    7. Checkpoint every 5,000 tasks
```

---

## ⚙️ HYPERPARAMETERS (from paper and config)

| Parameter | Paper | Config (LSA64) |
|---|---|---|
| Backbone | ResNet-50 (ImageNet) | `resnet50` |
| Feature dim D | 2048 | 2048 |
| Scales O | {1, 2, 4} | [1, 2, 4] |
| Mamba-2 d_state | 64 | 64 |
| Mamba-2 d_conv | 4 | 4 |
| Mamba-2 expand | 2 | 2 |
| Temperature τ | 0.07 | 0.07 |
| λ (loss weight) | 4.0 | 4.0 |
| Frames F | 8 (regular) / 16 (long) | 16 |
| Training tasks | 10,000 | 10,000 |
| Batch size | 1 episode/step | 1 |
| Optimizer | SGD | SGD |
| Learning rate | 0.001 | 0.001 |
| Momentum | 0.9 | 0.9 |
| Weight decay | 5e-4 | 0.0005 |
| Scheduler | Cosine annealing | Cosine (min_lr=1e-6) |
| Gradient clip | — | 10.0 |
| Val interval | 1,000 tasks | 1,000 |
| Val episodes | 600 | 600 |
| Test episodes | 10,000 | 10,000 |

---

## 📊 KẾT QUẢ TRAINING (LSA64, 5-way 1-shot, F=16)

```
Best validation accuracy: 96.07%
Final validation: 95.59% ± 0.42%
Total time: 4h 19m 30s
Model saved: checkpoints/lsa64_rn50_5way1shot_f16/best_model.pth
```

---

## 🧪 COMMANDS

### Training
```bash
python train.py --config configs/lsa64_resnet50.yaml
python train.py --config configs/lsa64_resnet50.yaml --resume checkpoints/lsa64_rn50_5way1shot_f16/checkpoint_task5000.pth
python train.py --config configs/lsa64_resnet50.yaml --seed 123
```

### Testing
```bash
python test.py --config configs/lsa64_resnet50.yaml --checkpoint checkpoints/lsa64_rn50_5way1shot_f16/best_model.pth
```

### Monitor
```bash
tensorboard --logdir logs/lsa64_rn50_5way1shot_f16