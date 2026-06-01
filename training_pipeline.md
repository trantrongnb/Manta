# MANTA — Luồng Training Chi Tiết

> **Paper:** "Manta: Enhancing Mamba for Few-Shot Action Recognition of Long Sub-Sequence"  
> **arXiv:** 2412.07481v7, AAAI 2025  
> **Paradigm:** Episodic Training cho Few-Shot Action Recognition (N-way K-shot)

---

## Mục lục

1. [Tổng quan Kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Sơ đồ Luồng Training Tổng thể](#2-sơ-đồ-luồng-training-tổng-thể)
3. [Sơ đồ Matryoshka Mamba (Algorithm 1)](#3-sơ-đồ-matryoshka-mamba-algorithm-1)
4. [Sơ đồ Loss Function](#4-sơ-đồ-loss-function)
5. [Sơ đồ Luồng Dữ liệu](#5-sơ-đồ-luồng-dữ-liệu)
6. [Bảng Tóm tắt Tham số](#6-bảng-tóm-tắt-tham-số)
7. [Chi tiết Từng Bước](#7-chi-tiết-từng-bước)

---

## 1. Tổng quan Kiến trúc

MANTA gồm 4 thành phần chính:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MANTA FRAMEWORK                                     │
│                                                                              │
│   Input: Support S = {(Vᶜₖ, yᶜ)}   +   Query Q = {Vʳ}                      │
│                                                                              │
│   ┌──────────────┐    ┌─────────────────────┐    ┌──────────────────┐        │
│   │ ① Backbone ψ │───▶│ ② Matryoshka Mamba  │───▶│ ③ Cross Distance │──▶ ŷ  │
│   │   (frozen)   │    │   Multi-scale SSM   │    │   4-directional  │        │
│   │  ResNet-50   │    │   Scales O={1,2,4}  │    │   L2 distance    │        │
│   └──────────────┘    └────────┬────────────┘    └──────────────────┘        │
│                                │                                              │
│                                ▼                                              │
│                      ┌─────────────────────┐                                 │
│                      │ ④ Contrastive Branch │ (song song với Mamba branch)   │
│                      │  L_hc = L_S+L_Q+L_SQ │                                │
│                      └─────────────────────┘                                 │
│                                                                              │
│   Loss:  L_total = λ × L_ce + L_hc   (λ = 4.0)                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Sơ đồ Luồng Training Tổng thể

Dưới đây là flowchart chi tiết toàn bộ một vòng lặp training (1 episode = 1 optimization step):

```
                              ┌──────────────────────────┐
                              │      BẮT ĐẦU TRAIN       │
                              │  python train.py         │
                              │  --config config.yaml    │
                              └────────────┬─────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      ▼                      │
                    │  ┌──────────────────────────────────────┐   │
                    │  │       STEP 1: KHỞI TẠO               │   │
                    │  │                                      │   │
                    │  │  ① Load Config (OmegaConf)          │   │
                    │  │  ② set_seed(42)                      │   │
                    │  │  ③ Build Model: Manta()             │   │
                    │  │     ├─ Backbone: ResNet-50          │   │
                    │  │     │  (frozen, ImageNet pretrained) │   │
                    │  │     ├─ MatryoshkaMamba()            │   │
                    │  │     │  ├─ InnerModule(scale=1)      │   │
                    │  │     │  ├─ InnerModule(scale=2)      │   │
                    │  │     │  ├─ InnerModule(scale=4)      │   │
                    │  │     │  └─ OuterModule (shared)      │   │
                    │  │     └─ HybridContrastiveLoss(τ=0.07)│   │
                    │  │  ④ Build Optimizer: SGD(lr=0.001,   │   │
                    │  │     momentum=0.9, wd=5e-4)          │   │
                    │  │  ⑤ Scheduler: CosineAnnealingLR     │   │
                    │  │     (T_max=10000, eta_min=1e-6)     │   │
                    │  │  ⑥ Load Train/Val Datasets          │   │
                    │  └──────────────────┬───────────────────┘   │
                    │                     │                       │
                    └─────────────────────┼───────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────┐
                    │  STEP 2: EPISODIC TRAINING LOOP         │
                    │  for task_idx in range(0, 10000):       │
                    └─────────────────┬───────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            ▼                            │
         │  ┌──────────────────────────────────────────────────┐   │
         │  │  2a. SAMPLE EPISODE                              │   │
         │  │  EpisodicSampler.sample_episode()                │   │
         │  │                                                  │   │
         │  │  Input:  train_dataset (all train classes)      │   │
         │  │          n_way=5, k_shot=1, n_query=5           │   │
         │  │                                                  │   │
         │  │  Process:                                        │   │
         │  │   ┌──────────────────────────────────────┐       │   │
         │  │   │ 1. Randomly select N=5 classes       │       │   │
         │  │   │    from available training classes   │       │   │
         │  │   │    (e.g. classes: 12, 3, 45, 8, 21) │       │   │
         │  │   ├──────────────────────────────────────┤       │   │
         │  │   │ 2. For each selected class:          │       │   │
         │  │   │    ├─ Get all video paths            │       │   │
         │  │   │    ├─ Random sample K+Q = 6 videos   │       │   │
         │  │   │    │  (disjoint: support ∩ query = ∅)│       │   │
         │  │   │    ├─ K=1 → Support set              │       │   │
         │  │   │    └─ Q=5 → Query set                │       │   │
         │  │   ├──────────────────────────────────────┤       │   │
         │  │   │ 3. Load video frames:                │       │   │
         │  │   │    ├─ Uniform sample F=16 frames     │       │   │
         │  │   │    │  (train: random jitter,         │       │   │
         │  │   │    │   test: deterministic center)   │       │   │
         │  │   │    ├─ Resize to 256×256              │       │   │
         │  │   │    ├─ RandomCrop to 224×224 (train)  │       │   │
         │  │   │    ├─ RandomHorizontalFlip (train)   │       │   │
         │  │   │    └─ Normalize (ImageNet stats)     │       │   │
         │  │   ├──────────────────────────────────────┤       │   │
         │  │   │ 4. Stack into tensors:               │       │   │
         │  │   │    Support: [N, K, F, C, H, W]       │       │   │
         │  │   │           = [5, 1, 16, 3, 224, 224]  │       │   │
         │  │   │    Query:   [N×Q, F, C, H, W]        │       │   │
         │  │   │           = [25, 16, 3, 224, 224]    │       │   │
         │  │   │    Labels:  Support [N×K]=[5],       │       │   │
         │  │   │             Query [N×Q]=[25]          │       │   │
         │  │   │             (relative: 0,1,2,3,4)    │       │   │
         │  │   └──────────────────────────────────────┘       │   │
         │  └─────────────────┬────────────────────────────────┘   │
         │                    │                                    │
         │                    ▼                                    │
         │  ┌──────────────────────────────────────────────────┐   │
         │  │  2b. FORWARD PASS (Manta.forward)               │   │
         │  │                                                  │   │
         │  │  ┌────────────────────────────────────────┐     │   │
         │  │  │  ① FEATURE EXTRACTION                  │     │   │
         │  │  │                                        │     │   │
         │  │  │  Backbone: ResNet-50 (FROZEN)          │     │   │
         │  │  │  ┌──────────────────────────────┐      │     │   │
         │  │  │  │ Support [5,1,16,3,224,224]   │      │     │   │
         │  │  │  │   → flatten [5,16,3,224,224] │      │     │   │
         │  │  │  │   → ResNet-50 (no_grad!)     │      │     │   │
         │  │  │  │   → sup_feats [5,16,2048]    │      │     │   │
         │  │  │  ├──────────────────────────────┤      │     │   │
         │  │  │  │ Query   [25,16,3,224,224]    │      │     │   │
         │  │  │  │   → ResNet-50 (no_grad!)     │      │     │   │
         │  │  │  │   → que_feats [25,16,2048]   │      │     │   │
         │  │  │  └──────────────────────────────┘      │     │   │
         │  │  └────────────────────────────────────────┘     │   │
         │  │                      │                           │   │
         │  │                      ▼                           │   │
         │  │  ┌────────────────────────────────────────┐     │   │
         │  │  │  ② MATRYOSHKA MAMBA BRANCH             │     │   │
         │  │  │  (Algorithm 1 — xem Section 3)         │     │   │
         │  │  │                                        │     │   │
         │  │  │  sup_enhanced = MatryoshkaMamba(       │     │   │
         │  │  │      sup_feats) → [5,16,2048]          │     │   │
         │  │  │  que_enhanced = MatryoshkaMamba(       │     │   │
         │  │  │      que_feats) → [25,16,2048]         │     │   │
         │  │  │                                        │     │   │
         │  │  │  Reshape support: [5,1,16,2048]        │     │   │
         │  │  └────────────────────────────────────────┘     │   │
         │  │                      │                           │   │
         │  │                      ▼                           │   │
         │  │  ┌────────────────────────────────────────┐     │   │
         │  │  │  ③ CROSS DISTANCE CALCULATION          │     │   │
         │  │  │  (Eq. 8-10)                            │     │   │
         │  │  │                                        │     │   │
         │  │  │  a) Build Prototypes (Eq.8):           │     │   │
         │  │  │     P̂ᶜ = mean(sup_enhanced, dim=K)    │     │   │
         │  │  │     [5,16,2048]  (1 per class)         │     │   │
         │  │  │                                        │     │   │
         │  │  │  b) Feature Inversion:                 │     │   │
         │  │  │     P̌ᶜ = -P̂ᶜ  (negation)              │     │   │
         │  │  │     Q̌ʳ = -Q̂ʳ  (negation)              │     │   │
         │  │  │                                        │     │   │
         │  │  │  c) 4-directional L2 (Eq.9):           │     │   │
         │  │  │     D1 = ||P̂ᶜ - Q̂ʳ||  (forward-fwd)   │     │   │
         │  │  │     D2 = ||P̌ᶜ - Q̌ʳ||  (inv-inv)       │     │   │
         │  │  │     D3 = 1/||P̂ᶜ - Q̌ʳ|| (fwd-inv, recip)│    │   │
         │  │  │     D4 = 1/||P̌ᶜ - Q̂ʳ|| (inv-fwd, recip)│    │   │
         │  │  │                                        │     │   │
         │  │  │  d) Final Distance (Eq.10):            │     │   │
         │  │  │     D = (D1+D2+D3+D4)/4 → [25, 5]     │     │   │
         │  │  └────────────────────────────────────────┘     │   │
         │  │                      │                           │   │
         │  │                      ▼                           │   │
         │  │  ┌────────────────────────────────────────┐     │   │
         │  │  │  ④ PREDICTION                           │     │   │
         │  │  │                                        │     │   │
         │  │  │  ŷ = argmin(D, dim=-1) → [25]          │     │   │
         │  │  │  (smallest distance = most similar)    │     │   │
         │  │  └────────────────────────────────────────┘     │   │
         │  │                      │                           │   │
         │  │         ┌────────────┴────────────┐              │   │
         │  │         │  mode?                  │              │   │
         │  │         ├──────────┬──────────────┤              │   │
         │  │         │  test    │  train       │              │   │
         │  │         ▼          ▼              │              │   │
         │  │  ┌──────────┐ ┌──────────────────┐│              │   │
         │  │  │ Return   │ │ ⑤ LOSS COMPUTE   ││              │   │
         │  │  │ pred +   │ │                  ││              │   │
         │  │  │ distances│ │ a) L_ce loss:    ││              │   │
         │  │  └──────────┘ │  logits = -D     ││              │   │
         │  │               │  (negate distance)││              │   │
         │  │               │  L_ce = Cross     ││              │   │
         │  │               │  Entropy(logits,  ││              │   │
         │  │               │  query_labels)   ││              │   │
         │  │               │                  ││              │   │
         │  │               │ b) L_hc loss:    ││              │   │
         │  │               │  HybridContrastive││              │   │
         │  │               │  Loss (xem §4)   ││              │   │
         │  │               │  ├─ L_S (support) ││              │   │
         │  │               │  ├─ L_Q (query)  ││              │   │
         │  │               │  └─ L_SQ (S∪Q)   ││              │   │
         │  │               │                  ││              │   │
         │  │               │ c) L_total:      ││              │   │
         │  │               │  L = 4.0×L_ce    ││              │   │
         │  │               │      + L_hc      ││              │   │
         │  │               │                  ││              │   │
         │  │               │ d) Accuracy:     ││              │   │
         │  │               │  acc = mean(ŷ==  ││              │   │
         │  │               │  query_labels)   ││              │   │
         │  │               └────────┬─────────┘│              │   │
         │  └────────────────────────┼──────────┘              │   │
         │                           │                         │   │
         │                           ▼                         │   │
         │  ┌──────────────────────────────────────────────┐   │   │
         │  │  2c. BACKWARD PASS                           │   │   │
         │  │                                              │   │   │
         │  │  ① optimizer.zero_grad()                     │   │   │
         │  │     (reset tất cả gradients về 0)            │   │   │
         │  │                                              │   │   │
         │  │  ② loss.backward()                           │   │   │
         │  │     (tính gradient cho toàn bộ               │   │   │
         │  │      trainable parameters:                   │   │   │
         │  │      MatryoshkaMamba + Contrastive)          │   │   │
         │  │     (backbone FROZEN → không có gradient)    │   │   │
         │  │                                              │   │   │
         │  │  ③ clip_grad_norm_(max_norm=10.0)            │   │   │
         │  │     (ngăn exploding gradients)               │   │   │
         │  │                                              │   │   │
         │  │  ④ optimizer.step()                          │   │   │
         │  │     (SGD update weights)                     │   │   │
         │  │                                              │   │   │
         │  │  ⑤ scheduler.step()                          │   │   │
         │  │     (CosineAnnealing: update learning rate)  │   │   │
         │  └──────────────────┬───────────────────────────┘   │   │
         │                     │                               │   │
         │         ┌───────────┼───────────┐                   │   │
         │         │           │           │                   │   │
         │    task%100=0  task%1000=0  task%5000=0             │   │
         │         │           │           │                   │   │
         │         ▼           ▼           ▼                   │   │
         │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
         │  │ 2d. LOG  │ │ 2e. VAL  │ │ 2f. SAVE │            │   │
         │  │          │ │          │ │          │            │   │
         │  │ Loss:    │ │ Eval 600 │ │ Save:    │            │   │
         │  │ L_total  │ │ episodes │ │ model    │            │   │
         │  │ L_ce     │ │ on val   │ │ optimizer│            │   │
         │  │ L_hc     │ │ split    │ │ scheduler│            │   │
         │  │ L_S,L_Q, │ │          │ │ task_idx │            │   │
         │  │ L_SQ     │ │ Save best│ │ + config │            │   │
         │  │ Acc      │ │ model    │ │          │            │   │
         │  │ LR       │ │ if val_acc│           │            │   │
         │  └──────────┘ │ improves │ └──────────┘            │   │
         │               └──────────┘                        │   │
         │                                                   │   │
         │  task_idx < 10000? ──── YES ───▶ loop tiếp       │   │
         │         │                                         │   │
         │        NO                                         │   │
         │         │                                         │   │
         └─────────┼─────────────────────────────────────────┘
                   │
                   ▼
          ┌───────────────────┐
          │  STEP 3: KẾT THÚC │
          │                   │
          │  • Log elapsed    │
          │    time           │
          │  • Report best    │
          │    val accuracy   │
          │  • Close TB       │
          │    logger         │
          └───────────────────┘
```

---

## 3. Sơ đồ Matryoshka Mamba (Algorithm 1)

Đây là thành phần **cốt lõi** của MANTA, xử lý feature ở nhiều tỉ lệ thời gian (multi-scale):

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   MATRYOSHKA MAMBA — Algorithm 1                         │
│                                                                          │
│  Input:  I ∈ ℝ^{F×D}  (per-frame features, F=16 frames, D=2048 dims)    │
│  Output: Î ∈ ℝ^{F×D}  (multi-scale enhanced features)                   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  FOR EACH SCALE o ∈ O = {1, 2, 4}:                              │    │
│  │                                                                  │    │
│  │  ╔══════════════════════════════════════════════════════════╗    │    │
│  │  ║  EXAMPLE: scale o=2, F=16                               ║    │    │
│  │  ║                                                          ║    │    │
│  │  ║  Step 1: Divide I into F/o = 8 fragments                ║    │    │
│  │  ║  ┌─────────┬─────────┬─────────┬───┬─────────┐          ║    │    │
│  │  ║  │ Frag 0  │ Frag 1  │ Frag 2  │...│ Frag 7  │          ║    │    │
│  │  ║  │ [2×2048]│ [2×2048]│ [2×2048]│   │ [2×2048]│          ║    │    │
│  │  ║  └─────────┴─────────┴─────────┴───┴─────────┘          ║    │    │
│  │  ║                                                          ║    │    │
│  │  ║  Step 2: Inner Module (IM) — PER SCALE independent       ║    │    │
│  │  ║                                                          ║    │    │
│  │  ║  For fragment i = 0:                                     ║    │    │
│  │  ║    ┌──────────────┐                                      ║    │    │
│  │  ║    │ InnerModule  │  (Mamba-2 SSM + residual)           ║    │    │
│  │  ║    │  - LayerNorm                                        ║    │    │
│  │  ║    │  - Conv1D (local)                                   ║    │    │
│  │  ║    │  - Mamba-2 SSM (selective scan)                     ║    │    │
│  │  ║    │  - Residual: E = IM(frag) + frag                    ║    │    │
│  │  ║    └──────┬───────┘                                      ║    │    │
│  │  ║           │ E₀ [2×2048]                                  ║    │    │
│  │  ║           ▼                                              ║    │    │
│  │  ║    Ĩ ← E₀   (first fragment = set Ĩ)                     ║    │    │
│  │  ║                                                          ║    │    │
│  │  ║  For fragment i = 1:                                     ║    │    │
│  │  ║    ┌──────────────┐                                      ║    │    │
│  │  ║    │ InnerModule  │ → E₁ [2×2048]                        ║    │    │
│  │  ║    └──────┬───────┘                                      ║    │    │
│  │  ║           │                                              ║    │    │
│  │  ║           ▼                                              ║    │    │
│  │  ║    Concat(Ĩ, E₁) → [4×2048]                              ║    │    │
│  │  ║           │                                              ║    │    │
│  │  ║           ▼                                              ║    │    │
│  │  ║    ┌────────────────────────────────────┐               ║    │    │
│  │  ║    │ Outer Module (OM) — SHARED         │               ║    │    │
│  │  ║    │  Bidirectional Mamba scan:         │               ║    │    │
│  │  ║    │   Forward scan  [0→4]             │               ║    │    │
│  │  ║    │   Backward scan [4→0]             │               ║    │    │
│  │  ║    │   Add + Gate: update Ĩ            │               ║    │    │
│  │  ║    ├────────────────────────────────────┤               ║    │    │
│  │  ║    │ Tác dụng: align fragment mới với  │               ║    │    │
│  │  ║    │ tất cả fragment trước đó trong    │               ║    │    │
│  │  ║    │ ngữ cảnh bidirectional            │               ║    │    │
│  │  ║    └────────────────┬───────────────────┘               ║    │    │
│  │  ║                     │ Ĩ [4×2048] (updated)              ║    │    │
│  │  ║                     ▼                                   ║    │    │
│  │  ║   ... lặp lại cho fragments 2..7 ...                    ║    │    │
│  │  ║   Cuối cùng: Ĩ [16×2048] (full sequence, OM-aligned)   ║    │    │
│  │  ║                                                          ║    │    │
│  │  ║  Step 3: Learnable Weights w (Cross-Attention gating)   ║    │    │
│  │  ║    ┌──────────────────────────────┐                     ║    │    │
│  │  ║    │ w = σ( CB( Ĩ ⊕ I ) )         │                     ║    │    │
│  │  ║    │   = Sigmoid( ConvBlock(      │                     ║    │    │
│  │  ║    │       [Ĩ, I] concat ) )      │                     ║    │    │
│  │  ║    │ → w ∈ [0,1]^{F×D}            │                     ║    │    │
│  │  ║    └──────────────┬───────────────┘                     ║    │    │
│  │  ║                   │                                      ║    │    │
│  │  ║  Step 4: Scale Output                                    ║    │    │
│  │  ║    ˚I_o = w ⊗ Ĩ   ([16×2048])                            ║    │    │
│  │  ╚══════════════════════════════════════════════════════════╝    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  FINAL OUTPUT (Eq. 7):                                           │    │
│  │                                                                  │    │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐ │    │
│  │  │ ˚I_{o=1}        │   │ ˚I_{o=2}        │   │ ˚I_{o=4}        │ │    │
│  │  │ fragment size 1 │ + │ fragment size 2 │ + │ fragment size 4 │ │    │
│  │  │ finest detail   │   │ medium detail   │   │ coarsest detail │ │    │
│  │  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘ │    │
│  │           │                     │                     │           │    │
│  │           └─────────────────────┼─────────────────────┘           │    │
│  │                                 │                                 │    │
│  │                                 ▼                                 │    │
│  │          Î = (1/|O|) × Σ_{o∈O} ˚I_o    ← AVERAGE over scales     │    │
│  │                 [B, F, D=2048]                                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  DESIGN CHOICES (from paper):                                           │
│  • Inner Module: PER SCALE independent parameters                       │
│  • Outer Module: SHARED across all scales (parameter efficiency)         │
│  • Default scales: O = {1,2,4} (Table 3 ablation)                       │
│  • Complexity: O(F × |O|) — LINEAR in sequence length                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Sơ đồ Loss Function

MANTA sử dụng hai loss song song: **Cross-Entropy** (từ Cross Distance) và **Hybrid Contrastive Loss**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        LOSS FUNCTION (Eq. 11-14)                         │
│                                                                          │
│                         L_total = λ × L_ce + L_hc                        │
│                                (λ = 4.0)                                 │
│                                                                          │
│  ┌─────────────────────────────────┐  ┌───────────────────────────────┐ │
│  │  L_ce: Cross-Entropy Loss       │  │  L_hc: Hybrid Contrastive     │ │
│  │  (Classification Branch)         │  │  (Contrastive Branch)         │ │
│  │                                 │  │                               │ │
│  │  Input: distances D ∈ [Q,N]    │  │  Input: enhanced features     │ │
│  │                                 │  │  sup [N,K,F,D], que [Q,F,D]  │ │
│  │  1. Convert to logits:         │  │                               │ │
│  │     logits = -D                │  │  1. Temporal Pooling (mean)   │ │
│  │     (negate: small dist        │  │     [N,K,F,D] → [N*K,D]      │ │
│  │      → large logit)            │  │     [Q,F,D]   → [Q,D]         │ │
│  │                                 │  │                               │ │
│  │  2. CrossEntropy:              │  │  2. Three contrastive losses: │ │
│  │     L_ce = CE(logits,          │  │                               │ │
│  │              query_labels)      │  │  ┌─────────────────────────┐ │ │
│  │                                 │  │  │ L^S_con                  │ │ │
│  │  Mục tiêu: query gần           │  │  │ Supervised contrastive   │ │ │
│  │  prototype của đúng class       │  │  │ on support set           │ │ │
│  │  nhất                          │  │  │                          │ │ │
│  │                                 │  │  │ For each support sample: │ │ │
│  └─────────────────────────────────┘  │  │   anchor: sample x      │ │ │
│                                       │  │   pos: same-class others│ │ │
│                                       │  │   neg: different-class  │ │ │
│  ┌─────────────────────────────────┐  │  │                          │ │ │
│  │  Gradient Flow                  │  │  │  L_S = -log(            │ │ │
│  │                                 │  │  │   exp(cos(a,pos)/τ) /  │ │ │
│  │  L_ce ← D ← Q̂,P̂ ← Mamba ← I    │  │  │   (exp(cos(a,pos)/τ) + │ │ │
│  │  (gradients qua Mamba,          │  │  │    Σ exp(cos(a,neg)/τ)) │ │ │
│  │   KHÔNG qua Backbone)           │  │  │  )                       │ │ │
│  └─────────────────────────────────┘  │  │                          │ │ │
│                                       │  └─────────────────────────┘ │ │
│                                       │                               │ │
│                                       │  ┌─────────────────────────┐ │ │
│                                       │  │ L^Q_con                  │ │ │
│                                       │  │ Unsupervised contrastive │ │ │
│                                       │  │ on query set             │ │ │
│                                       │  │                          │ │ │
│                                       │  │ Uses episodic structure: │ │ │
│                                       │  │ assumes queries are      │ │ │
│                                       │  │ ordered by class         │ │ │
│                                       │  │  [class_0×Q_per_class,  │ │ │
│                                       │  │   class_1×Q_per_class,  │ │ │
│                                       │  │   ..., class_N×Q_per]    │ │ │
│                                       │  │                          │ │ │
│                                       │  │ Same InfoNCE formula     │ │ │
│                                       │  └─────────────────────────┘ │ │
│                                       │                               │ │
│                                       │  ┌─────────────────────────┐ │ │
│                                       │  │ L^SQ_con                 │ │ │
│                                       │  │ Unsupervised contrastive │ │ │
│                                       │  │ on S ∪ Q combined        │ │ │
│                                       │  │                          │ │ │
│                                       │  │ Interleave support +     │ │ │
│                                       │  │ query per class:         │ │ │
│                                       │  │  [sup_c0, que_c0,        │ │ │
│                                       │  │   sup_c1, que_c1, ...]   │ │ │
│                                       │  │                          │ │ │
│                                       │  │ Each class now has       │ │ │
│                                       │  │ K + Q_per_class samples  │ │ │
│                                       │  │                          │ │ │
│                                       │  │ Cross-set relationships  │ │ │
│                                       │  └─────────────────────────┘ │ │
│                                       │                               │ │
│                                       │  3. Total (Eq. 13):           │ │
│                                       │     L_hc = L_S + L_Q + L_SQ  │ │
│                                       │                               │
│                                       │  4. InfoNCE temperature:     │ │
│                                       │     τ = 0.07                 │ │
│                                       └───────────────────────────────┘ │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  FINAL LOSS (Eq. 14):                                             │   │
│  │                                                                    │   │
│  │  L_total = 4.0 × L_ce + (L_S + L_Q + L_SQ)                       │   │
│  │                                                                    │   │
│  │  Weight λ=4.0: CE loss được ưu tiên hơn vì nó trực tiếp           │   │
│  │  ảnh hưởng đến classification accuracy                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Sơ đồ Luồng Dữ liệu

Mô tả kích thước tensor qua từng bước:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW DIAGRAM                                  │
│                                                                          │
│  Tham số episode: N=5 way, K=1 shot, Q=5 query/class, F=16 frames       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                     SUPPORT SET (S)                             │     │
│  │                                                                 │     │
│  │  Videos: [N, K, F, C, H, W] = [5, 1, 16, 3, 224, 224]         │     │
│  │     │                                                           │     │
│  │     ├── Flatten: [N×K, F, C, H, W] = [5, 16, 3, 224, 224]     │     │
│  │     │                                                           │     │
│  │     ├── Backbone (ResNet-50, FROZEN):                          │     │
│  │     │   per-frame: [224,224,3] → [2048]                        │     │
│  │     │   output: [5, 16, 2048]                                   │     │
│  │     │                                                           │     │
│  │     ├── Matryoshka Mamba: [5, 16, 2048] → [5, 16, 2048]       │     │
│  │     │                                                           │     │
│  │     └── Build Prototype (Eq.8):                                │     │
│  │         mean over K shots: [5, 1, 16, 2048] → [5, 16, 2048]   │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                      QUERY SET (Q)                              │     │
│  │                                                                 │     │
│  │  Videos: [N×Q, F, C, H, W] = [25, 16, 3, 224, 224]            │     │
│  │     │                                                           │     │
│  │     ├── Backbone (same, frozen):                               │     │
│  │     │   output: [25, 16, 2048]                                  │     │
│  │     │                                                           │     │
│  │     └── Matryoshka Mamba: [25, 16, 2048] → [25, 16, 2048]     │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                 CROSS DISTANCE                                   │     │
│  │                                                                 │     │
│  │  Prototypes P̂: [5, 16, 2048]                                   │     │
│  │  Queries    Q̂: [25, 16, 2048]                                   │     │
│  │                                                                 │     │
│  │  Compute D1..D4 → D: [25, 5]                                   │     │
│  │  (1 distance value per query-prototype pair)                    │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    PREDICTION                                    │     │
│  │                                                                 │     │
│  │  ŷ = argmin(D, dim=-1) → [25]                                  │     │
│  │  (values: 0,1,2,3,4 — relative class indices)                   │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    LOSSES                                        │     │
│  │                                                                 │     │
│  │  L_ce: [25,5] logits + [25] labels → scalar                    │     │
│  │  L_hc: [5, 16, 2048] sup + [25, 16, 2048] que → scalar        │     │
│  │    ├─ L_S:  sup pooled [5, 2048]                                │     │
│  │    ├─ L_Q:  que pooled [25, 2048]                               │     │
│  │    └─ L_SQ: combined [30, 2048]                                 │     │
│  │                                                                 │     │
│  │  L_total = 4.0 × L_ce + L_hc → scalar                          │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                  GRADIENT FLOW                                   │     │
│  │                                                                 │     │
│  │  L_total → backward()                                           │     │
│  │     │                                                           │     │
│  │     ├──▶ Matryoshka Mamba (trainable)                          │     │
│  │     │    ├─ InnerModule(scale=1) ✓ gradient                     │     │
│  │     │    ├─ InnerModule(scale=2) ✓ gradient                     │     │
│  │     │    ├─ InnerModule(scale=4) ✓ gradient                     │     │
│  │     │    └─ OuterModule        ✓ gradient                       │     │
│  │     │                                                           │     │
│  │     ├──▶ Contrastive Branch  ✓ gradient                        │     │
│  │     │                                                           │     │
│  │     └──▶ Backbone (ResNet-50)  ✗ FROZEN (no gradient)          │     │
│  └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Bảng Tóm tắt Tham số

| Component | Parameter | Value | Description |
|-----------|-----------|-------|-------------|
| **Episode** | N (n_way) | 5 | Số class trong mỗi episode |
| | K (k_shot) | 1 | Số support video mỗi class |
| | Q (n_query) | 5 | Số query video mỗi class |
| | F (num_frames) | 16 | Số frame lấy mỗi video |
| **Backbone** | ResNet-50 | - | Feature extractor |
| | pretrained | ImageNet | Pretrained weights |
| | frozen | True | Không update trong training |
| | D (d_model) | 2048 | Output dimension |
| **Matryoshka Mamba** | O (scales) | {1, 2, 4} | Multi-scale temporal set |
| | d_state | 64 | Mamba-2 state space dim |
| | d_conv | 4 | Conv1D kernel size |
| | expand | 2 | Mamba-2 expansion factor |
| | InnerModule | per-scale | Independent parameters |
| | OuterModule | shared | Single shared instance |
| **Contrastive** | τ (temperature) | 0.07 | InfoNCE temperature |
| | No projection | - | Direct on pooled features |
| **Loss** | λ (lambda_ce) | 4.0 | CE loss weight |
| | L_total | λ×L_ce + L_hc | Eq. 14 |
| **Optimizer** | Type | SGD | - |
| | lr | 0.001 | Initial learning rate |
| | momentum | 0.9 | - |
| | weight_decay | 5e-4 | L2 regularization |
| | grad_clip | 10.0 | Max gradient norm |
| **Scheduler** | Type | CosineAnnealing | - |
| | T_max | 10000 | = num_tasks |
| | eta_min | 1e-6 | Minimum LR |
| **Training** | num_tasks | 10000 | Total training episodes |
| | val_interval | 1000 | Validate every N tasks |
| | val_tasks | 600 | Val episodes per eval |
| | log_interval | 100 | Log every N tasks |
| | save_interval | 5000 | Checkpoint every N tasks |
| | seed | 42 | Random seed |

---

## 7. Chi tiết Từng Bước

### 7.1 Khởi tạo (train.py:139-202)

```python
# Pseudocode cho bước khởi tạo:
def train(cfg):
    set_seed(42)
    device = cuda if available else cpu
    
    # Build model
    model = Manta(
        backbone_name='resnet50',
        d_model=2048,
        scales=[1, 2, 4],
        d_state=64, d_conv=4, expand=2,
        temperature=0.07,
        lambda_ce=4.0,
        freeze_backbone=True  # backbone FROZEN
    ).to(device)
    
    # Optimizer: only non-frozen params
    optimizer = SGD(trainable_params, lr=0.001, mom=0.9, wd=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)
    
    # Datasets
    train_dataset = VideoDataset('data/splits/train.txt', F=16, mode='train')
    val_dataset   = VideoDataset('data/splits/val.txt',   F=16, mode='test')
```

### 7.2 Training Loop (train.py:209-307)

Mỗi iteration là **1 episode** (1 optimization step), gồm các bước con:

| Bước | Hàm gọi | Input → Output |
|------|---------|----------------|
| 2a | `EpisodicSampler.sample_episode()` | Dataset → Support[N,K,F,C,H,W] + Query[Q,F,C,H,W] |
| 2b | `model.forward()` | Videos → loss, predictions, accuracy |
| 2c | `loss.backward()` + `optimizer.step()` | loss → gradient → weight update |
| 2d | `tb_logger.log_train_step()` | metrics → TensorBoard |
| 2e | `evaluate()` (every 1000 tasks) | val_dataset → val_acc ± CI |
| 2f | `torch.save()` (every 5000 tasks) | checkpoint → disk |

### 7.3 Model Forward (manta.py:149-256)

Chi tiết từng sub-step trong forward pass:

```
model.forward(support, query, sup_labels, que_labels, n_way=5, k_shot=1):
    
    ① Feature Extraction:
       sup_feats = backbone(support)  [5, 16, 2048]
       que_feats = backbone(query)    [25, 16, 2048]
    
    ② Matryoshka Mamba:
       sup_enhanced = matryoshka_mamba(sup_feats)  [5, 16, 2048]
       que_enhanced = matryoshka_mamba(que_feats)  [25, 16, 2048]
       prototypes   = mean(sup_enhanced, dim=K)     [5, 16, 2048]
    
    ③ Cross Distance:
       D1 = ||P̂ - Q̂||, D2 = ||-P̂ - (-Q̂)||
       D3 = 1/||P̂ - (-Q̂)||, D4 = 1/||(-P̂) - Q̂||
       distances = (D1+D2+D3+D4)/4  [25, 5]
    
    ④ Prediction:
       pred_labels = argmin(distances, dim=-1)  [25]
    
    ⑤ Loss (train mode):
       logits = -distances  [25, 5]
       L_ce = CrossEntropy(logits, query_labels)
       
       L_S  = supervised_contrastive(sup_pooled, sup_labels)
       L_Q  = unsupervised_contrastive(que_pooled, n_way, Q_per_class)
       L_SQ = unsupervised_contrastive(combined, n_way, K+Q_per_class)
       L_hc = L_S + L_Q + L_SQ
       
       L_total = 4.0 * L_ce + L_hc
       accuracy = mean(pred_labels == query_labels)
```

### 7.4 Evaluation (train.py:89-136)

```python
def evaluate(model, val_dataset, n_way=5, k_shot=1, num_tasks=600):
    model.eval()
    accuracies = []
    
    for _ in range(600):  # 600 validation episodes
        episode = sample_episode(val_dataset, n_way, k_shot)
        result = model(episode, mode='test')  # No loss computation
        acc = compute_accuracy(result.pred_labels, episode.labels)
        accuracies.append(acc)
    
    mean_acc = mean(accuracies)
    ci = 1.96 * std(accuracies) / sqrt(600)  # 95% confidence
    return mean_acc, ci
```

### 7.5 Quan trọng: Backbone Frozen

Backbone (ResNet-50) **không bao giờ được training**:
- `model.train()` bị override để giữ backbone ở `eval()` mode
- `extract_features()` dùng `torch.no_grad()` khi backbone frozen
- Optimizer chỉ nhận `params.requires_grad == True` (chỉ MatryoshkaMamba + Contrastive)

---

*Generated from source code analysis of MANTA repository.*