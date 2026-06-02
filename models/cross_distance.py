"""
Cross Distance Calculation — Directional Distance Metric for FSAR.

Paper Section 3.3, Eq. (8)-(10):
    "We propose Cross Distance Calculation that leverages both forward and
     backward (inverted) representations to compute a robust distance metric
     between query samples and class prototypes."

Notation:
    P̂^c = forward prototype (original feature representation)
    P̌^c = inverted prototype (feature-level inversion, NOT temporal flip)
    Q̂^r = forward query
    Q̌^r = inverted query

    "Inversion" in the paper refers to FEATURE-LEVEL inversion:
        P̌ = -P̂  (negation of feature vectors)
    This creates a complementary representation in feature space.

Prototype Construction (Eq. 8):
    P̂^c = (1/K) × Σ_{k=1}^{K} Ŝ^c_k
    (Average K-shot enhanced features per class)

Cross Distance (Eq. 9-10):
    D1 = ||P̂^c - Q̂^r||₂     — forward-forward (same representation)
    D2 = ||P̌^c - Q̌^r||₂     — inverted-inverted (same representation)
    D3 = 1/||P̂^c - Q̌^r||₂   — forward-inverted (cross representation, INVERTED distance)
    D4 = 1/||P̌^c - Q̂^r||₂   — inverted-forward (cross representation, INVERTED distance)

    D(P^c, Q^r) = (D1 + D2 + D3 + D4) / 4

Intuition:
    - D1, D2: Same-representation distances → small if same class
    - D3, D4: Cross-representation distances → naturally LARGE for same class
      (because forward and inverted are far apart in feature space)
      → Taking 1/distance makes them small, contributing to low total D for same class
    - For different classes: D1,D2 are large; D3,D4 raw distances may be smaller
      → 1/distance makes D3,D4 larger, increasing total D
"""

import torch


def build_prototype(support_features: torch.Tensor) -> torch.Tensor:
    """
    Build class prototypes by averaging K-shot support features.

    Implements Eq. (8):
        P̂^c = (1/K) × Σ_{k=1}^{K} Ŝ^c_k

    Args:
        support_features: [N, K, F, D] — N classes, K shots per class,
                          F frames, D feature dimensions

    Returns:
        prototypes: [N, F, D] — one prototype per class (forward representation)
    """
    return support_features.mean(dim=1)  # Average over K shots


def invert_features(x: torch.Tensor) -> torch.Tensor:
    """
    Compute inverted (complementary) representation of features.
    
    Paper uses P̌ to denote the "inverted" version of P̂.
    This is a FEATURE-LEVEL inversion (negation), creating a complementary
    representation in the feature space.
    
    P̌ = -P̂
    
    Args:
        x: [*, F, D] — feature tensor in any batch shape

    Returns:
        Inverted features of same shape: -x
    """
    return -x


def compute_temporal_l2_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute L2 (Frobenius) norm distance between two temporal sequences.

    Paper Eq. (9): D = ||P̂ − Q̂|| where ||·|| denotes the L2 norm
    (Frobenius norm for matrices).

    For tensors [*, F, D], this computes:
        ||a - b||_F = sqrt(Σ_f Σ_d (a[f,d] - b[f,d])²)

    Args:
        a: [*, F, D] — first sequence
        b: [*, F, D] — second sequence
        eps: Small constant for numerical stability in sqrt

    Returns:
        [*] tensor — Frobenius norm of (a - b) over last two dims
    """
    # Frobenius norm: sqrt of sum of all squared differences over F and D
    diff_sq = (a - b).pow(2)                # [*, F, D]
    dist = diff_sq.sum(dim=(-2, -1))        # [*] — sum over F and D
    return dist.sqrt()                       # [*] — L2 norm (not squared!)


def cross_distance_calculation(
    prototypes: torch.Tensor,
    query_features: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Cross Distance between query samples and class prototypes.

    Implements Eq. (9)-(10):
        D1 = ||P̂^c - Q̂^r||  (forward-forward)
        D2 = ||P̌^c - Q̌^r||  (inverted-inverted)
        D3 = 1/||P̂^c - Q̌^r|| (forward-inverted, reciprocal)
        D4 = 1/||P̌^c - Q̂^r|| (inverted-forward, reciprocal)
        D = (D1 + D2 + D3 + D4) / 4

    Where:
        P̂ = forward representation (original features)
        P̌ = inverted representation (-P̂, feature-level negation)

    Args:
        prototypes:     [N, F, D] — N class prototypes (forward representation P̂)
        query_features: [Q, F, D] — Q query samples (forward representation Q̂)
        eps: Small constant to avoid division by zero

    Returns:
        distances: [Q, N] — distance from each query to each prototype
                   (smaller = more similar = likely same class)
    """
    Q, F, D = query_features.shape
    N = prototypes.shape[0]

    # Forward representations (original)
    P_hat = prototypes          # [N, F, D] — P̂^c
    Q_hat = query_features      # [Q, F, D] — Q̂^r

    # Inverted representations (feature-level negation)
    P_check = invert_features(prototypes)        # [N, F, D] — P̌^c = -P̂^c
    Q_check = invert_features(query_features)    # [Q, F, D] — Q̌^r = -Q̂^r

    # Expand for broadcasting: [Q, N, F, D]
    P_hat_exp = P_hat.unsqueeze(0).expand(Q, -1, -1, -1)      # [Q, N, F, D]
    P_check_exp = P_check.unsqueeze(0).expand(Q, -1, -1, -1)  # [Q, N, F, D]
    Q_hat_exp = Q_hat.unsqueeze(1).expand(-1, N, -1, -1)      # [Q, N, F, D]
    Q_check_exp = Q_check.unsqueeze(1).expand(-1, N, -1, -1)  # [Q, N, F, D]

    # D1: Forward-Forward distance ||P̂ - Q̂|| (Eq. 9)
    D1 = compute_temporal_l2_distance(P_hat_exp, Q_hat_exp)  # [Q, N]

    # D2: Inverted-Inverted distance ||P̌ - Q̌|| (Eq. 9)
    # Note: ||(-P) - (-Q)|| = ||Q - P|| = ||P - Q|| = D1 mathematically
    # But we keep it explicit to match paper's 4-directional formulation
    D2 = compute_temporal_l2_distance(P_check_exp, Q_check_exp)  # [Q, N]

    # D3: Forward-Inverted (cross representation) — RECIPROCAL (Eq. 9)
    # ||P̂ - Q̌||^{-1} = 1/||P̂ - (-Q̂)|| = 1/||P̂ + Q̂||
    # For same class: P̂ ≈ Q̂, so ||P̂ + Q̂|| ≈ ||2P̂|| = large → 1/large = small
    D3_raw = compute_temporal_l2_distance(P_hat_exp, Q_check_exp)  # [Q, N]
    D3 = 1.0 / (D3_raw + eps)  # Reciprocal

    # D4: Inverted-Forward (cross representation) — RECIPROCAL (Eq. 9)
    # ||P̌ - Q̂||^{-1} = 1/||(-P̂) - Q̂|| = 1/||P̂ + Q̂||
    D4_raw = compute_temporal_l2_distance(P_check_exp, Q_hat_exp)  # [Q, N]
    D4 = 1.0 / (D4_raw + eps)  # Reciprocal

    # Final distance (Eq. 10): average of 4 components
    D = (D1 + D2 + D3 + D4) / 4.0  # [Q, N]

    return D
