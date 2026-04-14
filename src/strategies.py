"""
Layer aggregation strategies for combining multiple layers.
Includes weighted sum methods, greedy selection, clustering, and PCA utilities.
"""

import numpy as np
import cvxpy as cp
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from typing import List, Tuple, Optional

from scipy.optimize import minimize

def compute_similarity_weights(
    similarity_matrix: np.ndarray,
    layer_quality: np.ndarray,
    lmbd: float = 0.5,
    nonnegative: bool = True,
    constraint=1.0,
    nonnegative_ids=None
) -> np.ndarray:
    """
    Compute weights using scipy (handles non-convex objectives).
    """
    num_layers = len(layer_quality)
    
    # Objective: minimize -(quality - lambda * redundancy)
    def objective(w):
        return -(layer_quality @ w - lmbd * (w @ similarity_matrix @ w))
    
    # Gradient for faster optimization
    def gradient(w):
        return -(layer_quality - 2 * lmbd * (similarity_matrix @ w))
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - constraint}, # sum(w) = 1,
        {'type': 'ineq', 'fun': lambda w: - np.sum(w**2) + 1}  # sum(w**2) < 1
    ]
    if not(nonnegative_ids is None):
        print(nonnegative_ids)
        for i in range(nonnegative_ids.shape[0]):
            if nonnegative_ids[i]:
                constraints.append({'type': 'ineq', 'fun': lambda w: w[i]})
    print(constraints)
    bounds = [(0, 1)] * num_layers if nonnegative else [(-1, 1)] * num_layers
    
    # Initial guess: uniform or quality-weighted
    w0 = layer_quality / layer_quality.sum()
    
    # Solve using SLSQP (handles nonlinear constraints)
    result = minimize(
        objective,
        x0=w0,
        method='SLSQP',
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")
    print(result)
    weights = result.x
    weights = np.maximum(weights, 0) if nonnegative else weights

    if constraint > 0.0:
        weights = weights / weights.sum() * constraint
    
    return weights


# def compute_similarity_weights(
#     similarity_matrix: np.ndarray,
#     layer_quality: np.ndarray,
#     lmbd: float = 0.5,
#     nonnegative: bool = True
# ) -> np.ndarray:
#     """
#     Compute layer weights using quadratic programming (standard QP).

#     Solves: maximize w^T quality - lambda * w^T S w
#             subject to: sum(w) = 1, w >= 0

#     Args:
#         similarity_matrix: (L, L) layer similarity matrix (higher = more similar)
#         layer_quality: (L,) quality score per layer
#         lmbd: Regularization strength (higher = more diversity)
#         nonnegative: Whether to enforce non-negative weights

#     Returns:
#         weights: (L,) normalized weights
#     """
#     num_layers = len(layer_quality)

#     # Define optimization variable
#     w = cp.Variable(num_layers)
#     #dissimilarity_matrix = 1.0 - similarity_matrix
#     # Objective: maximize accuracy - lambda * redundancy
#     objective = cp.Maximize(
#         layer_quality @ w - lmbd * cp.quad_form(w, similarity_matrix)
#     )

#     # Constraints
#     constraints = [cp.sum(w) == 1]
#     if nonnegative:
#         constraints.append(w >= 0)

#     # Solve
#     problem = cp.Problem(objective, constraints)
#     problem.solve()

#     if problem.status not in ["optimal", "optimal_inaccurate"]:
#         raise RuntimeError(f"Optimization failed with status {problem.status}")

#     weights = np.asarray(w.value, dtype=np.float32)

#     # Ensure sum to 1 (numerical stability)
#     weights = weights / weights.sum()

#     return weights


def compute_similarity_weights_greedy(
    similarity_matrix: np.ndarray,
    layer_quality: np.ndarray,
    lmbd: float = 0.5,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Compute layer weights using modified QP with diagonal penalty (qadrsolvgreedy from notebooks).

    This version modifies the similarity matrix diagonal before optimization:
    A = similarity_matrix.copy()
    A.diagonal = alpha

    Then minimizes: -q^T w + lambda * w^T A w

    Args:
        similarity_matrix: (L, L) layer similarity matrix
        layer_quality: (L,) quality scores
        lmbd: Regularization strength
        alpha: Diagonal penalty value (default 1.0 from notebooks)

    Returns:
        weights: (L,) normalized weights
    """
    n = len(layer_quality)
    #dissimilarity_matrix = 1.0 - similarity_matrix
    # Create mask for valid entries
    mask = (layer_quality[None, :] > layer_quality[:, None]).astype(float)
    A    = similarity_matrix * mask
    np.fill_diagonal(A, alpha)

    # Objective function for scipy.optimize
    def obj(w, q, A, lam):
        pen = w @ A @ w
        val = -q @ w + lam * pen  # Note: minimize, so negative quality
        grad = -q + lam * (A + A.T) @ w
        return val, grad

    # Initial guess
    w0 = np.full(n, 1/n)

    # Constraints
    cons = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},  # sum(w) = 1
        {'type': 'ineq', 'fun': lambda w: w}  # w >= 0
    ]

    # Optimize
    res = minimize(
        lambda w: obj(w, layer_quality, A, lmbd),
        w0,
        args=(),
        jac=True,
        method='SLSQP',
        constraints=cons
    )

    weights = np.asarray(res.x, dtype=np.float32)
    weights = weights / weights.sum()

    return weights


def compute_greedy_weights_rank(
    similarity_matrix: np.ndarray,
    layer_quality: np.ndarray,
    lmbd: float = 0.5,
    max_layers: Optional[int] = 4
) -> np.ndarray:
    """
    Greedy layer selection with equal weights (v1 from notebooks).

    At each step, add the layer that maximizes:
        marginal_accuracy - lambda * marginal_redundancy

    Final weights: rank-based (1/k normalization)

    Args:
        similarity_matrix: (L, L) similarity matrix
        layer_quality: (L,) quality scores
        lmbd: Regularization strength
        max_layers: Maximum layers to select (None = all)

    Returns:
        weights: (L,) weights with selected layers having rank-based weights
    """
    num_layers = len(layer_quality)
    if max_layers is None:
        max_layers = num_layers

    selected = []
    remaining = list(range(num_layers))

    while len(selected) < max_layers and remaining:
        best_gain = -np.inf
        best_layer = None

        for layer in remaining:
            # Compute marginal gain
            acc_gain = layer_quality[layer]

            if selected:
                # Redundancy with already selected layers
                redundancy = np.mean([similarity_matrix[layer, s] for s in selected])
            else:
                redundancy = 0.0

            gain = acc_gain - lmbd * redundancy

            if gain > best_gain:
                best_gain = gain
                best_layer = layer

        if best_layer is None or best_gain <= 0:
            break

        selected.append(best_layer)
        remaining.remove(best_layer)

    # Assign rank-based weights: 1/(k+1) for k-th selected
    weights = np.zeros(num_layers, dtype=np.float32)
    if selected:
        rank_weights = np.array([1.0 / (i+1) for i in range(len(selected))], dtype=np.float32)
        rank_weights = rank_weights / rank_weights.sum()
        for i, layer_idx in enumerate(selected):
            weights[layer_idx] = rank_weights[i]

    return weights


def compute_greedy_weights_value(
    similarity_matrix: np.ndarray,
    layer_quality: np.ndarray,
    lmbd: float = 0.5,
    max_layers: Optional[int] = 4
) -> np.ndarray:
    """
    Greedy selection with quality-based weighting (v2 from notebooks).

    Similar to _rank but assigns weights proportional
    to layer quality instead of equal weights.

    Args:
        similarity_matrix: (L, L) similarity matrix
        layer_quality: (L,) quality scores
        lmbd: Regularization strength
        max_layers: Maximum layers to select

    Returns:
        weights: (L,) normalized quality-based weights
    """
    num_layers = len(layer_quality)
    if max_layers is None:
        max_layers = num_layers

    selected = []
    remaining = list(range(num_layers))

    while len(selected) < max_layers and remaining:
        best_gain = -np.inf
        best_layer = None

        for layer in remaining:
            acc_gain = layer_quality[layer]

            if selected:
                redundancy = np.mean([similarity_matrix[layer, s] for s in selected])
            else:
                redundancy = 0.0

            gain = acc_gain - lmbd * redundancy

            if gain > best_gain:
                best_gain = gain
                best_layer = layer

        if best_layer is None or best_gain <= 0:
            break

        selected.append(best_layer)
        remaining.remove(best_layer)

    # Assign weights proportional to quality
    weights = np.zeros(num_layers, dtype=np.float32)
    if selected:
        selected_quality = np.array([layer_quality[i] for i in selected])
        selected_quality = selected_quality / selected_quality.sum()
        for i, layer_idx in enumerate(selected):
            weights[layer_idx] = selected_quality[i]

    return weights


def compute_greedy_weights_delta(
    similarity_matrix: np.ndarray,
    layer_quality: np.ndarray,
    lmbd: float = 0.5,
    drop_delta: float = 0.15,
    max_layers: Optional[int] = None,
    min_gain: float = 0.0,
) -> np.ndarray:
    """
    Greedy selection with drop-delta preprocessing (v3 from notebooks).

    Process:
    1. Pre-filter: keep only layers with quality >= (1 - drop_delta) * max_quality
    2. Run greedy selection on filtered layers
    3. Assign rank-based weights (1/k normalization)

    Args:
        similarity_matrix: (L, L) similarity matrix
        layer_quality: (L,) quality scores
        lmbd: Regularization strength
        drop_delta: Drop layers below (1-delta)*max_quality (0.15 = keep top 85%)
        max_layers: Maximum layers to select
        min_gain: Minimum gain threshold to continue

    Returns:
        weights: (L,) normalized weights
    """
    q_raw = np.asarray(layer_quality, dtype=np.float32)
    S = np.asarray(similarity_matrix, dtype=np.float32)

    # Step 1: Drop-delta filtering
    q_max = q_raw.max()
    keep = np.where(q_raw >= (1.0 - drop_delta) * q_max)[0]

    if keep.size == 0:
        raise ValueError(f"No layers passed drop_delta={drop_delta} threshold")

    # Filter quality and similarity
    q_filt = q_raw[keep]
    S_filt = S[np.ix_(keep, keep)]

    # Step 2: Greedy selection on filtered layers
    remaining = set(range(len(keep)))  # Indices in filtered space
    selected_local = []

    while remaining and (max_layers is None or len(selected_local) < max_layers):
        best_gain, best_idx = -np.inf, None

        for i in remaining:
            penalty = np.mean([S_filt[i, j] for j in selected_local]) if selected_local else 0.0
            gain = q_filt[i] - lmbd * penalty

            if gain > best_gain:
                best_gain, best_idx = gain, i

        if best_idx is None or best_gain <= min_gain:
            break

        selected_local.append(best_idx)
        remaining.remove(best_idx)

    # Step 3: Rank-based weights (1/k normalization)
    rank_w = np.array([1 / (k+1) for k in range(len(selected_local))], dtype=np.float32)
    rank_w = rank_w / rank_w.sum()

    # Map back to global layer indices
    selected_global = [int(keep[i]) for i in selected_local]

    weights = np.zeros(len(layer_quality), dtype=np.float32)
    for i, layer_idx in enumerate(selected_global):
        weights[layer_idx] = rank_w[i]

    return weights


def compute_layer_clusters(
    similarity_matrix: np.ndarray,
    num_clusters: int = 4,
    method: str = "average"
) -> List[List[int]]:
    """
    Perform hierarchical clustering on layers based on similarity.

    Args:
        similarity_matrix: (L, L) similarity matrix (higher = more similar)
        num_clusters: Number of clusters to form
        method: Linkage method for hierarchical clustering
                ('average', 'single', 'complete', 'ward')

    Returns:
        clusters: List of layer index lists, e.g., [[0,1,2], [3,4], ...]
    """
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0.0)  # Ensure diagonal is 0

    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage = sch.linkage(condensed, method=method)

    # Cut tree to get cluster assignments
    cluster_labels = sch.fcluster(linkage, num_clusters, criterion='maxclust')

    # Group layers by cluster
    clusters = []
    for cluster_id in range(1, num_clusters + 1):
        cluster_layers = np.where(cluster_labels == cluster_id)[0].tolist()
        if cluster_layers:  # Only add non-empty clusters
            clusters.append(cluster_layers)

    return clusters


def compute_cluster_weights_for_pca(
    similarity_matrix: np.ndarray,
    layer_quality: np.ndarray,
    num_clusters: int = 4
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Compute clusters and weights for PCA-based aggregation.

    Args:
        similarity_matrix: (L, L) similarity matrix
        layer_quality: (L,) quality scores for each layer
        num_clusters: Number of clusters

    Returns:
        clusters: List of layer index lists
        cluster_weights: (num_clusters,) normalized weights for each cluster
    """
    # Get clusters
    clusters = compute_layer_clusters(similarity_matrix, num_clusters)

    # Compute weight for each cluster (average quality)
    cluster_weights = np.zeros(len(clusters), dtype=np.float32)
    for i, cluster_layers in enumerate(clusters):
        cluster_weights[i] = np.mean([layer_quality[j] for j in cluster_layers])

    # Normalize weights
    cluster_weights = cluster_weights / cluster_weights.sum()

    return clusters, cluster_weights


def normalize_weights(
    weights: np.ndarray,
    threshold: float = 0.001
) -> np.ndarray:
    """
    Filter weights by threshold and re-normalize.

    This is critical for matching notebook behavior:
    1. Apply threshold: w[w < threshold] = 0
    2. Re-normalize: w = w / sum(w)

    Args:
        weights: (L,) unnormalized weights
        threshold: Minimum weight threshold

    Returns:
        weights: (L,) filtered and normalized weights
    """
    weights = np.asarray(weights, dtype=np.float32)

    # Apply threshold
    weights = np.where(weights > threshold, weights, 0.0)

    # Re-normalize
    weight_sum = weights.sum()
    if weight_sum > 1e-9:
        weights = weights / weight_sum
    else:
        # Fallback: uniform weights
        weights = np.ones_like(weights) / len(weights)

    return weights


def compute_cluster_weights(
    similarity_matrix: np.ndarray,
    layer_quality: np.ndarray,
    num_clusters: int = 4
) -> np.ndarray:
    """
    Compute layer weights based on hierarchical clustering.
    Each layer gets the average quality of its cluster.

    Args:
        similarity_matrix: (L, L) similarity matrix
        layer_quality: (L,) quality scores
        num_clusters: Number of clusters

    Returns:
        weights: (L,) normalized weights
    """
    clusters = compute_layer_clusters(similarity_matrix, num_clusters)
    num_layers = len(layer_quality)
    weights = np.zeros(num_layers, dtype=np.float32)

    for cluster in clusters:
        cluster_quality = np.mean([layer_quality[i] for i in cluster])
        for layer_idx in cluster:
            weights[layer_idx] = cluster_quality

    # Normalize
    weights = weights / weights.sum()

    return weights
