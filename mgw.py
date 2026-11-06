import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy import sparse as sp
import scanpy as sc


def same_label_mass_score(P, labels_A, labels_B, eps=1e-12):
    """
    Measure how much coupling mass connects spots with identical labels.

    Parameters
    ----------
    P : array-like or sparse matrix, shape (n_A, n_B)
        Coupling from section A to section B.

    labels_A : array-like, shape (n_A,)
        Cell-type labels for section A.

    labels_B : array-like, shape (n_B,)
        Cell-type labels for section B.

    eps : float
        Numerical tolerance.

    Returns
    -------
    dict containing
        identity_mass
            Fraction of total coupling mass joining identical labels.

        expected_identity
            Expected identity mass under an independent coupling with the
            same source and target label marginals.

        adjusted_identity
            Chance-corrected identity mass:
                (observed - expected) / (1 - expected).

        enrichment
            observed / expected.

        source_shared_coverage
            Fraction of source marginal mass belonging to labels present
            in both sections.

        target_shared_coverage
            Fraction of target marginal mass belonging to labels present
            in both sections.

        per_type_retention
            For each shared source type c, the fraction of its outgoing
            coupling mass transported to the same type c.

        label_mass_matrix
            Coupling aggregated to the cell-type level.
    """
    labels_A = np.asarray(labels_A).astype(str)
    labels_B = np.asarray(labels_B).astype(str)

    if P.shape != (len(labels_A), len(labels_B)):
        raise ValueError(
            f"P has shape {P.shape}, but expected "
            f"({len(labels_A)}, {len(labels_B)})."
        )

    total_mass = float(P.sum())
    if total_mass <= eps:
        raise ValueError("The coupling has zero total mass.")

    # Use one common label universe, so the cell-type matrix is square.
    all_labels = np.union1d(np.unique(labels_A), np.unique(labels_B))
    label_to_index = {lab: k for k, lab in enumerate(all_labels)}
    K = len(all_labels)

    codes_A = np.array([label_to_index[x] for x in labels_A], dtype=int)
    codes_B = np.array([label_to_index[x] for x in labels_B], dtype=int)

    # Aggregate P into F[c,d] = mass transported from type c to type d.
    if sp.issparse(P):
        Pcoo = P.tocoo()
        F = np.zeros((K, K), dtype=float)
        np.add.at(
            F,
            (codes_A[Pcoo.row], codes_B[Pcoo.col]),
            Pcoo.data,
        )
    else:
        P = np.asarray(P, dtype=float)
        F = np.zeros((K, K), dtype=float)

        for c in range(K):
            rows = codes_A == c
            if not np.any(rows):
                continue

            target_mass = P[rows, :].sum(axis=0)
            F[c, :] = np.bincount(
                codes_B,
                weights=target_mass,
                minlength=K,
            )

    # Convert to a probability distribution.
    F_prob = F / total_mass
    source_marginal = F_prob.sum(axis=1)
    target_marginal = F_prob.sum(axis=0)

    present_A = set(np.unique(labels_A))
    present_B = set(np.unique(labels_B))
    shared_labels = sorted(present_A.intersection(present_B))
    shared_idx = np.array(
        [label_to_index[x] for x in shared_labels],
        dtype=int,
    )

    # Observed same-label coupling mass.
    identity_mass = float(F_prob[shared_idx, shared_idx].sum())

    # Expected same-label mass under independence with fixed marginals.
    expected_identity = float(
        np.sum(
            source_marginal[shared_idx]
            * target_marginal[shared_idx]
        )
    )

    enrichment = (
        identity_mass / expected_identity
        if expected_identity > eps
        else np.nan
    )
    if not shared_labels:
        adjusted_identity = np.nan
    elif expected_identity < 1.0 - eps:
    # if expected_identity < 1.0 - eps:
        adjusted_identity = (
            (identity_mass - expected_identity)
            / (1.0 - expected_identity)
        )
    else:
        adjusted_identity = np.nan

    source_shared_coverage = float(
        source_marginal[shared_idx].sum()
    )
    target_shared_coverage = float(
        target_marginal[shared_idx].sum()
    )

    # Directional retention: fraction of each source type remaining itself.
    per_type_retention = {}
    for lab, k in zip(shared_labels, shared_idx):
        outgoing = source_marginal[k]
        per_type_retention[lab] = (
            float(F_prob[k, k] / outgoing)
            if outgoing > eps
            else np.nan
        )

    valid_retention = np.array(
        list(per_type_retention.values()),
        dtype=float,
    )
    macro_retention = (
        float(np.nanmean(valid_retention))
        if valid_retention.size
        else np.nan
    )

    return {
        "identity_mass": identity_mass,
        "expected_identity": expected_identity,
        "adjusted_identity": adjusted_identity,
        "enrichment": enrichment,
        "macro_retention": macro_retention,
        "source_shared_coverage": source_shared_coverage,
        "target_shared_coverage": target_shared_coverage,
        "per_type_retention": per_type_retention,
        "shared_labels": shared_labels,
        "all_labels": all_labels,
        "label_mass_matrix": F_prob,
    }

def scatter_colored(coords, values, s=10):
    plt.figure()
    plt.scatter(coords[:,0], coords[:,1], c=values, s=s)
    plt.axis('equal'); plt.title('Colored scatter'); plt.show()

def plot_alignment_lines_dense(xs, xt, P, k_per_point=3, alpha=0.05, lw=0.5):
    """
    Draw up to k_per_point connections per source x_i to target x_j
    with largest coupling weight. Covers all points but remains readable.
    """
    if not isinstance(P, np.ndarray):
        P = P.toarray() if hasattr(P, "toarray") else np.array(P)
    xs = np.asarray(xs)
    xt = np.asarray(xt)

    n1, n2 = P.shape
    offset = xs[:,0].max() - xt[:,0].min() + 1.25*(np.ptp(xs[:,0]) + np.ptp(xt[:,0]))

    plt.figure(figsize=(10,5))
    for i in range(n1):
        # find top-k_per_point targets
        j_top = np.argsort(-P[i,:])[:k_per_point]
        for j in j_top:
            w = P[i,j] / P[i,:].max()  # normalize per-row for transparency
            plt.plot([xs[i,0], xt[j,0]+offset],
                     [xs[i,1], xt[j,1]],
                     color="k", alpha=alpha*w, lw=lw)
    plt.scatter(xs[:,0], xs[:,1], s=8, c="C0", label="Source")
    plt.scatter(xt[:,0]+offset, xt[:,1], s=8, c="C1", label="Target")
    plt.axis("equal"); plt.axis("off")
    plt.title(f"GW Coupling: top {k_per_point} per point")
    plt.legend(frameon=False)
    plt.show()

def plot_alignment_lines(xs, xt, P, top=500, alpha=0.1):
    """
    Plot top-weight alignment lines between xs and xt, side-by-side.
    """
    if isinstance(P, np.ndarray):
        pass
    else:
        P = P.toarray() if hasattr(P, 'toarray') else np.array(P)

    n1, n2 = P.shape
    # pick top pairs
    flat = P.ravel()
    idx = np.argpartition(flat, -min(top, flat.size))[-min(top, flat.size):]
    idx = idx[np.argsort(-flat[idx])]

    xs = np.asarray(xs); xt = np.asarray(xt)
    offset = xs[:, 0].max() - xt[:, 0].min() + 1.25 * (np.ptp(xs[:, 0]) + np.ptp(xt[:, 0]))
    plt.figure(figsize=(10,5))
    for k in idx:
        i = k // n2; j = k % n2
        plt.plot([xs[i,0], xt[j,0]+offset], [xs[i,1], xt[j,1]], 'k-', alpha=alpha, lw=0.5)
    plt.scatter(xs[:,0], xs[:,1], s=10, c='C0')
    plt.scatter(xt[:,0]+offset, xt[:,1], s=10, c='C1')
    plt.axis('equal'); plt.axis('off'); plt.title('Top aligned pairs'); plt.show()

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def _edge_len_local(x_i, x_j, Gi, Gj):
    # symmetrized local arc length under pull-back metrics Gi, Gj
    dx_i = torch.as_tensor(x_j - x_i, dtype=Gi.dtype, device=Gi.device)
    dx_j = torch.as_tensor(x_i - x_j, dtype=Gj.dtype, device=Gj.device)
    li = torch.sqrt(torch.clamp(dx_i @ Gi @ dx_i, min=0.0))
    lj = torch.sqrt(torch.clamp(dx_j @ Gj @ dx_j, min=0.0))
    return 0.5 * (li + lj)

def geodesics_with_predecessors(coords_np, G_tensors, knn_csr):
    """
    coords_np: (n,2) numpy
    G_tensors: (n,2,2) torch (or numpy convertible) SPD tensors for each point
    knn_csr:   sparse kNN adjacency on coords (unweighted)
    Returns: (dist_matrix, predecessors)
    """
    if isinstance(G_tensors, np.ndarray):
        G_tensors = torch.from_numpy(G_tensors)
    device = G_tensors.device
    coords_t = torch.from_numpy(coords_np).to(device=device, dtype=G_tensors.dtype)

    rows, cols = knn_csr.nonzero()
    weights = []
    for i, j in zip(rows, cols):
        w = _edge_len_local(coords_np[i], coords_np[j], G_tensors[i], G_tensors[j])
        weights.append(float(w.detach().cpu().item()))
    W = csr_matrix((weights, (rows, cols)), shape=knn_csr.shape)
    D, P = shortest_path(W, directed=False, unweighted=False, return_predecessors=True)
    return D, P

def reconstruct_path(predecessors, i, j):
    """Return list of node indices along shortest path i -> j."""
    path = [j]
    k = j
    while k != i and k != -9999:
        k = predecessors[i, k]
        if k == -9999:
            return []  # no path
        path.append(k)
    return path[::-1]

def plot_geodesic_on_base(coordinates, predecessors, start, end, values=None, title='Geodesic path'):
    """
    coordinates: (n,2) numpy or torch
    predecessors: predecessor matrix from shortest_path
    values: optional scalar field to color the points (same length n)
    """
    coords = np.asarray(coordinates)
    #coordinates.detach().cpu().numpy() if hasattr(coordinates, 'device') else np.asarray(coordinates)
    path_idx = reconstruct_path(predecessors, start, end)

    plt.figure(figsize=(6,6))
    if values is not None:
        plt.scatter(coords[:,0], coords[:,1], c=np.asarray(values), s=6, cmap='viridis')
        plt.colorbar()
    else:
        plt.scatter(coords[:,0], coords[:,1], s=6, c='lightgray')

    if path_idx:
        P = coords[path_idx]
        plt.plot(P[:,0], P[:,1], '-o', ms=3, lw=1.5, c='k')
        plt.scatter([coords[start,0], coords[end,0]],
                    [coords[start,1], coords[end,1]],
                    c=['C3','C2'], s=40, marker='*')

    plt.title(title); plt.axis('equal'); plt.show()

import numpy as np, torch, matplotlib.pyplot as plt
from scipy.interpolate import griddata

def predict_on_model(model, arr_np):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    with torch.no_grad():
        return model(torch.from_numpy(arr_np).to(device, dtype)).cpu().numpy()

def field_to_grid(coords, values, nx=200, ny=160, method='linear'):
    x, y = coords[:,0], coords[:,1]
    gx = np.linspace(x.min(), x.max(), nx); gy = np.linspace(y.min(), y.max(), ny)
    GX, GY = np.meshgrid(gx, gy)
    GZ = griddata(np.c_[x,y], values, (GX, GY), method=method)
    if np.isnan(GZ).any():
        GZ = np.where(np.isnan(GZ), griddata(np.c_[x,y], values, (GX, GY), method='nearest'), GZ)
    return GX, GY, GZ

def plot_fit_on_cloud(coords, y_true, y_pred, title_true='raw', title_pred='pred'):
    lo = np.percentile(np.concatenate([y_true, y_pred]), 5)
    hi = np.percentile(np.concatenate([y_true, y_pred]), 95)

    GX, GY, GT = field_to_grid(coords, y_true)
    _,  _, GP = field_to_grid(coords, y_pred)

    fig, axes = plt.subplots(1,3, figsize=(13,4), constrained_layout=True)
    im0 = axes[0].imshow(GT, origin='lower',
                         extent=[GX.min(), GX.max(), GY.min(), GY.max()],
                         cmap='viridis', vmin=lo, vmax=hi, aspect='equal')
    axes[0].set_title(title_true); axes[0].axis('off')

    im1 = axes[1].imshow(GP, origin='lower',
                         extent=[GX.min(), GX.max(), GY.min(), GY.max()],
                         cmap='viridis', vmin=lo, vmax=hi, aspect='equal')
    axes[1].set_title(title_pred); axes[1].axis('off')

    im2 = axes[2].imshow(GP-GT, origin='lower',
                         extent=[GX.min(), GX.max(), GY.min(), GY.max()],
                         cmap='coolwarm', aspect='equal')
    axes[2].set_title('residual (pred - raw)'); axes[2].axis('off')

    fig.colorbar(im1, ax=axes[:2], fraction=0.03, pad=0.02)
    fig.colorbar(im2, ax=[axes[2]], fraction=0.03, pad=0.02)
    plt.show()

def fit_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1); y_pred = y_pred.reshape(-1)
    mse = np.mean((y_pred - y_true)**2)
    var = np.var(y_true) + 1e-12
    r2  = 1 - mse/var
    corr = np.corrcoef(y_true, y_pred)[0,1]
    return {'mse': mse, 'r2': r2, 'corr': corr}

def procrustes_from_coupling(X, Y, P, ensure_rotation=False):
    """
    Align Y to X using a weighted Procrustes fit with weights from coupling P.

    X: (n,d)  source coords
    Y: (m,d)  target coords
    P: (n,m)  coupling (sum(P) ~ 1 or any positive mass)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    P = np.asarray(P, dtype=float)

    m = P.sum()
    if m <= 0:
        raise ValueError("Coupling has zero mass.")

    # weights over points
    w_x = P.sum(axis=1)          # (n,)
    w_y = P.sum(axis=0)          # (m,)

    # weighted centroids
    x_bar = (w_x @ X) / m        # (d,)
    y_bar = (w_y @ Y) / m        # (d,)

    # center
    Xc = X - x_bar
    Yc = Y - y_bar

    # cross-covariance H = Yc^T P^T Xc  (d x m) @ (m x n) @ (n x d) -> (d x d)
    H = Yc.T @ P.T @ Xc

    # SVD -> rotation (map Y -> X)
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    R = U @ Vt                               # <-- key change
    
    if ensure_rotation and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1                      # standard Kabsch reflection guard
        R = U @ Vt

    # apply: align Y onto X’s frame
    Y_aligned = Yc @ R + x_bar
    # translation that maps Y -> Y_aligned is: y ↦ (y - y_bar) R + x_bar
    t = x_bar - y_bar @ R

    # X was centered then re-shifted, so return the original X for plotting
    return X, Y_aligned, R, t

def project_labels_via_P(P, labels_src, direction="A_to_B"):
    """
    Project discrete labels via GW coupling.
    direction="A_to_B": P shape (n_A, n_B), project A->B
    direction="B_to_A": P shape (n_A, n_B), project B->A
    Returns: pred_labels, confidence, posteriors (C x n_dest)
    """
    P = P.toarray() if sp.issparse(P) else np.asarray(P, float)
    labels_src = np.asarray(labels_src)

    classes, inv = np.unique(labels_src, return_inverse=True)     # C classes
    onehot = np.eye(len(classes))[inv]                            # (n_src, C)

    if direction == "A_to_B":
        # column-normalize (each B point gets a distribution over A)
        W = P / (P.sum(axis=0, keepdims=True) + 1e-12)           # (n_A, n_B)
        post = onehot.T @ W                                       # (C, n_B)
    else:
        # row-normalize (each A point gets a distribution over B)
        W = P / (P.sum(axis=1, keepdims=True) + 1e-12)           # (n_A, n_B)
        post = onehot.T @ W                                       # (C, n_B) but this
        # is actually labels on B; to get A<-B, swap roles:
        # Make onehot for B labels then multiply W^T. Provided here for symmetry:
        raise ValueError("For B_to_A, pass B labels as labels_src and use direction='A_to_B' with P.T")

    pred_idx = post.argmax(axis=0)
    conf = post.max(axis=0)
    return classes[pred_idx], conf, post

def plot_projected_labels(A_xy, B_xy, A_labels, B_labels_pred, B_conf=None, conf_thresh=None, 
                          title_A = "A: annotated", title_B = "B: projected labels", s=6):
    """Side-by-side scatter: A true labels vs B projected labels."""
    # color by categorical code
    cats = pd.Categorical(A_labels)
    colA = cats.codes
    # ensure same palette order on B
    catsB = pd.Categorical(B_labels_pred, categories=cats.categories)
    colB = catsB.codes

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].scatter(A_xy[:,0], A_xy[:,1], c=colA, s=s, cmap='tab20')
    ax[0].set_title(title_A); ax[0].axis('off'); ax[0].set_aspect('equal')

    alphaB = 1.0
    if B_conf is not None and conf_thresh is not None:
        alphaB = np.clip((B_conf - conf_thresh) / max(1e-8, 1 - conf_thresh), 0.15, 1.0)

    sc1 = ax[1].scatter(B_xy[:,0], B_xy[:,1], c=colB, s=s, cmap='tab20',
                        alpha=alphaB if np.isscalar(alphaB) else None)
    # if we used per-point alpha:
    if not np.isscalar(alphaB):
        for coll, a in zip(sc1.get_offsets(), alphaB): pass  # (mpl doesn't expose per-point alpha directly)
        # simple fallback: hide low-confidence as gray
        low = (B_conf < conf_thresh)
        ax[1].scatter(B_xy[low,0], B_xy[low,1], c='lightgray', s=s)

    ax[1].set_title(title_B); ax[1].axis('off'); ax[1].set_aspect('equal')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# --------------- #
# 1) Build weights
# --------------- #
def build_weighted_graph(coords, G, knn_csr):
    """
    coords: (n,2) float64/32
    G:      (n,2,2) pull-back metric (torch or np); we'll use numpy
    knn_csr: csr_matrix with 0/1 structure of kNN graph (undirected)
    returns: W (csr) with Riemannian arc-length weights on edges
    """
    if hasattr(G, "detach"):  # torch -> numpy
        G = G.detach().cpu().numpy()
    G = np.asarray(G)

    rows, cols = knn_csr.nonzero()
    # mid-point / symmetric metric on edge (i,j)
    M = 0.5*(G[rows] + G[cols])          # (nnz,2,2)
    v = coords[cols] - coords[rows]      # (nnz,2)
    # weight = sqrt( v^T M v )
    tmp = np.einsum('...i,...ij,...j->...', v, M, v)  # (nnz,)
    w = np.sqrt(np.maximum(tmp, 0.0))

    # Assemble weighted graph
    n = knn_csr.shape[0]
    W = csr_matrix((w, (rows, cols)), shape=(n, n))
    # Make sure it's symmetric (kNN is typically symmetrized, but be safe)
    W = 0.5*(W + W.T)
    return W

# -------------------------------- #
# 2) Single-source geodesic solver
# -------------------------------- #
def geodesics_from_source(W, src):
    """
    W:   weighted csr_matrix
    src: int (source index)
    returns: dist (n,), pred (n,) predecessor array
    """
    dist, pred = dijkstra(W, directed=False, indices=src, return_predecessors=True)
    return dist, pred

# ------------------------- #
# 3) Path reconstruct + plot
# ------------------------- #
def traceback_path(pred, i, j):
    """
    pred: predecessors array from dijkstra (shape (n,))
    returns: list of indices for the geodesic i -> j
    """
    path = [j]
    k = j
    while k != i and k != -9999:
        k = pred[k]
        path.append(k)
    if path[-1] != i:
        # no path (shouldn't happen on a connected kNN graph)
        return path[::-1], False
    return path[::-1], True

'''
def plot_geodesic(coords, path_idx, scatter=True, title=None, lw=2.5, alpha=0.9, s=6):
    """
    coords:   (n,2)
    path_idx: list of vertex indices
    """
    if scatter:
        plt.scatter(coords[:,0], coords[:,1], s=s, c='k', alpha=0.2)
    P = coords[path_idx]
    plt.plot(P[:,0], P[:,1], '-', lw=lw, alpha=alpha)
    plt.plot(P[0,0], P[0,1], marker='*', ms=12)         # start
    plt.plot(P[-1,0], P[-1,1], marker='*', ms=12)       # end
    plt.axis('equal'); plt.axis('off')
    if title: plt.title(title)
    plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

def scatter_big(ax, coords, color=None, max_points=50_000, s=2, alpha=0.15):
    """Fast, light background scatter."""
    n = coords.shape[0]
    if n > max_points:
        idx = np.random.RandomState(0).choice(n, size=max_points, replace=False)
        coords_ = coords[idx]
        c_ = None if color is None else np.asarray(color)[idx]
    else:
        coords_, c_ = coords, color

    sc = ax.scatter(
        coords_[:,0], coords_[:,1],
        c=c_, s=s, alpha=alpha,
        rasterized=True,  # PDF/SVG friendly
        linewidths=0,    # faster, cleaner
        edgecolors='none',
        marker='.',      # very small glyph
        antialiased=False
    )
    return sc

def draw_geodesic(ax, coords, path_idx, lw=3.0, alpha=0.95, zorder=5):
    """Crisp path with white halo so it reads over dense clouds."""
    P = coords[path_idx]
    lc = LineCollection([P], linewidths=lw, alpha=alpha, zorder=zorder, color='red')
    lc.set_path_effects([
        pe.Stroke(linewidth=lw+2.5, foreground="white", alpha=1.0),
        pe.Normal()
    ])
    ax.add_collection(lc)
    # start/end markers
    ax.plot(P[0,0],  P[0,1],  marker='*', ms=25, zorder=zorder+1, color='red')
    ax.plot(P[-1,0], P[-1,1], marker='*', ms=25, zorder=zorder+1, color='red')
    return lc

def plot_geodesic(coords, path_idx, color=None, title=None,
                         figsize=(5,5), max_points=50_000, s=10):
    fig, ax = plt.subplots(figsize=figsize)
    scatter_big(ax, coords, color=color, max_points=max_points, s=s, alpha=0.8)
    draw_geodesic(ax, coords, path_idx, lw=3.0, alpha=0.95)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    if title: ax.set_title(title)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def summarize_alignment(couplings, names, xs, xs2, A_labels, B_labels, A=None, B=None, plot_clus=True):
    """
    Plot migration (x-axis, lower better) vs mean projected AMI (y-axis, higher better)
    for each coupling in `couplings`.
    """
    mig_vals, ami_vals = [], []

    for i, P in enumerate(couplings):
        # Migration (expected displacement)
        mig = metrics.migration_metrics(xs, xs2, P)['expected_disp']
        mig_vals.append(float(mig))
        
        # Symmetric mean AMI
        ab = metrics.ami_on_projected_labels(P,   A_labels, B_labels)
        ba = metrics.ami_on_projected_labels(P.T, B_labels, A_labels)
        ami_vals.append(0.5 * (ab["AMI"] + ba["AMI"]))
        
        if A is not None and B is not None and plot_clus:
            B_pred, B_conf, _ = plotting.project_labels_via_P(P, A_labels, direction="A_to_B")
            plotting.plot_projected_labels(A.obsm['spatial'], B.obsm['spatial'],
                                  A_labels, B_pred, conf_thresh=0.5, title_A= ("Annotated "+names[i]))
            A_pred, A_conf, _ = plotting.project_labels_via_P(P.T, B_labels, direction="A_to_B")
            plotting.plot_projected_labels(B.obsm['spatial'], A.obsm['spatial'],
                                  B_labels, A_pred, conf_thresh=0.5, title_B = ("Projected "+names[i]))
    
    # Scatter
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(mig_vals, ami_vals, s=80, color='royalblue')

    # Label each point
    for i, name in enumerate(names):
        ax.text(mig_vals[i] * 1.01, ami_vals[i], name, fontsize=9, va='center')

    ax.set_xlabel("Expected migration distance (↓ better)")
    ax.set_ylabel("Mean projected AMI (↑ better)")
    ax.set_title("Alignment quality vs migration trade-off")

    # aesthetic touches
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {"names": names, "migration": mig_vals, "projAMI_mean": ami_vals}

import numpy as np
import matplotlib.pyplot as plt

def _labels_or_cluster(adata, key="annotation", n_pcs=30, res=1.0):
    """Return categorical labels from adata.obs[key] if present; otherwise Leiden clusters."""
    import scanpy as sc
    if key in adata.obs and adata.obs[key].notna().any():
        return adata.obs[key].astype(str).to_numpy()
    # quick unsupervised labels as fallback
    ad = adata.copy()
    if "X_pca" not in ad.obsm:
        sc.pp.normalize_total(ad); sc.pp.log1p(ad); sc.pp.pca(ad, n_comps=n_pcs)
    sc.pp.neighbors(ad); sc.tl.leiden(ad, resolution=res)
    return ad.obs["leiden"].astype(str).to_numpy()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from collections import OrderedDict
from typing import List, Dict, Optional
import math

def plot_alignment_summary_all(pairs_results,
    savepath_prefix=None, dpi=600):
    """
    pairs_results: list of dicts, each like:
      {
        "pair": "E11.5→E12.5",
        "methods": ["MGW","Spatial-GW","Feature-GW"],
        "metrics": [{"migration":...,"projAMI_mean":...}, ...]  # same order as methods
      }
    """
    all_methods = []
    for r in pairs_results:
        for m in r["methods"]:
            if m not in all_methods:
                all_methods.append(m)
    colors = plt.cm.tab10(np.linspace(0,1,len(all_methods)))
    method_style = {m: dict(color=colors[i], marker="o") for i,m in enumerate(all_methods)}

    fig, ax = plt.subplots(figsize=(7,5))
    for r in pairs_results:
        pair = r["pair"]
        for m, metr in zip(r["methods"], r["metrics"]):
            style = method_style[m]
            x, y = metr["migration"], metr["projAMI_mean"]
            ax.scatter(x, y, s=70, **style)
            ax.text(x*1.01, y, pair, fontsize=8, va="center")  # tiny offset label
            
    handles = [plt.Line2D([0],[0], linestyle="none", marker=style["marker"],
                          color=style["color"], label=m) 
               for m, style in method_style.items()]
    ax.legend(handles=handles, title="Method", loc="best", frameon=True)

    ax.set_xlabel("Expected migration distance (↓ better)")
    ax.set_ylabel("Mean projected AMI (A↔B) (↑ better)")
    ax.set_title("Alignment summary across all timepoint pairs")
    ax.grid(alpha=0.3)
    
    if savepath_prefix:
        fig.savefig(f"{savepath_prefix}.pdf")
        fig.savefig(f"{savepath_prefix}.svg")
        fig.savefig(f"{savepath_prefix}.png", dpi=dpi)
        
    plt.tight_layout(); plt.show()




def _to_numpy(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)

def per_gene_cosine_under_coupling(
    adata_source,   # e.g., Xenium (cells)
    adata_target,   # e.g., Visium (spots)
    P,              # shape (n_target, n_source)
    log1p=True,
    scale=True,
    eps=1e-12,
    return_pearson=False
):
    """
    Per-gene cosine similarity between adata_target.X and the barycentric
    projection of adata_source.X via coupling P.

    Returns:
        dict with:
          - 'genes': np.array of common gene names in order
          - 'cosine': np.array of per-gene cosine similarities
          - (optional) 'pearson': np.array of per-gene Pearson correlations
          - 'summary': dict with aggregates (mean/median)
    """
    a_src = adata_source.copy()
    a_tgt = adata_target.copy()
    a_src.var_names_make_unique()
    a_tgt.var_names_make_unique()

    if log1p:
        sc.pp.log1p(a_src); sc.pp.log1p(a_tgt)
    if scale:
        # scale per gene so cosine is not dominated by highly variable genes
        sc.pp.scale(a_src, zero_center=True, max_value=None)
        sc.pp.scale(a_tgt, zero_center=True, max_value=None)

    # Intersect genes and align order
    common = a_src.var_names.intersection(a_tgt.var_names)
    if len(common) == 0:
        raise ValueError("No shared genes between datasets.")
    common = common.sort_values()
    a_src = a_src[:, common].copy()
    a_tgt = a_tgt[:, common].copy()

    Xs = _to_numpy(a_src.X)  # (n_source, G)
    Xt = _to_numpy(a_tgt.X)  # (n_target, G)

    P = P.toarray() if sp.issparse(P) else np.asarray(P)
    n_t, n_s = Xt.shape[0], Xs.shape[0]
    if P.shape != (n_t, n_s):
        raise ValueError(f"P shape {P.shape} must be (n_target={n_t}, n_source={n_s}).")

    # Barycentric projection of source -> target
    row_sums = P.sum(axis=1, keepdims=True) + eps
    Xs_proj = (P @ Xs) / row_sums  # (n_target, G)

    # Per-gene cosine: cos(a,b) = (a·b) / (||a|| ||b||)
    num = (Xt * Xs_proj).sum(axis=0)
    Xt_norm = np.linalg.norm(Xt, axis=0) + eps
    Xp_norm = np.linalg.norm(Xs_proj, axis=0) + eps
    cos = num / (Xt_norm * Xp_norm)

    out = {
        "genes": np.asarray(common.values, dtype=str),
        "cosine": cos,
        "summary": {"mean_cosine": float(np.mean(cos)), "median_cosine": float(np.median(cos))}
    }

    if return_pearson:
        pear = np.array([pearsonr(Xt[:, g], Xs_proj[:, g])[0] for g in range(Xt.shape[1])])
        out["pearson"] = pear
        out["summary"].update(
            mean_pearson=float(np.nanmean(pear)),
            median_pearson=float(np.nanmedian(pear))
        )
    
    return out

def evaluate_coupling(xs, xs2, P, A_labels, B_labels, metrics):
    """Return dict with migration and symmetric mean projected AMI for one coupling."""
    mig = float(migration_metrics(xs, xs2, P)['expected_disp'])
    ab = ami_on_projected_labels(P,   A_labels, B_labels)
    ba = ami_on_projected_labels(P.T, B_labels, A_labels)
    ami_mean = 0.5 * (ab["AMI"] + ba["AMI"])
    identity = same_label_mass_score(P, A_labels, B_labels)
    # print('111')
    return {"migration": mig, "projAMI_mean": float(ami_mean),"adjusted_identity": identity["adjusted_identity"]}


# def evaluate_coupling(xs, xt, P, labels_A, labels_B):
#     identity = same_label_mass_score(P, labels_A, labels_B)

#     migration = float(
#         migration_metrics(xs, xt, P)["expected_disp"]
#     )

#     return {
#         "migration": migration,
#         "identity_mass": identity["identity_mass"],
#         "adjusted_identity": identity["adjusted_identity"],
#         "identity_enrichment": identity["enrichment"],
#         "macro_retention": identity["macro_retention"],
#         "source_shared_coverage":
#             identity["source_shared_coverage"],
#         "target_shared_coverage":
#             identity["target_shared_coverage"],
#     }

def _to_dense_col_norm(P_sub):
    """Return column-normalized dense matrix; safe for all-zero columns."""
    if sp.issparse(P_sub):
        P_sub = P_sub.tocsc(copy=True)
        col_sums = np.asarray(P_sub.sum(axis=0)).ravel()
        nz = col_sums > 0
        P_sub.data /= np.repeat(col_sums[nz], np.diff(P_sub.indptr)[nz])
        return P_sub.toarray(), col_sums
    else:
        col_sums = P_sub.sum(axis=0, keepdims=True)
        Pn = np.divide(P_sub, np.maximum(col_sums, 1e-12), where=True)
        return Pn, col_sums.ravel()

# def ami_on_projected_labels(P, labels_A, labels_B, conf_thresh=0.0):
#     """
#     Project A->B using P, restricting to the common label set, then compute AMI/ARI/Acc.
#     Returns dict with metrics and the projected labels/confidence for the evaluated B indices.
#     """
#     labels_A = np.asarray(labels_A)
#     labels_B = np.asarray(labels_B)
#     # common label universe
#     common = np.intersect1d(np.unique(labels_A), np.unique(labels_B))
#     maskA = np.isin(labels_A, common)
#     maskB = np.isin(labels_B, common)
#     if maskA.sum() == 0 or maskB.sum() == 0:
#         raise ValueError("No overlap in label sets between A and B.")

#     # restrict P to common labels
#     P_sub = P[maskA][:, maskB]
#     Pn, col_sums = _to_dense_col_norm(P_sub)  # shape (nA_common, nB_common)

#     # map common labels to row-indices in A-subset
#     common_to_rows = {lab: np.where(maskA & (labels_A == lab))[0] for lab in common}

#     # per B column: label probabilities and prediction
#     # Compute p(l | j) = sum_{i in A_l} Pn[i, j]
#     nB = Pn.shape[1]
#     probs = np.zeros((nB, common.size), dtype=float)
#     for ell, lab in enumerate(common):
#         probs[:, ell] = Pn[ np.ix_(np.isin(np.where(maskA)[0], common_to_rows[lab]), np.arange(nB)) ].sum(axis=0)

#     pred_idx = probs.argmax(axis=1)
#     pred_lab = common[pred_idx]
#     conf = probs.max(axis=1)

#     # filter by confidence threshold (optional)
#     keep = conf >= conf_thresh
#     true_lab = labels_B[maskB][keep]
#     pred_lab = pred_lab[keep]

#     # metrics
#     ami = adjusted_mutual_info_score(true_lab, pred_lab)
#     ari = adjusted_rand_score(true_lab, pred_lab)
#     acc = accuracy_score(true_lab, pred_lab)

#     return {
#         "AMI": ami,
#         "ARI": ari,
#         "ACC": acc,
#         "keep_mask_B_common": keep,
#         "B_true_eval": true_lab,
#         "B_pred_eval": pred_lab,
#         "B_conf_eval": conf[keep],
#         "common_labels": common
#     }

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             accuracy_score)


def ami_on_projected_labels(P, labels_A, labels_B, conf_thresh=0.0):
    """
    Score a coupling by partition agreement, without requiring the two label
    vocabularies to overlap.

    For each spot j in B, take the A-label carrying the most transported mass
    into j, then measure agreement between that assignment and B's own labels.
    AMI and ARI are invariant to cluster naming, so this stays meaningful across
    timepoints where cell identities genuinely change. ACC is only returned when
    the vocabularies overlap, since it compares label strings directly.
    """
    labels_A = np.asarray(labels_A).astype(str)
    labels_B = np.asarray(labels_B).astype(str)

    if P.shape != (len(labels_A), len(labels_B)):
        raise ValueError(
            f"P has shape {P.shape}, expected ({len(labels_A)}, {len(labels_B)})."
        )

    uniq_A, codes_A = np.unique(labels_A, return_inverse=True)
    nB = P.shape[1]

    # M[l, j] = coupling mass from A-label l into B spot j
    M = np.zeros((uniq_A.size, nB), dtype=float)
    if sp.issparse(P):
        Pcoo = P.tocoo()
        np.add.at(M, (codes_A[Pcoo.row], Pcoo.col), Pcoo.data)
    else:
        Pd = np.asarray(P, dtype=float)
        for l in range(uniq_A.size):
            M[l] = Pd[codes_A == l].sum(axis=0)

    col_sums = M.sum(axis=0)
    rec_idx  = np.flatnonzero(col_sums > 0)        # B spots receiving any mass

    top  = np.zeros(nB, dtype=int)
    conf = np.zeros(nB, dtype=float)               # share of incoming mass held
    top[rec_idx]  = M[:, rec_idx].argmax(axis=0)   # by the winning A-label
    conf[rec_idx] = M[top[rec_idx], rec_idx] / col_sums[rec_idx]

    keep = np.zeros(nB, dtype=bool)
    keep[rec_idx] = conf[rec_idx] >= conf_thresh
    if keep.sum() < 2:
        raise ValueError("Too few B spots retained to score.")

    pred = uniq_A[top[keep]]          # A vocabulary
    true = labels_B[keep]             # B vocabulary

    shared = np.intersect1d(uniq_A, np.unique(labels_B))
    acc = accuracy_score(true, pred) if shared.size else np.nan

    return {
        "AMI": adjusted_mutual_info_score(true, pred),
        "ARI": adjusted_rand_score(true, pred),
        "ACC": acc,                                   # nan when vocabularies disjoint
        "n_eval": int(keep.sum()),
        "coverage": float(keep.sum() / nB),           # fraction of B actually scored
        "n_unreceived": int(nB - rec_idx.size),       # B spots with zero incoming mass
        "mean_conf": float(conf[keep].mean()),
        "n_pred_labels": int(np.unique(pred).size),
        "n_true_labels": int(np.unique(true).size),
        "n_shared_labels": int(shared.size),
        "keep_mask_B": keep,
        "B_true_eval": true,
        "B_pred_eval": pred,
        "B_conf_eval": conf[keep],
    }
    
def expression_cosine_similarity(X, Z, P, eps=1e-12):
    from numpy.linalg import norm
    Xn = X / (norm(X, axis=1, keepdims=True) + eps)
    Zn = Z / (norm(Z, axis=1, keepdims=True) + eps)
    C = Xn @ Zn.T  # cosine matrix
    return (P * C).sum() / (P.sum() + eps)

def Alignment_Clus_Metrics(X, Z, P, k=4, random_state=0):
    
    # Cluster each modality independently
    clx = KMeans(k, random_state=random_state).fit_predict(X)
    clz = KMeans(k, random_state=random_state).fit_predict(Z)
    
    # Compute weighted AMI under coupling P (normalized joint)
    P_norm = P / P.sum()
    
    # Sample from P to estimate AMI under coupling
    n_samp = 20000
    idx_x = np.random.choice(P.shape[0], size=n_samp, p=P_norm.sum(1))
    idx_z = [np.random.choice(P.shape[1], p = P_norm[i,:] / P_norm[i,:].sum() ) for i in idx_x]
    
    ami = adjusted_mutual_info_score(clx[idx_x], clz[idx_z])
    print("AMI(X,Z) under coupling:", ami)
    ari = adjusted_rand_score(clx[idx_x], clz[idx_z])
    print("ARI(X,Z) under coupling:", ari)
    
    return ami, ari

def _safe_row_probs(row):
    s = row.sum()
    return row / s if s > 0 else np.zeros_like(row)

def _weighted_joint_from_P(P, clx, clz, kx=None, kz=None):
    """Return joint J[a,b] = sum_{i,j} P_ij 1[clx_i=a] 1[clz_j=b]."""
    P = np.asarray(P, dtype=np.float64)
    n1, n2 = P.shape
    if kx is None: kx = int(clx.max()) + 1
    if kz is None: kz = int(clz.max()) + 1
    J = np.zeros((kx, kz), dtype=np.float64)
    for a in range(kx):
        ia = (clx == a)
        if not ia.any(): continue
        Pa = P[ia]                      # (|Ia|, n2)
        # accumulate by columns grouped by clz label
        for b in range(kz):
            jb = (clz == b)
            if jb.any():
                J[a, b] = Pa[:, jb].sum()
    return J

def _entropy(p):
    p = p[p > 0]
    return -(p * np.log(p)).sum()

def _weighted_mi_nmi_from_P(P, clx, clz):
    """Compute MI, NMI from coupling-weighted joint contingency (deterministic)."""
    P = P / P.sum()
    kx = int(clx.max()) + 1
    kz = int(clz.max()) + 1
    J  = _weighted_joint_from_P(P, clx, clz, kx, kz)      # shape (kx, kz)
    if J.sum() <= 0:
        return dict(MI=0.0, NMI=0.0, Hx=0.0, Hz=0.0, J=J)
    J  = J / J.sum()
    px = J.sum(axis=1, keepdims=True)                    # (kx,1)
    pz = J.sum(axis=0, keepdims=True)                    # (1,kz)
    # Mutual information
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = J / (px @ pz)
        ratio[~np.isfinite(ratio)] = 1.0
        MI = (J * np.log(ratio)).sum()
    Hx = _entropy(px.ravel())
    Hz = _entropy(pz.ravel())
    NMI = 0.0 if (Hx + Hz) == 0 else (2 * MI) / (Hx + Hz)
    return dict(MI=MI, NMI=NMI, Hx=Hx, Hz=Hz, J=J)

def cluster_and_scores(X, Z, P, k=4, n_init=10, sample_pairs=20000, random_state=0):
    """
    KMeans each modality; compute:
      - deterministic coupling-weighted NMI/MI,
      - Monte-Carlo AMI/ARI by sampling pairs from P.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X); Z = np.asarray(Z); P = np.asarray(P, dtype=np.float64)
    # cluster
    clx = KMeans(k, n_init=n_init, random_state=random_state).fit_predict(X)
    clz = KMeans(k, n_init=n_init, random_state=random_state).fit_predict(Z)
    # deterministic (exact) weighted NMI/MI
    det = _weighted_mi_nmi_from_P(P, clx, clz)
    # Monte-Carlo AMI/ARI
    Pn = P / P.sum()
    row_m = Pn.sum(axis=1)
    row_m = row_m / row_m.sum()
    rows = rng.choice(Pn.shape[0], size=sample_pairs, p=row_m)
    cols = np.fromiter(
        (rng.choice(Pn.shape[1], p=_safe_row_probs(Pn[i])) for i in rows),
        dtype=np.int64, count=sample_pairs
    )
    ami = adjusted_mutual_info_score(clx[rows], clz[cols])
    ari = adjusted_rand_score(clx[rows], clz[cols])
    return {
        "clusters_X": clx,
        "clusters_Z": clz,
        "NMI_weighted": det["NMI"],
        "MI_weighted": det["MI"],
        "Hx": det["Hx"], "Hz": det["Hz"],
        "AMI_mc": ami, "ARI_mc": ari
    }

from scipy.spatial.distance import cdist

def migration_metrics(xs, xt, P, use_procrustes=True, p=2):
    """
    Compute displacement summaries under P:
      - E[||s - t||^p] and its sqrt (if p=2),
      - barycentric drift per-row, with summary stats.
    """
    xs = np.asarray(xs, dtype=np.float64)
    xt = np.asarray(xt, dtype=np.float64)
    P  = np.asarray(P,  dtype=np.float64)
    Pn = P / P.sum()

    if use_procrustes:
        s_al, t_al, R, tvec = procrustes_from_coupling(xs, xt, Pn)
        S, T = s_al, t_al
    else:
        S, T = xs, xt

    # expected displacement (all pairs)
    D = cdist(S, T)  # Euclidean
    
    disp_p = ( (D ** p) * Pn).sum()
    disp = disp_p ** (1.0 / p) if p != 1 else disp_p

    # barycentric drift per-row: || s_i - sum_j P_ij t_j / sum_j P_ij ||
    row_m = Pn.sum(axis=1, keepdims=True)  # (n,1)
    row_m[row_m == 0] = 1.0
    t_bar = (Pn @ T) / row_m               # (n,2)
    drift = np.linalg.norm(S - t_bar, axis=1)
    stats = {
        "mean": float(drift.mean()),
        "median": float(np.median(drift)),
        "p90": float(np.quantile(drift, 0.90)),
        "p95": float(np.quantile(drift, 0.95)),
        "max": float(drift.max())
    }

    return {
        "expected_disp_p": float(disp_p),
        "expected_disp": float(disp),
        "p": p,
        "barycentric_drift_stats": stats,
        "aligned_coords": dict(S=S, T=T)  # handy for plotting
    }

import numpy as np

def procrustes_from_coupling(X, Y, P, eps=1e-12):
    """
    Weighted Procrustes using P as soft correspondences.
    Returns Xc, Yc_aligned, R, t such that Y ≈ R X + t in least-squares(P).
    X: (n,d), Y: (m,d), P: (n,m) nonnegative
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    # normalize once; avoid divisions by ~0
    mass = P.sum()
    if mass <= eps:
        raise ValueError("Coupling P has (near) zero mass.")
    P = P / mass

    # 1D marginals
    wx = P.sum(axis=1)      # (n,)
    wy = P.sum(axis=0)      # (m,)

    # weighted centroids (shape (1,d)), robust wrt shapes
    X_mean = (wx[:, None] * X).sum(axis=0, keepdims=True) / (wx.sum() + eps)
    Y_mean = (wy[:, None] * Y).sum(axis=0, keepdims=True) / (wy.sum() + eps)

    # center
    Xc = X - X_mean         # (n,d)
    Yc = Y - Y_mean         # (m,d)

    # cross-covariance H = Yc^T P^T Xc  (d×d)
    H = Yc.T @ P.T @ Xc

    # SVD → rotation
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    R = Vt.T @ U.T
    # enforce proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # align Y to X: Y_aligned = (Y - Y_mean) @ R^T + t
    Yc_aligned = Yc @ R.T
    t = (Y_mean @ R.T) - X_mean  # translation to map X_mean to aligned Y_mean

    return Xc, Yc_aligned, R, t

