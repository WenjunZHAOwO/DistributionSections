import numpy as np
from scipy.spatial.distance import cdist, pdist
from mgw import evaluate_coupling
import torch, ot, time
import sys
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
import warnings

# sys.path.append('../../paste3/src')
# sys.path.append('../../paste3/src/paste3')
from paste_helper import dissimilarity_metric, to_dense_array, get_common_genes

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

def fused_upper_bounds(A, B, alpha=0.1, K=200,
                       exp_metric='kl',        # 'kl' | 'euclidean' | 'gkl' | 'pca' | 'glmpca'
                       spatial_norm='median',  # 'min' | 'median' | 'mean' | 'max' | 'none'
                       feat_norm='none',       # 'median' | 'mean' | 'max' | 'none'
                       solver='pot', epsilon=1e-2,
                       feat_scale=None, spatial_scale=None,
                       couplings=('tlb', 'spatial_stratified', 'expression_stratified'),
                       n_strata=4, max_spots=10000, seed=0,
                       return_coupling=False, return_all=False):
    """Feasible-coupling upper bounds on true_fgw. Mirrors fused_tlb's
    subsampling and normalization, so with matching args the 'tlb' entry
    sandwiches the fused_tlb value from above.

    Returns bounds dict, or (val, feat, strc, G, bounds) if return_all,
    where the triple is the 'tlb' coupling's and G is None unless
    return_coupling."""
    (A, B), _ = get_common_genes([A, B])
    rng = np.random.default_rng(seed)
    if A.n_obs > max_spots:
        A = A[rng.choice(A.n_obs, max_spots, replace=False)].copy()
    if B.n_obs > max_spots:
        B = B[rng.choice(B.n_obs, max_spots, replace=False)].copy()

    XA_t = to_dense_array(A.X).double()
    XB_t = to_dense_array(B.X).double()
    M = dissimilarity_metric(exp_metric, A, B, XA_t, XB_t).cpu().numpy()
    XA = XA_t.cpu().numpy(); XB = XB_t.cpu().numpy()

    cA = np.asarray(A.obsm['spatial'], float)
    cB = np.asarray(B.obsm['spatial'], float)
    DA = cdist(cA, cA); DB = cdist(cB, cB)

    M  = M  / (feat_scale    if feat_scale    is not None else _scale(M,  feat_norm))
    DA = DA / (spatial_scale if spatial_scale is not None else _scale(DA, spatial_norm))
    DB = DB / (spatial_scale if spatial_scale is not None else _scale(DB, spatial_norm))

    lv = (np.arange(K) + 0.5) / K
    QA = np.quantile(DA, lv, axis=1).T
    QB = np.quantile(DB, lv, axis=1).T
    C_struct = cdist(QA, QB, 'sqeuclidean') / K

    p = np.ones(len(cA)) / len(cA)
    q = np.ones(len(cB)) / len(cB)
    DA2 = DA ** 2; DB2 = DB ** 2

    def solve_ot(C):
        C = np.asarray(C, np.float64)
        if solver == 'pot':
            return ot.emd(p, q, C, numItermax=1_000_000)
        elif solver == 'ott':
            out = Sinkhorn()(LinearProblem(
                      Geometry(cost_matrix=jnp.asarray(C), epsilon=epsilon),
                      a=jnp.asarray(p), b=jnp.asarray(q)))
            G = np.asarray(out.matrix, dtype=np.float64)
            # Sinkhorn plans are only approximately in U(p, q); project back so
            # the evaluated objective is a genuine upper bound.
            G = G * np.minimum(p / np.maximum(G.sum(1), 1e-300), 1.0)[:, None]
            G = G * np.minimum(q / np.maximum(G.sum(0), 1e-300), 1.0)[None, :]
            er, ec = p - G.sum(1), q - G.sum(0)
            s = np.abs(er).sum()
            return G + np.outer(er, ec) / s if s > 0 else G
        else:
            raise ValueError(f"unknown solver: {solver}")

    def evaluate_fgw(G):
        feat = float((M * G).sum())
        strc = float(p @ DA2 @ p) + float(q @ DB2 @ q) \
             - 2.0 * float(((DA @ G @ DB.T) * G).sum())
        strc = max(strc, 0.0)
        return float((1 - alpha) * feat + alpha * strc), feat, strc

    def strata(v):
        v = np.asarray(v, float)
        r = (rankdata(v, 'average') - 1.0) / max(len(v) - 1, 1)
        return np.minimum(np.floor(n_strata * r).astype(int), n_strata - 1)

    def strat_cost(la, lb):
        return ((la[:, None] - lb[None, :]) / (n_strata - 1)) ** 2

    def gene_scores():
        ref = 0.5 * (XA.mean(0) + XB.mean(0))          # shared mean profile
        rn = max(np.linalg.norm(ref), 1e-15)
        gA = (XA @ ref) / (np.maximum(np.linalg.norm(XA, axis=1), 1e-15) * rn)
        gB = (XB @ ref) / (np.maximum(np.linalg.norm(XB, axis=1), 1e-15) * rn)
        return gA, gB

    builders = {
        'tlb': lambda: (1 - alpha) * M + alpha * C_struct,
        'expression_stratified': lambda: C_struct
            + max(float(np.quantile(C_struct, 0.95)), 1e-12)
            * strat_cost(*(strata(g) for g in gene_scores())),
        'spatial_stratified': lambda: M
            + max(float(np.quantile(M, 0.95)), 1e-12)
            * strat_cost(strata(DA.mean(1)), strata(DB.mean(1))),
    }

    unknown = set(couplings) - set(builders)
    if unknown:
        raise ValueError(f"unknown couplings: {sorted(unknown)}")

    bounds = {}
    for name in couplings:
        C = builders[name]()
        G = solve_ot(C)
        del C
        val, feat, strc = evaluate_fgw(G)
        bounds[name] = {'val': val, 'feat': feat, 'strc': strc}
        if return_coupling:
            bounds[name]['G'] = G

    if return_all:
        b = bounds.get('tlb')
        if b is None:
            raise ValueError("return_all=True requires 'tlb' in couplings")
        return b['val'], b['feat'], b['strc'], b.get('G'), bounds
    return bounds




def _scale(D, how):
    """Return the scalar to divide a distance matrix by."""
    d = D[D > 0]
    if how in (None, 'none'): return 1.0
    if how == 'min':    return d.min()
    if how == 'median': return np.median(d)
    if how == 'mean':   return d.mean()
    if how == 'max':    return d.max()
    raise ValueError(f"unknown norm: {how}")


def fused_tlb(A, B, alpha=0.1, K=200,
              exp_metric='kl',            # 'kl' | 'euclidean' | 'gkl' | 'pca' | 'glmpca'
              spatial_norm='median',      # 'min' | 'median' | 'mean' | 'max' | 'none'
              feat_norm='none',           # 'median' | 'mean' | 'max' | 'none'
              solver = 'pot',
              epsilon=1e-2,
              feat_scale=None, spatial_scale=None,
              max_spots=10000, seed=0, return_coupling=False):
    """Fused Third Lower Bound on true_fgw. Subsamples large sections to max_spots."""
    (A, B), _ = get_common_genes([A, B])
    rng = np.random.default_rng(seed)
    if A.n_obs > max_spots:
        A = A[rng.choice(A.n_obs, max_spots, replace=False)].copy()
    if B.n_obs > max_spots:
        B = B[rng.choice(B.n_obs, max_spots, replace=False)].copy()

    M = dissimilarity_metric(exp_metric, A, B,
            to_dense_array(A.X).double(),
            to_dense_array(B.X).double()).cpu().numpy()

    cA = np.asarray(A.obsm['spatial'], float)
    cB = np.asarray(B.obsm['spatial'], float)
    DA = cdist(cA, cA); DB = cdist(cB, cB)

    M = M / (feat_scale if feat_scale is not None else _scale(M, feat_norm))
    DA = DA / (spatial_scale if spatial_scale is not None else _scale(DA, spatial_norm))
    DB = DB / (spatial_scale if spatial_scale is not None else _scale(DB, spatial_norm))

    lv = (np.arange(K) + 0.5) / K
    QA = np.quantile(DA, lv, axis=1).T
    QB = np.quantile(DB, lv, axis=1).T
    C_struct = cdist(QA, QB, 'sqeuclidean') / K

    C = (1 - alpha) * M + alpha * C_struct
    p = np.ones(len(cA)) / len(cA)
    q = np.ones(len(cB)) / len(cB)
    if solver == 'pot':
        G = ot.emd(p, q, C, numItermax=1_000_000)
    elif solver == 'ott':
        # warnings.warn("solver='ott' uses entropic Sinkhorn; the TLB lower-bound "
        #               "guarantee does not hold for suboptimal couplings")
        out = Sinkhorn()(LinearProblem(
                  Geometry(cost_matrix=jnp.asarray(C), epsilon=epsilon),
                  a=jnp.asarray(p), b=jnp.asarray(q)))
        G = np.asarray(out.matrix, dtype=np.float64)
    else:
        raise ValueError(f"unknown solver: {solver}")

    val  = float((C * G).sum())
    feat = float((M * G).sum())
    strc = float((C_struct * G).sum())
    if return_coupling:
        return val, feat, strc, G
    return val, feat, strc




from ott.geometry.geometry import Geometry
import jax.numpy as jnp

def full_fgw(A, B, alpha=0.1,
             exp_metric='kl',          # same options as fused_tlb
             spatial_norm='median',    # 'min' | 'median' | 'mean' | 'max' | 'none'
             feat_norm='none',
             solver='pot',             # 'pot' (exact, cond. gradient) | 'ott' (entropic)
             epsilon=5e-3, max_iter=1000,
             feat_scale=None, spatial_scale=None,
             max_spots=None, seed=0, return_coupling=False):
    """Full fused Gromov-Wasserstein. Same cost construction as fused_tlb."""
    (A, B), _ = get_common_genes([A, B])
    rng = np.random.default_rng(seed)
    if max_spots is not None:
        if A.n_obs > max_spots: A = A[rng.choice(A.n_obs, max_spots, replace=False)].copy()
        if B.n_obs > max_spots: B = B[rng.choice(B.n_obs, max_spots, replace=False)].copy()

    M = dissimilarity_metric(exp_metric, A, B,
            to_dense_array(A.X).double(),
            to_dense_array(B.X).double()).cpu().numpy().astype(np.float64)


    cA = np.asarray(A.obsm['spatial'], float)
    cB = np.asarray(B.obsm['spatial'], float)
    DA = cdist(cA, cA); DB = cdist(cB, cB)
    M = M / (feat_scale if feat_scale is not None else _scale(M, feat_norm))
    DA = DA / (spatial_scale if spatial_scale is not None else _scale(DA, spatial_norm))
    DB = DB / (spatial_scale if spatial_scale is not None else _scale(DB, spatial_norm))


    p = np.ones(A.n_obs) / A.n_obs
    q = np.ones(B.n_obs) / B.n_obs

    if solver == 'pot':
        G = ot.gromov.fused_gromov_wasserstein(
                M, DA, DB, p, q, loss_fun='square_loss', alpha=alpha)
    elif solver == 'ott':
        prob = QuadraticProblem(
            Geometry(cost_matrix=jnp.asarray(DA)),
            Geometry(cost_matrix=jnp.asarray(DB)),
            geom_xy=Geometry(cost_matrix=jnp.asarray(M)),
            fused_penalty=(1 - alpha) / alpha,
            a=jnp.asarray(p), b=jnp.asarray(q))
        out = GromovWasserstein(linear_solver=Sinkhorn(), epsilon=epsilon)(prob)
        if not bool(out.converged):
            warnings.warn(f"OTT GW did not converge at epsilon={epsilon}; "
                          f"try a larger epsilon or relative_epsilon='mean'")
        G = np.asarray(out.matrix, dtype=np.float64)
    else:
        raise ValueError(f"unknown solver: {solver}")

    # terms computed identically for both solvers, so they're comparable
    feat = float((M * G).sum())
    constC, hC1, hC2 = ot.gromov.init_matrix(DA, DB, p, q, loss_fun='square_loss')
    strc = float(ot.gromov.gwloss(constC, hC1, hC2, G))
    val  = (1 - alpha) * feat + alpha * strc

    if return_coupling:
        return val, feat, strc, G
    return val, feat, strc

def coords(A):
    x = np.asarray(A.obsm['spatial'], float); x = x - x.mean(0)
    return x / np.sqrt((x**2).sum(1).mean())

def eval_pair(A_sec, B_sec, key=None, max_spots=2000, seed=0, annotation='Annotation',**fgw_kw):
    rng = np.random.default_rng(seed)
    A = A_sec[rng.choice(A_sec.n_obs, min(A_sec.n_obs, max_spots), replace=False)].copy()
    B = B_sec[rng.choice(B_sec.n_obs, min(B_sec.n_obs, max_spots), replace=False)].copy()
    _, _, _, P = full_fgw(A, B, return_coupling=True, **fgw_kw)
    r = evaluate_coupling(coords(A), coords(B), P,
                          A.obs[annotation].astype(str).values,
                          B.obs[annotation].astype(str).values, None)
    return r['migration'], r['projAMI_mean'], r['adjusted_identity']