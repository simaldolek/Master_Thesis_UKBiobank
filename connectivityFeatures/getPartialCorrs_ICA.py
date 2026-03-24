import re
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np


def _extract_eid_7digits(filename: str) -> str:
    m = re.search(r"(\d{7})", filename)
    return m.group(1)


def load_cov_stack_from_folder(
    cov_dir: Path,
    pattern: str = "*.npy",
    require_shape: Tuple[int, int] = (100, 100),
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[Path]]:
    cov_dir = Path(cov_dir)
    npy_paths = sorted(cov_dir.glob(pattern))
    if max_files is not None:
        npy_paths = npy_paths[:max_files]

    covs, eids, kept_paths = [], [], []
    for p in npy_paths:
            eid = _extract_eid_7digits(p.name)
            Sigma = np.load(p)

        if Sigma.shape != require_shape:
            print(f"Skipped wrong shape: {p.name} shape={Sigma.shape} expected={require_shape}")
            continue

        covs.append(Sigma)
        eids.append(eid)
        kept_paths.append(p)

    if len(covs) < 2:
        raise RuntimeError(f"Need at least 2 valid covariance matrices; found {len(covs)} in {cov_dir}")

    return np.stack(covs, axis=0), eids, kept_paths


def _precision_from_cov_cholesky(Sigma: np.ndarray, jitter: float = 0.0) -> np.ndarray:
    """
    Precision Ω = Σ^{-1} using Cholesky. Mirrors Eq (2) in Pervaiz et al. 2020:
    factorize Σ = U^T U (upper triangular U), then Ω = U^{-1} (U^{-1})^T.
    """
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma must be square; got {Sigma.shape}")
    if jitter < 0:
        raise ValueError("jitter must be >= 0")

    d = Sigma.shape[0]
    S = np.array(Sigma, dtype=float, copy=True)
    if jitter > 0:
        S[np.diag_indices(d)] += jitter

    L = np.linalg.cholesky(S)  # Σ = L L^T
    U = L.T                    # Σ = U^T U

    I = np.eye(d, dtype=float)
    U_inv = np.linalg.solve(U, I)     # U^{-1}
    Omega = U_inv @ U_inv.T           # Ω = U^{-1}(U^{-1})^T
    return Omega


def precision_to_partial_corr(Omega: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    ρ_ij = - w_ij / sqrt(w_ii w_jj), with diag(ρ)=1
    (Eq (3) in Pervaiz).
    """
    d = Omega.shape[0]
    w = np.diag(Omega).copy()
    w = np.maximum(w, eps)  
    denom = np.sqrt(np.outer(w, w))
    rho = -Omega / denom
    np.fill_diagonal(rho, 1.0)
    rho = 0.5 * (rho + rho.T) # symmetry
    return rho


def pervaiz_rms_score(diffs: np.ndarray) -> float:
    """
    square elements, sum over subjects -> dxd,
    sum upper triangle, sqrt. (Pervaiz et al. 2020 Footnote 4)
    """
    S = np.sum(diffs**2, axis=0)               # d x d
    iu = np.triu_indices(S.shape[0], k=1)
    return float(np.sqrt(np.sum(S[iu])))


def choose_alpha_pervaiz_exact(
    cov_stack: np.ndarray,
    alpha_grid: np.ndarray,
    jitter: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Choose single global alpha by minimizing RMS distance between
    regularized subject precision matrices and the group-average of the
    unregularized subject precision matrices (Pervaiz Eq (4) + footnote 4).
    """
    n, d, _ = cov_stack.shape
    I = np.eye(d)

    # unregularized precisions for all subjects
    Omegas_b = np.array([_precision_from_cov_cholesky(cov_stack[i], jitter=jitter)
                         for i in range(n)])
    Omega_b_bar = Omegas_b.mean(axis=0)

    scores = []
    for a in alpha_grid:
        diffs = np.empty((n, d, d), dtype=float)
        for i in range(n):
            Omega_a = _precision_from_cov_cholesky(cov_stack[i] + a * I, jitter=jitter)
            diffs[i] = Omega_a - Omega_b_bar
        scores.append(pervaiz_rms_score(diffs))

    scores = np.asarray(scores)
    best_idx = int(np.argmin(scores))
    alpha_star = float(alpha_grid[best_idx])
    diag = {
        "alpha_grid": np.asarray(alpha_grid),
        "rms_scores": scores,
        "best_index": best_idx,
        "best_alpha": alpha_star,
    }
    return alpha_star, diag


def compute_and_save_partial_corrs_pervaiz(
    cov_dir: str | Path,
    out_dir_unreg: str | Path,
    out_dir_tikh: str | Path,
    pattern: str = "*.npy",
    alpha_grid: Optional[np.ndarray] = None,
    jitter: float = 1e-8,
    require_shape: Tuple[int, int] = (100, 100),
    max_files: Optional[int] = None,
    save_dtype=np.float32,
    save_precision_too: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
      - Unregularized partial corr ρ_b from Ω_b = Σ^{-1}
      - Tikhonov partial corr      ρ_a from Ω_a = (Σ + α*I)^{-1}
      α* selected globally .
    """
    cov_dir = Path(cov_dir)
    out_dir_unreg = Path(out_dir_unreg)
    out_dir_tikh = Path(out_dir_tikh)
    out_dir_unreg.mkdir(parents=True, exist_ok=True)
    out_dir_tikh.mkdir(parents=True, exist_ok=True)

    if alpha_grid is None:
        alpha_grid = np.logspace(-6, 2, 60)

    cov_stack, eids, _paths = load_cov_stack_from_folder(
        cov_dir=cov_dir,
        pattern=pattern,
        require_shape=require_shape,
        max_files=max_files,
    )
    n, d, _ = cov_stack.shape
    print(f"Loaded {n} covariance matrices of shape {d}x{d} from {cov_dir}")

    alpha_star, diag = choose_alpha_pervaiz_exact(
        cov_stack=cov_stack,
        alpha_grid=alpha_grid,
        jitter=jitter,
    )
    print(f"Selected global alpha* = {alpha_star:.6g}")

    I = np.eye(d, dtype=float)

    for i in range(n):
        eid = eids[i]
        Sigma = cov_stack[i]

        # precisions
        Omega_b = _precision_from_cov_cholesky(Sigma, jitter=jitter)
        Omega_a = _precision_from_cov_cholesky(Sigma + alpha_star * I, jitter=jitter)

        # partial correlations (FINAL)
        rho_b = precision_to_partial_corr(Omega_b)
        rho_a = precision_to_partial_corr(Omega_a)

        # save
        out_rhob = out_dir_unreg / f"{eid}_partialcorr.npy"
        out_rhoa = out_dir_tikh / f"{eid}_partialcorr_tikh_alpha{alpha_star:.6g}.npy"
        np.save(out_rhob, rho_b.astype(save_dtype, copy=False))
        np.save(out_rhoa, rho_a.astype(save_dtype, copy=False))

        if save_precision_too:
            np.save(out_dir_unreg / f"{eid}_precision.npy", Omega_b.astype(save_dtype, copy=False))
            np.save(out_dir_tikh / f"{eid}_precision_tikh_alpha{alpha_star:.6g}.npy",
                    Omega_a.astype(save_dtype, copy=False))

        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"Saved {i+1}/{n}: {eid}")

    np.save(out_dir_tikh / "alpha_grid.npy", diag["alpha_grid"])
    np.save(out_dir_tikh / "rms_scores.npy", diag["rms_scores"])
    with open(out_dir_tikh / "alpha_star.txt", "w") as f:
        f.write(f"{alpha_star:.12g}\n")

    return alpha_star, diag


cov_dir = "/opt/notebooks/Cov_Matrices/ICA/PTSD"
out_unreg = "/opt/notebooks/Partial_Corr_Matrices/ICA/PTSD/Unregularized"
out_tikh  = "/opt/notebooks/Partial_Corr_Matrices/ICA/PTSD/Tikhonov"

alpha_star, diag = compute_and_save_partial_corrs_pervaiz(
    cov_dir=cov_dir,
    out_dir_unreg=out_unreg,
    out_dir_tikh=out_tikh,
    pattern="*.npy",
    alpha_grid=np.logspace(-6, 2, 60),
    jitter=1e-8,
    require_shape=(100, 100),
    save_precision_too=False,  
)

print("Done. alpha* =", alpha_star)
