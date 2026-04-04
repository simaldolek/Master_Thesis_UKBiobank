#!/usr/bin/env python3
"""
similarity_index_analysis.py
=============================
Similarity Index (SI) analysis for phase sequence bin size optimization,
exactly following Almeida et al. (2014) Figure 2B-C methodology.

For a single chosen subject and a single assembly detected at a LARGER bin size
("assembly A"), it:

  1. Re-runs assembly detection at both the LARGE bin size (coarse) and a
     SMALLER bin size (fine) using PCA + FastICA, identical to Lopes-dos-Santos
     et al. (2013) as used by Almeida et al. (2014).

  2. Projects the assembly activity time-series for all detected assemblies at
     both resolutions.

  3. For each pair (large-bin assembly vs. each small-bin assembly), computes
     the Similarity Index:
         SI = |u_A · u_B|
     where u_A, u_B are the L2-normalised weight vectors (unit vectors) of the
     two assembly patterns.
     SI ranges from 0 (orthogonal / no shared ICA components) to 1 (identical).

  4. Runs a 10,000-permutation bootstrap test to build the null SI distribution:
     - Shuffles the component weights within each pattern independently
     - Recomputes SI on the shuffled pair
     - Significance threshold: 99th percentile of null distribution (p = 0.01),
      as in Almeida et al. (2014).

  5. Determines whether a large-bin assembly was "split" into two or more
     distinct small-bin assemblies (both significantly similar to the large-bin
     assembly but NOT significantly similar to each other).

  6. Saves:
       si_results.csv            — all pairwise SI values + significance
       assembly_patterns_large.csv — ICA weight vectors at large bin size
       assembly_patterns_small.csv — ICA weight vectors at small bin size
       assembly_activity_large.csv — projected activity time-series (large bins)
       assembly_activity_small.csv — projected activity time-series (small bins)
       split_assemblies_report.csv — which large-bin assemblies were split

USAGE
-----
  Example:
    python similarity_index_analysis.py 1234567.txt --large_bin 20 --small_bin 10
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA, FastICA

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TR               = 0.735
N_ICA            = 100
TIMEPOINTS       = 490
BOLD_LOW_HZ      = 0.01
BOLD_HIGH_HZ     = 0.10
SEED             = 26

# Default bin sizes — override via CLI arguments
DEFAULT_LARGE_BIN_TRS = 60   # coarser resolution (fewer assemblies)
DEFAULT_SMALL_BIN_TRS = 10   # finer resolution (more assemblies)

N_PERMUTATIONS   = 10_000    # bootstrap iterations for null SI distribution
SI_ALPHA         = 0.01      # significance level (99th percentile)

OUTPUT_DIR = Path("/bin_size_analysis/ICA/")

# ── I/O & PREPROCESSING ───────────────────────────────────────────────────────
def load_timeseries(path, expected_tp=TIMEPOINTS, expected_ica=N_ICA):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    vals = (df.apply(pd.to_numeric, errors="coerce")
              .dropna(axis=0, how="all").dropna(axis=1, how="all")
              .to_numpy(dtype=np.float32))
    if vals.shape == (expected_tp, expected_ica):
        return vals.T                   # -> (N_ICA, TIMEPOINTS)
    if vals.shape == (expected_ica, expected_tp):
        return vals
    raise ValueError(f"Unexpected shape {vals.shape} in {path}")


def bandpass_and_zscore(Z, tr=TR):
    nyq  = 0.5 / tr
    low  = max(BOLD_LOW_HZ  / nyq, 1e-4)
    high = min(BOLD_HIGH_HZ / nyq, 0.99)
    b, a = butter(4, [low, high], btype="band")
    Zf   = np.zeros_like(Z, dtype=np.float64)
    for k in range(Z.shape[0]):
        s = Z[k].astype(np.float64)
        if np.std(s) > 0:
            Zf[k] = filtfilt(b, a, s)
    stds = Zf.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    Zf = (Zf - Zf.mean(axis=1, keepdims=True)) / stds
    return np.nan_to_num(Zf)


def bin_timeseries(Z, bin_size_trs):
    """Average into non-overlapping bins -> (N_ICA, n_bins)."""
    p, n   = Z.shape
    n_bins = n // bin_size_trs
    return Z[:, :n_bins * bin_size_trs].reshape(p, n_bins, bin_size_trs).mean(axis=2)

# ── MARCHENKO-PASTUR THRESHOLD ────────────────────────────────────────────────
def mp_upper(p, n, sigma2=1.0):
    """lambda_+ = sigma^2 * (1 + sqrt(p/n))^2"""
    return sigma2 * (1.0 + np.sqrt(p / n)) ** 2

# ── ASSEMBLY DETECTION (PCA + FastICA, Lopes-dos-Santos 2013) ─────────────────
def detect_assemblies(Z_binned, seed=SEED):
    """
    Parameters
    ----------
    Z_binned : np.ndarray (p, n)   p = ICA components, n = bins (already z-scored)

    Returns
    -------
    patterns : np.ndarray (n_assemblies, p)
        Each row is the L2-normalised ICA weight vector for one assembly.
    n_sig    : int
        Number of statistically significant assemblies found.
    pca      : fitted PCA object (kept for activity projection)
    ica      : fitted FastICA object
    """
    p, n = Z_binned.shape
    if n < 2 or p < 2:
        return np.empty((0, p)), 0, None, None

    # Step 1: PCA — count eigenvalues above Marchenko-Pastur upper bound
    cov     = np.cov(Z_binned)                         # (p, p)
    eigvals = np.linalg.eigvalsh(cov)                  # ascending
    sigma2  = float(np.median(eigvals))
    if sigma2 <= 0:
        sigma2 = 1.0
    lp      = mp_upper(p, n, sigma2)
    
    n_sig = int(np.sum(eigvals > lp))
    n_sig = min(n_sig, min(p, n) - 1)   # PCA hard cap: must be < min(samples, features)
    n_sig = max(n_sig, 1)                # ensure at least 1 if we proceed

    if n_sig == 0 or n < 2 or p < 2:
        return np.empty((0, p)), 0, None, None

    # Step 2: Project activity matrix onto significant PCs
    pca = PCA(n_components=n_sig, random_state=seed)
    # PCA expects (n_samples, n_features); our matrix is (p, n) -> transpose
    Z_pca = pca.fit_transform(Z_binned.T)              # (n, n_sig)

    # Step 3: FastICA on the PCA-reduced matrix to find assembly patterns
    ica = FastICA(
        n_components=n_sig,
        max_iter=2000,
        tol=1e-4,
        random_state=seed,
        whiten="unit-variance",
    )
    try:
        ica.fit(Z_pca)
    except Exception:
        return np.empty((0, p)), 0, pca, None

    # ICA mixing matrix in original (p-dimensional) space
    # mixing_ shape (n_sig, n_sig); rotate back via PCA components
    # assembly_pattern (n_sig, p) = ICA mixing * PCA components
    mixing       = ica.mixing_                             # (n_sig, n_sig)
    pca_comps    = pca.components_                         # (n_sig, p)
    patterns_raw = mixing.T @ pca_comps                    # (n_sig, p)

    # L2-normalise each pattern to a unit vector (required for SI computation)
    norms    = np.linalg.norm(patterns_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    patterns = patterns_raw / norms                        # (n_sig, p)

    return patterns, n_sig, pca, ica


def project_assembly_activity(Z_binned, patterns):
    """
    Compute assembly activity time-series.

    AA_b = W^T Z_b  (Lopes-dos-Santos et al. 2013, eq. 4)
    where W is the assembly weight vector (column), Z_b is the z-scored
    activity vector at bin b.

    Parameters
    ----------
    Z_binned : (p, n)
    patterns : (n_assemblies, p)  — each row is one unit-norm weight vector

    Returns
    -------
    activity : (n_assemblies, n)
    """
    # AA = patterns @ Z_binned  =>  (n_assemblies, n)
    return patterns @ Z_binned


# ── SIMILARITY INDEX ──────────────────────────────────────────────────────────
def similarity_index(u, v):
    """
    SI = |u · v|  where u, v are already L2-normalised unit vectors.
    Ranges [0, 1].  Exactly Almeida et al. (2014) definition.
    """
    return float(np.abs(np.dot(u, v)))


def permutation_test_si(u, v, n_perms=N_PERMUTATIONS, alpha=SI_ALPHA, rng=None):
    """
    Bootstrap null distribution for SI by shuffling weights independently
    within each pattern (Almeida et al. 2014).

    Returns
    -------
    si_real   : float  — observed SI
    threshold : float  — (1 - alpha) percentile of null distribution
    p_value   : float  — fraction of null SIs >= si_real
    is_sig    : bool   — True if si_real > threshold
    null_dist : np.ndarray (n_perms,)
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    si_real   = similarity_index(u, v)
    null_dist = np.empty(n_perms)

    for i in range(n_perms):
        u_shuf = rng.permutation(u)
        v_shuf = rng.permutation(v)
        # Re-normalise after shuffle (shuffling preserves L2 norm, but be safe)
        nu = np.linalg.norm(u_shuf)
        nv = np.linalg.norm(v_shuf)
        if nu > 0: u_shuf = u_shuf / nu
        if nv > 0: v_shuf = v_shuf / nv
        null_dist[i] = float(np.abs(np.dot(u_shuf, v_shuf)))

    threshold = float(np.percentile(null_dist, (1 - alpha) * 100))
    p_value   = float(np.mean(null_dist >= si_real))
    is_sig    = bool(si_real > threshold)

    return si_real, threshold, p_value, is_sig, null_dist


# ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────
def run_analysis(subject_file, large_bin, small_bin, subject_id=None):
    rng = np.random.default_rng(SEED)

    if subject_id is None:
        subject_id = Path(subject_file).stem[:7]

    print(f"\n{'='*65}")
    print(f"  Subject : {subject_id}")
    print(f"  File    : {subject_file}")
    print(f"  Large bin: {large_bin} TRs  ({large_bin * TR:.2f} s)")
    print(f"  Small bin: {small_bin} TRs  ({small_bin * TR:.2f} s)")
    print(f"  Permutations: {N_PERMUTATIONS:,}   alpha={SI_ALPHA}")
    print(f"{'='*65}\n")

    # ── Load & preprocess ─────────────────────────────────────────────────────
    raw = load_timeseries(subject_file)
    Z   = bandpass_and_zscore(raw)           # (N_ICA, TIMEPOINTS)

    # ── Bin at both resolutions ───────────────────────────────────────────────
    Z_large = bin_timeseries(Z, large_bin)   # (N_ICA, n_bins_large)
    Z_small = bin_timeseries(Z, small_bin)   # (N_ICA, n_bins_small)

    print(f"  Binned matrix shapes:")
    print(f"    Large bin ({large_bin} TRs): {Z_large.shape}  "
          f"[{Z_large.shape[1]} bins x {Z_large.shape[0]} components]")
    print(f"    Small bin ({small_bin} TRs): {Z_small.shape}  "
          f"[{Z_small.shape[1]} bins x {Z_small.shape[0]} components]")

    # ── Detect assemblies at both resolutions ─────────────────────────────────
    print("\n  Detecting assemblies...")
    pat_large, n_large, pca_large, ica_large = detect_assemblies(Z_large)
    pat_small, n_small, pca_small, ica_small = detect_assemblies(Z_small)

    print(f"    Large bin -> {n_large} assemblies detected")
    print(f"    Small bin -> {n_small} assemblies detected")

    if n_large == 0 or n_small == 0:
        print("\n  WARNING: No assemblies at one or both resolutions. "
              "Try a different subject or bin sizes.")
        return

    # ── Project assembly activity time-series ─────────────────────────────────
    # Use full TR-resolution z-scored data (not binned) for activity projection,
    # but projected via the patterns learned from binned data — this gives full
    # temporal resolution activity traces exactly as in Almeida et al. who
    # re-projected onto 1 ms bins after learning patterns from 5 ms bins.
    act_large = project_assembly_activity(Z, pat_large)   # (n_large, TIMEPOINTS)
    act_small = project_assembly_activity(Z, pat_small)   # (n_small, TIMEPOINTS)

    # ── Similarity Index: all large-bin assemblies vs all small-bin assemblies ─
    print(f"\n  Computing SI for all {n_large} x {n_small} pairs "
          f"({n_large * n_small} comparisons) + {N_PERMUTATIONS:,} permutations each...")

    si_rows = []
    for i in range(n_large):
        for j in range(n_small):
            si, thr, pval, sig, _ = permutation_test_si(
                pat_large[i], pat_small[j], rng=rng)
            si_rows.append({
                "large_assembly": i,
                "small_assembly": j,
                "SI":             round(si,  4),
                "null_threshold": round(thr, 4),
                "p_value":        round(pval, 5),
                "significant":    sig,
                "large_bin_trs":  large_bin,
                "small_bin_trs":  small_bin,
                "large_bin_sec":  round(large_bin * TR, 3),
                "small_bin_sec":  round(small_bin * TR, 3),
                "subject_id":     subject_id,
            })

    si_df = pd.DataFrame(si_rows)

    # ── Within-small-bin SI matrix: are the small-bin assemblies themselves ────
    # orthogonal? (Almeida: A' and A'' should have SI~0.016 to each other)
    print(f"\n  Computing within-small-bin SI matrix "
          f"({n_small} x {n_small} pairs)...")
    wsi_rows = []
    for i in range(n_small):
        for j in range(i + 1, n_small):
            si, thr, pval, sig, _ = permutation_test_si(
                pat_small[i], pat_small[j], rng=rng)
            wsi_rows.append({
                "assembly_i":     i,
                "assembly_j":     j,
                "SI":             round(si,  4),
                "null_threshold": round(thr, 4),
                "p_value":        round(pval, 5),
                "significant":    sig,
                "subject_id":     subject_id,
            })
    wsi_df = pd.DataFrame(wsi_rows)

    # ── Identify split assemblies ─────────────────────────────────────────────
    # A large-bin assembly A is "split" if >= 2 small-bin assemblies are
    # significantly similar to A (both SI > threshold with A) but are NOT
    # significantly similar to EACH OTHER (SI ~ 0 between them), mirroring
    # Almeida Figure 2C.
    print("\n  Identifying split assemblies...")
    split_rows = []
    for i in range(n_large):
        sig_children = si_df[(si_df["large_assembly"] == i) &
                             (si_df["significant"] == True)]["small_assembly"].tolist()

        if len(sig_children) < 2:
            split_rows.append({
                "large_assembly":  i,
                "n_sig_children":  len(sig_children),
                "child_assemblies": str(sig_children),
                "is_split":        False,
                "children_orthogonal": None,
                "subject_id": subject_id,
            })
            continue

        # Check whether the children are mutually orthogonal (not sig similar)
        children_orthogonal = []
        for ci in range(len(sig_children)):
            for cj in range(ci + 1, len(sig_children)):
                a, b = sig_children[ci], sig_children[cj]
                match = wsi_df[
                    ((wsi_df["assembly_i"] == min(a, b)) &
                     (wsi_df["assembly_j"] == max(a, b)))
                ]
                if len(match):
                    children_orthogonal.append(not bool(match.iloc[0]["significant"]))
                else:
                    children_orthogonal.append(True)   # trivially orthogonal if not found

        all_orth = all(children_orthogonal)
        split_rows.append({
            "large_assembly":    i,
            "n_sig_children":    len(sig_children),
            "child_assemblies":  str(sig_children),
            "is_split":          all_orth,
            "children_orthogonal": all_orth,
            "subject_id":        subject_id,
        })

    split_df = pd.DataFrame(split_rows)

    # ── Save all outputs ───────────────────────────────────────────────────────
    tag = f"{subject_id}_large{large_bin}_small{small_bin}"

    # SI results
    si_path = OUTPUT_DIR / f"si_results_{tag}.csv"
    si_df.to_csv(si_path, index=False)

    # Within-small-bin SI
    wsi_path = OUTPUT_DIR / f"si_within_small_{tag}.csv"
    wsi_df.to_csv(wsi_path, index=False)

    # Assembly patterns (ICA weight vectors)
    pat_large_df = pd.DataFrame(
        pat_large,
        index=[f"assembly_{i}" for i in range(n_large)],
        columns=[f"ICA_{k}" for k in range(N_ICA)],
    )
    pat_small_df = pd.DataFrame(
        pat_small,
        index=[f"assembly_{i}" for i in range(n_small)],
        columns=[f"ICA_{k}" for k in range(N_ICA)],
    )
    pat_large_df.to_csv(OUTPUT_DIR / f"assembly_patterns_large_{tag}.csv")
    pat_small_df.to_csv(OUTPUT_DIR / f"assembly_patterns_small_{tag}.csv")

    # Assembly activity time-series (projected onto full TR resolution)
    act_large_df = pd.DataFrame(
        act_large,
        index=[f"assembly_{i}" for i in range(n_large)],
        columns=[f"TR_{t}" for t in range(act_large.shape[1])],
    )
    act_small_df = pd.DataFrame(
        act_small,
        index=[f"assembly_{i}" for i in range(n_small)],
        columns=[f"TR_{t}" for t in range(act_small.shape[1])],
    )
    act_large_df.to_csv(OUTPUT_DIR / f"assembly_activity_large_{tag}.csv")
    act_small_df.to_csv(OUTPUT_DIR / f"assembly_activity_small_{tag}.csv")

    # Split assembly report
    split_path = OUTPUT_DIR / f"split_assemblies_report_{tag}.csv"
    split_df.to_csv(split_path, index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  RESULTS SUMMARY")
    print(f"{'─'*65}")
    print(f"  Large bin ({large_bin} TRs = {large_bin*TR:.2f} s): "
          f"{n_large} assemblies")
    print(f"  Small bin ({small_bin} TRs = {small_bin*TR:.2f} s): "
          f"{n_small} assemblies")
    print()
    print("  Cross-resolution SI (significant pairs, p < 0.01):")
    sig_pairs = si_df[si_df["significant"] == True]
    if len(sig_pairs):
        for _, row in sig_pairs.iterrows():
            print(f"    Large-A{int(row.large_assembly)} <-> "
                  f"Small-A{int(row.small_assembly)}: "
                  f"SI={row.SI:.3f}  (null thr={row.null_threshold:.3f})")
    else:
        print("    None found.")

    print()
    print("  Split assembly detection:")
    split_found = split_df[split_df["is_split"] == True]
    if len(split_found):
        for _, row in split_found.iterrows():
            print(f"    Large-A{int(row.large_assembly)} SPLIT into "
                  f"small-bin assemblies {row.child_assemblies}  "
                  f"(children are mutually orthogonal)")
    else:
        print("    No split assemblies detected (no large-bin assembly "
              "has >= 2 mutually orthogonal significant children).")

    print(f"\n  Files saved:")
    for p in [si_path, wsi_path, split_path]:
        print(f"    {p}")
    print(f"    {OUTPUT_DIR}/assembly_patterns_large_{tag}.csv")
    print(f"    {OUTPUT_DIR}/assembly_patterns_small_{tag}.csv")
    print(f"    {OUTPUT_DIR}/assembly_activity_large_{tag}.csv")
    print(f"    {OUTPUT_DIR}/assembly_activity_small_{tag}.csv")
    print()

    return si_df, wsi_df, split_df, pat_large_df, pat_small_df


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser(
        description="Similarity Index analysis (Almeida et al. 2014) for fMRI assemblies"
    )
    parser.add_argument("subject_file", nargs="?", default=None,
                        help=".txt ICA timeseries file for one subject")
    parser.add_argument("--large_bin", type=int, default=DEFAULT_LARGE_BIN_TRS,
                        help=f"Coarse bin size in TRs (default: {DEFAULT_LARGE_BIN_TRS})")
    parser.add_argument("--small_bin", type=int, default=DEFAULT_SMALL_BIN_TRS,
                        help=f"Fine bin size in TRs (default: {DEFAULT_SMALL_BIN_TRS})")
    parser.add_argument("--subject_id", type=str, default=None,
                        help="Optional label for the subject")

    args = parser.parse_args()

    if args.subject_file is None:
        candidates = sorted(glob.glob("*.txt"))
        if not candidates:
            print("No .txt file provided or found in current directory.")
            sys.exit(1)
        args.subject_file = candidates[0]
        print(f"Auto-selected: {args.subject_file}")

    if args.large_bin <= args.small_bin:
        print("ERROR: --large_bin must be strictly greater than --small_bin")
        sys.exit(1)

    run_analysis(
        subject_file=args.subject_file,
        large_bin=args.large_bin,
        small_bin=args.small_bin,
        subject_id=args.subject_id,
    )
