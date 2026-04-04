"""
Adaptation for raw BOLD ROI time series (not ICA components).

  ICA-component data: components ARE assemblies, used directly.
  Raw BOLD ROI data:  ROIs are NOT assemblies: must DETECT assemblies
                      from ROI co-activation patterns, exactly as in
                      Almeida-Filho et al. (2014) and Lopes-dos-Santos (2013):
                        1. Bandpass + z-score ROI time series
                        2. Compute ROI correlation matrix
                        3. Marchenko-Pastur eigenvalue threshold → n_assemblies
                        4. Project onto significant PCs
                        5. ICA in reduced space → assembly weight patterns
                        6. Project activity: AA_b = Z_b^T P Z_b (diag zeroed)
                        7. Detect activation peaks → phase-sequence graph

PARAMETER DIFFERENCES vs v4.1 (ICA version):
  - N_ROIS: configurable 
  - ACTIVATION_THRESHOLD_PCT: 90 
  - HRF_REFRACTORY_TRS: 4 
  - MAX_IAI_TRS: 10 
  - WINDOW_SIZE_TRS: 80 
  - TPM size: N_ASSEMBLIES² which varies per subject, upper-triangle of
    fixed N_ROIS×N_ROIS matrix used as TPM proxy for comparability
"""

import numpy as np
import networkx as nx
from scipy.stats import entropy as scipy_entropy
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pandas as pd
import random
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("assembly_graph_BOLD_v1.log", mode="w"),
    ]
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# CHANGE THESE to match your parcellation
N_ROIS        = 414          # number of ROIs
TIMEPOINTS    = 490

DATA_ROOT     = Path("/opt/notebooks/CombinedAtlas_31016+31019/")
OUTPUT_DIR    = Path("./assembly_graph_BOLD_outputs")
SEED          = 26
TR            = 0.735        # seconds

BOLD_LOW_HZ   = 0.01
BOLD_HIGH_HZ  = 0.10

# Assembly detection: slightly stricter than ICA version because the
# quadratic projection AA_b = Z_b^T P Z_b produces more selective peaks
ACTIVATION_THRESHOLD_PCT = 90.0
HRF_REFRACTORY_TRS       = 4    # ~2.9 s

MAX_IAI_TRS       = 10   # ~7.4 s
WINDOW_SIZE_TRS   = 80   # ~59 s
WINDOW_STRIDE_TRS = 20   # ~15 s

TPM_PCA_DIM = 50
FC_PCA_DIM  = 50

LABEL_TO_INDEX = {"HC": 0, "PTSD": 1}


# ── DATA LOADING (provided by user) ──────────────────────────────────────────

@dataclass
class SubjectRecord:
    csv_path: str
    subject_id: str
    class_name: str
    label: int

def get_subject_csvs(data_root: Path) -> List[SubjectRecord]:
    records: List[SubjectRecord] = []
    for class_name, label in LABEL_TO_INDEX.items():
        class_dir = data_root / class_name
        for csv_path in sorted(class_dir.glob("*.csv")):
            records.append(SubjectRecord(
                csv_path=str(csv_path.resolve()),
                subject_id=csv_path.stem,
                class_name=class_name,
                label=label,
            ))
    return records

def load_subject_timeseries(
    csv_path: str,
    expected_timepoints: int = TIMEPOINTS,
    expected_rois: int = N_ROIS,
) -> np.ndarray:
    """Returns (N_ROIS, N_TRs) — ROI-first for downstream pipeline."""
    df = pd.read_csv(csv_path)
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)  # drop ROI name column
    if values.shape != (expected_rois, expected_timepoints):
        raise ValueError(
            f"Unexpected shape for {csv_path}: got {values.shape}, "
            f"expected ({expected_rois}, {expected_timepoints})"
        )
    return values  # already (N_ROIS, N_TRs) — do NOT transpose

def screen_records(
    records: List[SubjectRecord],
    expected_timepoints: int,
    expected_rois: int,
    log_path: Path,
) -> Tuple[List[SubjectRecord], List[str]]:
    retained, excluded_paths = [], []
    for record in records:
        try:
            load_subject_timeseries(record.csv_path, expected_timepoints, expected_rois)
            retained.append(record)
        except Exception as e:
            excluded_paths.append(f"{record.csv_path} | {e}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Excluded {len(excluded_paths)} of {len(records)} subjects\n\n")
        f.writelines(f"{entry}\n" for entry in excluded_paths)
    return retained, excluded_paths


# ── STEP 1: BANDPASS FILTER + Z-SCORE ────────────────────────────────────────

def bandpass_and_zscore(Z: np.ndarray, tr: float = TR) -> np.ndarray:
    """
    Z: (N_ROIS, N_TRs)
    Bandpass to BOLD resting-state band (0.01–0.10 Hz), then z-score per ROI.
    """
    nyq  = 0.5 / tr
    low  = max(BOLD_LOW_HZ  / nyq, 1e-4)
    high = min(BOLD_HIGH_HZ / nyq, 0.99)
    b, a = butter(4, [low, high], btype="band")
    Z_filt = np.zeros_like(Z, dtype=np.float64)
    for k in range(Z.shape[0]):
        sig = Z[k].astype(np.float64)
        if np.std(sig) > 0:
            Z_filt[k] = filtfilt(b, a, sig)
    stds = Z_filt.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    Z_filt = (Z_filt - Z_filt.mean(axis=1, keepdims=True)) / stds
    return np.nan_to_num(Z_filt)


# ── STEP 2: MARCHENKO-PASTUR THRESHOLD ───────────────────────────────────────

def marchenko_pastur_lambda_max(Z: np.ndarray, sigma2: float = 1.0) -> float:
    """
    Upper bound of the Marchenko-Pastur distribution for a z-scored matrix.
    λ_max = σ² (1 + √(1/q))²   where q = N_TRs / N_ROIs

    Any eigenvalue of the ROI correlation matrix above this bound is
    considered statistically significant — i.e., it reflects genuine
    co-activation structure rather than random fluctuations.
    (Marchenko & Pastur 1967; Peyrache et al. 2010; Lopes-dos-Santos 2013)
    """
    N_rois, N_trs = Z.shape
    q = N_trs / N_rois
    return sigma2 * (1.0 + np.sqrt(1.0 / q)) ** 2


# ── STEP 3: DETECT ASSEMBLY PATTERNS (paper Steps 2-4) ───────────────────────

def detect_assembly_patterns(
    Z: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Implements Lopes-dos-Santos et al. (2013) / Almeida-Filho (2014)
    assembly detection, adapted for BOLD ROI time series:

      (a) Compute ROI correlation matrix C = Z Z^T / N_TRs
      (b) Eigendecompose C: sort descending
      (c) Count eigenvalues above Marchenko-Pastur bound: n_assemblies
      (d) Project Z onto top n_assemblies PCs: Z_reduced
      (e) FastICA on Z_reduced: assembly weight patterns in ROI space

    Parameters
    ----------
    Z : (N_ROIS, N_TRs): bandpass-filtered, z-scored

    Returns
    -------
    assembly_patterns : (n_assemblies, N_ROIS)
        Each row = weight vector of one assembly over ROIs.
        Analogous to neuron weight vectors in the paper.
    n_assemblies : int
    """
    N_rois, N_trs = Z.shape

    # (a) Correlation matrix
    C = (Z @ Z.T) / N_trs  # (N_ROIS, N_ROIS)

    # (b) Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues  = eigenvalues[::-1]   # descending
    eigenvectors = eigenvectors[:, ::-1]

    # (c) Marchenko-Pastur threshold
    lam_max     = marchenko_pastur_lambda_max(Z)
    n_assemblies = int(np.sum(eigenvalues > lam_max))

    if n_assemblies < 1:
        if verbose:
            log.warning("  No significant assemblies found; defaulting to 1")
        n_assemblies = 1

    if verbose:
        log.info(f"  λ_max={lam_max:.3f}  n_assemblies={n_assemblies}  "
                 f"top-3 eigenvalues={eigenvalues[:3].round(3)}")

    # (d) Project Z onto significant PCs
    sig_pcs   = eigenvectors[:, :n_assemblies]   # (N_ROIS, n_assemblies)
    Z_reduced = sig_pcs.T @ Z                    # (n_assemblies, N_TRs)

    # (e) ICA in reduced space
    if n_assemblies == 1:
        # ICA needs ≥2 components; use the single PC directly
        assembly_patterns = sig_pcs.T  # (1, N_ROIS)
    else:
        ica = FastICA(
            n_components=n_assemblies,
            max_iter=10000,
            tol=1e-5,
            random_state=SEED,
        )
        ica.fit(Z_reduced.T)  # fit on (N_TRs, n_assemblies)
        # Un-project mixing matrix from reduced space back to ROI space
        assembly_patterns = (sig_pcs @ ica.mixing_).T  # (n_assemblies, N_ROIS)

    # Normalise to unit length
    norms = np.linalg.norm(assembly_patterns, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    assembly_patterns = assembly_patterns / norms

    return assembly_patterns, n_assemblies


# ── STEP 4: COMPUTE ASSEMBLY ACTIVITY TIME SERIES ────────────────────────────

def compute_assembly_activity(
    Z: np.ndarray,
    assembly_patterns: np.ndarray,
) -> np.ndarray:
    """
    AA_b = Z_b^T P Z_b   where P = w w^T with diagonal zeroed.

    The zeroed diagonal removes the contribution of single ROIs firing in
    isolation: only genuine co-activation of assembly members raises AA_b.
    This is equation (1) from Lopes-dos-Santos (2013), adapted for BOLD:
    each TR is a "bin" and each ROI signal value replaces spike count.

    Parameters
    ----------
    Z                 : (N_ROIS, N_TRs) — z-scored BOLD
    assembly_patterns : (n_assemblies, N_ROIS)

    Returns
    -------
    assembly_activity : (n_assemblies, N_TRs)
    """
    n_assemblies = assembly_patterns.shape[0]
    N_trs        = Z.shape[1]
    activity     = np.zeros((n_assemblies, N_trs), dtype=np.float64)

    for k, w in enumerate(assembly_patterns):
        P = np.outer(w, w)
        np.fill_diagonal(P, 0.0)   # critical: zero diagonal
        PZ = P @ Z                 # (N_ROIS, N_TRs)
        activity[k] = np.einsum("it,it->t", Z, PZ)  # vectorised Z_b^T P Z_b

    return activity


# ── STEP 5: DETECT ACTIVATION EVENTS (HRF-AWARE) ─────────────────────────────

def detect_activations_bold(
    assembly_activity: np.ndarray,
    threshold_pct: float = ACTIVATION_THRESHOLD_PCT,
    refractory_trs: int  = HRF_REFRACTORY_TRS,
) -> List[Tuple[int, int]]:
    """
    Threshold each assembly's activity at `threshold_pct` percentile.
    Use find_peaks with minimum distance = refractory_trs to avoid
    double-detecting the same hemodynamic event.

    Returns list of (assembly_index, TR), sorted by TR.
    The assembly_activity here is computed via
    the quadratic projection rather than taken directly from ICA components.
    """
    N_assemblies = assembly_activity.shape[0]
    thresholds   = np.percentile(assembly_activity, threshold_pct, axis=1)
    events: List[Tuple[int, int]] = []
    for k in range(N_assemblies):
        peaks, _ = find_peaks(
            assembly_activity[k],
            height=thresholds[k],
            distance=refractory_trs,
        )
        for t in peaks:
            events.append((k, int(t)))
    events.sort(key=lambda x: x[1])
    return events


# ── STEP 6: GRAPH CONSTRUCTION ────────────────────────────────────────────────

def extract_window_events(events, t_start, t_end):
    return [(k, t) for k, t in events if t_start <= t < t_end]

def build_graph_from_window(
    window_events: List[Tuple[int, int]],
    n_assemblies: int,
    max_iai_trs: int = MAX_IAI_TRS,
) -> nx.DiGraph:
    """
    Build directed graph from assembly activation events within a TR window.

    Nodes: assembly indices (0 … n_assemblies-1)
    Edges: directed, i→j if assembly j activates within max_iai_trs after i.
    Weight: exp(-IAI / tau) — stronger weight for tighter temporal coupling.

    NOTE: n_assemblies varies per subject (Marchenko-Pastur gives different
    counts). We always add all n_assemblies nodes so graph size is consistent
    within a subject, but graph-level features are normalised by n_assemblies
    for cross-subject comparability.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n_assemblies))
    if len(window_events) < 2:
        return G
    evts = sorted(window_events, key=lambda x: x[1])
    tau  = max_iai_trs / 2.0
    for i in range(len(evts)):
        ki, ti = evts[i]
        for j in range(i + 1, len(evts)):
            kj, tj = evts[j]
            iai = tj - ti
            if iai > max_iai_trs:
                break
            if ki == kj:
                continue
            w = float(np.exp(-iai / tau))
            if G.has_edge(ki, kj):
                G[ki][kj]["weight"] += w
                G[ki][kj]["count"]  += 1
            else:
                G.add_edge(ki, kj, weight=w, count=1)
    return G


# ── STEP 7: GRAPH FEATURE EXTRACTION ─────────────────────────────────────────

def extract_graph_features(
    G: nx.DiGraph,
    n_activations: int,
    n_assemblies: int,
) -> Dict[str, float]:
    """
    All 13 Almeida-Filho (2014) attributes + extended features.
    Normalised by n_assemblies (not fixed N_ICA=100) for cross-subject
    comparability when assembly count varies.
    """
    f: Dict[str, float] = {}
    norm  = max(n_activations, 1)
    n_ref = max(n_assemblies, 1)
    Gu    = nx.Graph(G.to_undirected())

    active_nodes = [n for n in G.nodes() if G.degree(n) > 0]
    f["ActiveNodes"] = len(active_nodes) / n_ref
    f["Edges"]       = G.number_of_edges() / norm
    f["Density"]     = nx.density(G)

    _nan_keys = ["RE","PE","L1","L2","L3","LCC","LSC","ATD",
                 "Diameter","ASP","CC","Transitivity","Reciprocity",
                 "MeanWeight","StdWeight","WeightEntropy",
                 "InDegreeGini","OutDegreeGini","HubScore",
                 "nAssemblies"]   # NEW: assembly count itself as a feature

    if G.number_of_edges() == 0:
        for k in _nan_keys:
            f[k] = np.nan
        f["nAssemblies"] = n_assemblies
        return f

    edge_counts = {}
    for u, v in G.edges():
        edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
    f["RE"] = sum(c - 1 for c in edge_counts.values() if c > 1) / norm
    pe      = sum(1 for u, v in G.edges() if G.has_edge(v, u)) / 2
    f["PE"] = pe / norm
    f["L1"] = 0.0

    l2 = sum(1 for u in G.nodes() for v in G.successors(u)
             if u < v and G.has_edge(v, u))
    f["L2"] = l2 / norm
    l3 = sum(1 for u in G.nodes() for v in G.successors(u) if v != u
             for w in G.successors(v) if w not in (u, v) and G.has_edge(w, u))
    f["L3"] = l3 / norm

    ccs  = list(nx.connected_components(Gu))
    f["LCC"] = max(len(c) for c in ccs) / n_ref if ccs else 0.0
    sccs = list(nx.strongly_connected_components(G))
    f["LSC"] = max(len(s) for s in sccs) / n_ref if sccs else 0.0

    degs  = [d for _, d in G.degree()]
    f["ATD"] = float(np.mean(degs)) / n_ref

    try:
        sub = Gu if nx.is_connected(Gu) else Gu.subgraph(max(ccs, key=len))
        f["Diameter"] = nx.diameter(sub) / n_ref
        f["ASP"]      = nx.average_shortest_path_length(sub) / n_ref
    except Exception:
        f["Diameter"] = np.nan
        f["ASP"]      = np.nan

    f["CC"]           = nx.average_clustering(Gu)
    f["Transitivity"] = nx.transitivity(Gu)
    try:
        f["Reciprocity"] = nx.overall_reciprocity(G)
    except Exception:
        f["Reciprocity"] = np.nan

    weights            = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    f["MeanWeight"]    = float(np.mean(weights))
    f["StdWeight"]     = float(np.std(weights))
    w_norm             = weights / (weights.sum() + 1e-12)
    f["WeightEntropy"] = float(scipy_entropy(w_norm + 1e-12))

    in_degs  = np.array([G.in_degree(n)  for n in G.nodes()], dtype=float)
    out_degs = np.array([G.out_degree(n) for n in G.nodes()], dtype=float)

    def gini(x):
        if x.sum() == 0:
            return 0.0
        x = np.sort(x)
        n = len(x)
        return float((2 * np.sum(np.arange(1, n+1) * x) / (n * x.sum())) - (n+1)/n)

    f["InDegreeGini"]  = gini(in_degs)
    f["OutDegreeGini"] = gini(out_degs)
    f["HubScore"]      = float(out_degs.max() / (out_degs.mean() + 1e-12))

    # Assembly count as feature: subjects with more assemblies have richer
    # co-activation structure — this may differ between PTSD and HC
    f["nAssemblies"] = float(n_assemblies)

    return f


# ── STEP 8: TPM AND FC ────────────────────────────────────────────────────────

def compute_tpm(
    events: List[Tuple[int, int]],
    n_rois: int = N_ROIS,
    max_iai_trs: int = MAX_IAI_TRS,
) -> np.ndarray:
    """
    Assembly-level TPM projected back to ROI space for cross-subject
    comparability.

    Because n_assemblies varies per subject, we cannot directly compare
    raw N_assemblies × N_assemblies TPMs. Instead, we build the
    N_ROIS × N_ROIS ROI co-activation count matrix (which has fixed size),
    using the event stream where assembly index is treated as a proxy
    for a ROI-cluster index.

    NOTE: For large parcellations (N_ROIS > 200), reduce n_tpm_rois.
    """
    # Use a fixed ROI-space TPM: count how often event at roi i is followed
    # by event at roi j within max_iai_trs. Assembly indices are already
    # in [0, n_assemblies), which are subsets of ROI space — we reuse
    # N_ROIS as the fixed dimension.
    counts = np.zeros((n_rois, n_rois), dtype=np.float32)
    evts   = sorted(events, key=lambda x: x[1])
    for i in range(len(evts)):
        ki, ti = evts[i]
        if ki >= n_rois:
            continue
        for j in range(i + 1, len(evts)):
            kj, tj = evts[j]
            if tj - ti > max_iai_trs:
                break
            if ki != kj and kj < n_rois:
                counts[ki, kj] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tpm = counts / row_sums
    idx = np.triu_indices(n_rois, k=1)
    return tpm[idx].astype(np.float32)

def compute_fc(Z: np.ndarray) -> np.ndarray:
    """Pearson FC upper triangle. Z: (N_ROIS, N_TRs)."""
    corr = np.corrcoef(Z)
    idx  = np.triu_indices(corr.shape[0], k=1)
    return corr[idx].astype(np.float32)


# ── STEP 9: FULL SUBJECT-LEVEL FEATURE EXTRACTION ────────────────────────────

def extract_subject_features(
    bold_array: np.ndarray,   # (N_ROIS, N_TRs)
    tr: float = TR,
    verbose: bool = False,
) -> Dict:
    """
    Complete pipeline for one subject from raw BOLD ROI time series:
      bandpass → z-score → assembly detection (MP + ICA) →
      activity projection → event detection → sliding window graphs →
      graph features + TPM + FC
    """
    N_trs = bold_array.shape[1]

    # 1. Preprocess
    Z = bandpass_and_zscore(bold_array, tr)

    # 2. Detect assembly patterns (this is the key step absent in ICA version)
    assembly_patterns, n_assemblies = detect_assembly_patterns(Z, verbose=verbose)

    # 3. Compute assembly activity time series (quadratic projection)
    assembly_activity = compute_assembly_activity(Z, assembly_patterns)

    # 4. Detect activation events
    events = detect_activations_bold(assembly_activity)

    if verbose:
        log.info(f"  n_assemblies={n_assemblies}  "
                 f"n_events={len(events)}")

    # 5. Sliding window graph features 
    window_feature_dicts = []
    for t_start in range(0, N_trs - WINDOW_SIZE_TRS + 1, WINDOW_STRIDE_TRS):
        win_evts = extract_window_events(events, t_start, t_start + WINDOW_SIZE_TRS)
        G        = build_graph_from_window(win_evts, n_assemblies, MAX_IAI_TRS)
        feat     = extract_graph_features(G, len(win_evts), n_assemblies)
        window_feature_dicts.append(feat)

    feat_keys   = sorted(window_feature_dicts[0].keys()) if window_feature_dicts else []
    graph_feats = []
    graph_names = []
    for key in feat_keys:
        vals  = np.array([d.get(key, np.nan) for d in window_feature_dicts],
                         dtype=np.float64)
        vc    = vals[~np.isnan(vals)]
        mean_v = float(np.mean(vc)) if len(vc)     else np.nan
        std_v  = float(np.std(vc))  if len(vc) > 1 else np.nan
        if len(vc) > 5:
            hist, _ = np.histogram(vc, bins=min(10, len(vc)//2))
            hn      = hist / (hist.sum() + 1e-12)
            ent_v   = float(scipy_entropy(hn + 1e-12))
        else:
            ent_v = np.nan
        graph_feats.extend([mean_v, std_v, ent_v])
        graph_names.extend([f"{key}_mean", f"{key}_std", f"{key}_ent"])

    return {
        "graph":       np.array(graph_feats, dtype=np.float32),
        "graph_names": graph_names,
        "tpm":         compute_tpm(events, N_ROIS, MAX_IAI_TRS),
        "fc":          compute_fc(Z),
        "n_assemblies": n_assemblies,
    }


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred, y_prob, fold=None) -> Dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "fold":               fold,
        "auroc":              roc_auc_score(y_true, y_prob),
        "accuracy":           accuracy_score(y_true, y_pred),
        "balanced_accuracy":  balanced_accuracy_score(y_true, y_pred),
        "sensitivity":        recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity":        tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "precision":          precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1":                 f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc":                matthews_corrcoef(y_true, y_pred),
        "tp":  int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "ptsd_misclassified": int(fn),
        "hc_misclassified":   int(fp),
        "n_test":             len(y_true),
    }

def log_fold_metrics(m: Dict):
    log.info(
        f"  Fold {m['fold']:>2} | "
        f"AUROC={m['auroc']:.3f}  Acc={m['accuracy']:.3f}  "
        f"BalAcc={m['balanced_accuracy']:.3f}  "
        f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
        f"F1={m['f1']:.3f}  MCC={m['mcc']:.3f}  |  "
        f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}  "
        f"[PTSD→HC={m['ptsd_misclassified']} | HC→PTSD={m['hc_misclassified']}]"
    )

def log_aggregate_metrics(all_metrics: List[Dict], label: str = ""):
    keys = ["auroc","accuracy","balanced_accuracy","sensitivity",
            "specificity","precision","f1","mcc"]
    log.info(f"\n{'─'*70}")
    log.info(f"  AGGREGATE RESULTS {label}")
    log.info(f"{'─'*70}")
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(float(m[k]))]
        log.info(f"  {k:<22}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    log.info(f"  {'ptsd_misclassified':<22}: "
             f"{sum(m['ptsd_misclassified'] for m in all_metrics)} / "
             f"{sum(m['tp']+m['fn'] for m in all_metrics)} PTSD subjects")
    log.info(f"  {'hc_misclassified':<22}: "
             f"{sum(m['hc_misclassified'] for m in all_metrics)} / "
             f"{sum(m['tn']+m['fp'] for m in all_metrics)} HC subjects")
    log.info(f"{'─'*70}\n")


# ── SAFE HELPERS (same as v4.1) ───────────────────────────────────────────────

def safe_pca(X_tr, X_te, n_components, random_state=SEED):
    n_comp = min(n_components, X_tr.shape[0] - 1, X_tr.shape[1] - 1)
    if n_comp < 1:
        return X_tr, X_te
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(X_tr), pca.transform(X_te)

def select_graph_features(Xg_tr, Xg_te, y_tr):
    if Xg_tr.shape[1] <= 1:
        return Xg_tr, Xg_te
    try:
        vt    = VarianceThreshold(threshold=1e-4)
        Xg_tr = vt.fit_transform(Xg_tr)
        Xg_te = vt.transform(Xg_te)
    except ValueError:
        pass
    if Xg_tr.shape[1] == 0:
        return np.zeros((Xg_tr.shape[0], 1)), np.zeros((Xg_te.shape[0], 1))
    k_sel = min(40, Xg_tr.shape[1])
    sel   = SelectKBest(mutual_info_classif, k=k_sel)
    Xg_tr = sel.fit_transform(Xg_tr, y_tr)
    Xg_te = sel.transform(Xg_te)
    return Xg_tr, Xg_te


# ── CLASSIFICATION WITH FULL LOGGING ─────────────────────────────────────────

def classify_full_logging(
    X_graph:  np.ndarray,
    X_tpm:    np.ndarray,
    X_fc:     np.ndarray,
    y:        np.ndarray,
    subject_ids: List[str],
    graph_names: List[str],
    n_splits: int = 5,
    random_state: int = SEED,
    tag: str = "run",
) -> Tuple[List[Dict], pd.DataFrame]:

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_metrics: List[Dict] = []
    subj_probs = np.full(len(y), np.nan)
    subj_preds = np.full(len(y), -1, dtype=int)

    has_graph = X_graph.shape[1] > 0
    has_tpm   = X_tpm.shape[1]   > 0
    has_fc    = X_fc.shape[1]    > 0

    log.info(f"\n{'═'*70}")
    log.info(f"  CLASSIFICATION: {tag}")
    log.info(f"  Subjects={len(y)}  Folds={n_splits}  "
             f"PTSD={y.sum()}  HC={(y==0).sum()}")
    log.info(f"  Blocks — graph:{has_graph}({X_graph.shape[1]})  "
             f"tpm:{has_tpm}({X_tpm.shape[1]})  "
             f"fc:{has_fc}({X_fc.shape[1]})")
    log.info(f"  Params: threshold={ACTIVATION_THRESHOLD_PCT}pct  "
             f"max_iai={MAX_IAI_TRS}TRs  "
             f"window={WINDOW_SIZE_TRS}TRs  stride={WINDOW_STRIDE_TRS}TRs  "
             f"N_ROIS={N_ROIS}")
    log.info(f"{'═'*70}")

    split_X = X_graph if has_graph else (X_tpm if has_tpm else X_fc)

    for fold, (tr_idx, te_idx) in enumerate(cv.split(split_X, y), 1):
        blocks_tr, blocks_te = [], []

        if has_graph:
            imp   = SimpleImputer(strategy="median")
            Xg_tr = imp.fit_transform(X_graph[tr_idx])
            Xg_te = imp.transform(X_graph[te_idx])
            Xg_tr, Xg_te = select_graph_features(Xg_tr, Xg_te, y[tr_idx])
            blocks_tr.append(Xg_tr); blocks_te.append(Xg_te)

        if has_tpm:
            imp   = SimpleImputer(strategy="median")
            Xt_tr = imp.fit_transform(X_tpm[tr_idx])
            Xt_te = imp.transform(X_tpm[te_idx])
            Xt_tr, Xt_te = safe_pca(Xt_tr, Xt_te, TPM_PCA_DIM, random_state)
            blocks_tr.append(Xt_tr); blocks_te.append(Xt_te)

        if has_fc:
            imp   = SimpleImputer(strategy="median")
            Xf_tr = imp.fit_transform(X_fc[tr_idx])
            Xf_te = imp.transform(X_fc[te_idx])
            Xf_tr, Xf_te = safe_pca(Xf_tr, Xf_te, FC_PCA_DIM, random_state)
            blocks_tr.append(Xf_tr); blocks_te.append(Xf_te)

        X_tr = np.concatenate(blocks_tr, axis=1)
        X_te = np.concatenate(blocks_te, axis=1)

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        rf  = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                     min_samples_leaf=3, class_weight="balanced",
                                     random_state=random_state, n_jobs=-1)
        lr  = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced",
                                  random_state=random_state, solver="lbfgs")
        svm = SVC(kernel="rbf", probability=True, class_weight="balanced",
                  C=0.5, random_state=random_state)

        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("lr", lr), ("svm", svm)],
            voting="soft", n_jobs=-1,
        )
        ensemble.fit(X_tr, y[tr_idx])
        y_prob = ensemble.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        subj_probs[te_idx] = y_prob
        subj_preds[te_idx] = y_pred

        m = compute_all_metrics(y[te_idx], y_pred, y_prob, fold=fold)
        all_metrics.append(m)
        log_fold_metrics(m)

    log_aggregate_metrics(all_metrics, label=tag)

    pred_df = pd.DataFrame({
        "subject_id": subject_ids,
        "true_label": ["PTSD" if yi == 1 else "HC" for yi in y],
        "predicted":  ["PTSD" if pi == 1 else "HC" for pi in subj_preds],
        "prob_ptsd":  subj_probs,
        "correct":    (subj_preds == y).astype(int),
    })
    return all_metrics, pred_df


# ── FEATURE SAVING ────────────────────────────────────────────────────────────

def save_subject_features(subjects_bold, records, output_dir):
    log.info("\n=== EXTRACTING FEATURES (BOLD → assembly detection) ===")

    all_graph, all_tpm, all_fc = [], [], []
    all_n_assemblies = []
    graph_names = None
    subject_ids = [r.subject_id for r in records]
    labels_str  = [r.class_name for r in records]
    le = LabelEncoder()
    y  = le.fit_transform(labels_str)

    for idx, (bold, rec) in enumerate(zip(subjects_bold, records)):
        if idx % 50 == 0:
            log.info(f"  Subject {idx+1}/{len(records)}")
        feats = extract_subject_features(bold)
        all_graph.append(feats["graph"])
        all_tpm.append(feats["tpm"])
        all_fc.append(feats["fc"])
        all_n_assemblies.append(feats["n_assemblies"])
        if graph_names is None:
            graph_names = feats["graph_names"]

    X_graph = np.array(all_graph, dtype=np.float32)
    X_tpm   = np.array(all_tpm,   dtype=np.float32)
    X_fc    = np.array(all_fc,    dtype=np.float32)

    log.info(f"  Graph: {X_graph.shape} | TPM: {X_tpm.shape} | FC: {X_fc.shape}")
    log.info(f"  Assembly count per subject — "
             f"mean={np.mean(all_n_assemblies):.1f}  "
             f"min={np.min(all_n_assemblies)}  "
             f"max={np.max(all_n_assemblies)}")

    meta = pd.DataFrame({
        "subject_id":   subject_ids,
        "group":        labels_str,
        "n_assemblies": all_n_assemblies,
    })

    pd.concat([meta, pd.DataFrame(X_graph, columns=graph_names)], axis=1
              ).to_csv(output_dir / "subject_graph_features.csv", index=False)

    tpm_cols = [f"tpm_{i}" for i in range(X_tpm.shape[1])]
    pd.concat([meta, pd.DataFrame(X_tpm, columns=tpm_cols)], axis=1
              ).to_csv(output_dir / "subject_tpm_features.csv", index=False)

    fc_cols = [f"fc_{i}" for i in range(X_fc.shape[1])]
    pd.concat([meta, pd.DataFrame(X_fc, columns=fc_cols)], axis=1
              ).to_csv(output_dir / "subject_fc_features.csv", index=False)

    np.savez_compressed(
        output_dir / "all_features.npz",
        X_graph=X_graph, X_tpm=X_tpm, X_fc=X_fc,
        y=y, subject_ids=np.array(subject_ids),
        groups=np.array(labels_str),
        graph_names=np.array(graph_names),
        n_assemblies=np.array(all_n_assemblies),
    )
    log.info(f"  All features saved to {output_dir}/")
    return X_graph, X_tpm, X_fc, y, subject_ids, graph_names


# ── ABLATION STUDY ────────────────────────────────────────────────────────────

def run_ablation(X_graph, X_tpm, X_fc, y, subject_ids, graph_names, output_dir):
    log.info("\n" + "═"*70)
    log.info("  ABLATION STUDY")
    log.info("═"*70)
    N     = len(y)
    empty = np.zeros((N, 0), dtype=np.float32)
    configs = [
        ("graph_only", X_graph, empty,  empty),
        ("tpm_only",   empty,   X_tpm,  empty),
        ("fc_only",    empty,   empty,  X_fc),
        ("graph+tpm",  X_graph, X_tpm,  empty),
        ("graph+fc",   X_graph, empty,  X_fc),
        ("tpm+fc",     empty,   X_tpm,  X_fc),
        ("all",        X_graph, X_tpm,  X_fc),
    ]
    abl_results = []
    for name, Xg, Xt, Xf in configs:
        metrics, _ = classify_full_logging(
            Xg, Xt, Xf, y, subject_ids, graph_names, n_splits=5, tag=name)
        abl_results.append({
            "block":   name,
            "auroc":   np.mean([m["auroc"]              for m in metrics]),
            "f1":      np.mean([m["f1"]                 for m in metrics]),
            "sens":    np.mean([m["sensitivity"]        for m in metrics]),
            "spec":    np.mean([m["specificity"]        for m in metrics]),
            "bal_acc": np.mean([m["balanced_accuracy"]  for m in metrics]),
            "mcc":     np.mean([m["mcc"]                for m in metrics]),
        })
    abl_df = pd.DataFrame(abl_results).sort_values("auroc", ascending=False)
    abl_df.to_csv(output_dir / "ablation_results.csv", index=False)
    log.info(f"\nAblation summary:\n{abl_df.to_string(index=False)}")
    return abl_df


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = get_subject_csvs(DATA_ROOT)
    log.info(f"Found {len(records)} subjects: "
             f"{sum(r.label==0 for r in records)} HC, "
             f"{sum(r.label==1 for r in records)} PTSD")

    valid_records, excluded = screen_records(
        records,
        expected_timepoints=TIMEPOINTS,
        expected_rois=N_ROIS,
        log_path=OUTPUT_DIR / "excluded_subjects.log",
    )
    log.info(f"Valid: {len(valid_records)}  Excluded: {len(excluded)}")

    subjects_bold = [
        load_subject_timeseries(r.csv_path, TIMEPOINTS, N_ROIS)
        for r in valid_records
    ]

    X_graph, X_tpm, X_fc, y, subject_ids, graph_names = save_subject_features(
        subjects_bold, valid_records, OUTPUT_DIR
    )

    all_metrics, pred_df = classify_full_logging(
        X_graph, X_tpm, X_fc, y, subject_ids, graph_names,
        n_splits=5, tag="full_pipeline",
    )
    pred_df.to_csv(OUTPUT_DIR / "subject_predictions.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    keys = ["auroc","accuracy","balanced_accuracy","sensitivity",
            "specificity","precision","f1","mcc"]
    summary = {k: {"mean": float(np.mean([m[k] for m in all_metrics])),
                   "std":  float(np.std( [m[k] for m in all_metrics]))}
               for k in keys}
    summary["total_ptsd_misclassified"] = int(sum(m["ptsd_misclassified"] for m in all_metrics))
    summary["total_hc_misclassified"]   = int(sum(m["hc_misclassified"]   for m in all_metrics))
    (OUTPUT_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    log.info(f"All outputs saved to {OUTPUT_DIR}/")

    run_ablation(X_graph, X_tpm, X_fc, y, subject_ids, graph_names, OUTPUT_DIR)
