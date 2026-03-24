import numpy as np
import networkx as nx
from scipy.stats import zscore, entropy as scipy_entropy
from scipy.signal import find_peaks, butter, filtfilt
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
import glob
import sys

# ── OUTPUT DIR (matches SexClass_STAGIN_ICA.py) ───────────────────────────────
OUTPUT_DIR = Path("/home/dnanexus/out/out/")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "hebbian_graph_sex_classification.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_ICA = 100
TIMEPOINTS = 490
SEED = 26
TR = 0.735

BOLD_LOW_HZ = 0.01
BOLD_HIGH_HZ = 0.10

ACTIVATION_THRESHOLD_PCT = 85.0
HRF_REFRACTORY_TRS = 4

MAX_IAI_TRS = 10
WINDOW_SIZE_TRS = 80
WINDOW_STRIDE_TRS = 20

TPM_PCA_DIM = 50
FC_PCA_DIM = 50

# ── SEX LABEL CONFIG (matches SexClass_STAGIN_ICA.py) ────────────────────────
# CSV with at least [EID, sex_31] columns (UK Biobank encoding: 0=Female, 1=Male)
SEX_CSV_PATH = "ICA_Pearson_Full_Features.csv"
INDEX_TO_LABEL = {0: "Female", 1: "Male"}

# ── LABEL LOADING ─────────────────────────────────────────────────────────────

def load_sex_labels(sex_csv_path: str) -> Dict[str, int]:
    """Return {eid_str: sex_int} from the phenotype CSV.

    Reads only EID and sex_31 columns; sex_31: 0=Female, 1=Male (UK Biobank).
    """
    df = pd.read_csv(sex_csv_path, usecols=lambda c: c in {"EID", "eid", "sex_31"})
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"eid": "EID"})
    df = df.dropna(subset=["EID", "sex_31"])
    df["EID"] = df["EID"].astype(int).astype(str).str.zfill(7)
    df["sex_31"] = df["sex_31"].astype(int)
    return dict(zip(df["EID"], df["sex_31"]))

# ── DATA LOADING ──────────────────────────────────────────────────────────────

@dataclass
class SubjectRecord:
    csv_path: str
    subject_id: str   # 7-digit EID string
    class_name: str   # "Female" or "Male"
    label: int        # 0 or 1

def build_records_from_files(
    txt_files: List[str],
    sex_labels: Dict[str, int],
) -> Tuple[List[SubjectRecord], List[str]]:
    """Match .txt ICA files to sex labels via the first 7 characters of the filename."""
    records: List[SubjectRecord] = []
    skipped: List[str] = []
    for fpath in sorted(txt_files):
        stem = Path(fpath).stem
        eid = stem[:7]
        if eid not in sex_labels:
            skipped.append(f"{fpath} | reason=EID {eid!r} not in sex CSV")
            continue
        label = sex_labels[eid]
        records.append(SubjectRecord(
            csv_path=str(Path(fpath).resolve()),
            subject_id=eid,
            class_name=INDEX_TO_LABEL[label],
            label=label,
        ))
    return records, skipped

def load_subject_timeseries(
    csv_path: str,
    expected_tp: int = TIMEPOINTS,
    expected_ica: int = N_ICA,
) -> np.ndarray:
    df = pd.read_csv(csv_path, sep=r"\s+", header=None)
    values = (df.apply(pd.to_numeric, errors="coerce")
              .dropna(axis=0, how="all").dropna(axis=1, how="all")
              .to_numpy(dtype=np.float32))
    if values.shape == (expected_tp, expected_ica):
        return values.T
    if values.shape == (expected_ica, expected_tp):
        return values
    raise ValueError(f"Unexpected shape {values.shape} in {csv_path}")

def screen_records(
    records: List[SubjectRecord],
    log_path: Path,
) -> Tuple[List[SubjectRecord], List[str]]:
    retained, excluded = [], []
    for rec in records:
        try:
            load_subject_timeseries(rec.csv_path, TIMEPOINTS, N_ICA)
            retained.append(rec)
        except Exception as e:
            excluded.append(f"{rec.csv_path} | {e}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Excluded {len(excluded)} of {len(records)} subjects\n\n")
        f.writelines(f"{entry}\n" for entry in excluded)
    return retained, excluded

# ── PREPROCESSING ─────────────────────────────────────────────────────────────

def bandpass_and_zscore(Z: np.ndarray, tr: float = TR) -> np.ndarray:
    nyq = 0.5 / tr
    low = max(BOLD_LOW_HZ / nyq, 1e-4)
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

# ── ACTIVATION DETECTION ──────────────────────────────────────────────────────

def detect_activations_bold(
    Z: np.ndarray,
    threshold_pct: float = ACTIVATION_THRESHOLD_PCT,
    refractory_trs: int = HRF_REFRACTORY_TRS,
) -> List[Tuple[int, int]]:
    N_caps = Z.shape[0]
    thresholds = np.percentile(Z, threshold_pct, axis=1)
    events: List[Tuple[int, int]] = []
    for k in range(N_caps):
        peaks, _ = find_peaks(Z[k], height=thresholds[k], distance=refractory_trs)
        for t in peaks:
            events.append((k, int(t)))
    events.sort(key=lambda x: x[1])
    return events

# ── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────

def extract_window_events(events, t_start, t_end):
    return [(k, t) for k, t in events if t_start <= t < t_end]

def build_graph_from_window(
    window_events: List[Tuple[int, int]],
    max_iai_trs: int = MAX_IAI_TRS,
) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(N_ICA))
    if len(window_events) < 2:
        return G
    evts = sorted(window_events, key=lambda x: x[1])
    tau = max_iai_trs / 2.0
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
                G[ki][kj]["count"] += 1
            else:
                G.add_edge(ki, kj, weight=w, count=1)
    return G

# ── GRAPH FEATURE EXTRACTION ─────────────────────────────────────────────────

def extract_graph_features(G: nx.DiGraph, n_activations: int) -> Dict[str, float]:
    f: Dict[str, float] = {}
    norm = max(n_activations, 1)
    Gu = nx.Graph(G.to_undirected())

    active_nodes = [n for n in G.nodes() if G.degree(n) > 0]
    f["ActiveNodes"] = len(active_nodes) / N_ICA
    f["Edges"] = G.number_of_edges() / norm
    f["Density"] = nx.density(G)

    _nan_keys = ["RE","PE","L1","L2","L3","LCC","LSC","ATD",
                 "Diameter","ASP","CC","Transitivity","Reciprocity",
                 "MeanWeight","StdWeight","WeightEntropy",
                 "InDegreeGini","OutDegreeGini","HubScore"]

    if G.number_of_edges() == 0:
        for k in _nan_keys:
            f[k] = np.nan
        return f

    edge_counts = {}
    for u, v in G.edges():
        edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
    f["RE"] = sum(c - 1 for c in edge_counts.values() if c > 1) / norm
    pe = sum(1 for u, v in G.edges() if G.has_edge(v, u)) / 2
    f["PE"] = pe / norm
    f["L1"] = 0.0

    l2 = sum(1 for u in G.nodes() for v in G.successors(u)
             if u < v and G.has_edge(v, u))
    f["L2"] = l2 / norm
    l3 = sum(1 for u in G.nodes() for v in G.successors(u) if v != u
             for w in G.successors(v) if w not in (u, v) and G.has_edge(w, u))
    f["L3"] = l3 / norm

    ccs = list(nx.connected_components(Gu))
    f["LCC"] = max(len(c) for c in ccs) / N_ICA if ccs else 0.0
    sccs = list(nx.strongly_connected_components(G))
    f["LSC"] = max(len(s) for s in sccs) / N_ICA if sccs else 0.0

    degs = [d for _, d in G.degree()]
    f["ATD"] = float(np.mean(degs)) / N_ICA

    try:
        sub = Gu if nx.is_connected(Gu) else Gu.subgraph(max(ccs, key=len))
        f["Diameter"] = nx.diameter(sub) / N_ICA
        f["ASP"] = nx.average_shortest_path_length(sub) / N_ICA
    except Exception:
        f["Diameter"] = np.nan
        f["ASP"] = np.nan

    f["CC"] = nx.average_clustering(Gu)
    f["Transitivity"] = nx.transitivity(Gu)
    try:
        f["Reciprocity"] = nx.overall_reciprocity(G)
    except Exception:
        f["Reciprocity"] = np.nan

    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    f["MeanWeight"] = float(np.mean(weights))
    f["StdWeight"] = float(np.std(weights))
    w_norm = weights / (weights.sum() + 1e-12)
    f["WeightEntropy"] = float(scipy_entropy(w_norm + 1e-12))

    in_degs = np.array([G.in_degree(n) for n in G.nodes()], dtype=float)
    out_degs = np.array([G.out_degree(n) for n in G.nodes()], dtype=float)

    def gini(x):
        if x.sum() == 0:
            return 0.0
        x = np.sort(x)
        n = len(x)
        return float((2 * np.sum(np.arange(1, n+1) * x) / (n * x.sum())) - (n+1)/n)

    f["InDegreeGini"] = gini(in_degs)
    f["OutDegreeGini"] = gini(out_degs)
    f["HubScore"] = float(out_degs.max() / (out_degs.mean() + 1e-12))
    return f

# ── TPM AND FC ────────────────────────────────────────────────────────────────

def compute_tpm(events, max_iai_trs=MAX_IAI_TRS, n=N_ICA):
    counts = np.zeros((n, n), dtype=np.float32)
    evts = sorted(events, key=lambda x: x[1])
    for i in range(len(evts)):
        ki, ti = evts[i]
        for j in range(i + 1, len(evts)):
            kj, tj = evts[j]
            if tj - ti > max_iai_trs:
                break
            if ki != kj:
                counts[ki, kj] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tpm = counts / row_sums
    idx = np.triu_indices(n, k=1)
    return tpm[idx].astype(np.float32)

def compute_fc(Z):
    corr = np.corrcoef(Z)
    idx = np.triu_indices(corr.shape[0], k=1)
    return corr[idx].astype(np.float32)

# ── SUBJECT-LEVEL FEATURE EXTRACTION ─────────────────────────────────────────

def extract_subject_features(bold_array: np.ndarray, tr: float = TR) -> Dict:
    N_trs = bold_array.shape[1]
    Z = bandpass_and_zscore(bold_array, tr)
    events = detect_activations_bold(Z)

    window_feature_dicts = []
    for t_start in range(0, N_trs - WINDOW_SIZE_TRS + 1, WINDOW_STRIDE_TRS):
        win_evts = extract_window_events(events, t_start, t_start + WINDOW_SIZE_TRS)
        G = build_graph_from_window(win_evts, MAX_IAI_TRS)
        feat = extract_graph_features(G, len(win_evts))
        window_feature_dicts.append(feat)

    feat_keys = sorted(window_feature_dicts[0].keys()) if window_feature_dicts else []
    graph_feats = []
    graph_names = []
    for key in feat_keys:
        vals = np.array([d.get(key, np.nan) for d in window_feature_dicts], dtype=np.float64)
        vc = vals[~np.isnan(vals)]
        mean_v = float(np.mean(vc)) if len(vc) else np.nan
        std_v = float(np.std(vc)) if len(vc) > 1 else np.nan
        if len(vc) > 5:
            hist, _ = np.histogram(vc, bins=min(10, len(vc)//2))
            hn = hist / (hist.sum() + 1e-12)
            ent_v = float(scipy_entropy(hn + 1e-12))
        else:
            ent_v = np.nan
        graph_feats.extend([mean_v, std_v, ent_v])
        graph_names.extend([f"{key}_mean", f"{key}_std", f"{key}_ent"])

    return {
        "graph": np.array(graph_feats, dtype=np.float32),
        "graph_names": graph_names,
        "tpm": compute_tpm(events),
        "fc": compute_fc(Z),
    }

# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred, y_prob, fold=None) -> Dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "fold": fold,
        "auroc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_male": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity_female": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_male": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "male_misclassified": int(fn),
        "female_misclassified": int(fp),
        "n_test": len(y_true),
    }

def log_fold_metrics(m: Dict):
    log.info(
        f"  Fold {m['fold']:>2} | "
        f"AUROC={m['auroc']:.3f} Acc={m['accuracy']:.3f} "
        f"BalAcc={m['balanced_accuracy']:.3f} "
        f"Sens(M)={m['sensitivity_male']:.3f} Spec(F)={m['specificity_female']:.3f} "
        f"F1={m['f1_male']:.3f} MCC={m['mcc']:.3f} | "
        f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']} "
        f"[Male→Female={m['male_misclassified']} | Female→Male={m['female_misclassified']}]"
    )

def log_aggregate_metrics(all_metrics: List[Dict], label: str = ""):
    keys = ["auroc","accuracy","balanced_accuracy","sensitivity_male",
            "specificity_female","precision","f1_male","mcc"]
    log.info(f"\n{'─'*70}")
    log.info(f" AGGREGATE RESULTS {label}")
    log.info(f"{'─'*70}")
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(float(m[k]))]
        log.info(f"  {k:<26}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    log.info(
        f"  {'male_misclassified':<26}: "
        f"{sum(m['male_misclassified'] for m in all_metrics)} / "
        f"{sum(m['tp']+m['fn'] for m in all_metrics)} Male subjects"
    )
    log.info(
        f"  {'female_misclassified':<26}: "
        f"{sum(m['female_misclassified'] for m in all_metrics)} / "
        f"{sum(m['tn']+m['fp'] for m in all_metrics)} Female subjects"
    )
    log.info(f"{'─'*70}\n")

# ── SAFE PCA HELPER ───────────────────────────────────────────────────────────

def safe_pca(X_tr, X_te, n_components, random_state=SEED):
    n_comp = min(n_components, X_tr.shape[0] - 1, X_tr.shape[1] - 1)
    if n_comp < 1:
        return X_tr, X_te
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(X_tr), pca.transform(X_te)

# ── SAFE GRAPH FEATURE SELECTION ─────────────────────────────────────────────

def select_graph_features(Xg_tr, Xg_te, y_tr):
    if Xg_tr.shape[1] <= 1:
        return Xg_tr, Xg_te
    try:
        vt = VarianceThreshold(threshold=1e-4)
        Xg_tr = vt.fit_transform(Xg_tr)
        Xg_te = vt.transform(Xg_te)
    except ValueError:
        pass
    if Xg_tr.shape[1] == 0:
        return np.zeros((Xg_tr.shape[0], 1)), np.zeros((Xg_te.shape[0], 1))
    k_sel = min(40, Xg_tr.shape[1])
    sel = SelectKBest(mutual_info_classif, k=k_sel)
    Xg_tr = sel.fit_transform(Xg_tr, y_tr)
    Xg_te = sel.transform(Xg_te)
    return Xg_tr, Xg_te

# ── CLASSIFICATION WITH FULL LOGGING ─────────────────────────────────────────

def classify_full_logging(
    X_graph: np.ndarray,
    X_tpm: np.ndarray,
    X_fc: np.ndarray,
    y: np.ndarray,
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
    has_tpm   = X_tpm.shape[1] > 0
    has_fc    = X_fc.shape[1] > 0

    log.info(f"\n{'═'*70}")
    log.info(f" CLASSIFICATION: {tag}")
    log.info(
        f" Subjects={len(y)} Folds={n_splits} "
        f"Male={y.sum()} Female={(y==0).sum()}"
    )
    log.info(
        f" Blocks active — graph:{has_graph} ({X_graph.shape[1]}) "
        f"tpm:{has_tpm} ({X_tpm.shape[1]}) "
        f"fc:{has_fc} ({X_fc.shape[1]})"
    )
    log.info(
        f" Params — threshold_pct={ACTIVATION_THRESHOLD_PCT} "
        f"max_iai={MAX_IAI_TRS}TRs "
        f"window={WINDOW_SIZE_TRS}TRs stride={WINDOW_STRIDE_TRS}TRs"
    )
    log.info(f"{'═'*70}")

    for fold, (tr_idx, te_idx) in enumerate(cv.split(
            X_graph if has_graph else X_tpm, y), 1):
        blocks_tr, blocks_te = [], []

        if has_graph:
            imp = SimpleImputer(strategy="median")
            Xg_tr = imp.fit_transform(X_graph[tr_idx])
            Xg_te = imp.transform(X_graph[te_idx])
            Xg_tr, Xg_te = select_graph_features(Xg_tr, Xg_te, y[tr_idx])
            blocks_tr.append(Xg_tr)
            blocks_te.append(Xg_te)

        if has_tpm:
            imp = SimpleImputer(strategy="median")
            Xt_tr = imp.fit_transform(X_tpm[tr_idx])
            Xt_te = imp.transform(X_tpm[te_idx])
            Xt_tr, Xt_te = safe_pca(Xt_tr, Xt_te, TPM_PCA_DIM, random_state)
            blocks_tr.append(Xt_tr)
            blocks_te.append(Xt_te)

        if has_fc:
            imp = SimpleImputer(strategy="median")
            Xf_tr = imp.fit_transform(X_fc[tr_idx])
            Xf_te = imp.transform(X_fc[te_idx])
            Xf_tr, Xf_te = safe_pca(Xf_tr, Xf_te, FC_PCA_DIM, random_state)
            blocks_tr.append(Xf_tr)
            blocks_te.append(Xf_te)

        X_tr = np.concatenate(blocks_tr, axis=1)
        X_te = np.concatenate(blocks_te, axis=1)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=3, class_weight="balanced",
            random_state=random_state, n_jobs=-1)
        lr = LogisticRegression(
            C=0.1, max_iter=2000, class_weight="balanced",
            random_state=random_state, solver="lbfgs")
        svm = SVC(
            kernel="rbf", probability=True, class_weight="balanced",
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
        "true_label": [INDEX_TO_LABEL[yi] for yi in y],
        "predicted":  [INDEX_TO_LABEL[pi] for pi in subj_preds],
        "prob_male":  subj_probs,
        "correct":    (subj_preds == y).astype(int),
    })
    return all_metrics, pred_df

# ── FEATURE SAVING ────────────────────────────────────────────────────────────

def save_subject_features(subjects_bold, records, output_dir):
    log.info("\n=== EXTRACTING FEATURES FOR ALL SUBJECTS ===")

    all_graph, all_tpm, all_fc = [], [], []
    graph_names = None
    subject_ids  = [r.subject_id for r in records]
    labels_str   = [r.class_name  for r in records]
    y            = np.array([r.label for r in records], dtype=int)

    for idx, (bold, rec) in enumerate(zip(subjects_bold, records)):
        if idx % 50 == 0:
            log.info(f"  Extracting subject {idx+1}/{len(records)}")
        feats = extract_subject_features(bold)
        all_graph.append(feats["graph"])
        all_tpm.append(feats["tpm"])
        all_fc.append(feats["fc"])
        if graph_names is None:
            graph_names = feats["graph_names"]

    X_graph = np.array(all_graph, dtype=np.float32)
    X_tpm   = np.array(all_tpm,   dtype=np.float32)
    X_fc    = np.array(all_fc,    dtype=np.float32)

    log.info(f"  Graph: {X_graph.shape} | TPM: {X_tpm.shape} | FC: {X_fc.shape}")

    meta = pd.DataFrame({"subject_id": subject_ids, "sex": labels_str})

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
    )

    log.info(f"  All features saved to {output_dir}/")
    return X_graph, X_tpm, X_fc, y, subject_ids, graph_names

# ── ABLATION STUDY ────────────────────────────────────────────────────────────

def run_ablation(X_graph, X_tpm, X_fc, y, subject_ids, graph_names, output_dir):
    log.info("\n" + "═"*70)
    log.info(" ABLATION STUDY")
    log.info("═"*70)

    N = len(y)
    empty = np.zeros((N, 0), dtype=np.float32)

    configs = [
        ("graph_only",  X_graph, empty,   empty),
        ("tpm_only",    empty,   X_tpm,   empty),
        ("fc_only",     empty,   empty,   X_fc),
        ("graph+tpm",   X_graph, X_tpm,   empty),
        ("graph+fc",    X_graph, empty,   X_fc),
        ("tpm+fc",      empty,   X_tpm,   X_fc),
        ("all",         X_graph, X_tpm,   X_fc),
    ]

    abl_results = []
    for name, Xg, Xt, Xf in configs:
        metrics, _ = classify_full_logging(
            Xg, Xt, Xf, y, subject_ids, graph_names,
            n_splits=5, tag=name,
        )
        abl_results.append({
            "block":    name,
            "auroc":    np.mean([m["auroc"]             for m in metrics]),
            "f1_male":  np.mean([m["f1_male"]           for m in metrics]),
            "sens_male": np.mean([m["sensitivity_male"] for m in metrics]),
            "spec_fem": np.mean([m["specificity_female"]for m in metrics]),
            "bal_acc":  np.mean([m["balanced_accuracy"] for m in metrics]),
            "mcc":      np.mean([m["mcc"]               for m in metrics]),
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

    # ── 1. Load sex labels from phenotype CSV ─────────────────────────────────
    log.info(f"Loading sex labels from: {SEX_CSV_PATH}")
    sex_labels = load_sex_labels(SEX_CSV_PATH)
    log.info(f"  {len(sex_labels)} subjects with sex_31 labels loaded.")

    # ── 2. Collect input .txt files (CLI args or current-dir glob) ────────────
    #  Swiss Army Knife passes all selected files as positional CLI arguments.
    if len(sys.argv) > 1:
        txt_files = sys.argv[1:]
    else:
        txt_files = sorted(glob.glob("*.txt", recursive=True))
    log.info(f"  {len(txt_files)} .txt ICA timeseries files supplied.")

    # ── 3. Match files to sex labels via 7-digit EID prefix ──────────────────
    records, skipped = build_records_from_files(txt_files, sex_labels)

    skipped_log = OUTPUT_DIR / "skipped_no_sex_label.txt"
    skipped_log.parent.mkdir(parents=True, exist_ok=True)
    with open(skipped_log, "w", encoding="utf-8") as f:
        f.write(f"Skipped {len(skipped)} files (EID not found in sex CSV)\n\n")
        f.writelines(f"{s}\n" for s in skipped)
    log.info(f"  {len(records)} files matched to sex labels; {len(skipped)} skipped.")

    # ── 4. Screen for correct shape ───────────────────────────────────────────
    records, _ = screen_records(
        records=records,
        log_path=OUTPUT_DIR / "excluded_subjects.txt",
    )
    log.info(f"  {len(records)} subjects retained after shape screening.")

    class_counts = pd.Series([r.class_name for r in records]).value_counts().to_dict()
    log.info(f"Class counts: {class_counts}")

    subjects_bold = [load_subject_timeseries(r.csv_path) for r in records]

    # ── 5. Extract features ───────────────────────────────────────────────────
    X_graph, X_tpm, X_fc, y, subject_ids, graph_names = save_subject_features(
        subjects_bold, records, OUTPUT_DIR
    )

    # ── 6. Full pipeline classification ──────────────────────────────────────
    all_metrics, pred_df = classify_full_logging(
        X_graph, X_tpm, X_fc, y, subject_ids, graph_names,
        n_splits=5, tag="full_pipeline",
    )

    pred_df.to_csv(OUTPUT_DIR / "all_fold_predictions.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    keys = ["auroc","accuracy","balanced_accuracy","sensitivity_male",
            "specificity_female","precision","f1_male","mcc"]
    summary = {
        k: {
            "mean": float(np.mean([m[k] for m in all_metrics])),
            "std":  float(np.std( [m[k] for m in all_metrics])),
        }
        for k in keys
    }
    summary["total_male_misclassified"]   = int(sum(m["male_misclassified"]   for m in all_metrics))
    summary["total_female_misclassified"] = int(sum(m["female_misclassified"] for m in all_metrics))
    (OUTPUT_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Saved all outputs to {OUTPUT_DIR}/")

    # ── 7. Ablation study ─────────────────────────────────────────────────────
    run_ablation(X_graph, X_tpm, X_fc, y, subject_ids, graph_names, OUTPUT_DIR)
