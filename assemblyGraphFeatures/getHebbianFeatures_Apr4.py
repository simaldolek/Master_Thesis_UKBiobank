import warnings
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Optional
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA, FastICA
from itertools import combinations

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TR           = 0.735
N_ICA        = 100
TIMEPOINTS   = 490
BOLD_LOW_HZ  = 0.01
BOLD_HIGH_HZ = 0.10
SEED         = 26
BIN_TRS      = 27

ACT_THRESHOLDS = [99, 90, 85]

OUTPUT_DIR   = Path("/home/dnanexus/out/out/")
SEX_CSV_PATH = "ICA_Pearson_Full_Features.csv"

INDEX_TO_SEX  = {0: "Female", 1: "Male"}
INDEX_TO_PTSD = {0: "Healthy", 1: "PTSD"}


# ── LABEL LOADING ─────────────────────────────────────────────────────────────
def load_labels(csv_path: str) -> Dict[str, dict]:
    wanted = {"EID", "eid", "sex_31", "ptsd"}
    df = pd.read_csv(csv_path, usecols=lambda c: c in wanted)
    df.columns = [c.lower() for c in df.columns]
    if "eid" not in df.columns:
        raise ValueError(f"No EID/eid column found in {csv_path}")

    df = df.rename(columns={"eid": "EID"})
    df = df.dropna(subset=["EID"])
    df["EID"] = df["EID"].astype(int).astype(str).str.zfill(7)

    label_map: Dict[str, dict] = {}
    for _, row in df.iterrows():
        entry: dict = {}
        if "sex_31" in df.columns and pd.notna(row.get("sex_31")):
            s = int(row["sex_31"])
            entry["sex"]       = s
            entry["sex_label"] = INDEX_TO_SEX.get(s, "Unknown")
        else:
            entry["sex"]       = np.nan
            entry["sex_label"] = "Unknown"

        if "ptsd" in df.columns and pd.notna(row.get("ptsd")):
            p = int(row["ptsd"])
            entry["ptsd"]       = p
            entry["ptsd_label"] = INDEX_TO_PTSD.get(p, "Unknown")
        else:
            entry["ptsd"]       = np.nan
            entry["ptsd_label"] = "Unknown"

        label_map[row["EID"]] = entry

    return label_map


# ── I/O & PREPROCESSING ───────────────────────────────────────────────────────
def load_timeseries(path, expected_tp=TIMEPOINTS, expected_ica=N_ICA):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    vals = (df.apply(pd.to_numeric, errors="coerce")
              .dropna(axis=0, how="all").dropna(axis=1, how="all")
              .to_numpy(dtype=np.float32))
    if vals.shape == (expected_tp, expected_ica):
        return vals.T
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
    p, n   = Z.shape
    n_bins = n // bin_size_trs
    return Z[:, :n_bins * bin_size_trs].reshape(p, n_bins, bin_size_trs).mean(axis=2)


# ── MARCHENKO-PASTUR THRESHOLD ────────────────────────────────────────────────
def mp_upper(p, n, sigma2=1.0):
    return sigma2 * (1.0 + np.sqrt(p / n)) ** 2


# ── ASSEMBLY DETECTION ────────────────────────────────────────────────────────
def detect_assemblies(Z_binned, seed=SEED):
    p, n = Z_binned.shape
    if n < 2 or p < 2:
        return np.empty((0, p)), 0, None, None

    cov     = np.cov(Z_binned)
    eigvals = np.linalg.eigvalsh(cov)
    sigma2  = float(np.median(eigvals))
    if sigma2 <= 0:
        sigma2 = 1.0
    lp    = mp_upper(p, n, sigma2)
    n_sig = int(np.sum(eigvals > lp))
    n_sig = min(n_sig, min(p, n) - 1)

    if n_sig == 0:
        return np.empty((0, p)), 0, None, None

    pca   = PCA(n_components=n_sig, random_state=seed)
    Z_pca = pca.fit_transform(Z_binned.T)

    ica = FastICA(n_components=n_sig, max_iter=2000, tol=1e-4,
                  random_state=seed, whiten="unit-variance")
    try:
        ica.fit(Z_pca)
    except Exception:
        return np.empty((0, p)), 0, pca, None

    mixing       = ica.mixing_
    pca_comps    = pca.components_
    patterns_raw = mixing.T @ pca_comps

    norms = np.linalg.norm(patterns_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    patterns = patterns_raw / norms

    return patterns, n_sig, pca, ica


# ── ASSEMBLY ACTIVITY PROJECTIONS ─────────────────────────────────────────────
def project_assembly_activity_linear(Z, patterns):
    return patterns @ Z


def project_assembly_activity_quadratic(Z, patterns):
    n_assemblies = len(patterns)
    n_timepoints = Z.shape[1]
    act = np.zeros((n_assemblies, n_timepoints))
    for i, w in enumerate(patterns):
        P = np.outer(w, w)
        np.fill_diagonal(P, 0)
        act[i] = np.einsum("ib,ij,jb->b", Z, P, Z)
    return act


# ── GRAPH CONSTRUCTION & ATTRIBUTES ──────────────────────────────────────────
def get_activation_sequence(act_matrix, percentile):
    n_assemblies, n_tp = act_matrix.shape
    thresholds = np.percentile(act_matrix, percentile, axis=1)
    active     = act_matrix > thresholds[:, None]
    sequence   = []
    for t in range(n_tp):
        fired = np.where(active[:, t])[0].tolist()
        if len(fired) == 1:
            sequence.append((fired[0], t))
        elif len(fired) > 1:
            sequence.append((tuple(sorted(fired)), t))
    return sequence


MAX_IAI_TRS    = [10, 20, 50, 100, 200]
ACT_COUNT_PCTS = [10, 20, 50, 100, 120, 150, 200]


def build_assembly_graph(sequence, max_iai_trs):
    G = nx.DiGraph()
    if len(sequence) < 2:
        return G
    for label, _ in sequence:
        if not G.has_node(label):
            G.add_node(label)
    for i in range(len(sequence) - 1):
        label_a, t_a = sequence[i]
        label_b, t_b = sequence[i + 1]
        if (t_b - t_a) <= max_iai_trs:
            G.add_edge(label_a, label_b, iai=(t_b - t_a))
    return G


def compute_graph_attributes(G):
    if G.number_of_nodes() == 0:
        return {k: np.nan for k in
                ["Nodes", "RE", "PE", "L1", "L2", "L3",
                 "LCC", "LSC", "ATD", "Density", "Diameter", "ASP", "CC"]}

    attrs = {"Nodes": G.number_of_nodes()}

    edge_counts = {}
    for u, v in G.edges():
        edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1

    attrs["L1"] = sum(1 for (u, v) in edge_counts if u == v)
    attrs["RE"] = sum(1 for (u, v), c in edge_counts.items() if u != v and c > 1)

    undirected_pairs = {}
    for (u, v), c in edge_counts.items():
        if u == v:
            continue
        key = tuple(sorted([str(u), str(v)]))
        undirected_pairs[key] = undirected_pairs.get(key, 0) + c
    attrs["PE"] = sum(1 for total in undirected_pairs.values() if total > 1)

    attrs["L2"] = sum(
        1 for (u, v) in edge_counts if u != v and (v, u) in edge_counts
    ) // 2

    simple_G   = nx.DiGraph(G)
    nodes_list = list(simple_G.nodes())
    l3 = 0
    for a, b, c in combinations(nodes_list, 3):
        if simple_G.has_edge(a, b) and simple_G.has_edge(b, c) and simple_G.has_edge(c, a):
            l3 += 1
        if simple_G.has_edge(a, c) and simple_G.has_edge(c, b) and simple_G.has_edge(b, a):
            l3 += 1
    attrs["L3"] = l3

    undirected = G.to_undirected()
    wcc        = list(nx.connected_components(undirected))
    attrs["LCC"] = max(len(c) for c in wcc) if wcc else 0
    scc          = list(nx.strongly_connected_components(G))
    attrs["LSC"] = max(len(c) for c in scc) if scc else 0

    total_degrees = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
    attrs["ATD"]  = float(np.mean(total_degrees)) if total_degrees else 0.0

    n            = G.number_of_nodes()
    max_edges    = n * (n - 1)
    actual_edges = sum(c for (u, v), c in edge_counts.items() if u != v)
    attrs["Density"] = actual_edges / max_edges if max_edges > 0 else 0.0

    largest_wcc_nodes = max(wcc, key=len) if wcc else set()
    sub = undirected.subgraph(largest_wcc_nodes)
    if nx.is_connected(sub) and sub.number_of_nodes() > 1:
        attrs["Diameter"] = nx.diameter(sub)
        attrs["ASP"]      = nx.average_shortest_path_length(sub)
    else:
        attrs["Diameter"] = np.nan
        attrs["ASP"]      = np.nan

    attrs["CC"] = nx.average_clustering(undirected)
    return attrs


def build_graph_features(act_matrix, act_type_label, tag, n_assemblies):
    rows = []
    for pct_thr in ACT_THRESHOLDS:
        seq = get_activation_sequence(act_matrix, pct_thr)
        for max_iai in MAX_IAI_TRS:
            for act_pct in ACT_COUNT_PCTS:
                target_count = max(2, int((act_pct / 100.0) * (n_assemblies ** 2)))
                window_attrs = []
                i = 0
                while i + target_count <= len(seq):
                    window = seq[i : i + target_count]
                    valid  = all(
                        window[j+1][1] - window[j][1] <= max_iai
                        for j in range(len(window) - 1)
                    )
                    if valid:
                        G    = build_assembly_graph(window, max_iai)
                        attr = compute_graph_attributes(G)
                        window_attrs.append(attr)
                    i += target_count

                if not window_attrs:
                    continue

                mean_attr = {k: float(np.nanmean([a[k] for a in window_attrs]))
                             for k in window_attrs[0]}

                rows.append({
                    "subject_tag"     : tag,
                    "activity_type"   : act_type_label,
                    "threshold_pct"   : pct_thr,
                    "max_iai_trs"     : max_iai,
                    "max_iai_seconds" : round(max_iai * TR, 3),
                    "activation_count": target_count,
                    "act_count_pct"   : act_pct,
                    "n_windows"       : len(window_attrs),
                    "n_assemblies"    : n_assemblies,
                    **mean_attr,
                })
    return rows


# ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────
def run_analysis(subject_file, tr_bin, subject_id=None,
                 label_map: Optional[Dict[str, dict]] = None):

    if subject_id is None:
        subject_id = Path(subject_file).stem[:7]

    print(f"\n{'='*65}")
    print(f"  Subject : {subject_id}")
    print(f"  File    : {subject_file}")
    print(f"  TR bin  : {tr_bin} TRs  ({tr_bin * TR:.2f} s)")
    print(f"{'='*65}\n")

    subj_labels = label_map.get(subject_id, {}) if label_map else {}
    sex_val     = subj_labels.get("sex",        np.nan)
    sex_label   = subj_labels.get("sex_label",  "Unknown")
    ptsd_val    = subj_labels.get("ptsd",       np.nan)
    ptsd_label  = subj_labels.get("ptsd_label", "Unknown")

    print(f"  Sex  : {sex_label} ({sex_val})")
    print(f"  PTSD : {ptsd_label} ({ptsd_val})\n")

    raw   = load_timeseries(subject_file)
    Z     = bandpass_and_zscore(raw)
    Z_bin = bin_timeseries(Z, tr_bin)

    print(f"  Binned matrix shape: {Z_bin.shape}  "
          f"[{Z_bin.shape[1]} bins x {Z_bin.shape[0]} components]")

    print("\n  Detecting assemblies...")
    pat_tr, n_tr, pca_tr, ica_tr = detect_assemblies(Z_bin)
    print(f"    {n_tr} assemblies detected")

    if n_tr == 0:
        print("\n  WARNING: No assemblies detected. Skipping subject.")
        return None

    act_linear    = project_assembly_activity_linear(Z, pat_tr)
    act_quadratic = project_assembly_activity_quadratic(Z, pat_tr)

    tag = f"{subject_id}_tr{tr_bin}"

    print("\n  Building assembly graphs and extracting features...")
    all_rows = []
    print("  -> Linear activity:")
    all_rows.extend(build_graph_features(act_linear,    "linear",    tag, n_tr))
    print("  -> Quadratic activity:")
    all_rows.extend(build_graph_features(act_quadratic, "quadratic", tag, n_tr))

    for row in all_rows:
        row["EID"]        = subject_id
        row["sex"]        = sex_val
        row["sex_label"]  = sex_label
        row["ptsd"]       = ptsd_val
        row["ptsd_label"] = ptsd_label

    print(f"  Done: {len(all_rows)} feature rows generated.")
    return all_rows


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob
    import sys

    label_map = None
    try:
        label_map = load_labels(SEX_CSV_PATH)
        print(f"Loaded labels for {len(label_map)} subjects from {SEX_CSV_PATH}")
    except Exception as e:
        print(f"WARNING: Could not load labels: {e}. Continuing without.")

    if len(sys.argv) > 1:
        txt_files = sys.argv[1:]
    else:
        txt_files = sorted(glob.glob("*.txt", recursive=True))
    print(f"  {len(txt_files)} .txt ICA timeseries files found.")

    if not txt_files:
        print("ERROR: No subject files found.")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_subjects_rows = []
    failed = []

    for i, fpath in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] {fpath}")
        try:
            result = run_analysis(
                subject_file=fpath,
                tr_bin=BIN_TRS,
                subject_id=None,
                label_map=label_map,
            )
            if result:
                all_subjects_rows.extend(result)
        except Exception as e:
            print(f"  ERROR processing {fpath}: {e}")
            failed.append((fpath, str(e)))

    # ── Save single combined CSV ──────────────────────────────────────────────
    if all_subjects_rows:
        all_df = pd.DataFrame(all_subjects_rows)

        id_cols = ["EID", "subject_tag", "activity_type", "threshold_pct",
                   "max_iai_trs", "max_iai_seconds", "activation_count",
                   "act_count_pct", "n_windows", "n_assemblies"]
        label_cols = ["sex", "sex_label", "ptsd", "ptsd_label"]
        feat_cols  = [c for c in all_df.columns
                      if c not in id_cols and c not in label_cols]

        all_df = all_df[id_cols + feat_cols + label_cols]

        out_path = OUTPUT_DIR / "graph_features_ALL_subjects.csv"
        all_df.to_csv(out_path, index=False)
        print(f"\n  Saved combined features -> {out_path}")
        print(f"  Shape: {all_df.shape[0]} rows x {all_df.shape[1]} columns")
    else:
        print("\n  WARNING: No features were generated for any subject.")

    print(f"\n{'='*65}")
    print(f"  Done. {len(txt_files) - len(failed)}/{len(txt_files)} subjects completed.")
    if failed:
        print(f"  {len(failed)} failed:")
        for fpath, reason in failed:
            print(f"    {fpath}  ->  {reason}")
        fail_log = OUTPUT_DIR / "failed_subjects.txt"
        with open(fail_log, "w") as f:
            f.writelines(f"{p}\t{r}\n" for p, r in failed)
        print(f"  Failure log -> {fail_log}")
