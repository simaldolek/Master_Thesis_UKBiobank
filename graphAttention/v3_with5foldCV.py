from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# CONFIG
# =============================================================================

N_ROIS = 100
DATA_ROOT = Path("/opt/notebooks/CompTimeSeries/100/")
TIMEPOINTS = 490
OUTPUT_DIR = Path("./stagin_outputs")
SEED = 26
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.10

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Dynamic graph
WINDOW_SIZE = 50
WINDOW_STRIDE = 3
TOP_PERCENT = 30.0
ADD_SELF_LOOPS = True
STANDARDIZE_EACH_ROI = True

# STAGIN architecture
HIDDEN_DIM = 128
NUM_LAYERS = 4
NUM_CLASSES = 2
TEMPORAL_NUM_HEADS = 1
DROPOUT = 0.4
READOUT = "sero"   # "sero" | "garo" | "mean"
TIME_POOL = "sum"  # "sum"  | "mean" | "last" | "param"
REG_LAMBDA = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATA LOADING
# =============================================================================

LABEL_TO_INDEX = {"HC": 0, "PTSD": 1}
INDEX_TO_LABEL = {0: "HC", 1: "PTSD"}


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
        for csv_path in sorted(class_dir.rglob("*.txt")):
            records.append(SubjectRecord(
                csv_path=str(csv_path.resolve()),
                subject_id=csv_path.stem,
                class_name=class_name,
                label=label,
            ))
    return records


def load_subject_timeseries(
    csv_path: str, expected_timepoints: int, expected_rois: int
) -> np.ndarray:
    df = pd.read_csv(csv_path, sep=r"\s+", header=None)
    numeric = (
        df.apply(pd.to_numeric, errors="coerce")
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
    )
    values = numeric.to_numpy(dtype=np.float32)

    if values.shape == (expected_timepoints, expected_rois):
        return values
    if values.shape == (expected_rois, expected_timepoints):
        return values.T
    raise ValueError(
        f"Unexpected shape for {csv_path}: got {values.shape}, "
        f"expected ({expected_timepoints}, {expected_rois}) or transposed"
    )


def screen_records(
    records: List[SubjectRecord],
    expected_timepoints: int,
    expected_rois: int,
    log_path: Path,
) -> Tuple[List[SubjectRecord], List[str]]:
    retained, excluded_paths = [], []
    for record in records:
        try:
            ts = load_subject_timeseries(record.csv_path, expected_timepoints, expected_rois)
            if ts.shape[0] == expected_timepoints:
                retained.append(record)
            else:
                excluded_paths.append(f"{record.csv_path} | shape={ts.shape}")
        except Exception as e:
            excluded_paths.append(f"{record.csv_path} | error={e}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Excluded {len(excluded_paths)} of {len(records)} subjects\n\n")
        f.writelines(f"{entry}\n" for entry in excluded_paths)

    return retained, excluded_paths


def standardize_timeseries(timeseries: np.ndarray) -> np.ndarray:
    mean = timeseries.mean(axis=0, keepdims=True)
    std = timeseries.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (timeseries - mean) / std


# =============================================================================
# DYNAMIC GRAPH CONSTRUCTION
# =============================================================================

def corrcoef_time_by_roi(windowed_timeseries: torch.Tensor) -> torch.Tensor:
    """Compute ROI×ROI Pearson correlation from a [window, roi] matrix."""
    x = windowed_timeseries - windowed_timeseries.mean(dim=0, keepdim=True)
    cov = x.T @ x / max(x.shape[0] - 1, 1)
    std = cov.diag().clamp_min(1e-8).sqrt()
    return (cov / (std[:, None] * std[None, :])).clamp(-1.0, 1.0)


def sliding_window_endpoints(
    total_timepoints: int, window_size: int, window_stride: int
) -> List[int]:
    if total_timepoints <= window_size:
        raise ValueError("total_timepoints must be larger than window_size")
    return [
        start + window_size
        for start in range(0, total_timepoints - window_size + 1, window_stride)
    ]


def build_dynamic_fc_graphs(
    standardized_timeseries: torch.Tensor,
    window_size: int,
    window_stride: int,
    top_percent: float,
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Input : standardized_timeseries [T, N]
    Output: node_identity [W, N, N], adjacency [W, N, N], endpoints list[int]
    """
    total_timepoints, num_rois = standardized_timeseries.shape
    endpoints = sliding_window_endpoints(total_timepoints, window_size, window_stride)

    adjacency_per_window: List[torch.Tensor] = []
    for endpoint in endpoints:
        fc = corrcoef_time_by_roi(standardized_timeseries[endpoint - window_size : endpoint])
        threshold = torch.quantile(fc.reshape(-1), 1.0 - top_percent / 100.0)
        adjacency = (fc > threshold).to(fc.dtype)
        if add_self_loops:
            adjacency.fill_diagonal_(1.0)
        adjacency_per_window.append(adjacency)

    adjacency = torch.stack(adjacency_per_window, dim=0)
    node_identity = (
        torch.eye(num_rois, device=standardized_timeseries.device)
        .unsqueeze(0)
        .repeat(len(endpoints), 1, 1)
    )
    return node_identity, adjacency, endpoints


# =============================================================================
# DATASET
# =============================================================================

class SubjectGraphDataset(Dataset):
    """Loads one subject CSV and builds STAGIN inputs on-the-fly."""

    def __init__(
        self,
        records: Sequence[SubjectRecord],
        n_rois: int,
        timepoints: int,
        window_size: int,
        window_stride: int,
        top_percent: float,
        add_self_loops: bool,
        standardize_each_roi: bool,
    ):
        self.records = list(records)
        self.n_rois = n_rois
        self.timepoints = timepoints
        self.window_size = window_size
        self.window_stride = window_stride
        self.top_percent = top_percent
        self.add_self_loops = add_self_loops
        self.standardize_each_roi = standardize_each_roi
        self.endpoints = sliding_window_endpoints(timepoints, window_size, window_stride)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        timeseries_np = load_subject_timeseries(
            record.csv_path, self.timepoints, self.n_rois
        )
        if self.standardize_each_roi:
            timeseries_np = standardize_timeseries(timeseries_np)

        timeseries = torch.from_numpy(timeseries_np)
        node_identity, adjacency, endpoints = build_dynamic_fc_graphs(
            standardized_timeseries=timeseries,
            window_size=self.window_size,
            window_stride=self.window_stride,
            top_percent=self.top_percent,
            add_self_loops=self.add_self_loops,
        )
        return {
            "node_identity": node_identity,
            "adjacency": adjacency,
            "timeseries": timeseries,
            "label": torch.tensor(record.label, dtype=torch.long),
            "subject_id": record.subject_id,
            "class_name": record.class_name,
            "csv_path": record.csv_path,
            "endpoints": endpoints,
        }


def collate_subject_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    endpoints = batch[0]["endpoints"]
    return {
        "node_identity": torch.stack([s["node_identity"] for s in batch], dim=0),
        "adjacency": torch.stack([s["adjacency"] for s in batch], dim=0),
        "timeseries": torch.stack([s["timeseries"] for s in batch], dim=1),  # [T, B, N]
        "labels": torch.stack([s["label"] for s in batch], dim=0),
        "endpoints": endpoints,
        "subject_ids": [s["subject_id"] for s in batch],
        "class_names": [s["class_name"] for s in batch],
        "csv_paths": [s["csv_path"] for s in batch],
    }


# =============================================================================
# MODEL
# =============================================================================

class TimestampEncoder(nn.Module):
    """GRU that encodes the raw timeseries into window-level embeddings.
    Input:  t_seq [T, B, N]
    Output: [W, B, D]
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, t_seq: torch.Tensor, endpoints: Sequence[int]) -> torch.Tensor:
        if not endpoints:
            raise ValueError("endpoints must not be empty")
        gru_output, _ = self.rnn(t_seq[: endpoints[-1]])
        endpoint_indices = torch.as_tensor([e - 1 for e in endpoints], device=t_seq.device)
        return gru_output[endpoint_indices]


class GINLayer(nn.Module):
    """Sparse GIN update operating on flattened (B*W*N, D) tensors."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.sparse.mm(a, v) + self.eps * v)


class MeanReadout(nn.Module):
    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return node_features.mean(dim=2), torch.zeros(
            node_features.shape[:3], device=node_features.device
        )


class GAROReadout(nn.Module):
    """Graph attention readout (Kim et al., 2021)."""

    def __init__(self, feature_dim: int, dropout: float = 0.1, upscale: float = 1.0):
        super().__init__()
        projected_dim = max(1, round(upscale * feature_dim))
        self.key = nn.Linear(feature_dim, projected_dim)
        self.query = nn.Linear(feature_dim, projected_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.query(node_features.mean(dim=2, keepdim=True))
        scores = torch.matmul(query, self.key(node_features).transpose(-1, -2)).squeeze(2)
        attention = torch.sigmoid(scores / math.sqrt(query.shape[-1]))
        graph_features = (node_features * self.dropout(attention.unsqueeze(-1))).mean(dim=2)
        return graph_features, attention


class SEROReadout(nn.Module):
    """Squeeze-excitation readout (Kim et al., 2021)."""

    def __init__(self, feature_dim: int, num_nodes: int, dropout: float = 0.1, upscale: float = 1.0):
        super().__init__()
        projected_dim = max(1, round(upscale * feature_dim))
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, projected_dim),
            nn.BatchNorm1d(projected_dim),
            nn.GELU(),
        )
        self.attend = nn.Linear(projected_dim, num_nodes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, W, N, D = node_features.shape
        embedded = self.embed(node_features.mean(dim=2).reshape(B * W, D))
        attention = torch.sigmoid(self.attend(embedded)).reshape(B, W, -1)
        graph_features = (node_features * self.dropout(attention.unsqueeze(-1))).mean(dim=2)
        return graph_features, attention


class TemporalTransformer(nn.Module):
    """Single temporal Transformer encoder block."""

    def __init__(self, feature_dim: int, mlp_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attended, time_attention = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        attended = self.norm1(self.dropout1(attended))
        attended = self.norm2(attended + self.dropout2(self.mlp(attended)))
        return attended, time_attention


@dataclass
class STAGINOutput:
    logits: torch.Tensor
    node_attention: torch.Tensor       # [B, K, W, N]
    time_attention: torch.Tensor       # [B, K, W, W]
    layer_latent: torch.Tensor
    final_representation: torch.Tensor
    orthogonality_penalty: torch.Tensor


class STAGIN(nn.Module):
    """Subject-level STAGIN classifier with per-layer logit accumulation."""

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        readout: str = "sero",
        time_pool: str = "sum",
        temporal_num_heads: int = 1,
        dropout: float = 0.1,
        reg_lambda: float = 0.0,
    ):
        super().__init__()
        if readout not in {"sero", "garo", "mean"}:
            raise ValueError("readout must be one of: 'sero', 'garo', 'mean'")
        if time_pool not in {"sum", "mean", "last", "param"}:
            raise ValueError("time_pool must be one of: 'sum', 'mean', 'last', 'param'")

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_pool = time_pool
        self.reg_lambda = reg_lambda

        self.timestamp_encoder = TimestampEncoder(input_dim=num_nodes, hidden_dim=hidden_dim)
        self.initial_linear = nn.Linear(num_nodes + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gin_layers = nn.ModuleList([GINLayer(hidden_dim) for _ in range(num_layers)])

        if readout == "sero":
            self.readouts = nn.ModuleList(
                [SEROReadout(hidden_dim, num_nodes, dropout=dropout) for _ in range(num_layers)]
            )
        elif readout == "garo":
            self.readouts = nn.ModuleList(
                [GAROReadout(hidden_dim, dropout=dropout) for _ in range(num_layers)]
            )
        else:
            self.readouts = nn.ModuleList([MeanReadout() for _ in range(num_layers)])

        self.temporal_blocks = nn.ModuleList([
            TemporalTransformer(
                feature_dim=hidden_dim,
                mlp_dim=2 * hidden_dim,
                num_heads=temporal_num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.time_tokens = (
            nn.Parameter(torch.randn(num_layers, 1, 1, hidden_dim))
            if time_pool == "param"
            else None
        )

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])

    def _collate_adjacency(self, a: torch.Tensor) -> torch.Tensor:
        """Convert dense [B, W, N, N] adjacency to sparse [(B*W*N)×(B*W*N)]."""
        i_list, v_list = [], []
        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                _i = _a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i = _i + sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        size = a.shape[0] * a.shape[1] * a.shape[2]
        return torch.sparse_coo_tensor(_i, _v, (size, size))

    def _pool_time(self, temporal_features: torch.Tensor, layer_index: int) -> torch.Tensor:
        if self.time_pool == "sum":
            return temporal_features.sum(dim=0)
        if self.time_pool == "mean":
            return temporal_features.mean(dim=0)
        return temporal_features[-1]

    def forward(
        self,
        node_identity: torch.Tensor,
        adjacency: torch.Tensor,
        t_seq: torch.Tensor,
        endpoints: Sequence[int],
    ) -> STAGINOutput:
        batch_size, num_windows, num_nodes, _ = node_identity.shape

        encoded_time = self.timestamp_encoder(t_seq, endpoints)          # [W, B, D]
        encoded_time = (
            encoded_time.permute(1, 0, 2)
            .unsqueeze(2)
            .expand(batch_size, num_windows, num_nodes, self.hidden_dim)
        )

        node_features = self.initial_linear(
            torch.cat([node_identity, encoded_time], dim=-1)
        )  # [B, W, N, D]
        h = node_features.reshape(batch_size * num_windows * num_nodes, self.hidden_dim)
        sparse_a = self._collate_adjacency(adjacency)

        logit = 0.0
        ortho_penalty = node_features.new_tensor(0.0)
        per_layer_latents: List[torch.Tensor] = []
        per_layer_node_attention: List[torch.Tensor] = []
        per_layer_time_attention: List[torch.Tensor] = []

        for layer_index, (gin, readout, temporal_block, linear) in enumerate(
            zip(self.gin_layers, self.readouts, self.temporal_blocks, self.linear_layers)
        ):
            h = gin(h, sparse_a)
            h_bridge = h.reshape(batch_size, num_windows, num_nodes, self.hidden_dim)

            graph_features, node_attention = readout(h_bridge)   # [B, W, D], [B, W, N]

            temporal_input = graph_features.permute(1, 0, 2)     # [W, B, D]
            if self.time_tokens is not None:
                cls_token = self.time_tokens[layer_index].expand(1, temporal_input.shape[1], -1)
                temporal_input = torch.cat([temporal_input, cls_token], dim=0)

            temporal_output, time_attention = temporal_block(temporal_input)

            pooled = self._pool_time(temporal_output, layer_index)  # [B, D]
            logit = logit + self.dropout(linear(pooled))

            ortho_h = h_bridge.reshape(batch_size * num_windows, num_nodes, self.hidden_dim)
            matrix_inner = torch.bmm(ortho_h, ortho_h.permute(0, 2, 1))
            reg = (
                matrix_inner / matrix_inner.max(-1)[0].unsqueeze(-1)
                - torch.eye(num_nodes, device=h.device)
            ).triu().norm(dim=(1, 2)).mean()
            ortho_penalty = ortho_penalty + reg

            per_layer_latents.append(pooled)
            per_layer_node_attention.append(node_attention)
            per_layer_time_attention.append(time_attention)

        layer_latent = torch.stack(per_layer_latents, dim=1)
        return STAGINOutput(
            logits=logit,
            node_attention=torch.stack(per_layer_node_attention, dim=1),   # [B, K, W, N]
            time_attention=torch.stack(per_layer_time_attention, dim=1),   # [B, K, W, W]
            layer_latent=layer_latent,
            final_representation=layer_latent.reshape(batch_size, self.num_layers * self.hidden_dim),
            orthogonality_penalty=ortho_penalty,
        )


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def compute_loss(output: STAGINOutput, labels: torch.Tensor, reg_lambda: float) -> torch.Tensor:
    return F.cross_entropy(output.logits, labels) + reg_lambda * output.orthogonality_penalty


def _compute_metrics(
    all_labels: List[int],
    all_pred_labels: List[int],
    all_prob_ptsd: List[float],
    total_loss: float,
    total_subjects: int,
) -> Dict[str, float]:
    """Compute all scalar metrics from collected predictions."""
    labels_np = np.array(all_labels)
    preds_np = np.array(all_pred_labels)

    accuracy = float(np.mean(labels_np == preds_np))
    try:
        auroc = float(roc_auc_score(all_labels, all_prob_ptsd))
    except ValueError:
        auroc = float("nan")

    f1 = float(f1_score(labels_np, preds_np, average="binary", pos_label=1, zero_division=0))
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")  # recall for PTSD
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")  # recall for HC

    miss_hc = 100.0 * fp / (tn + fp) if (tn + fp) > 0 else float("nan")
    miss_ptsd = 100.0 * fn / (tp + fn) if (tp + fn) > 0 else float("nan")

    return {
        "loss": float(total_loss / max(total_subjects, 1)),
        "accuracy": accuracy,
        "auroc": auroc,
        "f1_ptsd": f1,
        "sensitivity_ptsd": sensitivity,
        "specificity_hc": specificity,
        "misclassification_percent_HC": miss_hc,
        "misclassification_percent_PTSD": miss_ptsd,
        "n_subjects": int(total_subjects),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


@torch.no_grad()
def evaluate_model(
    model: STAGIN,
    loader: DataLoader,
    reg_lambda: float,
    device: str,
    collect_attention: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame, torch.Tensor | None, torch.Tensor | None]:
    """
    Evaluate model on loader.

    collect_attention=True  →  returns mean node/time attention tensors
                                [K, W, N] and [K, W, W] (averaged over subjects).
                                Only do this for the final evaluation of the best
                                fold to avoid storing large tensors for every fold.
    collect_attention=False →  returns (None, None) for the attention outputs.
    """
    model.eval()
    total_loss = 0.0
    total_subjects = 0
    all_labels: List[int] = []
    all_pred_labels: List[int] = []
    all_prob_ptsd: List[float] = []
    prediction_rows: List[Dict[str, object]] = []

    node_attn_accum: torch.Tensor | None = None
    time_attn_accum: torch.Tensor | None = None
    n_attn_subjects = 0

    for batch in loader:
        node_identity = batch["node_identity"].to(device)
        adjacency = batch["adjacency"].to(device)
        t_seq = batch["timeseries"].to(device)
        labels = batch["labels"].to(device)
        endpoints = batch["endpoints"]

        output = model(node_identity, adjacency, t_seq, endpoints)
        loss = compute_loss(output, labels, reg_lambda)

        probabilities = torch.softmax(output.logits, dim=1)
        pred_labels = probabilities.argmax(dim=1)

        b = labels.shape[0]
        total_loss += float(loss.item()) * b
        total_subjects += b

        labels_np = labels.cpu().numpy()
        pred_np = pred_labels.cpu().numpy()
        prob_ptsd_np = probabilities[:, 1].cpu().numpy()

        all_labels.extend(labels_np.tolist())
        all_pred_labels.extend(pred_np.tolist())
        all_prob_ptsd.extend(prob_ptsd_np.tolist())

        # Accumulate attention as running mean (memory-efficient)
        if collect_attention:
            node_batch = output.node_attention.cpu()   # [B, K, W, N]
            time_batch = output.time_attention.cpu()   # [B, K, W, W]
            node_sum = node_batch.sum(dim=0)
            time_sum = time_batch.sum(dim=0)
            if node_attn_accum is None:
                node_attn_accum = node_sum
                time_attn_accum = time_sum
            else:
                node_attn_accum += node_sum
                time_attn_accum += time_sum
            n_attn_subjects += b

        for subject_id, true_label, pred_label, prob_ptsd, class_name, csv_path in zip(
            batch["subject_ids"], labels_np.tolist(), pred_np.tolist(),
            prob_ptsd_np.tolist(), batch["class_names"], batch["csv_paths"],
        ):
            prediction_rows.append({
                "subject_id": subject_id,
                "true_label_index": true_label,
                "true_label_name": INDEX_TO_LABEL[true_label],
                "pred_label_index": pred_label,
                "pred_label_name": INDEX_TO_LABEL[pred_label],
                "probability_PTSD": prob_ptsd,
                "probability_HC": 1.0 - prob_ptsd,
                "correct": int(true_label == pred_label),
                "source_folder_label": class_name,
                "csv_path": csv_path,
            })

    metrics = _compute_metrics(all_labels, all_pred_labels, all_prob_ptsd, total_loss, total_subjects)
    predictions_df = pd.DataFrame(prediction_rows)

    mean_node_attn = node_attn_accum / n_attn_subjects if collect_attention else None
    mean_time_attn = time_attn_accum / n_attn_subjects if collect_attention else None

    return metrics, predictions_df, mean_node_attn, mean_time_attn


def run_one_epoch(
    model: STAGIN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    reg_lambda: float,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_subjects = 0
    for batch in loader:
        node_identity = batch["node_identity"].to(device)
        adjacency = batch["adjacency"].to(device)
        t_seq = batch["timeseries"].to(device)
        labels = batch["labels"].to(device)
        endpoints = batch["endpoints"]

        optimizer.zero_grad()
        output = model(node_identity, adjacency, t_seq, endpoints)
        loss = compute_loss(output, labels, reg_lambda)
        loss.backward()
        optimizer.step()

        b = labels.shape[0]
        total_loss += float(loss.item()) * b
        total_subjects += b

    return total_loss / max(total_subjects, 1)


# =============================================================================
# HELPERS: SPLITS & SAVING
# =============================================================================

def save_records_csv(records: Sequence[SubjectRecord], save_path: Path) -> None:
    pd.DataFrame([asdict(r) for r in records]).to_csv(save_path, index=False)


def save_metrics_json(metrics: Dict[str, object], save_path: Path) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_confusion_matrix(cm_dict: Dict[str, int], save_path: Path) -> None:
    pd.DataFrame(
        [[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]],
        index=["True HC", "True PTSD"],
        columns=["Pred HC", "Pred PTSD"],
    ).to_csv(save_path)


def append_history_csv(history_rows: List[Dict], path: Path) -> None:
    """Append epoch rows to a global CSV (created on first fold)."""
    df = pd.DataFrame(history_rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Scanning data under: {DATA_ROOT.resolve()}")

    records = get_subject_csvs(DATA_ROOT)
    print(f"Found {len(records)} subject CSVs.")

    records, _ = screen_records(
        records=records,
        expected_timepoints=TIMEPOINTS,
        expected_rois=N_ROIS,
        log_path=OUTPUT_DIR / "excluded_subjects.txt",
    )
    print(f"{len(records)} subjects retained after screening.")

    random.Random(SEED).shuffle(records)
    class_counts = pd.Series([r.class_name for r in records]).value_counts().to_dict()
    print(f"Class counts: {class_counts}")

    labels_array = np.array([r.label for r in records])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    dataset_kwargs = dict(
        n_rois=N_ROIS, timepoints=TIMEPOINTS, window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE, top_percent=TOP_PERCENT,
        add_self_loops=ADD_SELF_LOOPS, standardize_each_roi=STANDARDIZE_EACH_ROI,
    )

    fold_metrics: List[Dict[str, float]] = []
    all_fold_predictions: List[pd.DataFrame] = []
    history_csv_path = OUTPUT_DIR / "training_history.csv"

    best_cv_auroc = -float("inf")
    best_cv_fold = -1

    for fold_idx, (train_indices, test_indices) in enumerate(
        skf.split(records, labels_array), start=1
    ):
        print(f"\n{'='*60}\n FOLD {fold_idx} / 5\n{'='*60}")

        fold_dir = OUTPUT_DIR / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_records = [records[i] for i in train_indices]
        test_records = [records[i] for i in test_indices]

        save_records_csv(train_records, fold_dir / "train_split.csv")
        save_records_csv(test_records, fold_dir / "test_split.csv")

        train_loader = DataLoader(
            SubjectGraphDataset(train_records, **dataset_kwargs),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_subject_batch,
        )
        test_loader = DataLoader(
            SubjectGraphDataset(test_records, **dataset_kwargs),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_subject_batch,
        )

        set_seed(SEED + fold_idx)
        model = STAGIN(
            num_nodes=N_ROIS, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
            num_layers=NUM_LAYERS, readout=READOUT, time_pool=TIME_POOL,
            temporal_num_heads=TEMPORAL_NUM_HEADS, dropout=DROPOUT, reg_lambda=REG_LAMBDA,
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_fold_auroc = -float("inf")
        best_model_path = fold_dir / "best_model.pt"
        fold_history: List[Dict] = []

        for epoch in range(1, EPOCHS + 1):
            train_loss = run_one_epoch(model, train_loader, optimizer, REG_LAMBDA, DEVICE)
            # collect_attention=False during training loop to save memory
            val_metrics, _, _, _ = evaluate_model(model, test_loader, REG_LAMBDA, DEVICE)

            row = {
                "fold": fold_idx, "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_auroc": val_metrics["auroc"],
                "val_f1_ptsd": val_metrics["f1_ptsd"],
                "val_sensitivity_ptsd": val_metrics["sensitivity_ptsd"],
                "val_specificity_hc": val_metrics["specificity_hc"],
                "val_misclassification_HC": val_metrics["misclassification_percent_HC"],
                "val_misclassification_PTSD": val_metrics["misclassification_percent_PTSD"],
            }
            fold_history.append(row)

            print(
                f"  Fold {fold_idx} | Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_auroc={val_metrics['auroc']:.4f}"
            )

            current_auroc = val_metrics["auroc"]
            if not math.isnan(current_auroc) and current_auroc > best_fold_auroc:
                best_fold_auroc = current_auroc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": {
                        "num_nodes": N_ROIS, "hidden_dim": HIDDEN_DIM,
                        "num_classes": NUM_CLASSES, "num_layers": NUM_LAYERS,
                        "readout": READOUT, "time_pool": TIME_POOL,
                        "temporal_num_heads": TEMPORAL_NUM_HEADS,
                        "dropout": DROPOUT, "reg_lambda": REG_LAMBDA,
                    },
                    "fold": fold_idx, "epoch": epoch, "best_val_auroc": best_fold_auroc,
                }, best_model_path)

        # Incrementally flush this fold's history to disk
        append_history_csv(fold_history, history_csv_path)

        # --- Final evaluation on best checkpoint ---
        checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        # collect_attention=True only for this final pass
        test_metrics, test_preds, mean_node_attn, mean_time_attn = evaluate_model(
            model, test_loader, REG_LAMBDA, DEVICE, collect_attention=True
        )
        test_preds["fold"] = fold_idx
        all_fold_predictions.append(test_preds)

        # Save compact mean attention: [K, W, N] and [K, W, W]
        torch.save(mean_node_attn, fold_dir / "mean_node_attention.pt")
        torch.save(mean_time_attn, fold_dir / "mean_time_attention.pt")

        # Save per-fold confusion matrix
        save_confusion_matrix(test_metrics, fold_dir / "confusion_matrix.csv")

        save_metrics_json(
            {
                "fold": fold_idx,
                "best_val_auroc": best_fold_auroc,
                "best_epoch": int(checkpoint["epoch"]),
                "test_metrics": test_metrics,
            },
            fold_dir / "metrics.json",
        )

        fold_metrics.append({
            "fold": fold_idx,
            "accuracy": test_metrics["accuracy"],
            "auroc": test_metrics["auroc"],
            "f1_ptsd": test_metrics["f1_ptsd"],
            "sensitivity_ptsd": test_metrics["sensitivity_ptsd"],
            "specificity_hc": test_metrics["specificity_hc"],
            "misclassification_percent_HC": test_metrics["misclassification_percent_HC"],
            "misclassification_percent_PTSD": test_metrics["misclassification_percent_PTSD"],
        })

        if best_fold_auroc > best_cv_auroc:
            best_cv_auroc = best_fold_auroc
            best_cv_fold = fold_idx

        print(
            f"\n  Fold {fold_idx} final → "
            f"acc={test_metrics['accuracy']:.4f}  "
            f"AUROC={test_metrics['auroc']:.4f}  "
            f"F1={test_metrics['f1_ptsd']:.4f}"
        )

    # -------------------------------------------------------------------------
    # Aggregate across folds
    # -------------------------------------------------------------------------
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    all_predictions_df = pd.concat(all_fold_predictions, ignore_index=True)
    all_predictions_df.to_csv(OUTPUT_DIR / "all_fold_predictions.csv", index=False)

    def mean_std(col: str) -> Tuple[float, float]:
        return float(metrics_df[col].mean()), float(metrics_df[col].std())

    acc_m, acc_s = mean_std("accuracy")
    auc_m, auc_s = mean_std("auroc")
    f1_m,  f1_s  = mean_std("f1_ptsd")
    sen_m, sen_s = mean_std("sensitivity_ptsd")
    spe_m, spe_s = mean_std("specificity_hc")
    mhc_m, mhc_s = mean_std("misclassification_percent_HC")
    mpt_m, mpt_s = mean_std("misclassification_percent_PTSD")

    summary = {
        "accuracy":             {"mean": acc_m, "std": acc_s},
        "auroc":                {"mean": auc_m, "std": auc_s},
        "f1_ptsd":              {"mean": f1_m,  "std": f1_s},
        "sensitivity_ptsd":     {"mean": sen_m, "std": sen_s},
        "specificity_hc":       {"mean": spe_m, "std": spe_s},
        "misclassification_HC": {"mean": mhc_m, "std": mhc_s},
        "misclassification_PTSD": {"mean": mpt_m, "std": mpt_s},
        "best_fold_by_auroc": best_cv_fold,
        "per_fold": fold_metrics,
    }
    save_metrics_json(summary, OUTPUT_DIR / "cv_summary.json")

    print(f"\n{'='*60}\n 5-FOLD CV RESULTS\n{'='*60}")
    print(f"  Accuracy    : {acc_m*100:.2f} ± {acc_s*100:.2f}%")
    print(f"  AUROC       : {auc_m:.4f} ± {auc_s:.4f}")
    print(f"  F1 (PTSD)   : {f1_m:.4f} ± {f1_s:.4f}")
    print(f"  Sensitivity : {sen_m:.4f} ± {sen_s:.4f}")
    print(f"  Specificity : {spe_m:.4f} ± {spe_s:.4f}")
    print(f"  Miss % HC   : {mhc_m:.2f} ± {mhc_s:.2f}%")
    print(f"  Miss % PTSD : {mpt_m:.2f} ± {mpt_s:.2f}%")
    print(f"  Best fold   : Fold {best_cv_fold} (val AUROC={best_cv_auroc:.4f})")
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
