from __future__ import annotations

"""
Subject-level classification of entire sample (n=1000).
Labels are derived entirely from the parent folders.  HC = 0, PTSD = 1

Saves:
    - trained model weights
    - split membership
    - per-subject test predictions
    - metrics JSON containing AUROC, accuracy, and class-wise misclassification rates

"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# CONFIG
# =============================================================================

N_ROIS = 100

# HC/ and PTSD/ directly inside this directory
DATA_ROOT = Path("opt/notebooks/CombinedAtlas_31016+31019/100/")
TIMEPOINTS = 490
OUTPUT_DIR = Path("./stagin_outputs")
SEED = 26
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.10

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Dynamic graph 
WINDOW_SIZE = 30
WINDOW_STRIDE = 5
TOP_PERCENT = 30.0
ADD_SELF_LOOPS = True
STANDARDIZE_EACH_ROI = True

# STAGIN architecture
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 2
TEMPORAL_NUM_HEADS = 1
DROPOUT = 0.1
READOUT = "sero"      # "sero", "garo", "mean"
TIME_POOL = "sum"     # "sum", "mean", "last", "param"
REG_LAMBDA = 1e-4

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# LOAD
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
            records.append(
                SubjectRecord(
                    csv_path=str(csv_path.resolve()),
                    subject_id=csv_path.stem,
                    class_name=class_name,
                    label=label,
                )
            )

    return records


def load_subject_timeseries(csv_path: str, expected_timepoints: int, expected_rois: int) -> np.ndarray:
    df = pd.read_csv(csv_path, sep=r"\s+", header=None)  # change for atlas csv / ICA space separated txt

    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
    values = numeric.to_numpy(dtype=np.float32)

    if values.shape == (expected_timepoints, expected_rois):
        return values
    if values.shape == (expected_rois, expected_timepoints):
        return values.T

    raise ValueError(
        f"Unexpected shape for {csv_path}: got {values.shape}, "
        f"expected ({expected_timepoints}, {expected_rois}) or ({expected_rois}, {expected_timepoints})"
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
    """Compute ROI-by-ROI correlation from a [window, roi] matrix."""
    x = windowed_timeseries - windowed_timeseries.mean(dim=0, keepdim=True)
    cov = x.T @ x / max(x.shape[0] - 1, 1)
    std = cov.diag().clamp_min(1e-8).sqrt()
    corr = cov / (std[:, None] * std[None, :])
    return corr.clamp(-1.0, 1.0)


def sliding_window_endpoints(total_timepoints: int, window_size: int, window_stride: int) -> List[int]:
    """Return 1-based endpoints for all sliding windows."""
    if total_timepoints <= window_size:
        raise ValueError("total_timepoints must be larger than window_size")

    endpoints: List[int] = []
    for start in range(0, total_timepoints - window_size + 1, window_stride):
        endpoints.append(start + window_size)
    return endpoints


def build_dynamic_fc_graphs(
    standardized_timeseries: torch.Tensor,
    window_size: int,
    window_stride: int,
    top_percent: float,
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    for one subject
    Input: standardized_timeseries: [T, N]
    Output: node_identity: [W, N, N]; adjacency: [W, N, N] ;  endpoints: list[int]
    """
    total_timepoints, num_rois = standardized_timeseries.shape
    endpoints = sliding_window_endpoints(total_timepoints, window_size, window_stride)

    adjacency_per_window: List[torch.Tensor] = []
    for endpoint in endpoints:
        start = endpoint - window_size
        fc = corrcoef_time_by_roi(standardized_timeseries[start:endpoint])

        threshold = torch.quantile(fc.reshape(-1), 1.0 - top_percent / 100.0)
        adjacency = (fc > threshold).to(fc.dtype)

        if add_self_loops:
            adjacency.fill_diagonal_(1.0)

        adjacency_per_window.append(adjacency)

    adjacency = torch.stack(adjacency_per_window, dim=0)
    node_identity = torch.eye(num_rois, device=standardized_timeseries.device).unsqueeze(0).repeat(len(endpoints), 1, 1)
    return node_identity, adjacency, endpoints



# DATASET

class SubjectGraphDataset(Dataset):
    """Load one subject CSV and convert it into STAGIN inputs on the fly."""

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
            csv_path=record.csv_path,
            expected_timepoints=self.timepoints,
            expected_rois=self.n_rois,
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
    """
    Stack a batch of subjects into STAGIN-ready tensors.
    """
    endpoints = batch[0]["endpoints"]

    node_identity = torch.stack([sample["node_identity"] for sample in batch], dim=0)
    adjacency = torch.stack([sample["adjacency"] for sample in batch], dim=0)
    timeseries = torch.stack([sample["timeseries"] for sample in batch], dim=1)  # [T, B, N]
    labels = torch.stack([sample["label"] for sample in batch], dim=0)

    return {
        "node_identity": node_identity,
        "adjacency": adjacency,
        "timeseries": timeseries,
        "labels": labels,
        "endpoints": endpoints,
        "subject_ids": [sample["subject_id"] for sample in batch],
        "class_names": [sample["class_name"] for sample in batch],
        "csv_paths": [sample["csv_path"] for sample in batch],
    }


# =============================================================================
# MODEL PIECES
# =============================================================================

class TimestampEncoder(nn.Module):
    """GRU timestamp encoder.
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
        if len(endpoints) == 0:
            raise ValueError("endpoints must not be empty")
        gru_output, _ = self.rnn(t_seq[: endpoints[-1]])
        endpoint_indices = torch.as_tensor([e - 1 for e in endpoints], device=t_seq.device)
        return gru_output[endpoint_indices]


class GINLayer(nn.Module):
    """Dense GIN update: H_k = MLP((A + eps I) H_{k-1})."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        aggregated = torch.matmul(a, x) + self.eps * x
        batch_size, num_windows, num_nodes, feature_dim = aggregated.shape
        flat = aggregated.reshape(batch_size * num_windows * num_nodes, feature_dim)
        return self.mlp(flat).reshape(batch_size, num_windows, num_nodes, feature_dim)


class MeanReadout(nn.Module):
    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_features = node_features.mean(dim=2)
        node_attention = torch.zeros(
            node_features.shape[0],
            node_features.shape[1],
            node_features.shape[2],
            device=node_features.device,
        )
        return graph_features, node_attention


class GAROReadout(nn.Module):
    """Graph attention readout from Kim et al. (2021)."""

    def __init__(self, feature_dim: int, dropout: float = 0.1, upscale: float = 1.0):
        super().__init__()
        projected_dim = max(1, round(upscale * feature_dim))
        self.key = nn.Linear(feature_dim, projected_dim)
        self.query = nn.Linear(feature_dim, projected_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unattended_graph = node_features.mean(dim=2, keepdim=True)
        query = self.query(unattended_graph)
        key = self.key(node_features)
        scores = torch.matmul(query, key.transpose(-1, -2)).squeeze(2)
        attention = torch.sigmoid(scores / math.sqrt(query.shape[-1]))
        graph_features = (node_features * self.dropout(attention.unsqueeze(-1))).mean(dim=2)
        return graph_features, attention


class SEROReadout(nn.Module):
    """Squeeze-excitation readout from Kim et al. (2021)."""

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
        squeezed_graph = node_features.mean(dim=2)
        batch_size, num_windows, feature_dim = squeezed_graph.shape
        embedded = self.embed(squeezed_graph.reshape(batch_size * num_windows, feature_dim))
        attention = torch.sigmoid(self.attend(embedded)).reshape(batch_size, num_windows, -1)
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
        ff = self.mlp(attended)
        attended = self.norm2(attended + self.dropout2(ff))
        return attended, time_attention


# =============================================================================
# FULL MODEL
# =============================================================================

@dataclass
class STAGINOutput:
    logits: torch.Tensor
    node_attention: torch.Tensor
    time_attention: torch.Tensor
    layer_latent: torch.Tensor
    final_representation: torch.Tensor
    orthogonality_penalty: torch.Tensor


class STAGIN(nn.Module):
    """
    Subject-level STAGIN classifier.

    Final classification uses the concatenation of all per-layer temporal graph
    representations.
    """

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
        self.readout_name = readout
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

        self.temporal_blocks = nn.ModuleList(
            [
                TemporalTransformer(
                    feature_dim=hidden_dim,
                    mlp_dim=2 * hidden_dim,
                    num_heads=temporal_num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        if time_pool == "param":
            self.time_tokens = nn.Parameter(torch.randn(num_layers, 1, 1, hidden_dim))
        else:
            self.time_tokens = None

        self.classifier = nn.Linear(num_layers * hidden_dim, num_classes)

    def _pool_time(self, temporal_features: torch.Tensor, layer_index: int) -> torch.Tensor:
        del layer_index
        if self.time_pool == "sum":
            return temporal_features.sum(dim=0)
        if self.time_pool == "mean":
            return temporal_features.mean(dim=0)
        return temporal_features[-1]

    @staticmethod
    def orthogonality_penalty(node_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_windows, num_nodes, feature_dim = node_features.shape
        h = node_features.reshape(batch_size * num_windows, num_nodes, feature_dim)
        inner = torch.bmm(h, h.transpose(1, 2))
        scale = inner.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        normalized = inner / scale
        identity = torch.eye(num_nodes, device=node_features.device).unsqueeze(0)
        return torch.triu(normalized - identity, diagonal=1).norm(dim=(1, 2)).mean()

    def forward(
        self,
        node_identity: torch.Tensor,
        adjacency: torch.Tensor,
        t_seq: torch.Tensor,
        endpoints: Sequence[int],
    ) -> STAGINOutput:
        batch_size, num_windows, num_nodes, _ = node_identity.shape

        encoded_time = self.timestamp_encoder(t_seq, endpoints)
        encoded_time = encoded_time.permute(1, 0, 2).unsqueeze(2)
        encoded_time = encoded_time.expand(batch_size, num_windows, num_nodes, self.hidden_dim)

        node_features = torch.cat([node_identity, encoded_time], dim=-1)
        node_features = self.initial_linear(node_features)

        per_layer_latents: List[torch.Tensor] = []
        per_layer_node_attention: List[torch.Tensor] = []
        per_layer_time_attention: List[torch.Tensor] = []
        ortho_penalty = node_features.new_tensor(0.0)

        h = node_features
        for layer_index, (gin, readout, temporal_block) in enumerate(
            zip(self.gin_layers, self.readouts, self.temporal_blocks)
        ):
            h = gin(h, adjacency)
            graph_features, node_attention = readout(h)
            temporal_input = graph_features.permute(1, 0, 2)

            if self.time_tokens is not None:
                cls_token = self.time_tokens[layer_index].expand(1, temporal_input.shape[1], -1)
                temporal_input = torch.cat([temporal_input, cls_token], dim=0)

            temporal_output, time_attention = temporal_block(temporal_input)
            pooled_layer_representation = self._pool_time(temporal_output, layer_index)

            per_layer_latents.append(pooled_layer_representation)
            per_layer_node_attention.append(node_attention)
            per_layer_time_attention.append(time_attention)
            ortho_penalty = ortho_penalty + self.orthogonality_penalty(h)

        layer_latent = torch.stack(per_layer_latents, dim=1)
        final_representation = layer_latent.reshape(batch_size, self.num_layers * self.hidden_dim)
        logits = self.classifier(self.dropout(final_representation))

        node_attention = torch.stack(per_layer_node_attention, dim=1)
        time_attention = torch.stack(per_layer_time_attention, dim=1)

        return STAGINOutput(
            logits=logits,
            node_attention=node_attention,
            time_attention=time_attention,
            layer_latent=layer_latent,
            final_representation=final_representation,
            orthogonality_penalty=ortho_penalty,
        )


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def compute_loss(output: STAGINOutput, labels: torch.Tensor, reg_lambda: float) -> torch.Tensor:
    classification_loss = F.cross_entropy(output.logits, labels)
    return classification_loss + reg_lambda * output.orthogonality_penalty


@torch.no_grad()
def evaluate_model(
    model: STAGIN,
    loader: DataLoader,
    reg_lambda: float,
    device: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()

    total_loss = 0.0
    total_subjects = 0

    all_labels: List[int] = []
    all_pred_labels: List[int] = []
    all_prob_ptsd: List[float] = []
    prediction_rows: List[Dict[str, object]] = []

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

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_subjects += batch_size

        labels_np = labels.cpu().numpy()
        pred_np = pred_labels.cpu().numpy()
        prob_ptsd_np = probabilities[:, 1].cpu().numpy()

        all_labels.extend(labels_np.tolist())
        all_pred_labels.extend(pred_np.tolist())
        all_prob_ptsd.extend(prob_ptsd_np.tolist())

        for subject_id, true_label, pred_label, prob_ptsd, class_name, csv_path in zip(
            batch["subject_ids"],
            labels_np.tolist(),
            pred_np.tolist(),
            prob_ptsd_np.tolist(),
            batch["class_names"],
            batch["csv_paths"],
        ):
            prediction_rows.append(
                {
                    "subject_id": subject_id,
                    "true_label_index": true_label,
                    "true_label_name": INDEX_TO_LABEL[true_label],
                    "pred_label_index": pred_label,
                    "pred_label_name": INDEX_TO_LABEL[pred_label],
                    "probability_PTSD": prob_ptsd,
                    "probability_HC": 1.0 - prob_ptsd,
                    "source_folder_label": class_name,
                    "csv_path": csv_path,
                }
            )

    mean_loss = total_loss / max(total_subjects, 1)
    accuracy = float(np.mean(np.array(all_labels) == np.array(all_pred_labels))) if all_labels else float("nan")

    try:
        auroc = float(roc_auc_score(all_labels, all_prob_ptsd))
    except ValueError:
        auroc = float("nan")

    cm = confusion_matrix(all_labels, all_pred_labels, labels=[0, 1])
    class_misclassification_percent: Dict[str, float] = {}
    for class_index, class_name in INDEX_TO_LABEL.items():
        class_total = cm[class_index].sum()
        correct = cm[class_index, class_index]
        misclassified = class_total - correct
        misclassification_percent = 100.0 * misclassified / class_total if class_total > 0 else float("nan")
        class_misclassification_percent[class_name] = float(misclassification_percent)

    metrics = {
        "loss": float(mean_loss),
        "accuracy": accuracy,
        "auroc": auroc,
        "misclassification_percent_HC": class_misclassification_percent["HC"],
        "misclassification_percent_PTSD": class_misclassification_percent["PTSD"],
        "n_subjects": int(total_subjects),
    }
    return metrics, pd.DataFrame(prediction_rows)


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

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_subjects += batch_size

    return total_loss / max(total_subjects, 1)


# =============================================================================
# SPLITS AND SAVING
# =============================================================================

def stratified_split_records(
    records: Sequence[SubjectRecord],
    test_size: float,
    val_size_within_train: float,
    seed: int,
) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
    labels = [record.label for record in records]
    train_val, test = train_test_split(
        list(records),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_val_labels = [record.label for record in train_val]
    train, val = train_test_split(
        train_val,
        test_size=val_size_within_train,
        random_state=seed,
        stratify=train_val_labels,
    )
    return train, val, test


def save_records_csv(records: Sequence[SubjectRecord], save_path: Path) -> None:
    df = pd.DataFrame([asdict(record) for record in records])
    df.to_csv(save_path, index=False)


def save_metrics_json(metrics: Dict[str, object], save_path: Path) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


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

    records, excluded_paths = screen_records(records=records,expected_timepoints=TIMEPOINTS, expected_rois=N_ROIS,
                                             log_path=OUTPUT_DIR / "excluded_subjects.txt",)
    print(f"{len(records)} subjects retained after screening.")


    class_counts = pd.Series([record.class_name for record in records]).value_counts().to_dict()
    print(f"Class counts: {class_counts}")

    train_records, val_records, test_records = stratified_split_records(
        records=records,
        test_size=TEST_SIZE,
        val_size_within_train=VAL_SIZE_WITHIN_TRAIN,
        seed=SEED,
    )

    save_records_csv(train_records, OUTPUT_DIR / "train_split.csv")
    save_records_csv(val_records, OUTPUT_DIR / "val_split.csv")
    save_records_csv(test_records, OUTPUT_DIR / "test_split.csv")

    dataset_kwargs = dict(
        n_rois=N_ROIS,
        timepoints=TIMEPOINTS,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
        top_percent=TOP_PERCENT,
        add_self_loops=ADD_SELF_LOOPS,
        standardize_each_roi=STANDARDIZE_EACH_ROI,
    )

    train_dataset = SubjectGraphDataset(train_records, **dataset_kwargs)
    val_dataset = SubjectGraphDataset(val_records, **dataset_kwargs)
    test_dataset = SubjectGraphDataset(test_records, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_subject_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_subject_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_subject_batch)

    model = STAGIN(
        num_nodes=N_ROIS,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        readout=READOUT,
        time_pool=TIME_POOL,
        temporal_num_heads=TEMPORAL_NUM_HEADS,
        dropout=DROPOUT,
        reg_lambda=REG_LAMBDA,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_auroc = -float("inf")
    best_model_path = OUTPUT_DIR / "best_model.pt"
    training_history: List[Dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_one_epoch(model, train_loader, optimizer, REG_LAMBDA, DEVICE)
        val_metrics, _ = evaluate_model(model, val_loader, REG_LAMBDA, DEVICE)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_auroc": float(val_metrics["auroc"]),
            "val_misclassification_percent_HC": float(val_metrics["misclassification_percent_HC"]),
            "val_misclassification_percent_PTSD": float(val_metrics["misclassification_percent_PTSD"]),
        }
        training_history.append(epoch_summary)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={epoch_summary['train_loss']:.4f} | "
            f"val_loss={epoch_summary['val_loss']:.4f} | "
            f"val_acc={epoch_summary['val_accuracy']:.4f} | "
            f"val_auroc={epoch_summary['val_auroc']:.4f} | "
            f"val_miss_HC={epoch_summary['val_misclassification_percent_HC']:.2f}% | "
            f"val_miss_PTSD={epoch_summary['val_misclassification_percent_PTSD']:.2f}%"
        )

        current_val_auroc = val_metrics["auroc"]
        if not math.isnan(current_val_auroc) and current_val_auroc > best_val_auroc:
            best_val_auroc = current_val_auroc
            torch.save(model.state_dict(), best_model_path)

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_metrics, test_predictions = evaluate_model(model, test_loader, REG_LAMBDA, DEVICE)
    test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    metrics_payload = {
        "config": {
            "N_ROIS": N_ROIS,
            "DATA_ROOT": str(DATA_ROOT),
            "TIMEPOINTS": TIMEPOINTS,
            "WINDOW_SIZE": WINDOW_SIZE,
            "WINDOW_STRIDE": WINDOW_STRIDE,
            "TOP_PERCENT": TOP_PERCENT,
            "HIDDEN_DIM": HIDDEN_DIM,
            "NUM_LAYERS": NUM_LAYERS,
            "NUM_CLASSES": NUM_CLASSES,
            "TEMPORAL_NUM_HEADS": TEMPORAL_NUM_HEADS,
            "DROPOUT": DROPOUT,
            "READOUT": READOUT,
            "TIME_POOL": TIME_POOL,
            "REG_LAMBDA": REG_LAMBDA,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "TEST_SIZE": TEST_SIZE,
            "VAL_SIZE_WITHIN_TRAIN": VAL_SIZE_WITHIN_TRAIN,
            "SEED": SEED,
            "DEVICE": DEVICE,
        },
        "dataset_summary": {
            "n_total_subjects": len(records),
            "n_train": len(train_records),
            "n_val": len(val_records),
            "n_test": len(test_records),
            "class_counts_total": class_counts,
        },
        "best_val_auroc": float(best_val_auroc) if best_val_auroc > -float("inf") else None,
        "test_metrics": test_metrics,
        "training_history": training_history,
    }
    save_metrics_json(metrics_payload, OUTPUT_DIR / "metrics.json")

    print("\nFinal test performance")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Misclassification % HC: {test_metrics['misclassification_percent_HC']:.2f}%")
    print(f"  Misclassification % PTSD: {test_metrics['misclassification_percent_PTSD']:.2f}%")
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
