"""
STAGIN workflow inspired by Kim et al. (NeurIPS 2021)

Workflow:
1. Standardize ROI time series across time [UKB already standardized.]
2. Build dynamic FC graphs with a sliding window.
3. Binarize each FC matrix by keeping the top percentile entries.
4. Encode time with a GRU using ROI time series up to each window endpoint.
5. Build node features as Linear([one-hot ROI || encoded timestamp]).
6. Pass node features through stacked GIN layers.
7. Apply GARO or SERO graph readout at each layer.
8. Apply a temporal Transformer encoder to the graph sequence.
9. Pool across time per layer.
10. Concatenate per-layer dynamic graph representations.
11. Classify from the concatenated representation.
12. Optionally add the orthogonality penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIG
# =============================================================================

# Set to 414 for atlas, and 100 for ICA
N_ROIS = 100

TIMEPOINTS = 490
WINDOW_SIZE = 30 # Kim et al. used window size of 50 with a stride of 3
WINDOW_STRIDE = 5
TOP_PERCENT = 30.0
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 2
TEMPORAL_NUM_HEADS = 1  # single-headed temporal Transformer
DROPOUT = 0.1
READOUT = "sero"        # "sero", "garo", "mean"
TIME_POOL = "sum"       # "sum", "mean", "last", "param"
REG_LAMBDA = 1e-4
DEVICE = "cpu"


# =============================================================================
# DYNAMIC GRAPH CONSTRUCTION
# =============================================================================

def corrcoef_time_by_roi(windowed_timeseries: torch.Tensor) -> torch.Tensor:
    """
    Compute ROI-by-ROI correlation from a [window, roi] matrix.
    Returns a dense correlation matrix of shape [100, 100].
    """
    x = windowed_timeseries - windowed_timeseries.mean(dim=0, keepdim=True)
    cov = x.T @ x / max(x.shape[0] - 1, 1)
    std = cov.diag().clamp_min(1e-8).sqrt()
    corr = cov / (std[:, None] * std[None, :])
    return corr.clamp(-1.0, 1.0)


def sliding_window_endpoints(
    total_timepoints: int,
    window_size: int,
    window_stride: int,
) -> List[int]:
    """
    Return the 1-based window endpoints used by the timestamp encoder.

    If a window starts at s and ends at e = s + window_size, the endpoint stored is e.
    The idea is to feed the GRU up to each window endpoint.
    """
    if total_timepoints <= window_size:
        raise ValueError("total_timepoints must be larger than window_size")

    endpoints: List[int] = []
    for start in range(0, total_timepoints - window_size, window_stride):
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
    Build dynamic graph inputs from one subject.

    Input: standardized_timeseries: [490, 100]

    Output:  node_identity: [W, 100, 100], One-hot node identity matrix repeated across windows.
        adjacency: [W, 100, 100], Binary adjacency matrices from thresholded windowed FC.
        endpoints: list[int], Endpoints of the sliding windows, used by the timestamp encoder.
    """
    T, N = standardized_timeseries.shape
    endpoints = sliding_window_endpoints(T, window_size, window_stride)

    adjacency_per_window: List[torch.Tensor] = []
    for endpoint in endpoints:
        start = endpoint - window_size
        fc = corrcoef_time_by_roi(standardized_timeseries[start:endpoint])

        threshold = torch.quantile(fc.reshape(-1), 1.0 - top_percent / 100.0)
        binary_adjacency = (fc > threshold).to(fc.dtype)

        if add_self_loops:
            binary_adjacency.fill_diagonal_(1.0)

        adjacency_per_window.append(binary_adjacency)

    adjacency = torch.stack(adjacency_per_window, dim=0)                 # [W, 100, 100]
    node_identity = torch.eye(N, device=standardized_timeseries.device)
    node_identity = node_identity.unsqueeze(0).repeat(len(endpoints), 1, 1)  # [W, 100, 100]
    return node_identity, adjacency, endpoints


# =============================================================================
# MODEL PIECES
# =============================================================================

class TimestampEncoder(nn.Module):
    """
    GRU timestamp encoder.

    Input to GRU: t_seq: [T, B, N],  ROI time series for each subject in the batch.
    Output: encoded_timestamps: [W, B, D]   
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
    """
    Dense GIN update implementing: H_k = MLP((A + eps * I) H_{k-1})
    Shapes: x: [B, W, N, D] ; a: [B, W, N, N] ;  out: [B, W, N, D]
    """

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
        B, W, N, D = aggregated.shape
        return self.mlp(aggregated.reshape(B * W * N, D)).reshape(B, W, N, D)


class MeanReadout(nn.Module):
    """Plain mean readout baseline."""

    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_features = node_features.mean(dim=2)
        node_attention = torch.zeros(
            node_features.shape[0], node_features.shape[1], node_features.shape[2],
            device=node_features.device,
        )
        return graph_features, node_attention


class GAROReadout(nn.Module):
    """
    Graph-Attention ReadOut from the paper:
        K = W_key H
        q = W_query H phi_mean
        z_space = sigmoid(q^T K / sqrt(D))
        h_tilde_G = H z_space

    Input: node_features [B, W, N, D]
    Output: graph_features: [B, W, D] ; node_attention: [B, W, N]
    """

    def __init__(self, feature_dim: int, dropout: float = 0.1, upscale: float = 1.0):
        super().__init__()
        projected_dim = max(1, round(upscale * feature_dim))
        self.key = nn.Linear(feature_dim, projected_dim)
        self.query = nn.Linear(feature_dim, projected_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unattended_graph = node_features.mean(dim=2, keepdim=True)       # [B, W, 1, D]
        q = self.query(unattended_graph)                                 # [B, W, 1, Dq]
        k = self.key(node_features)                                      # [B, W, N, Dq]
        scores = torch.matmul(q, k.transpose(-1, -2)).squeeze(2)         # [B, W, N]
        attention = torch.sigmoid(scores / math.sqrt(q.shape[-1]))       # [B, W, N]
        graph_features = (node_features * self.dropout(attention.unsqueeze(-1))).mean(dim=2)
        return graph_features, attention


class SEROReadout(nn.Module):
    """
    Squeeze-Excitation ReadOut from Kim et al. (2021):  z_space = sigmoid(W2 sigma(W1 H phi_mean))

    Input: node_features: [B, W, N, D]
    Output: graph_features: [B, W, D] ; node_attention: [B, W, N]
    """

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
        squeezed_graph = node_features.mean(dim=2)                       # [B, W, D]
        B, W, D = squeezed_graph.shape
        embedded = self.embed(squeezed_graph.reshape(B * W, D))
        attention = torch.sigmoid(self.attend(embedded)).reshape(B, W, -1)
        graph_features = (node_features * self.dropout(attention.unsqueeze(-1))).mean(dim=2)
        return graph_features, attention


class TemporalTransformer(nn.Module):
    """
    Single temporal Transformer encoder block.

    Input: x: [W, B, D]
    Output: attended: [W, B, D] ; time_attention: [B, W, W]
    """
    
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
        attended, time_attention = self.attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=True,
        )
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
    Spatio-temporal Attention Graph Isomprhism Network implementation:
    The final classifier operates on the concatenation of per-layer dynamic graph
    representations, matching Equation (16) in Kim et al. (2021) rather than summing
    separate layer logits.
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
        """
        Pool across the temporal axis after temporal attention.

        Input: temporal_features: [W, B, D] or [W+1, B, D] (with param token)
        Output: pooled: [B, D]
        """
        if self.time_pool == "sum":
            return temporal_features.sum(dim=0)
        if self.time_pool == "mean":
            return temporal_features.mean(dim=0)
        if self.time_pool == "last":
            return temporal_features[-1]
        return temporal_features[-1]

    @staticmethod
    def orthogonality_penalty(node_features: torch.Tensor) -> torch.Tensor:
        """
        Orthogonality regularizer.
        Input:node_features: [B, W, N, D]
        """
        B, W, N, D = node_features.shape
        h = node_features.reshape(B * W, N, D)
        inner = torch.bmm(h, h.transpose(1, 2))
        scale = inner.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        normalized = inner / scale
        identity = torch.eye(N, device=node_features.device).unsqueeze(0)
        return torch.triu(normalized - identity, diagonal=1).norm(dim=(1, 2)).mean()

    def forward(
        self,
        node_identity: torch.Tensor,
        adjacency: torch.Tensor,
        t_seq: torch.Tensor,
        endpoints: Sequence[int],
    ) -> STAGINOutput:
        """
        Inputs:
            node_identity: [B, W, N, N]
            adjacency: [B, W, N, N]
            t_seq: [T, B, N]
            endpoints: sequence of window endpoints
        """
        B, W, N, _ = node_identity.shape

        encoded_time = self.timestamp_encoder(t_seq, endpoints)                  # [W, B, D]
        encoded_time = encoded_time.permute(1, 0, 2).unsqueeze(2)              # [B, W, 1, D]
        encoded_time = encoded_time.expand(B, W, N, self.hidden_dim)           # [B, W, N, D]

        node_features = torch.cat([node_identity, encoded_time], dim=-1)       # [B, W, N, N+D]
        node_features = self.initial_linear(node_features)                      # [B, W, N, D]

        per_layer_latents: List[torch.Tensor] = []
        per_layer_node_attention: List[torch.Tensor] = []
        per_layer_time_attention: List[torch.Tensor] = []
        ortho_penalty = node_features.new_tensor(0.0)

        h = node_features
        for layer_index, (gin, readout, temporal_block) in enumerate(
            zip(self.gin_layers, self.readouts, self.temporal_blocks)
        ):
            h = gin(h, adjacency)                                               # [B, W, N, D]

            graph_features, node_attention = readout(h)                         # [B, W, D], [B, W, N]
            temporal_input = graph_features.permute(1, 0, 2)                    # [W, B, D]

            if self.time_tokens is not None:
                cls_token = self.time_tokens[layer_index].expand(1, temporal_input.shape[1], -1)
                temporal_input = torch.cat([temporal_input, cls_token], dim=0)  # [W+1, B, D]

            temporal_output, time_attention = temporal_block(temporal_input)    # [W, B, D], [B, W, W]
            pooled_layer_representation = self._pool_time(temporal_output, layer_index)

            per_layer_latents.append(pooled_layer_representation)                # [B, D]
            per_layer_node_attention.append(node_attention)                      # [B, W, N]
            per_layer_time_attention.append(time_attention)                      # [B, W, W]
            ortho_penalty = ortho_penalty + self.orthogonality_penalty(h)

        layer_latent = torch.stack(per_layer_latents, dim=1)                    # [B, L, D]
        final_representation = layer_latent.reshape(B, self.num_layers * self.hidden_dim)
        logits = self.classifier(self.dropout(final_representation))             # [B, C]

        node_attention = torch.stack(per_layer_node_attention, dim=1)           # [B, L, W, N]
        time_attention = torch.stack(per_layer_time_attention, dim=1)           # [B, L, W, W]

        return STAGINOutput(
            logits=logits,
            node_attention=node_attention,
            time_attention=time_attention,
            layer_latent=layer_latent,
            final_representation=final_representation,
            orthogonality_penalty=ortho_penalty,
        )


# =============================================================================
# TRAINING 
# =============================================================================


def compute_loss(output: STAGINOutput, labels: torch.Tensor, reg_lambda: float) -> torch.Tensor:
    classification_loss = F.cross_entropy(output.logits, labels)
    return classification_loss + reg_lambda * output.orthogonality_penalty


def train_step(
    model: STAGIN,
    optimizer: torch.optim.Optimizer,
    node_identity: torch.Tensor,
    adjacency: torch.Tensor,
    t_seq: torch.Tensor,
    endpoints: Sequence[int],
    labels: torch.Tensor,
    reg_lambda: float,
) -> Dict[str, torch.Tensor]:
    model.train()
    optimizer.zero_grad()

    output = model(node_identity, adjacency, t_seq, endpoints)
    loss = compute_loss(output, labels, reg_lambda)
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.detach(),
        "logits": output.logits.detach(),
        "node_attention": output.node_attention.detach(),
        "time_attention": output.time_attention.detach(),
        "layer_latent": output.layer_latent.detach(),
        "final_representation": output.final_representation.detach(),
        "orthogonality_penalty": output.orthogonality_penalty.detach(),
    }


# =============================================================================
# MINIMAL DEMO
# =============================================================================


if __name__ == "__main__":
    torch.manual_seed(7)

    # -------------------------------------------------------------------------
    # 1. Fake BOLD data for confidentiality (single subject)
    # -------------------------------------------------------------------------
    bold_timeseries = torch.randn(TIMEPOINTS, N_ROIS, device=DEVICE)
    standardized_timeseries = bold_timeseries # already standardized in UKB

    # -------------------------------------------------------------------------
    # 2. Dynamic graphs 
    # -------------------------------------------------------------------------
    node_identity, adjacency, endpoints = build_dynamic_fc_graphs(
        standardized_timeseries=bold_timeseries,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
        top_percent=TOP_PERCENT,
        add_self_loops=True,
    )

    # batch dimension
    node_identity = node_identity.unsqueeze(0)                                  # [B, W, N, N]
    adjacency = adjacency.unsqueeze(0)                                          # [B, W, N, N]

    # GRU expects [T, B, N].
    t_seq = bold_timeseries.unsqueeze(1)                                        # [T, B, N]
    labels = torch.tensor([1], device=DEVICE)                                   # dummy class label

    # -------------------------------------------------------------------------
    # 3. STAGIN
    # -------------------------------------------------------------------------
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step_output = train_step(
        model=model,
        optimizer=optimizer,
        node_identity=node_identity,
        adjacency=adjacency,
        t_seq=t_seq,
        endpoints=endpoints,
        labels=labels,
        reg_lambda=REG_LAMBDA,
    )

    print(f"loss: {step_output['loss'].item():.6f}")
    print("logits:", tuple(step_output["logits"].shape))
    print("node_attention:", tuple(step_output["node_attention"].shape))
    print("time_attention:", tuple(step_output["time_attention"].shape))
    print("layer_latent:", tuple(step_output["layer_latent"].shape))
    print("final_representation:", tuple(step_output["final_representation"].shape))
    print(f"orthogonality_penalty: {step_output['orthogonality_penalty'].item():.6f}")
