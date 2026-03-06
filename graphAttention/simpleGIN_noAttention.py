# for troubleshooting 
# GIN for single subj, single train step, no attention, no regularization

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# for ICA timeseries
def corrcoef(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.T @ x / (x.shape[0] - 1)
    std = cov.diag().clamp_min(1e-8).sqrt()
    corr = cov / (std[:, None] * std[None, :])
    return corr.clamp(-1, 1)

def make_dynamic_fc(timeseries: torch.Tensor, window=30, stride=5, topk=0.3):
    """
    timeseries: [T, 100]
    returns
        X: node features (identity, same for every window)
        A: binary adjacency from windowed FC
    """
    T, N = timeseries.shape

    A_list = []
    for s in range(0, T - window + 1, stride):
        fc = corrcoef(timeseries[s:s + window])          
        thresh = torch.quantile(fc.flatten(), 1 - topk)  # 30% 
        A = (fc >= thresh).float()
        A.fill_diagonal_(1.0)
        A_list.append(A)

    A = torch.stack(A_list)                               
    X = torch.eye(N).unsqueeze(0).repeat(len(A), 1, 1)   
    return X, A


class GINLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, a):
        agg = torch.matmul(a, x)
        return self.mlp((1 + self.eps) * x + agg)


class SpatialReadout(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        node_attn = torch.softmax(self.score(x).squeeze(-1), dim=-1) 
        graph_emb = (x * node_attn.unsqueeze(-1)).sum(dim=2)       
        return graph_emb, node_attn

class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h):
        time_attn = torch.softmax(self.score(h).squeeze(-1), dim=-1)  
        out = (h * time_attn.unsqueeze(-1)).sum(dim=1)                
        return out, time_attn

class SimpleSTAGIN(nn.Module):
    def __init__(self, n_nodes=100, hidden=64, n_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(n_nodes, hidden)  
        self.gin1 = GINLayer(hidden)
        self.gin2 = GINLayer(hidden)
        self.readout = SpatialReadout(hidden)
        self.time_pool = TemporalAttention(hidden)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x, a):
        x = self.input_proj(x)
        x = self.gin1(x, a)
        x = self.gin2(x, a)

        graph_seq, node_attn = self.readout(x)   
        subject_emb, time_attn = self.time_pool(graph_seq)
        logits = self.classifier(subject_emb)

        return logits, node_attn, time_attn


def train_step(model, batch_x, batch_a, batch_y, optimizer):
    model.train()
    optimizer.zero_grad()
    logits, node_attn, time_attn = model(batch_x, batch_a)
    loss = F.cross_entropy(logits, batch_y)
    loss.backward()
    optimizer.step()
    return loss.item(), node_attn.detach(), time_attn.detach()


if __name__ == "__main__":
    # UKB confidentiality guidelines
    ts = torch.randn(490, 100)
    #path = "XXX"
    #ts = np.loadtxt(path)
    #ts = torch.tensor(ts, dtype=torch.float32)

    X, A = make_dynamic_fc(ts, window=30, stride=5, topk=0.3)

    X = X.unsqueeze(0)               
    A = A.unsqueeze(0)           
    y = torch.tensor([1])      

    model = SimpleSTAGIN(n_nodes=100, hidden=64, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss, node_attn, time_attn = train_step(model, X, A, y, optimizer)
    print("loss:", loss)
    print("node_attn:", node_attn.shape)   
    print("time_attn:", time_attn.shape)   