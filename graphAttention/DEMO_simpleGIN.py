# single troubleshooting version
# with GRU encoder, GIN+ attention-based readout + temporal transformer

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def corrcoef(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.T @ x / (x.shape[0] - 1)
    std = cov.diag().clamp_min(1e-8).sqrt()
    corr = cov / (std[:, None] * std[None, :])
    return corr.clamp(-1, 1)


def make_dynamic_fc(timeseries: torch.Tensor, window=30, stride=5, topk=0.3):
    T, N = timeseries.shape
    A_list, endpoints = [], []

    for s in range(0, T - window + 1, stride):
        e = s + window
        fc = corrcoef(timeseries[s:e])                      
        thresh = torch.quantile(fc.flatten(), 1 - topk)
        A = (fc >= thresh).float()
        A.fill_diagonal_(1.0)
        A_list.append(A)
        endpoints.append(e)

    A = torch.stack(A_list)                               
    X = torch.eye(N, device=timeseries.device).unsqueeze(0).repeat(len(A), 1, 1)
    return X, A, endpoints


class TimestampEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim)

    def forward(self, t_seq, endpoints):
        full_out, _ = self.rnn(t_seq[:endpoints[-1]])      
        idx = torch.tensor([e - 1 for e in endpoints], device=t_seq.device)
        return full_out[idx]                                


class GINLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

    def forward(self, x, a):
        agg = torch.matmul(a, x) + self.eps * x
        b, w, n, d = agg.shape
        out = self.mlp(agg.reshape(b * w * n, d))
        return out.reshape(b, w, n, d)


class MeanReadout(nn.Module):
    def forward(self, x):
        w, b, n, _ = x.shape
        graph_emb = x.mean(dim=2)
        node_attn = torch.zeros(b, w, n, device=x.device)
        return graph_emb, node_attn

# squeeze-excitation
class SERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        mid = round(upscale * hidden_dim)
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.BatchNorm1d(mid),
            nn.GELU(),
        )
        self.attend = nn.Linear(mid, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        w, b, n, d = x.shape
        pooled = x.mean(dim=2)                             
        z = self.embed(pooled.reshape(w * b, d))          
        attn = torch.sigmoid(self.attend(z)).reshape(w, b, n)
        graph_emb = (x * self.dropout(attn.unsqueeze(-1))).mean(dim=2)
        return graph_emb, attn.permute(1, 0, 2)            

# graph attention readout
class GARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        mid = round(upscale * hidden_dim)
        self.embed_query = nn.Linear(hidden_dim, mid)
        self.embed_key = nn.Linear(hidden_dim, mid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.embed_query(x.mean(dim=2, keepdim=True))  
        k = self.embed_key(x)                               
        attn = torch.sigmoid((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])).squeeze(2)
        graph_emb = (x * self.dropout(attn.unsqueeze(-1))).mean(dim=2)
        return graph_emb, attn.permute(1, 0, 2)            


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        attended, attn = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(self.drop1(attended))               
        x2 = self.mlp(x)
        x = self.norm2(x + self.drop2(x2))
        return x, attn


class STAGIN(nn.Module):
    def __init__(
        self,
        n_nodes=100,
        hidden=64,
        n_classes=2,
        num_layers=2,
        num_heads=4,
        readout="sero",        
        cls_token="sum",       
        dropout=0.1,
        reg_lambda=0.0,
    ):
        super().__init__()
        assert readout in {"sero", "garo", "mean"}
        assert cls_token in {"sum", "mean", "param"}

        self.n_nodes = n_nodes
        self.hidden = hidden
        self.num_layers = num_layers
        self.reg_lambda = reg_lambda
        self.cls_token_mode = cls_token

        self.timestamp_encoder = TimestampEncoder(input_dim=n_nodes, hidden_dim=hidden)
        self.initial_linear = nn.Linear(n_nodes + hidden, hidden)

        self.gnn_layers = nn.ModuleList([GINLayer(hidden) for _ in range(num_layers)])

        if readout == "sero":
            self.readouts = nn.ModuleList([SERO(hidden, n_nodes, dropout=dropout) for _ in range(num_layers)])
        elif readout == "garo":
            self.readouts = nn.ModuleList([GARO(hidden, dropout=dropout) for _ in range(num_layers)])
        else:
            self.readouts = nn.ModuleList([MeanReadout() for _ in range(num_layers)])

        self.transformers = nn.ModuleList([
            TemporalTransformer(hidden, 2 * hidden, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.classifiers = nn.ModuleList([nn.Linear(hidden, n_classes) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        if cls_token == "param":
            self.token_parameter = nn.Parameter(torch.randn(num_layers, 1, 1, hidden))
        else:
            self.token_parameter = None

    def pool_cls(self, h, layer_idx):
        """
        h: [W, B, D] or [W+1, B, D] if param token used
        """
        if self.cls_token_mode == "sum":
            return h.sum(dim=0)
        if self.cls_token_mode == "mean":
            return h.mean(dim=0)
        return h[-1]  

    def orthogonality_penalty(self, h_bridge):
        """
        h_bridge: [W, B, N, D]
        """
        w, b, n, d = h_bridge.shape
        z = h_bridge.reshape(w * b, n, d)
        inner = z @ z.transpose(1, 2)                    
        inner = inner / inner.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        eye = torch.eye(n, device=h_bridge.device).unsqueeze(0)
        return (torch.triu(inner - eye, diagonal=1).norm(dim=(1, 2))).mean()

    def forward(self, x, a, t_seq, endpoints):
        b, w, n, _ = x.shape

        time_enc = self.timestamp_encoder(t_seq, endpoints)        
        time_enc = time_enc.permute(1, 0, 2).unsqueeze(2).expand(b, w, n, self.hidden)

        h = torch.cat([x, time_enc], dim=-1)                         
        h = self.initial_linear(h)                                

        logits = 0.0
        node_attn_all = []
        time_attn_all = []
        latent_all = []
        reg_ortho = 0.0

        for layer_idx, (gin, readout, transformer, clf) in enumerate(
            zip(self.gnn_layers, self.readouts, self.transformers, self.classifiers)
        ):
            h = gin(h, a)                                           
            h_bridge = h.permute(1, 0, 2, 3)                       

            h_readout, node_attn = readout(h_bridge)               

            if self.token_parameter is not None:
                token = self.token_parameter[layer_idx].expand(1, h_readout.shape[1], -1)
                h_readout = torch.cat([h_readout, token], dim=0)    

            h_attend, time_attn = transformer(h_readout)            
            latent = self.pool_cls(h_attend, layer_idx)              

            logits = logits + self.dropout(clf(latent))
            latent_all.append(latent)
            node_attn_all.append(node_attn)
            time_attn_all.append(time_attn)

            if self.reg_lambda > 0:
                reg_ortho = reg_ortho + self.orthogonality_penalty(h_bridge)

        node_attn_all = torch.stack(node_attn_all, dim=1)          
        time_attn_all = torch.stack(time_attn_all, dim=1)           
        latent_all = torch.stack(latent_all, dim=1)                

        return logits, node_attn_all, time_attn_all, latent_all, reg_ortho


def train_step(model, batch_x, batch_a, batch_t, endpoints, batch_y, optimizer):
    model.train()
    optimizer.zero_grad()

    logits, node_attn, time_attn, latent, reg_ortho = model(batch_x, batch_a, batch_t, endpoints)
    loss = F.cross_entropy(logits, batch_y)

    if model.reg_lambda > 0:
        loss = loss + model.reg_lambda * reg_ortho

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "logits": logits.detach(),
        "node_attn": node_attn.detach(),
        "time_attn": time_attn.detach(),
        "latent": latent.detach(),
        "reg_ortho": reg_ortho.detach() if torch.is_tensor(reg_ortho) else torch.tensor(reg_ortho),
    }


if __name__ == "__main__":
    # single subj : rand data for confidentiality
    ts = torch.randn(490, 100)                               

    X, A, endpoints = make_dynamic_fc(ts, window=30, stride=5, topk=0.3)
    X = X.unsqueeze(0)                                       
    A = A.unsqueeze(0)                                        
    t_seq = ts.unsqueeze(1)                                 
    y = torch.tensor([1])

    model = STAGIN(
        n_nodes=100,
        hidden=64,
        n_classes=2,
        num_layers=2,
        num_heads=4,
        readout="sero",   
        cls_token="sum",   
        dropout=0.1,
        reg_lambda=1e-4,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    out = train_step(model, X, A, t_seq, endpoints, y, optimizer)

    print("loss:", out["loss"])
    print("node_attn:", out["node_attn"].shape)  
    print("time_attn:", out["time_attn"].shape)   
    print("latent:", out["latent"].shape)         
    print("reg_ortho:", out["reg_ortho"].item())
