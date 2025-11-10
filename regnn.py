import numpy as np
import torch
import torch.nn as nn

def _poly_powers(H, K):
    """Return [I, H, H^2, ..., H^{K-1}] for a batch of graphs."""
    B, n, _ = H.shape
    I = torch.eye(n, device=H.device).expand(B, n, n)
    out = [I]
    for _ in range(1, K):
        out.append(out[-1] @ H)
    return out

def _capacities_torch(P, H, N0, eps=1e-12):
    """
    P: (B, n)   nonnegative powers
    H: (B, n, n)
    returns capacities (B, n)
    """
    diagH = torch.diagonal(H, dim1=1, dim2=2)            
    signal = diagH * P

    Hp = torch.einsum('bij,bi->bj', H.transpose(1, 2), P)
    denom = N0 + (Hp - signal)
    denom = torch.clamp(denom, min=eps)

    sinr = signal / denom
    sinr = torch.clamp(sinr, min=0.0, max=1e6) 
    cap = torch.log1p(sinr)
    return cap

class REGNNLayer(nn.Module):
    """MIMO graph filter layer: Y = sum_k H^k X A_k, then ReLU."""
    def __init__(self, inF, outF, K):
        super().__init__()
        self.K = K
        self.A = nn.ParameterList([nn.Parameter(torch.empty(inF, outF)) for _ in range(K)])
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for A in self.A:
            nn.init.xavier_uniform_(A, gain=0.1) 

    def forward(self, H, X):
        Ks = _poly_powers(H, self.K)
        Y = 0.0
        for k in range(self.K):
            Y = Y + (Ks[k] @ X) @ self.A[k]
        return self.act(Y)

class REGNN(nn.Module):
    """
    REGNN with final bounded power head:
      raw -> Softplus -> Sigmoid -> scale by pmax
    """
    def __init__(self, n_features=[8,4,1], K=5, pmax=1e-1):
        super().__init__()
        self.K = K
        self.pmax = pmax
        dims = [1] + n_features
        self.layers = nn.ModuleList([REGNNLayer(dims[i], dims[i+1], K) for i in range(len(dims)-1)])
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, H):
        B, n, _ = H.shape
        X = torch.ones(B, n, 1, device=H.device)
        for layer in self.layers:
            X = layer(H, X)
        raw = X.squeeze(-1)                  
        P = self.pmax * torch.sigmoid(self.softplus(raw))
        return P

def batch_objective_penalized(model, H_batch, N0, mu=0.05):
    P = model(H_batch)                                        
    cap = _capacities_torch(P, H_batch, N0)                   
    reward = cap.sum(dim=1).mean()
    power_pen = mu * P.mean()
    return reward - power_pen

def train_unconstrained(model, net_sampler, steps=200, batch_graphs=100,
                        mu=0.05, lr=5e-4, weight_decay=1e-5, device='cpu'):
    torch.manual_seed(0)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    N0 = torch.tensor(net_sampler.N0, device=device, dtype=torch.float32)

    for _ in range(steps):
        H_np = net_sampler.sample_H(q=batch_graphs)
        H = torch.tensor(H_np, dtype=torch.float32, device=device)
        opt.zero_grad()
        obj = batch_objective_penalized(model, H, N0, mu=mu)
        loss = -obj
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
    return model

def train_primal_dual(model, net_sampler, steps=200, batch_graphs=100,
                      p_budget=1e-3, lr_primal=5e-4, lr_dual=1e-4,
                      weight_decay=1e-5, device='cpu'):
    torch.manual_seed(0)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr_primal, weight_decay=weight_decay)
    N0 = torch.tensor(net_sampler.N0, device=device, dtype=torch.float32)
    n = net_sampler.n
    mu = torch.full((n,), 1.0, device=device, dtype=torch.float32)

    for _ in range(steps):
        H_np = net_sampler.sample_H(q=batch_graphs)
        H = torch.tensor(H_np, dtype=torch.float32, device=device)

        P = model(H)                                 
        cap = _capacities_torch(P, H, N0)            
        avg_pow = P.mean(dim=0)                      

        reward = cap.sum(dim=1).mean()
        lag = reward - torch.dot(mu, (avg_pow - torch.full_like(avg_pow, p_budget)))

        opt.zero_grad()
        (-lag).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        with torch.no_grad():
            mu += lr_dual * (avg_pow - p_budget)
            mu.clamp_(min=0.0)

    return model, mu.detach().cpu().numpy()
