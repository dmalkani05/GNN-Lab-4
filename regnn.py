import numpy as np
import torch
import torch.nn as nn

def _poly_powers(H, K):
    """
    Compute [I, H, H^2, ..., H^{K-1}] for a batch of graphs.
    H: (B, n, n)  -> returns list of length K, each (B, n, n)
    """
    B, n, _ = H.shape
    powers = [torch.eye(n, device=H.device).expand(B, n, n)]
    for _ in range(1, K):
        powers.append(powers[-1] @ H)
    return powers

class REGNNLayer(nn.Module):
    """
    One MIMO graph-convolution layer:
      Y = sum_{k=0}^{K-1} H^k X A_k,  then nonlinearity
    """
    def __init__(self, inF, outF, K):
        super().__init__()
        self.K = K
        self.A = nn.ParameterList([nn.Parameter(torch.randn(inF, outF)*0.01) for _ in range(K)])
        self.act = nn.ReLU()

    def forward(self, H, X):
        B, n, _ = H.shape
        Ks = _poly_powers(H, self.K)
        out = 0.0
        for k in range(self.K):
            out = out + (Ks[k] @ X) @ self.A[k]
        return self.act(out)

class REGNN(nn.Module):
    def __init__(self, n_features=[8,4,1], K=5):
        super().__init__()
        self.K = K
        layers = []
        dims = [1] + n_features
        for i in range(len(dims)-1):
            layers.append(REGNNLayer(dims[i], dims[i+1], K))
        self.layers = nn.ModuleList(layers)

    def forward(self, H):
        B, n, _ = H.shape
        X = torch.ones(B, n, 1, device=H.device) 
        for layer in self.layers:
            X = layer(H, X)
        return X.squeeze(-1)

def batch_loss_penalized(model, H_batch, N0, mu=0.01):
    """
    H_batch: (B, n, n) torch
    Returns scalar loss we *maximize*. We’ll train by minimizing (-loss).
    """
    P = model(H_batch)                       
    diagH = torch.diagonal(H_batch, dim1=1, dim2=2)      
    signal = diagH * P
    Hp = torch.einsum('bij,bi->bj', H_batch.transpose(1,2), P) 
    sinr = signal / (N0 + (Hp - signal))
    cap = torch.log1p(sinr)                 

    reward = cap.sum(dim=1).mean()           
    power_pen = mu * P.mean()
    return reward - power_pen

def train_unconstrained(model, net_sampler, steps=200, batch_graphs=100, mu=0.01, lr=1e-2, device='cpu'):
    """
    net_sampler: callable(q) -> numpy H with shape (q, n, n)
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N0 = torch.tensor(net_sampler.N0, device=device, dtype=torch.float32)

    for it in range(steps):
        H_np = net_sampler.sample_H(q=batch_graphs)    
        H = torch.tensor(H_np, dtype=torch.float32, device=device)

        opt.zero_grad()
        obj = batch_loss_penalized(model, H, N0, mu=mu)
        loss = -obj
        loss.backward()
        opt.step()

    return model

def train_primal_dual(model, net_sampler, steps=200, batch_graphs=100,
                      p_budget=1e-3, lr_primal=1e-2, lr_dual=1e-3, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr_primal)
    N0 = torch.tensor(net_sampler.N0, device=device, dtype=torch.float32)
    n = net_sampler.n
    mu = torch.zeros(n, device=device, dtype=torch.float32)

    for it in range(steps):
        H_np = net_sampler.sample_H(q=batch_graphs)
        H = torch.tensor(H_np, dtype=torch.float32, device=device)
        P = model(H) 

        
        diagH = torch.diagonal(H, dim1=1, dim2=2)
        signal = diagH * P
        Hp = torch.einsum('bij,bi->bj', H.transpose(1,2), P)
        sinr = signal / (N0 + (Hp - signal))
        cap = torch.log1p(sinr)              

        avg_pow = P.mean(dim=0)              
        reward = cap.sum(dim=1).mean()
        lagrangian = reward - torch.dot(mu, avg_pow - torch.full_like(avg_pow, p_budget))

        opt.zero_grad()
        (-lagrangian).backward()
        opt.step()

        with torch.no_grad():
            mu -= lr_dual * (avg_pow - p_budget)
            mu.clamp_(min=0.0)

    return model, mu.detach().cpu().numpy()
