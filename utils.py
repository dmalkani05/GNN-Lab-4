import numpy as np
import matplotlib.pyplot as plt

def plot_pathloss(d, Eh):
    plt.figure()
    plt.plot(d, Eh)
    plt.yscale('log')
    plt.xlabel('Distance (m)')
    plt.ylabel('E[h] (log scale)')
    plt.title('Pathloss vs. distance')
    plt.show()

def ribbon_by_distance(d_grid, samples_by_d, ylabel, title):
    """
    samples_by_d: list of arrays (Q,) per distance
    Plots median with 10–90 percentile ribbon.
    """
    meds, p10, p90 = [], [], []
    for arr in samples_by_d:
        meds.append(np.median(arr))
        p10.append(np.percentile(arr, 10))
        p90.append(np.percentile(arr, 90))
    meds, p10, p90 = map(np.array, (meds, p10, p90))
    plt.figure()
    plt.fill_between(d_grid, p10, p90, alpha=0.3)
    plt.plot(d_grid, meds)
    plt.xlabel('Distance (m)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def scatter_network(tx, rx, wx, wy):
    plt.figure()
    plt.scatter(tx[:,0], tx[:,1], s=20, label='TX', marker='s')
    plt.scatter(rx[:,0], rx[:,1], s=20, label='RX', marker='o')
    plt.xlim(0, wx); plt.ylim(0, wy)
    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    plt.title('Network layout')
    plt.show()

def avg_loss_cap_pow(model, H_np, N0):
    import torch
    H = torch.tensor(H_np, dtype=torch.float32)
    N0t = torch.tensor(N0, dtype=torch.float32)
    with torch.no_grad():
        P = model(H)                       # (B, n)
        diagH = torch.diagonal(H, dim1=1, dim2=2)
        signal = diagH * P
        Hp = torch.einsum('bij,bi->bj', H.transpose(1,2), P)
        sinr = signal / (N0t + (Hp - signal))
        cap = torch.log1p(sinr).cpu().numpy()
        pow_avg = P.mean(dim=0).cpu().numpy()
    return cap.mean(), cap.mean(axis=0), pow_avg
