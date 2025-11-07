import numpy as np

class WirelessNetwork:
    """
    Random spatial wireless network:
      - n TX uniformly in [0, wx] x [0, wy]
      - RX_i within a circle radius wc around TX_i
      - Pathloss matrix E[H] via distances + (d0/d)^gamma
      - Fading samples H via exponential-based model
      - Network capacities with SINR formula
    """
    def __init__(self, n, wx, wy, wc, d0=1.0, gamma=2.2, s=2.0, N0=1e-6, rng=None):
        self.n   = n
        self.wx  = wx
        self.wy  = wy
        self.wc  = wc
        self.d0  = d0
        self.gamma = gamma
        self.s   = s
        self.N0  = N0
        self.rng = np.random.default_rng(rng)

        self.tx = np.column_stack([
            self.rng.uniform(0, wx, size=n),
            self.rng.uniform(0, wy, size=n)
        ])
    
        ang = self.rng.uniform(0, 2*np.pi, size=n)
        rad = self.rng.uniform(0, wc, size=n)
        self.rx = self.tx + np.column_stack([rad*np.cos(ang), rad*np.sin(ang)])

        self.EH = self._compute_pathloss_matrix()

    def _compute_pathloss_matrix(self):
        n = self.n
        EH = np.zeros((n, n), dtype=float)
        for i in range(n):
            diff = self.rx - self.tx[i]  
            dij = np.linalg.norm(diff, axis=1)  
            EH[i, :] = (self.d0 / np.clip(dij, 1e-12, None)) ** self.gamma
        return EH

    def sample_H(self, q=1):
        """
        Return q samples of the interference matrix H.
        h_ij = E[h_ij] * (exp(mean=s) / s)
        """
        n = self.n
        tilde = self.rng.exponential(scale=self.s, size=(q, n, n))
        return self.EH[None, :, :] * (tilde / self.s)

    def capacities(self, P, H):
        """
        Compute capacities for each sample (q) and node (n).
        Inputs:
          P: array shape (q, n) or (n,)   transmit powers
          H: array shape (q, n, n)        channel gains
        Returns:
          C: array shape (q, n)
        """
        H = np.asarray(H, dtype=float)
        q, n, _ = H.shape
        P = np.asarray(P, dtype=float)
        if P.ndim == 1:
            P = np.repeat(P[None, :], q, axis=0) 

        diagH = np.einsum('qii->qi', H)  
        signal = diagH * P

        Hp = np.einsum('qji,qj->qi', H, P) 
        sinr = signal / (self.N0 + (Hp - signal))
        return np.log(1.0 + sinr)