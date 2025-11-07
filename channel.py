import numpy as np

class WirelessChannel:
    """
    Handles large-scale pathloss, small-scale fading samples, and capacity evals
    for a single point-to-point link model.
    """
    def __init__(self, d0=1.0, gamma=2.2, s=2.0, N0=1e-6, rng=None):
        self.d0 = d0
        self.gamma = gamma
        self.s = s
        self.N0 = N0
        self.rng = np.random.default_rng(rng)

    def pathloss(self, d):
        """E[h] = (d0/d)^gamma, d>0"""
        d = np.asarray(d, dtype=float)
        return (self.d0 / np.clip(d, 1e-12, None)) ** self.gamma

    def sample_fading(self, q=1, d=1.0):
        """
        Draw q i.i.d. realizations of h using:
        h = E[h] * ( h_tilde / E[h_tilde] ),  where h_tilde ~ Exp(scale=s)
        """
        Eh = self.pathloss(d)
        h_tilde = self.rng.exponential(scale=self.s, size=q)
        return Eh * (h_tilde / self.s)

    def capacity(self, p, h):
        """c = log(1 + h*p / N0)"""
        p = np.asarray(p, dtype=float)
        h = np.asarray(h, dtype=float)
        return np.log(1.0 + (h * p) / self.N0)