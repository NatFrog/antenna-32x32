# Dataset loading and stratified splitting

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATASET = Path(__file__).parent / "data" / "dataset.npz"


@dataclass
class Data:
    weights: torch.Tensor  # complex64 [N,32,32]
    gain_db: torch.Tensor  # float32 [N,T,P]
    theta_deg: torch.Tensor
    phi_deg: torch.Tensor
    n_beams: np.ndarray
    names: np.ndarray

    def to(self, device):
        self.weights = self.weights.to(device)
        self.gain_db = self.gain_db.to(device)
        self.theta_deg = self.theta_deg.to(device)
        self.phi_deg = self.phi_deg.to(device)
        return self


def load(path=DATASET) -> Data:
    d = np.load(path)
    return Data(
        weights=torch.from_numpy(d["weights"]),
        gain_db=torch.from_numpy(d["gain_db"]),
        theta_deg=torch.from_numpy(d["theta_deg"]),
        phi_deg=torch.from_numpy(d["phi_deg"]),
        n_beams=d["n_beams"],
        names=d["names"],
    )


def split(n_beams: np.ndarray, seed: int = 0, val_frac=0.1, test_frac=0.1):
    """Stratified (per beam count) train/val/test index split."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for k in np.unique(n_beams):
        idx = rng.permutation(np.flatnonzero(n_beams == k))
        n_te = int(round(len(idx) * test_frac))
        n_va = int(round(len(idx) * val_frac))
        te += idx[:n_te].tolist()
        va += idx[n_te:n_te + n_va].tolist()
        tr += idx[n_te + n_va:].tolist()
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))
