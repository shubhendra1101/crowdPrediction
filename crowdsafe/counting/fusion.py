"""Confidence-weighted fusion of detector and density counts (architecture.md §2.2, T3.7).

    N = w · N_det + (1 − w) · N_density,   w = sigmoid(θ · [1, standardised f])

f = reliability features (mean detection confidence, detection count, share of overlapping boxes,
mean box height, density-model count). θ is fitted on validation frames by minimising the mean
absolute error of N against ground truth. Also provides the "hard switch" baseline: detector count
below a threshold τ (fitted on validation), density count above it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class Fusion:
    """Learned per-frame (or per-zone) weight between a detector count and a density count."""

    def __init__(self, l2: float = 1e-3, seed: int = 42) -> None:
        self.l2, self.seed = l2, seed
        self.theta: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def _design(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.mu) / self.sd
        return np.column_stack([np.ones(len(z)), z])

    def weights(self, feats: np.ndarray) -> np.ndarray:
        """Detector weight w in [0, 1] per row."""
        return sigmoid(self._design(np.asarray(feats, float)) @ self.theta)

    def fit(self, feats: np.ndarray, n_det: np.ndarray, n_dens: np.ndarray, target: np.ndarray) -> "Fusion":
        """Fit θ by minimising MAE (+ small L2) of the fused count on validation rows."""
        feats = np.asarray(feats, float)
        self.mu, self.sd = feats.mean(0), feats.std(0) + 1e-9
        X = self._design(feats)
        n_det, n_dens, target = (np.asarray(v, float) for v in (n_det, n_dens, target))

        def loss(th: np.ndarray) -> float:
            w = sigmoid(X @ th)
            err = w * n_det + (1 - w) * n_dens - target
            return float(np.mean(np.sqrt(err ** 2 + 1e-6)) + self.l2 * np.sum(th[1:] ** 2))   # smooth |err|

        rng = np.random.default_rng(self.seed)
        best = None
        for start in [np.zeros(X.shape[1])] + [rng.normal(0, 1, X.shape[1]) for _ in range(4)]:
            r = minimize(loss, start, method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x
        return self

    def __call__(self, feats: np.ndarray, n_det: np.ndarray, n_dens: np.ndarray) -> np.ndarray:
        """Fused counts."""
        w = self.weights(feats)
        return w * np.asarray(n_det, float) + (1 - w) * np.asarray(n_dens, float)


def fit_hard_switch(n_det: np.ndarray, n_dens: np.ndarray, target: np.ndarray) -> float:
    """Threshold τ on the density count: use the detector when density count < τ, else density. Minimises MAE."""
    n_det, n_dens, target = (np.asarray(v, float) for v in (n_det, n_dens, target))
    cands = np.unique(np.concatenate([[-np.inf, np.inf], n_dens]))
    maes = [np.mean(np.abs(np.where(n_dens < t, n_det, n_dens) - target)) for t in cands]
    return float(cands[int(np.argmin(maes))])


def hard_switch(n_det: np.ndarray, n_dens: np.ndarray, tau: float) -> np.ndarray:
    return np.where(np.asarray(n_dens, float) < tau, n_det, n_dens).astype(float)


def fit_constant_weight(n_det: np.ndarray, n_dens: np.ndarray, target: np.ndarray) -> float:
    """Single detector weight w in [0, 1] (same for every frame) minimising MAE — the no-features baseline."""
    from scipy.optimize import minimize_scalar

    n_det, n_dens, target = (np.asarray(v, float) for v in (n_det, n_dens, target))
    r = minimize_scalar(lambda w: np.mean(np.abs(w * n_det + (1 - w) * n_dens - target)), bounds=(0, 1), method="bounded")
    return float(r.x)
