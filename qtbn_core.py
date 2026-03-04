from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class QTBNConfig:
    """Minimal config for a toy Quantum Temporal Bayesian Network (QTBN) engine."""

    regimes: List[str]
    transition_matrix: np.ndarray
    drift_by_regime: Dict[str, float]
    risk_on_by_regime: Dict[str, float]
    step_horizon_days: int = 10

    def check(self) -> None:
        r = len(self.regimes)
        if self.transition_matrix.shape != (r, r):
            raise ValueError("transition_matrix must be (R, R)")
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Each row of transition_matrix must sum to 1.")
        for regime in self.regimes:
            if regime not in self.drift_by_regime:
                raise ValueError(f"Missing drift for regime '{regime}'")
            if regime not in self.risk_on_by_regime:
                raise ValueError(f"Missing risk_on for regime '{regime}'")


class QTBNEngine:
    """Tiny engine that rolls regime probabilities forward over multiple time steps."""

    def __init__(self, config: QTBNConfig, prior_regime_probs: np.ndarray):
        self.config = config
        self.config.check()
        probs = np.asarray(prior_regime_probs, dtype=float)
        if probs.shape != (len(config.regimes),):
            raise ValueError("prior_regime_probs must have shape (R,)")
        total = probs.sum()
        if total <= 0:
            raise ValueError("prior_regime_probs must have positive mass")
        self.prior = probs / total

    def forward(self, n_steps: int = 3):
        """Roll regime distribution forward n_steps and summarize drift/risk-on paths."""

        regime_paths = [self.prior.copy()]
        drift_path: List[float] = []
        risk_on_path: List[float] = []

        def summarize(p_reg: np.ndarray) -> tuple[float, float]:
            mu = 0.0
            risk_on = 0.0
            for idx, regime in enumerate(self.config.regimes):
                mu += p_reg[idx] * float(self.config.drift_by_regime[regime])
                risk_on += p_reg[idx] * float(self.config.risk_on_by_regime[regime])
            return mu, risk_on

        mu0, ro0 = summarize(regime_paths[0])
        drift_path.append(mu0)
        risk_on_path.append(ro0)

        transition = self.config.transition_matrix
        for _ in range(n_steps):
            next_probs = regime_paths[-1] @ transition
            regime_paths.append(next_probs)
            mu_t, ro_t = summarize(next_probs)
            drift_path.append(mu_t)
            risk_on_path.append(ro_t)

        return {
            "regime_paths": regime_paths,
            "drift_path": drift_path,
            "risk_on_path": risk_on_path,
        }
