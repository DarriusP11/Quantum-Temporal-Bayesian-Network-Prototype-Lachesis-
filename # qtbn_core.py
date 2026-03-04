# qtbn_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class QTBNConfig:
    """
    Minimal config for a toy Quantum Temporal Bayesian Network (QTBN) engine.
    This is *classical* under the hood but structured like a temporal BN.
    """
    regimes: List[str]                      # e.g. ["calm", "stressed", "crisis"]
    transition_matrix: np.ndarray           # shape (R, R) : P(Regime_{t+1} | Regime_t)
    drift_by_regime: Dict[str, float]       # μ per regime
    risk_on_by_regime: Dict[str, float]     # P(risk-on) per regime
    step_horizon_days: int = 10             # interpret each step as N days

    def check(self) -> None:
        R = len(self.regimes)
        if self.transition_matrix.shape != (R, R):
            raise ValueError("transition_matrix must be (R, R)")
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Each row of transition_matrix must sum to 1.")
        for r in self.regimes:
            if r not in self.drift_by_regime:
                raise ValueError(f"Missing drift for regime '{r}'")
            if r not in self.risk_on_by_regime:
                raise ValueError(f"Missing risk_on for regime '{r}'")


class QTBNEngine:
    """
    Tiny engine that rolls regime probabilities forward over multiple time steps.
    This is a toy stand-in for a full Quantum Temporal Bayesian Network.
    """

    def __init__(self, config: QTBNConfig, prior_regime_probs: np.ndarray):
        self.config = config
        self.config.check()
        prior_regime_probs = np.asarray(prior_regime_probs, dtype=float)
        if prior_regime_probs.shape != (len(config.regimes),):
            raise ValueError("prior_regime_probs must have shape (R,)")
        if prior_regime_probs.sum() <= 0:
            raise ValueError("prior_regime_probs must have positive mass")
        self.prior = prior_regime_probs / prior_regime_probs.sum()

    def forward(self, n_steps: int = 3):
        """
        Roll regime distribution forward n_steps.
        Returns a dict with:
          - 'regime_paths': list of np.ndarray, length n_steps+1 (including t0)
          - 'drift_path':   list of floats
          - 'risk_on_path': list of floats
        """
        R = len(self.config.regimes)
        T = n_steps

        regime_paths = [self.prior.copy()]
        drift_path: List[float] = []
        risk_on_path: List[float] = []

        def summarize(p_reg: np.ndarray):
            mu = 0.0
            risk_on = 0.0
            for i, r in enumerate(self.config.regimes):
                mu += p_reg[i] * float(self.config.drift_by_regime[r])
                risk_on += p_reg[i] * float(self.config.risk_on_by_regime[r])
            return mu, risk_on

        # t0 summary
        mu0, ro0 = summarize(regime_paths[0])
        drift_path.append(mu0)
        risk_on_path.append(ro0)

        # propagate
        P = self.config.transition_matrix
        for _ in range(T):
            next_p = regime_paths[-1] @ P  # shape (R,)
            regime_paths.append(next_p)
            mu_t, ro_t = summarize(next_p)
            drift_path.append(mu_t)
            risk_on_path.append(ro_t)

        return {
            "regime_paths": regime_paths,   # list of length T+1
            "drift_path": drift_path,       # length T+1
            "risk_on_path": risk_on_path,   # length T+1
        }
