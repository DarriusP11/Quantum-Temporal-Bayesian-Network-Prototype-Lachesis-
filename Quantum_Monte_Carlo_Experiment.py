# Quantum Monte Carlo Experiment 

"""Mimicked Quantum Monte Carlo for portfolio weights.

This follows the Medium article's idea: generate uniform random weights using
Hadamard superposition and measurement, then post-process to weights.
Note: This is NOT true QMC; it is a quantum-inspired sampler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# Optional plotting and data sources
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - plotting is optional
    plt = None
    sns = None

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "yfinance is required for this example. Install with: pip install yfinance"
    ) from exc


@dataclass
class PortfolioSamples:
    returns: List[float]
    risks: List[float]
    sharpes: List[float]
    weights: List[np.ndarray]


def bitstring_to_frac(bitstring: str) -> float:
    """Convert a bitstring to a fraction in [0, 1)."""
    num = 0.0
    for i, b in enumerate(bitstring):
        num += int(b) / (2 ** (i + 1))
    return num


def quantum_bitstrings(num_bits: int, shots: int) -> List[str]:
    """Sample bitstrings from a uniform superposition circuit.

    Falls back to a classical uniform random generator if Qiskit is unavailable.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit(num_bits)
        qc.h(range(num_bits))
        qc.measure_all()

        backend = AerSimulator()
        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()

        samples: List[str] = []
        for bitstring, count in counts.items():
            bitstring = bitstring.replace(" ", "")
            samples.extend([bitstring] * count)
        return samples
    except Exception:
        rng = np.random.default_rng()
        return [
            "".join(rng.integers(0, 2, size=num_bits).astype(str))
            for _ in range(shots)
        ]


def sample_portfolios(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    num_assets: int,
    qubits_per_asset: int,
    shots: int,
) -> Tuple[PortfolioSamples, PortfolioSamples]:
    """Generate quantum-inspired and classical MC samples."""

    num_bits = num_assets * qubits_per_asset
    bitstrings = quantum_bitstrings(num_bits, shots)

    q_returns, q_risks, q_sharpes, q_weights = [], [], [], []
    c_returns, c_risks, c_sharpes, c_weights = [], [], [], []

    for state in bitstrings:
        # Quantum-inspired weights
        q_nums = []
        for k in range(0, len(state), qubits_per_asset):
            sub = state[k : k + qubits_per_asset]
            q_nums.append(bitstring_to_frac(sub))
        q_nums = np.array(q_nums, dtype=float)
        if q_nums.sum() == 0:
            q_w = np.ones(num_assets) / num_assets
        else:
            q_w = q_nums / q_nums.sum()
        q_weights.append(q_w)

        q_ret = float(q_w @ mean_returns) * 100.0
        q_risk = float(math.sqrt(q_w @ (covariance * 252.0) @ q_w)) * 100.0
        q_sharpe = q_ret / q_risk if q_risk != 0 else 0.0

        q_returns.append(q_ret)
        q_risks.append(q_risk)
        q_sharpes.append(q_sharpe)

        # Classical MC weights
        c_nums = np.random.random(num_assets)
        c_w = c_nums / c_nums.sum()
        c_weights.append(c_w)

        c_ret = float(c_w @ mean_returns) * 100.0
        c_risk = float(math.sqrt(c_w @ (covariance * 252.0) @ c_w)) * 100.0
        c_sharpe = c_ret / c_risk if c_risk != 0 else 0.0

        c_returns.append(c_ret)
        c_risks.append(c_risk)
        c_sharpes.append(c_sharpe)

    return (
        PortfolioSamples(q_returns, q_risks, q_sharpes, q_weights),
        PortfolioSamples(c_returns, c_risks, c_sharpes, c_weights),
    )


def main() -> None:
    assets = ["MSFT", "GOOGL", "AMZN", "TSLA"]
    data = yf.download(assets, start="2018-01-01", end="2020-01-31")["Close"]
    returns = data.pct_change().dropna()

    mean_returns = returns.mean().to_numpy() * 252.0
    covariance = returns.cov().to_numpy()

    if sns is not None:
        sns.heatmap(returns.corr())
        if plt is not None:
            plt.title("Asset Correlation")
            plt.show()

    qubits_per_asset = 4
    shots = 500

    q_samples, c_samples = sample_portfolios(
        mean_returns, covariance, len(assets), qubits_per_asset, shots
    )

    best_q_idx = int(np.argmax(q_samples.sharpes))
    best_c_idx = int(np.argmax(c_samples.sharpes))

    print("Best Quantum-Inspired Sample")
    print(
        {
            "Sharpe": q_samples.sharpes[best_q_idx],
            "Return": q_samples.returns[best_q_idx],
            "Risk": q_samples.risks[best_q_idx],
            "weights": q_samples.weights[best_q_idx],
        }
    )

    print("Best Classical MC Sample")
    print(
        {
            "Sharpe": c_samples.sharpes[best_c_idx],
            "Return": c_samples.returns[best_c_idx],
            "Risk": c_samples.risks[best_c_idx],
            "weights": c_samples.weights[best_c_idx],
        }
    )

    if plt is not None:
        plt.scatter(c_samples.risks, c_samples.returns, alpha=0.5, label="Classical")
        plt.scatter(q_samples.risks, q_samples.returns, alpha=0.6, label="Quantum")
        plt.scatter(
            c_samples.risks[best_c_idx],
            c_samples.returns[best_c_idx],
            color="red",
            label="Best Classical",
        )
        plt.scatter(
            q_samples.risks[best_q_idx],
            q_samples.returns[best_q_idx],
            color="green",
            label="Best Quantum",
        )
        plt.xlabel("Risk")
        plt.ylabel("Return")
        plt.legend(loc="lower right")
        plt.title("Risk vs Return")
        plt.show()


if __name__ == "__main__":
    main()
