# qaoa_scenario1.py


from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# -------------------------------
# Basic config & demo portfolios
# -------------------------------

try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit.algorithms.minimum_eigensolvers import (
        NumPyMinimumEigensolver,
        QAOA,
    )
    from qiskit.algorithms.optimizers import COBYLA

    _HAS_QISKIT_OPT = True
except Exception:
    _HAS_QISKIT_OPT = False

LOG_PATH = "qaoa_runs_log.csv"
SCENARIO_PATH = "qaoa_scenarios.json"
QAOA_SNAPSHOT_PATH = "qaoa_snapshot.json"  # snapshot for QTBN foresight engine

TOY_QAOA_PORTFOLIO: Dict[str, Any] = {
    "name": "Toy 3-asset tech portfolio",
    "assets": ["AAPL", "MSFT", "GOOG"],
    "mu": [0.10, 0.12, 0.08],
    "cov": [
        [0.0400, 0.0280, 0.0220],
        [0.0280, 0.0500, 0.0240],
        [0.0220, 0.0240, 0.0450],
    ],
    "risk_aversion": 2.0,
}

LACHESIS_BENCHMARK_PORTFOLIO = {
    "name": "Lachesis benchmark (equities + bond + gold)",
    "assets": ["AAPL", "MSFT", "QQQ", "TLT", "GLD"],
    "mu": [0.11, 0.12, 0.09, 0.04, 0.06],
    "cov": [
        [0.0500, 0.0400, 0.0450, 0.0100, 0.0150],
        [0.0400, 0.0550, 0.0480, 0.0120, 0.0180],
        [0.0450, 0.0480, 0.0600, 0.0150, 0.0200],
        [0.0100, 0.0120, 0.0150, 0.0200, 0.0080],
        [0.0150, 0.0180, 0.0200, 0.0080, 0.0300],
    ],
    "risk_aversion": 2.0,
}

PRICE_CSV_PATH = "lachesis_benchmark_prices.csv"

ASSET_CLASS_MAP = {
    "AAPL": "Equity",
    "MSFT": "Equity",
    "GOOG": "Equity",
    "QQQ": "Equity ETF",
    "TLT": "Bond ETF",
    "GLD": "Gold ETF",
}

PERSONA_LAMBDA = {
    "Conservative": 0.6,
    "Balanced": 1.0,
    "Aggressive": 1.6,
}

# -------------------------------
# Data utilities
# -------------------------------


def load_price_csv(path: str = PRICE_CSV_PATH):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df


def compute_mu_cov_from_prices(df: pd.DataFrame):
    rets = df.pct_change().dropna()
    mu = rets.mean().values
    cov = rets.cov().values
    return mu, cov


def get_qaoa_portfolio_config(selection: str) -> dict:
    if selection.startswith("Lachesis benchmark"):
        cfg = dict(LACHESIS_BENCHMARK_PORTFOLIO)
        price_df = load_price_csv()
        if price_df is not None:
            price_df = price_df[cfg["assets"]]
            mu, cov = compute_mu_cov_from_prices(price_df)
            cfg["mu"] = mu.tolist()
            cfg["cov"] = cov.tolist()
            cfg["data_source"] = "CSV prices (lachesis_benchmark_prices.csv)"
        else:
            cfg["data_source"] = "built-in demo parameters"
        return cfg
    return TOY_QAOA_PORTFOLIO


# -------------------------------
# Regime-aware adjustments
# -------------------------------


def apply_regime_to_cfg(cfg: Dict[str, Any], regime: str) -> Dict[str, Any]:
    """
    Take a base portfolio config and apply a market-regime lens:
    - Bull: boost equity μ, slightly lower Σ
    - Bear: cut equity μ, bump Σ
    - Shock: keep μ but inflate Σ heavily
    """
    new_cfg = dict(cfg)
    assets = cfg["assets"]
    base_mu = np.array(cfg["mu"], dtype=float)
    base_cov = np.array(cfg["cov"], dtype=float)

    adj_mu = base_mu.copy()
    adj_cov = base_cov.copy()
    vol_factor = 1.0

    for i, a in enumerate(assets):
        cls = ASSET_CLASS_MAP.get(a, "Other")
        if regime == "Bull regime":
            if cls in ("Equity", "Equity ETF"):
                adj_mu[i] += 0.03
            elif cls == "Bond ETF":
                adj_mu[i] += 0.005
            elif cls == "Gold ETF":
                adj_mu[i] += 0.01
            vol_factor = 0.9
        elif regime == "Bear regime":
            if cls in ("Equity", "Equity ETF"):
                adj_mu[i] -= 0.04
            elif cls == "Bond ETF":
                adj_mu[i] += 0.01
            elif cls == "Gold ETF":
                adj_mu[i] += 0.015
            vol_factor = 1.3
        elif regime == "Shock regime":
            vol_factor = 1.8

    adj_cov = adj_cov * vol_factor
    new_cfg["mu"] = adj_mu.tolist()
    new_cfg["cov"] = adj_cov.tolist()
    new_cfg["regime"] = regime
    return new_cfg


# -------------------------------
# Core QAOA / portfolio logic
# -------------------------------

@dataclass
class PortfolioResult:
    backend: str
    bitstring: str
    selected_assets: List[str]
    energy: float
    expected_return: float
    risk: float
    objective: float
    lam: float
    shots: int
    note: str = ""


def _evaluate_portfolio(mu: np.ndarray, cov: np.ndarray, z_bits: np.ndarray) -> (float, float):
    z = z_bits.astype(float)
    expected = float(mu @ z)
    risk = float(z.T @ cov @ z)
    return expected, risk


def run_qaoa_portfolio(
    cfg: dict,
    depth: int,
    shots: int,
    lam: float,
    backend: str,
) -> dict:
    assets = cfg["assets"]
    mu = np.array(cfg["mu"], dtype=float)
    cov = np.array(cfg["cov"], dtype=float)
    n = len(assets)

    def bitstring_to_stats(bits: str, backend_label: str) -> dict:
        x = np.array([int(b) for b in bits], dtype=float)
        expected_return = float(mu @ x)
        risk = float(x @ cov @ x)
        objective = lam * expected_return - (1.0 - lam) * risk
        energy = -objective
        selected_assets = [a for a, b in zip(assets, bits) if b == "1"]
        if not selected_assets:
            selected_assets = ["(none)"]
        return {
            "bitstring": bits,
            "selected_assets": selected_assets,
            "expected_return": expected_return,
            "risk": risk,
            "objective": objective,
            "energy": energy,
            "backend": backend_label,
            "lam": lam,
        }

    def solve_bruteforce(label: str) -> dict:
        best_obj = -1e9
        best_bits = "0" * n
        for i in range(2 ** n):
            bits = format(i, f"0{n}b")
            stats = bitstring_to_stats(bits, label)
            if stats["objective"] > best_obj:
                best_obj = stats["objective"]
                best_bits = bits
        return bitstring_to_stats(best_bits, label)

    # Pure classical
    if backend == "Classical brute-force":
        return solve_bruteforce("Classical brute-force")

    # No qiskit-optimization available => fallback
    if not _HAS_QISKIT_OPT:
        return solve_bruteforce(f"{backend} (no qiskit-optimization, classical fallback)")

    # QAOA / quantum path
    try:
        qp = QuadraticProgram()
        for a in assets:
            qp.binary_var(a)

        # Objective J(x) = λV - (1-λ)R encoded into QUBO
        linear = {a: lam * mu[i] for i, a in enumerate(assets)}
        quadratic: Dict[tuple, float] = {}
        for i, a in enumerate(assets):
            for j, b in enumerate(assets):
                if i <= j:
                    coef = -(1.0 - lam) * cov[i, j]
                    if abs(coef) > 1e-12:
                        quadratic[(a, b)] = coef
        qp.maximize(linear=linear, quadratic=quadratic)

        conv = QuadraticProgramToQubo()
        qubo = conv.convert(qp)

        if backend == "Qiskit QAOA":
            try:
                from qiskit.primitives import Sampler

                sampler = Sampler()
                mes = QAOA(
                    sampler=sampler,
                    reps=depth,
                    optimizer=COBYLA(maxiter=100),
                )
                backend_label = "Qiskit QAOA"
            except Exception:
                mes = NumPyMinimumEigensolver()
                backend_label = "Qiskit QAOA (fallback to classical)"
        elif backend == "QAOA (Aer Sampler)":
            try:
                from qiskit_aer.primitives import Sampler as AerSampler

                sampler = AerSampler(shots=shots)
                mes = QAOA(
                    sampler=sampler,
                    reps=depth,
                    optimizer=COBYLA(maxiter=100),
                )
                backend_label = "QAOA (Aer Sampler)"
            except Exception:
                mes = NumPyMinimumEigensolver()
                backend_label = "QAOA (Aer Sampler, fallback to classical)"
        else:
            mes = NumPyMinimumEigensolver()
            backend_label = f"{backend} (using NumPyMinimumEigensolver)"

        optimizer = MinimumEigenOptimizer(mes)
        meo_result = optimizer.solve(qubo)
        bits = "".join("1" if v > 0.5 else "0" for v in meo_result.x)
        return bitstring_to_stats(bits, backend_label)
    except Exception:
        stats = solve_bruteforce(f"{backend} (fallback to brute-force)")
        return stats


# -------------------------------
# Custom Pauli Hamiltonian QAOA
# -------------------------------


def _parse_custom_pauli(pauli_text: str) -> List[tuple]:
    """Parse 'ZZ:1.0, XI:0.4, IX:0.4' → [(coef, pauli_str), ...]"""
    terms = []
    for token in pauli_text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            pauli_part, coef_part = token.rsplit(":", 1)
        else:
            parts = token.split()
            if len(parts) == 2:
                pauli_part, coef_part = parts[0], parts[1]
            else:
                continue
        terms.append((float(coef_part.strip()), pauli_part.strip().upper()))
    return terms


def run_qaoa_custom_hamiltonian(
    pauli_text: str,
    depth: int,
    shots: int,
    backend: str,
) -> dict:
    """
    Run QAOA (or brute-force) directly on a user-supplied Pauli string Hamiltonian.

    The number of qubits is inferred from the length of the first Pauli word.
    Returns a result dict compatible with the QAOA endpoint response shape.
    """
    terms = _parse_custom_pauli(pauli_text)
    if not terms:
        raise ValueError("No valid Pauli terms found in the input string.")

    n_qubits = len(terms[0][1])

    # ── Classical brute-force over all 2^n bitstrings ────────────────────────
    def eval_hamiltonian(bitstring: str) -> float:
        """Compute ⟨ψ|H|ψ⟩ for a computational basis state."""
        energy = 0.0
        for coef, pauli in terms:
            val = 1.0
            for bit_char, p_char in zip(bitstring, pauli):
                z = 1 if bit_char == "0" else -1  # Z eigenvalue
                if p_char == "Z":
                    val *= z
                elif p_char == "I":
                    pass
                else:
                    val = 0.0; break  # X/Y flip out of diagonal basis
            energy += coef * val
        return energy

    if backend == "Classical brute-force" or not _HAS_QISKIT_OPT:
        best_e, best_bits = float("inf"), "0" * n_qubits
        for i in range(2 ** n_qubits):
            bits = format(i, f"0{n_qubits}b")
            e = eval_hamiltonian(bits)
            if e < best_e:
                best_e, best_bits = e, bits
        return {
            "bitstring": best_bits,
            "selected_assets": [f"q{i}" for i, b in enumerate(best_bits) if b == "1"] or ["(none)"],
            "expected_return": 0.0,
            "risk": 0.0,
            "objective": -best_e,
            "energy": best_e,
            "backend": "Classical brute-force (custom Hamiltonian)",
            "lam": 1.0,
            "num_qubits": n_qubits,
            "pauli_terms": [{"coef": c, "pauli": p} for c, p in terms],
        }

    # ── QAOA path ─────────────────────────────────────────────────────────────
    try:
        from qiskit.quantum_info import SparsePauliOp  # type: ignore
        from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver  # type: ignore
        from qiskit.algorithms.optimizers import COBYLA  # type: ignore

        hamiltonian = SparsePauliOp.from_list([(p, c) for c, p in terms])

        try:
            from qiskit_aer.primitives import Sampler as AerSampler  # type: ignore
            sampler = AerSampler(shots=shots)
            label = "QAOA (Aer Sampler, custom H)"
        except Exception:
            from qiskit.primitives import Sampler  # type: ignore
            sampler = Sampler()
            label = "QAOA (Sampler, custom H)"

        qaoa = QAOA(sampler=sampler, reps=depth, optimizer=COBYLA(maxiter=100))
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        energy = float(np.real(result.eigenvalue))

        # Read most probable bitstring from eigenstate distribution
        if hasattr(result, "eigenstate") and hasattr(result.eigenstate, "binary_probabilities"):
            probs = result.eigenstate.binary_probabilities()
            best_bits = max(probs, key=probs.__getitem__)
        else:
            best_bits = "0" * n_qubits
    except Exception:
        # Final fallback
        best_e, best_bits = float("inf"), "0" * n_qubits
        for i in range(2 ** n_qubits):
            bits = format(i, f"0{n_qubits}b")
            e = eval_hamiltonian(bits)
            if e < best_e:
                best_e, best_bits = e, bits
        energy = best_e
        label = "Classical fallback (custom H)"

    return {
        "bitstring": best_bits,
        "selected_assets": [f"q{i}" for i, b in enumerate(best_bits) if b == "1"] or ["(none)"],
        "expected_return": 0.0,
        "risk": 0.0,
        "objective": -energy,
        "energy": energy,
        "backend": label,
        "lam": 1.0,
        "num_qubits": n_qubits,
        "pauli_terms": [{"coef": c, "pauli": p} for c, p in terms],
    }


# -------------------------------
# Logging + λ-sweep
# -------------------------------


def _ensure_log_header() -> None:
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "backend",
                    "lambda",
                    "bitstring",
                    "selected_assets",
                    "expected_return",
                    "risk",
                    "objective",
                    "energy",
                    "shots",
                    "note",
                ]
            )


def log_qaoa_run(result: Dict[str, Any]) -> None:
    _ensure_log_header()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        lam_val = result.get("lam", None)
        selected_assets = result.get("selected_assets", [])
        shots = int(result.get("shots", 0))
        note = result.get("note", "")
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                result.get("backend", ""),
                f"{lam_val:.4f}" if lam_val is not None else "",
                result.get("bitstring", ""),
                "|".join(selected_assets),
                f"{result.get('expected_return', 0.0):.6f}",
                f"{result.get('risk', 0.0):.6f}",
                f"{result.get('objective', 0.0):.6f}",
                f"{result.get('energy', 0.0):.6f}",
                shots,
                note,
            ]
        )


def load_qaoa_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_PATH)
    except Exception:
        return pd.read_csv(LOG_PATH, on_bad_lines="skip")


def lambda_sweep_classical(cfg: dict, lam_min: float, lam_max: float, n_points: int) -> pd.DataFrame:
    lam_values = np.linspace(lam_min, lam_max, n_points)
    rows = []
    for lam in lam_values:
        result = run_qaoa_portfolio(
            cfg,
            depth=1,
            shots=1024,
            lam=lam,
            backend="Classical brute-force",
        )
        rows.append(
            {
                "lambda": lam,
                "expected_return": result["expected_return"],
                "risk": result["risk"],
                "objective": result["objective"],
                "bitstring": result["bitstring"],
            }
        )
    sweep_df = pd.DataFrame(rows)
    sweep_df.set_index("lambda", inplace=True)
    return sweep_df


# -------------------------------
# Narrative generator
# -------------------------------


def generate_portfolio_narrative(
    result: Dict[str, Any],
    cfg: Dict[str, Any],
    regime_label: Optional[str] = None,
) -> str:
    assets = cfg["assets"]
    mu_vec = np.array(cfg["mu"], dtype=float)
    cov_mat = np.array(cfg["cov"], dtype=float)

    lam = float(result.get("lam", 0.0))
    V_sel = float(result.get("expected_return", 0.0))
    R_sel = float(result.get("risk", 0.0))
    selected_assets = result.get("selected_assets", [])

    if selected_assets == ["(none)"] or len(selected_assets) == 0:
        base = f"For λ = {lam:.2f}, Lachesis chooses to **stay in cash** (no assets selected). "
        if regime_label:
            base = f"Under a **{regime_label}** regime, " + base
        return (
            base
            + "This corresponds to a portfolio with zero modeled return and zero modeled variance — "
            "a fully capital-preserving stance because the risk-aversion setting heavily penalizes volatility."
        )

    # Compare to entire subset universe
    n = len(assets)
    all_returns = []
    all_risks = []
    for k in range(1, 1 << n):
        bits = np.array([(k >> i) & 1 for i in range(n)], dtype=float)
        V, R = _evaluate_portfolio(mu_vec, cov_mat, bits)
        all_returns.append(V)
        all_risks.append(R)
    all_returns = np.array(all_returns)
    all_risks = np.array(all_risks)

    low_risk, high_risk = np.quantile(all_risks, [0.33, 0.66])
    low_ret, high_ret = np.quantile(all_returns, [0.33, 0.66])

    if R_sel <= low_risk:
        risk_label = "conservative"
    elif R_sel >= high_risk:
        risk_label = "aggressive"
    else:
        risk_label = "balanced"

    if V_sel <= low_ret:
        return_label = "low-return"
    elif V_sel >= high_ret:
        return_label = "high-return"
    else:
        return_label = "moderate-return"

    # Correlation diagnostics
    std = np.sqrt(np.diag(cov_mat))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_mat = np.where(denom > 0, cov_mat / denom, 0.0)
    if n > 1:
        off_diag = corr_mat[np.triu_indices(n, k=1)]
        avg_corr = float(np.nanmean(off_diag))
    else:
        avg_corr = 0.0

    if avg_corr > 0.7:
        corr_label = "tightly coupled – shocks tend to hit most holdings together"
    elif avg_corr > 0.3:
        corr_label = "moderately correlated – some diversification, but cycles are shared"
    else:
        corr_label = "loosely correlated – the mix is naturally diversifying"

    # Asset-class breakdown
    class_counts: Dict[str, int] = {}
    for a in selected_assets:
        cls = ASSET_CLASS_MAP.get(a, "Other")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    total_sel = sum(class_counts.values())
    if total_sel > 0:
        class_breakdown_parts = []
        for cls, count in class_counts.items():
            pct = 100.0 * count / total_sel
            class_breakdown_parts.append(f"{pct:.0f}% {cls.lower()}")
        class_phrase = ", ".join(class_breakdown_parts)
    else:
        class_phrase = "no invested asset classes (all cash)"

    asset_list_str = ", ".join(selected_assets)

    intro = f"For λ = **{lam:.2f}**, "
    if regime_label:
        intro += f"under a **{regime_label}** regime, "
    intro += f"Lachesis selects the portfolio **[{asset_list_str}]**.  \n"

    narrative = (
        intro
        + f"- The model estimates an expected return of **{V_sel:.2%}** per period with "
        f"variance **{R_sel:.4f}**, placing it in the **{risk_label} / {return_label}** zone "
        "relative to other feasible portfolios in this universe.  \n"
        f"- In terms of asset classes, the mix is roughly **{class_phrase}**, "
        "so investors are mainly exposed to those risk premia.  \n"
        f"- Correlations implied by Σ are **{corr_label}**, which shapes how drawdowns "
        "are likely to propagate across the portfolio.  \n\n"
        "In plain language: this is a "
        f"**{risk_label} but {return_label.replace('-',' ')} stance** at this risk-aversion setting λ, "
        "balancing growth versus stability according to the investor’s preferences."
    )
    return narrative


# -------------------------------
# Scenario storage + crash metrics
# -------------------------------


def load_qaoa_scenarios() -> List[Dict[str, Any]]:
    if not os.path.exists(SCENARIO_PATH):
        return []
    try:
        with open(SCENARIO_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_qaoa_scenario(scenario: Dict[str, Any]) -> None:
    scenarios = load_qaoa_scenarios()
    scenarios.append(scenario)
    with open(SCENARIO_PATH, "w") as f:
        json.dump(scenarios, f, indent=2)


def compute_crash_index_and_label(scen: Dict[str, Any]) -> (float, str):
    """
    Simple downside metric:

    - Uses scenario risk R and equity share based on ASSET_CLASS_MAP.
    - Higher equity share + higher R => more crash-sensitive.

    Returns:
        crash_index (float), crash_label (str)
    """
    risk = float(scen.get("risk", 0.0))
    selected_assets = scen.get("selected_assets") or []

    # All cash => extremely resilient
    if not selected_assets or selected_assets == ["(none)"]:
        return 0.0, "Crash-resilient 🟢 (all cash)"

    equity_like = 0
    defensive_like = 0
    for a in selected_assets:
        cls = ASSET_CLASS_MAP.get(a, "Other")
        if "Equity" in cls:
            equity_like += 1
        elif "Bond" in cls or "Gold" in cls:
            defensive_like += 1

    if equity_like + defensive_like == 0:
        equity_share = 0.5  # unknown mix
    else:
        equity_share = equity_like / float(equity_like + defensive_like)

    # 0.5–1.5 multiplier on risk based on equity share
    multiplier = 0.5 + equity_share
    crash_index = risk * multiplier

    if crash_index < 0.015:
        label = "Crash-resilient 🟢"
    elif crash_index < 0.04:
        label = "Moderate risk 🟡"
    else:
        label = "Crash-sensitive 🔴"

    return float(crash_index), label


def export_qaoa_snapshot(
    result: Dict[str, Any],
    cfg: Dict[str, Any],
    portfolio_choice: str,
    regime_choice: str,
    persona_label: Optional[str],
) -> None:
    """
    Export a compact JSON snapshot that the QTBN foresight / sweeps engine
    can read as a 'stance prior' from the QAOA mini-lab.

    This version ALSO writes:
      - expected_return
      - risk
      - objective
      - crash_index / crash_label
      - assets / weights (equal-weight across selected assets)
    so the foresight tab can display meaningful numbers.
    """

    # Normalize selected assets (treat "(none)" as all-cash)
    selected_assets = result.get("selected_assets", []) or []
    if selected_assets == ["(none)"]:
        selected_assets = []

    # Equal-weight weights for now (you can change this later)
    if selected_assets:
        weights = [1.0 / len(selected_assets)] * len(selected_assets)
    else:
        weights = []

    # Reuse our crash metric using only risk + selected assets
    temp_scen = {
        "risk": result.get("risk", 0.0),
        "selected_assets": selected_assets,
    }
    crash_index, crash_label = compute_crash_index_and_label(temp_scen)

    snapshot = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "portfolio_choice": portfolio_choice,
        "portfolio_name": cfg.get("name", portfolio_choice),
        "regime": regime_choice,
        "persona": persona_label,
        "lambda": float(result.get("lam", 0.0)),

        # Portfolio stance
        "bitstring": result.get("bitstring"),
        "selected_assets": selected_assets,
        "assets": selected_assets,          # for foresight display
        "weights": weights,                 # equal-weight

        # Risk/return numbers
        "expected_return": float(result.get("expected_return", 0.0)),
        "risk": float(result.get("risk", 0.0)),
        "objective": float(result.get("objective", 0.0)),

        # Crash index for QTBN priors
        "crash_index": float(crash_index),
        "crash_label": crash_label,

        # Backend metadata
        "backend": result.get("backend", ""),
        "shots": int(result.get("shots", 0)),
        "note": result.get("note", ""),
    }
    with open(QAOA_SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)


def load_qaoa_snapshot(path: str = QAOA_SNAPSHOT_PATH) -> Optional[Dict[str, Any]]:
    """
    Load the QAOA snapshot and upgrade it if it's in an older format:
      - fill selected_assets from assets/weights if needed
      - recompute expected_return / risk / objective if missing or zero
      - recompute crash_index / crash_label if missing
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            snap = json.load(f)
    except Exception:
        return None

    # 1) Ensure selected_assets exists
    if not snap.get("selected_assets"):
        assets = snap.get("assets", [])
        weights = snap.get("weights", [])
        if assets and weights and len(assets) == len(weights):
            sel = [
                a for a, w in zip(assets, weights)
                if isinstance(w, (int, float)) and abs(w) > 1e-12
            ]
        else:
            sel = assets
        snap["selected_assets"] = sel or ["(none)"]

    # 2) Recompute expected_return / risk / objective if missing or zero
    need_recompute = (
        "expected_return" not in snap
        or "risk" not in snap
        or (
            float(snap.get("expected_return", 0.0)) == 0.0
            and float(snap.get("risk", 0.0)) == 0.0
        )
    )

    if need_recompute:
        try:
            portfolio_choice = snap.get(
                "portfolio_choice",
                "Lachesis benchmark (5-asset mix)",
            )
            regime_choice = snap.get("regime", "Baseline")
            lam = float(snap.get("lambda", 1.0))

            cfg_base = get_qaoa_portfolio_config(portfolio_choice)
            cfg = apply_regime_to_cfg(cfg_base, regime_choice)

            assets = cfg["assets"]
            mu = np.array(cfg["mu"], dtype=float)
            cov = np.array(cfg["cov"], dtype=float)

            bitstring = snap.get("bitstring")
            if isinstance(bitstring, str) and len(bitstring) == len(assets):
                x = np.array([int(b) for b in bitstring], dtype=float)
            else:
                sel = snap.get("selected_assets", [])
                x = np.array([1.0 if a in sel else 0.0 for a in assets], dtype=float)

            V = float(mu @ x)
            R = float(x @ cov @ x)
            obj = lam * V - (1.0 - lam) * R

            snap["expected_return"] = V
            snap["risk"] = R
            snap["objective"] = obj
        except Exception:
            # If anything fails, just leave the current values (likely zeros)
            pass

    # 3) Ensure crash_index / crash_label are present and consistent
    if (
        "crash_index" not in snap
        or "crash_label" not in snap
        or (
            float(snap.get("crash_index", 0.0)) == 0.0
            and float(snap.get("risk", 0.0)) > 0.0
        )
    ):
        ci, cl = compute_crash_index_and_label(
            {
                "risk": snap.get("risk", 0.0),
                "selected_assets": snap.get("selected_assets", []),
            }
        )
        snap["crash_index"] = float(ci)
        snap["crash_label"] = cl

    return snap


# -------------------------------
# Streamlit UI for QAOA tab
# -------------------------------


def render_qaoa_tab(st) -> None:
    # ---------------------------------------------------------
    # 1) Apply any pending scenario/persona state BEFORE widgets
    # ---------------------------------------------------------
    pending_scen = st.session_state.pop("pending_scenario", None)
    if pending_scen is not None:
        lam_val = pending_scen.get("lambda")
        if isinstance(lam_val, (int, float)):
            st.session_state["qaoa_lambda"] = float(lam_val)

        st.session_state["qaoa_portfolio_choice"] = pending_scen.get(
            "portfolio_choice",
            "Lachesis benchmark (5-asset mix)",
        )
        st.session_state["qaoa_regime_choice"] = pending_scen.get(
            "regime",
            "Baseline",
        )

    pending_lambda = st.session_state.pop("pending_lambda", None)
    if isinstance(pending_lambda, (int, float)):
        st.session_state["qaoa_lambda"] = float(pending_lambda)

    if "pending_persona" not in st.session_state:
        st.session_state["pending_persona"] = None

    # ---------------------------------------------------------
    # 2) Main UI
    # ---------------------------------------------------------
    st.markdown("## Toy QAOA – Portfolio Selection")
    st.markdown(
        """
This tab shows a **toy portfolio optimization** problem that mirrors the
kind of binary decision QAOA is good at:

- Universe of assets (toy): AAPL, MSFT, GOOG  
- Decision variable: include each asset (1) or not (0)  
- Objective: maximize  
  \\[
    J(x) = \\lambda V(x) - (1 - \\lambda) R(x)
  \\]  
  where \(V\) is expected return and \(R\) is risk (variance).
        """
    )

    # --- Backend selection ---
    st.markdown("### Backend selection")
    backend_choice = st.radio(
        "Choose backend (algorithm mode)",
        (
            "Classical brute-force",
            "Qiskit QAOA (Sampler stub)",
            "Qiskit QAOA (Aer Sampler stub)",
        ),
        horizontal=True,
        key="qaoa_backend_mode",
    )

    # Map display labels -> internal engine names + human note
    if backend_choice == "Classical brute-force":
        backend_internal = "Classical brute-force"
        backend_label = "Classical brute-force"
        backend_note = "Exact search over all bitstrings (ground truth)."
    elif backend_choice == "Qiskit QAOA (Sampler stub)":
        backend_internal = "Qiskit QAOA"  # triggers Sampler+QAOA branch
        backend_label = "Qiskit QAOA (Sampler stub)"
        backend_note = (
            "Currently using the same classical optimizer, but labeled as "
            "a sampler-based QAOA mode for UX and logging. "
            "Can be wired to real Qiskit QAOA later."
        )
    else:  # "Qiskit QAOA (Aer Sampler stub)"
        backend_internal = "QAOA (Aer Sampler)"  # triggers AerSampler+QAOA branch
        backend_label = "Qiskit QAOA (Aer Sampler stub)"
        backend_note = (
            "Aer Sampler path is stubbed due to version mismatches. "
            "Using the classical optimizer while preserving the investor-facing story."
        )

    st.markdown(f"*Selected backend:* **{backend_label}**  \n_{backend_note}_")

    # --- Universe selection ---
    st.markdown("#### Portfolio universe")
    portfolio_choice = st.radio(
        "Choose which portfolio Lachesis optimizes:",
        [
            "Toy 3-asset tech portfolio",
            "Lachesis benchmark (5-asset mix)",
        ],
        key="qaoa_portfolio_choice",
    )

    cfg_base = get_qaoa_portfolio_config(portfolio_choice)

    # --- Regime selection ---
    st.markdown("#### Market regime")
    regime_choice = st.selectbox(
        "Choose the market regime context for this run:",
        ["Baseline", "Bull regime", "Bear regime", "Shock regime"],
        index=0,
        key="qaoa_regime_choice",
    )

    cfg = apply_regime_to_cfg(cfg_base, regime_choice)

    st.markdown("#### Current portfolio config (after regime adjustment)")
    st.json(cfg)

    # --- Historical price preview (if CSV present) ---
    price_df = None
    if portfolio_choice.startswith("Lachesis benchmark"):
        try:
            price_df = load_price_csv()
        except Exception:
            price_df = None

    if price_df is not None:
        with st.expander("Show historical price data (CSV)", expanded=False):
            st.dataframe(price_df.tail(20))
    elif portfolio_choice.startswith("Lachesis benchmark"):
        st.info(
            "No CSV found at 'lachesis_benchmark_prices.csv' – using built-in demo μ and Σ."
        )

    # --- Quick explainer block ---
    with st.expander("Quick guide for investors: what μ, Σ, and λ mean", expanded=False):
        st.markdown(
            """
**μ (mu) – expected return**

- For each asset, μ is the model’s estimate of its average annual return.  
- Example: μ = 0.12 means “about **12% expected annual return**” under the assumptions of the model.  

**Σ (sigma) – risk & correlation (covariance matrix)**

- Σ captures both **volatility** and **how assets move together**.  
- The **diagonal** entries show how “noisy” each asset is on its own.  
- The **off-diagonal** entries show how tightly two assets are linked.

**λ (lambda) – risk aversion knob**

- λ is the **risk-return dial** used in the QAOA objective.  
- Lower λ ⇒ Lachesis behaves **more risk-averse**.  
- Higher λ ⇒ Lachesis behaves **more return-seeking**.  
"""
        )

    # --- QAOA hyperparameters ---
    st.markdown("### QAOA Hyperparameters")
    depth = st.slider(
        "QAOA depth p (layer count)",
        min_value=1,
        max_value=3,
        value=1,
        key="qaoa_depth",
    )
    shots = st.slider(
        "Number of shots",
        min_value=128,
        max_value=4096,
        value=1024,
        step=128,
        key="qaoa_shots",
    )

    if "qaoa_lambda" not in st.session_state:
        st.session_state["qaoa_lambda"] = 0.8

    lam = st.slider(
        "Risk aversion λ",
        min_value=0.5,
        max_value=2.0,
        value=float(st.session_state["qaoa_lambda"]),
        step=0.1,
        key="qaoa_lambda",
    )

    # ---------------- Investor personas (now using pending_lambda) ------------
    st.markdown("### Investor personas")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Conservative", key="persona_conservative"):
            st.session_state["pending_lambda"] = PERSONA_LAMBDA["Conservative"]
            st.session_state["pending_persona"] = "Conservative"
            st.rerun()
    with c2:
        if st.button("Balanced", key="persona_balanced"):
            st.session_state["pending_lambda"] = PERSONA_LAMBDA["Balanced"]
            st.session_state["pending_persona"] = "Balanced"
            st.rerun()
    with c3:
        if st.button("Aggressive", key="persona_aggressive"):
            st.session_state["pending_lambda"] = PERSONA_LAMBDA["Aggressive"]
            st.session_state["pending_persona"] = "Aggressive"
            st.rerun()

    # --- Run QAOA ---
    run_clicked = st.button("Run QAOA (selected backend)", type="primary")
    if run_clicked:
        result = run_qaoa_portfolio(
            cfg,
            depth=depth,
            shots=shots,
            lam=lam,
            backend=backend_internal,
        )
        classical_result = run_qaoa_portfolio(
            cfg,
            depth=depth,
            shots=shots,
            lam=lam,
            backend="Classical brute-force",
        )

        # annotate & log
        result["shots"] = shots
        result["note"] = "user_backend"
        result["lam"] = lam

        classical_result["shots"] = shots
        classical_result["note"] = "classical_baseline"
        classical_result["lam"] = lam

        log_qaoa_run(classical_result)
        log_qaoa_run(result)

        # cache last run for scenario saving
        st.session_state["last_qaoa_result"] = {
            "result": result,
            "classical_result": classical_result,
            "cfg": cfg,
            "portfolio_choice": portfolio_choice,
            "regime_choice": regime_choice,
            "lam": lam,
            "depth": depth,
            "shots": shots,
            "backend_choice": backend_choice,
        }

        # --- Results ---
        st.markdown("### Result")
        st.write(f"Backend: {result['backend']}")
        st.write(f"Market regime: {regime_choice}")
        st.write(f"Bitstring: {result['bitstring']}")
        st.write(f"Selected assets: {', '.join(result['selected_assets'])}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Energy (−objective)", f"{result['energy']:.4f}")
        with col2:
            st.metric("Expected return", f"{result['expected_return']:.4f}")
        with col3:
            st.metric("Risk (variance)", f"{result['risk']:.4f}")
        with col4:
            st.metric("Objective J(x)", f"{result['objective']:.4f}")

        st.caption(
            f"Objective J(x) = λV − (1−λ)R at λ = {lam:.2f}. "
            "Lower energy corresponds to higher objective."
        )

        # Classical vs QAOA comparison
        st.markdown("### Classical vs QAOA benchmark")
        comp_df = pd.DataFrame(
            [
                {
                    "Engine": classical_result["backend"],
                    "Bitstring": classical_result["bitstring"],
                    "Expected return": classical_result["expected_return"],
                    "Risk (variance)": classical_result["risk"],
                    "Objective J(x)": classical_result["objective"],
                },
                {
                    "Engine": result["backend"],
                    "Bitstring": result["bitstring"],
                    "Expected return": result["expected_return"],
                    "Risk (variance)": result["risk"],
                    "Objective J(x)": result["objective"],
                },
            ]
        )
        st.table(comp_df)

        st.markdown("### Lachesis narrative")
        narrative = generate_portfolio_narrative(result, cfg, regime_choice)
        st.write(narrative)

        # --- Bridge to QTBN foresight engine ---
        st.markdown("#### Send this stance to the QTBN foresight engine")
        persona_label = st.session_state.get("pending_persona", None)
        if st.button("Send to foresight engine", key="btn_export_snapshot"):
            export_qaoa_snapshot(
                result=result,
                cfg=cfg,
                portfolio_choice=portfolio_choice,
                regime_choice=regime_choice,
                persona_label=persona_label,
            )
            st.success(
                f"Exported QAOA snapshot for QTBN foresight engine → '{QAOA_SNAPSHOT_PATH}'"
            )

    # --- Uplift analytics ---
    with st.expander("Lachesis uplift analytics (classical vs QAOA)"):
        df_log = load_qaoa_log()
        if df_log.empty:
            st.info(
                "No QAOA runs logged yet. Run at least one classical "
                "and one QAOA backend above to see uplift analytics."
            )
        else:
            if "lambda" in df_log.columns:
                df_log["lambda"] = pd.to_numeric(df_log["lambda"], errors="coerce")
            classical_mask = df_log["backend"].str.contains(
                "Classical brute-force", case=False, na=False
            )
            qaoa_mask = df_log["backend"].str.contains(
                "Qiskit QAOA", case=False, na=False
            ) | df_log["backend"].str.contains(
                "Aer Sampler", case=False, na=False
            )
            classical_df = df_log[classical_mask].copy()
            qaoa_df = df_log[qaoa_mask].copy()
            if classical_df.empty or qaoa_df.empty:
                st.info(
                    "To compute uplift, we need runs from both the classical "
                    "baseline and at least one QAOA backend at overlapping "
                    "λ values."
                )
            else:
                join_cols = ["lambda"]
                for col in ["shots"]:
                    if col in df_log.columns:
                        join_cols.append(col)
                merged = pd.merge(
                    classical_df,
                    qaoa_df,
                    on=join_cols,
                    suffixes=("_classical", "_qaoa"),
                )
                if merged.empty:
                    st.info(
                        "No overlapping λ between classical and QAOA runs yet. "
                        "Try re-running both backends at the same λ (and shots)."
                    )
                else:
                    merged["delta_objective"] = (
                        merged["objective_qaoa"] - merged["objective_classical"]
                    )
                    merged["delta_return"] = (
                        merged["expected_return_qaoa"]
                        - merged["expected_return_classical"]
                    )
                    merged["delta_risk"] = (
                        merged["risk_classical"] - merged["risk_qaoa"]
                    )

                    avg_obj = float(merged["delta_objective"].mean())
                    max_obj = float(merged["delta_objective"].max())
                    avg_ret = float(merged["delta_return"].mean())
                    avg_risk_red = float(merged["delta_risk"].mean())

                    st.markdown("#### Lachesis uplift vs classical (global view)")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "Avg objective uplift ΔJ",
                            f"{avg_obj:.4f}",
                        )
                    with c2:
                        st.metric(
                            "Max objective uplift ΔJ",
                            f"{max_obj:.4f}",
                        )
                    with c3:
                        st.metric(
                            "Avg return uplift ΔV",
                            f"{avg_ret:.4f}",
                        )
                    with c4:
                        st.metric(
                            "Avg risk reduction ΔR",
                            f"{avg_risk_red:.4f}",
                        )
                    uplift_cols = [
                        "lambda",
                        "backend_classical",
                        "backend_qaoa",
                        "bitstring_classical",
                        "bitstring_qaoa",
                        "delta_objective",
                        "delta_return",
                        "delta_risk",
                    ]
                    uplift_cols = [c for c in uplift_cols if c in merged.columns]
                    st.markdown("##### Per-λ uplift table")
                    st.dataframe(
                        merged[uplift_cols].sort_values("lambda"),
                        use_container_width=True,
                    )
                    chart_df = merged[["lambda", "delta_objective"]].copy()
                    chart_df = chart_df.sort_values("lambda").set_index("lambda")
                    st.bar_chart(chart_df, height=260)

    # --- λ-sweep ---
    st.markdown("### λ-sweep: risk aversion vs optimal portfolio")
    lam_min = st.slider("λ min", 0.0, 5.0, 0.5, 0.1, key="lambda_min")
    lam_max = st.slider("λ max", 0.0, 5.0, 3.0, 0.1, key="lambda_max")
    lam_points = st.slider("Number of λ points", 3, 15, 5, 1, key="lambda_points")

    if st.button("Run λ-sweep (classical)", type="secondary"):
        sweep_df = lambda_sweep_classical(
            cfg=cfg,
            lam_min=lam_min,
            lam_max=lam_max,
            n_points=lam_points,
        )
        st.line_chart(
            sweep_df[["expected_return", "risk", "objective"]],
            height=300,
        )
        st.caption(
            "Each point on the λ-sweep curve is the **best portfolio** Lachesis finds "
            "for that risk preference λ."
        )
        try:
            import matplotlib.pyplot as plt

            fig1, ax1 = plt.subplots()
            ax1.plot(sweep_df.index, sweep_df["expected_return"], marker="o")
            ax1.set_xlabel("λ (risk aversion)")
            ax1.set_ylabel("Expected return")
            ax1.set_title("Expected return vs λ")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(sweep_df.index, sweep_df["risk"], marker="o")
            ax2.set_xlabel("λ (risk aversion)")
            ax2.set_ylabel("Risk (variance)")
            ax2.set_title("Risk vs λ")
            st.pyplot(fig2)
        except Exception:
            st.info(
                "λ-sweep completed. Install matplotlib if you’d like inline plots."
            )

    # --- Portfolio summary + correlations ---
    st.markdown("### Portfolio summary")
    with st.expander("μ, Σ, and ρ (correlation) for current portfolio", expanded=False):
        assets = cfg["assets"]
        mu_vec = np.array(cfg["mu"], dtype=float)
        cov_mat = np.array(cfg["cov"], dtype=float)

        std_vec = np.sqrt(np.diag(cov_mat))
        summary_df = pd.DataFrame(
            {
                "μ (expected return)": mu_vec,
                "σ (volatility)": std_vec,
            },
            index=assets,
        )
        st.markdown("**Per-asset summary**")
        st.dataframe(summary_df.style.format("{:.4f}"))

        denom = np.outer(std_vec, std_vec)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(denom > 0, cov_mat / denom, 0.0)
        corr_df = pd.DataFrame(rho, index=assets, columns=assets)
        st.markdown("**Correlation matrix ρ from Σ**")
        st.dataframe(corr_df.style.format("{:.2f}"))

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            im = ax.imshow(rho, origin="lower")
            ax.set_xticks(range(len(assets)))
            ax.set_xticklabels(assets)
            ax.set_yticks(range(len(assets)))
            ax.set_yticklabels(assets)
            ax.set_title("Correlation matrix ρ from Σ")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
        except Exception:
            st.info("Install matplotlib to see a heatmap of the correlation matrix.")

        st.markdown("**Asset class mix (equal-weight across universe)**")
        classes = [ASSET_CLASS_MAP.get(a, "Other") for a in assets]
        class_counts = pd.Series(classes, index=assets).value_counts()
        try:
            import matplotlib.pyplot as plt

            fig_ac, ax_ac = plt.subplots()
            ax_ac.pie(
                class_counts.values,
                labels=class_counts.index,
                autopct="%1.0f%%",
            )
            ax_ac.set_title("Asset class mix (equal-weight)")
            st.pyplot(fig_ac)
        except Exception:
            st.dataframe(
                class_counts.to_frame(name="count"),
                use_container_width=True,
            )

    st.markdown("### Portfolio (reference)")
    st.json(cfg)

    # --- Scenario library (save / load + compare) ---
    st.markdown("### Saved QAOA scenarios")
    with st.expander("Scenario library", expanded=False):
        scenarios = load_qaoa_scenarios()

        # Save
        if "last_qaoa_result" in st.session_state:
            st.markdown("#### Save current run as a scenario")
            default_name = time.strftime("Scenario %Y-%m-%d %H:%M:%S")
            scen_name = st.text_input(
                "Scenario name",
                value=default_name,
                key="scenario_name_input",
            )
            if st.button("Save this scenario", key="btn_save_scenario"):
                last = st.session_state["last_qaoa_result"]
                r = last["result"]
                cfg_saved = last["cfg"]
                scenario = {
                    "name": scen_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "portfolio_choice": last.get("portfolio_choice"),
                    "portfolio_name": cfg_saved.get("name", last.get("portfolio_choice")),
                    "regime": last.get("regime_choice"),
                    "backend": r.get("backend"),
                    "lambda": float(last.get("lam")),
                    "depth": int(last.get("depth")),
                    "shots": int(last.get("shots")),
                    "bitstring": r.get("bitstring"),
                    "selected_assets": r.get("selected_assets"),
                    "expected_return": float(r.get("expected_return", 0.0)),
                    "risk": float(r.get("risk", 0.0)),
                    "objective": float(r.get("objective", 0.0)),
                    "note": r.get("note", ""),
                }
                crash_score, crash_label = compute_crash_index_and_label(scenario)
                scenario["crash_score"] = float(crash_score)
                scenario["crash_label"] = crash_label

                save_qaoa_scenario(scenario)
                st.success(f"Saved scenario '{scen_name}'")
                scenarios = load_qaoa_scenarios()
        else:
            st.info("Run QAOA at least once to enable scenario saving.")

        # Load/display
        st.markdown("#### Existing scenarios")
        scenarios = load_qaoa_scenarios()
        if not scenarios:
            st.write("No scenarios saved yet.")
        else:
            for idx, scen in enumerate(scenarios):
                cols = st.columns([3, 2, 2, 2, 3])
                crash_score, crash_label = compute_crash_index_and_label(scen)
                with cols[0]:
                    st.markdown(f"**{scen.get('name','(unnamed)')}**")
                    st.caption(
                        f"{scen.get('portfolio_name','')} – {scen.get('regime','Baseline')}"
                    )
                    st.write(crash_label)
                with cols[1]:
                    lam_val = scen.get("lambda", None)
                    lam_str = (
                        f"{lam_val:.2f}"
                        if isinstance(lam_val, (int, float))
                        else "—"
                    )
                    st.write(f"λ = {lam_str}")
                    st.write(scen.get("backend", ""))
                with cols[2]:
                    obj = float(scen.get("objective", 0.0))
                    ret = float(scen.get("expected_return", 0.0))
                    st.write(f"J(x) = {obj:.4f}")
                    st.write(f"V = {ret:.4f}")
                with cols[3]:
                    risk = float(scen.get("risk", 0.0))
                    assets_sel = scen.get("selected_assets", []) or []
                    st.write(f"R = {risk:.4f}")
                    st.write(
                        ", ".join(assets_sel[:3])
                        + ("..." if len(assets_sel) > 3 else "")
                    )
                with cols[4]:
                    st.write(f"Crash idx: {crash_score:.4f}")
                    if st.button("Load", key=f"load_scen_{idx}"):
                        # Mark this scenario as pending and rerun;
                        # top-of-function logic will apply it
                        st.session_state["pending_scenario"] = scen
                        st.rerun()

            # ---- Scenario comparison ----
            if len(scenarios) >= 2:
                st.markdown("---")
                st.markdown("#### Compare two scenarios")

                index_options = list(range(len(scenarios)))
                left_idx = st.selectbox(
                    "Left scenario",
                    index_options,
                    format_func=lambda i: f"{i+1}. {scenarios[i].get('name','(unnamed)')}",
                    key="scenario_compare_left",
                )
                right_idx = st.selectbox(
                    "Right scenario",
                    index_options,
                    format_func=lambda i: f"{i+1}. {scenarios[i].get('name','(unnamed)')}",
                    key="scenario_compare_right",
                )

                if left_idx == right_idx:
                    st.info("Select two different scenarios to compare.")
                else:
                    left_scen = scenarios[left_idx]
                    right_scen = scenarios[right_idx]

                    lam_L = float(left_scen.get("lambda", 0.0))
                    lam_R = float(right_scen.get("lambda", 0.0))
                    obj_L = float(left_scen.get("objective", 0.0))
                    obj_R = float(right_scen.get("objective", 0.0))
                    ret_L = float(left_scen.get("expected_return", 0.0))
                    ret_R = float(right_scen.get("expected_return", 0.0))
                    risk_L = float(left_scen.get("risk", 0.0))
                    risk_R = float(right_scen.get("risk", 0.0))

                    crash_L, badge_L = compute_crash_index_and_label(left_scen)
                    crash_R, badge_R = compute_crash_index_and_label(right_scen)

                    # Winner by objective
                    if obj_L > obj_R:
                        winner_side = "left"
                        winner_name = left_scen.get("name", "(unnamed)")
                        best_obj = obj_L
                    elif obj_R > obj_L:
                        winner_side = "right"
                        winner_name = right_scen.get("name", "(unnamed)")
                        best_obj = obj_R
                    else:
                        winner_side = "tie"
                        winner_name = None
                        best_obj = obj_L

                    # Winner by crash-resilience (lower crash index)
                    if crash_L < crash_R:
                        crash_winner_side = "left"
                        crash_winner_name = left_scen.get("name", "(unnamed)")
                    elif crash_R < crash_L:
                        crash_winner_side = "right"
                        crash_winner_name = right_scen.get("name", "(unnamed)")
                    else:
                        crash_winner_side = "tie"
                        crash_winner_name = None

                    left_name = left_scen.get("name", "(unnamed)")
                    right_name = right_scen.get("name", "(unnamed)")

                    left_label = left_name + (" 🏆" if winner_side == "left" else "")
                    right_label = right_name + (" 🏆" if winner_side == "right" else "")

                    comp_rows = [
                        {
                            "Metric": "λ (risk aversion)",
                            left_label: f"{lam_L:.2f}",
                            right_label: f"{lam_R:.2f}",
                        },
                        {
                            "Metric": "J(x) (objective)",
                            left_label: f"{obj_L:.4f}",
                            right_label: f"{obj_R:.4f}",
                        },
                        {
                            "Metric": "Expected return V",
                            left_label: f"{ret_L:.4f}",
                            right_label: f"{ret_R:.4f}",
                        },
                        {
                            "Metric": "Risk R (variance)",
                            left_label: f"{risk_L:.4f}",
                            right_label: f"{risk_R:.4f}",
                        },
                        {
                            "Metric": "Crash index (lower is safer)",
                            left_label: f"{crash_L:.4f}",
                            right_label: f"{crash_R:.4f}",
                        },
                        {
                            "Metric": "Crash badge",
                            left_label: badge_L,
                            right_label: badge_R,
                        },
                    ]
                    comp_df = pd.DataFrame(comp_rows).set_index("Metric")
                    st.table(comp_df)

                    if winner_side == "tie":
                        st.info(
                            f"By J(x), both scenarios are tied at **{best_obj:.4f}**."
                        )
                    else:
                        st.success(
                            f"🏆 **Winner by J(x)**: {winner_name} with J(x) = {best_obj:.4f}"
                        )

                    if crash_winner_side == "tie":
                        st.info(
                            "By crash-resilience, both scenarios have similar downside exposure."
                        )
                    else:
                        st.markdown(
                            f"🛡️ **More crash-resilient:** {crash_winner_name} "
                            "(lower crash index, better expected behavior in a selloff)."
                        )

            # ---------- Scenario storyline chart ----------
            if len(scenarios) >= 1:
                st.markdown("#### Scenario storyline (J(x) vs λ)")
                story_df = pd.DataFrame(
                    {
                        "lambda": [
                            float(s.get("lambda", 0.0)) for s in scenarios
                        ],
                        "objective": [
                            float(s.get("objective", 0.0)) for s in scenarios
                        ],
                        "name": [s.get("name", "") for s in scenarios],
                    }
                ).sort_values("lambda")

                story_df = story_df.set_index("lambda")
                st.line_chart(story_df[["objective"]], height=260)
                st.caption(
                    "Each point is a saved scenario. The curve shows how J(x) evolves as you change λ "
                    "and lock in different QAOA-driven portfolios."
                )
