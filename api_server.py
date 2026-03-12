"""
api_server.py  –  Lachesis FastAPI REST bridge
Exposes Python backend computation as JSON endpoints for the React frontend.
Run: uvicorn api_server:app --reload --port 8000
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── project root on sys.path ─────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

app = FastAPI(title="Lachesis API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173",
        "http://localhost:8080", "http://127.0.0.1:8080",
        "http://localhost:8082", "http://127.0.0.1:8082",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL IMPORTS (graceful fallback if a dependency is missing)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
    )
    _HAS_QISKIT = True
except Exception:
    _HAS_QISKIT = False

# ── QAE optional imports (qiskit-finance) ────────────────────────────────────
_HAS_QAE = False
_IAE     = None   # IterativeAmplitudeEstimation
_NormDist = None  # NormalDistribution
_EstProb  = None  # EstimationProblem
try:
    try:
        from qiskit_finance.circuit.library.probability_distributions import NormalDistribution as _NormDist  # type: ignore  # noqa
    except ImportError:
        from qiskit.circuit.library import NormalDistribution as _NormDist  # type: ignore  # noqa
    try:
        from qiskit.algorithms.amplitude_estimators import (  # type: ignore
            IterativeAmplitudeEstimation as _IAE, EstimationProblem as _EstProb,
        )
    except ImportError:
        from qiskit.algorithms import (  # type: ignore
            IterativeAmplitudeEstimation as _IAE, EstimationProblem as _EstProb,
        )
    _HAS_QAE = True
except Exception:
    pass

try:
    from qtbn_core import QTBNConfig, QTBNEngine
    _HAS_QTBN_CORE = True
except ImportError:
    _HAS_QTBN_CORE = False

try:
    from qaoa_scenario1 import (
        run_qaoa_portfolio,
        get_qaoa_portfolio_config,
        apply_regime_to_cfg,
        lambda_sweep_classical,
        log_qaoa_run,
        load_qaoa_log,
        load_qaoa_scenarios,
        save_qaoa_scenario,
        export_qaoa_snapshot,
        generate_portfolio_narrative,
        TOY_QAOA_PORTFOLIO,
        LACHESIS_BENCHMARK_PORTFOLIO,
    )
    _HAS_QAOA = True
except Exception as _qaoa_err:
    _HAS_QAOA = False
    _qaoa_err_msg = str(_qaoa_err)

try:
    from vqe_tab import (
        apply_risk_gates,
        build_scaled_risk_limits,
        estimate_order_risk,
        _try_run_real_vqe,
        _run_toy_energy,
        _build_problem_paulis,
        _energy_to_risk_multiplier,
    )
    _HAS_VQE = True
except Exception:
    _HAS_VQE = False

try:
    from foresight import SweepSpec, aggregate_counts, kl_div, blend
    _HAS_FORESIGHT = True
except Exception:
    _HAS_FORESIGHT = False

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _np_to_py(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _np_to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_to_py(v) for v in obj]
    return obj


def _log_return_frame(prices_df) -> "pd.DataFrame":
    """Compute log-return frame from a price DataFrame."""
    import numpy as np
    return np.log(prices_df / prices_df.shift(1)).dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "1.0",
        "capabilities": {
            "qiskit": _HAS_QISKIT,
            "yfinance": _HAS_YF,
            "qaoa": _HAS_QAOA,
            "vqe": _HAS_VQE,
            "foresight": _HAS_FORESIGHT,
            "qtbn_core": _HAS_QTBN_CORE,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH (SerpAPI)
# ═══════════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    serpapi_key: str
    num_results: int = 8


@app.post("/api/search")
def web_search(req: SearchRequest):
    import requests as _req
    if not req.serpapi_key.strip():
        raise HTTPException(400, "SerpAPI key required")
    params = {
        "q": req.query,
        "api_key": req.serpapi_key,
        "num": req.num_results,
        "engine": "google",
        "gl": "us",
        "hl": "en",
        "tbs": "qdr:m",  # results from the past month only
    }
    r = _req.get("https://serpapi.com/search.json", params=params, timeout=12)
    if not r.ok:
        raise HTTPException(r.status_code, f"SerpAPI error: {r.text[:200]}")
    data = r.json()
    results = [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        }
        for item in data.get("organic_results", [])[:req.num_results]
    ]
    return {"results": results, "query": req.query}


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class GateStep(BaseModel):
    q0: str = "None"
    q0_angle: float = 0.0
    q1: str = "None"
    q1_angle: float = 0.0
    q2: str = "None"
    q2_angle: float = 0.0
    q3: str = "None"
    q3_angle: float = 0.0
    cnot_01: bool = False
    cnot_12: bool = False
    cnot_23: bool = False


class NoiseParams(BaseModel):
    enable_depolarizing: bool = False
    depolarizing_prob: float = 0.01
    enable_amplitude_damping: bool = False
    amplitude_damping_prob: float = 0.02
    enable_phase_damping: bool = False
    phase_damping_prob: float = 0.02
    enable_cnot_noise: bool = False
    cnot_noise_prob: float = 0.02


class QuantumSimulateRequest(BaseModel):
    num_qubits: int = Field(1, ge=1, le=4)
    shots: int = Field(2048, ge=64, le=32768)
    seed: Optional[int] = 17
    step0: GateStep = GateStep(q0="H", q0_angle=0.5)
    step1: GateStep = GateStep()
    step2: GateStep = GateStep()
    noise: NoiseParams = NoiseParams()


def _apply_gate(qc, qubit: int, gate: str, angle: float):
    g = (gate or "None").upper()
    if g == "H":  qc.h(qubit)
    elif g == "X": qc.x(qubit)
    elif g == "Y": qc.y(qubit)
    elif g == "Z": qc.z(qubit)
    elif g == "RX": qc.rx(float(angle) * math.pi, qubit)
    elif g == "RY": qc.ry(float(angle) * math.pi, qubit)
    elif g == "RZ": qc.rz(float(angle) * math.pi, qubit)
    elif g == "S":  qc.s(qubit)
    elif g == "T":  qc.t(qubit)


def _build_circuit(req: QuantumSimulateRequest, measure: bool = False):
    nq = req.num_qubits
    qc = QuantumCircuit(nq, nq)
    steps = [req.step0, req.step1, req.step2]
    gate_keys = ["q0", "q1", "q2", "q3"]
    angle_keys = ["q0_angle", "q1_angle", "q2_angle", "q3_angle"]
    cnot_pairs = [("cnot_01", 0, 1), ("cnot_12", 1, 2), ("cnot_23", 2, 3)]

    for step in steps:
        for qi in range(nq):
            gate = getattr(step, gate_keys[qi], "None")
            angle = getattr(step, angle_keys[qi], 0.0)
            _apply_gate(qc, qi, gate, angle)
        for attr, ctrl, tgt in cnot_pairs:
            if getattr(step, attr, False) and nq > tgt:
                qc.cx(ctrl, tgt)
        qc.barrier()

    if measure:
        qc.measure(range(nq), range(nq))
    return qc


def _build_noise_model(noise: NoiseParams) -> Optional["NoiseModel"]:
    if not _HAS_QISKIT:
        return None
    nm = NoiseModel()
    if noise.enable_depolarizing and noise.depolarizing_prob > 0:
        err = depolarizing_error(noise.depolarizing_prob, 1)
        nm.add_all_qubit_quantum_error(err, ["u1", "u2", "u3", "h", "x", "y", "z", "s", "t", "rx", "ry", "rz"])
    if noise.enable_amplitude_damping and noise.amplitude_damping_prob > 0:
        err = amplitude_damping_error(noise.amplitude_damping_prob)
        nm.add_all_qubit_quantum_error(err, ["u1", "u2", "u3", "h", "x", "y", "z", "rx", "ry", "rz"])
    if noise.enable_phase_damping and noise.phase_damping_prob > 0:
        err = phase_damping_error(noise.phase_damping_prob)
        nm.add_all_qubit_quantum_error(err, ["u1", "u2", "u3", "h", "z", "s", "t", "rz"])
    if noise.enable_cnot_noise and noise.cnot_noise_prob > 0:
        err2 = depolarizing_error(noise.cnot_noise_prob, 2)
        nm.add_all_qubit_quantum_error(err2, ["cx"])
    return nm


@app.post("/api/quantum/simulate")
def quantum_simulate(req: QuantumSimulateRequest):
    if not _HAS_QISKIT:
        raise HTTPException(503, "Qiskit not available on this server")
    try:
        # Statevector (ideal)
        ideal_qc = _build_circuit(req, measure=False)
        sv_sim = AerSimulator(method="statevector")
        ideal_qc_sv = ideal_qc.copy()
        ideal_qc_sv.save_statevector()
        t_ideal = transpile(ideal_qc_sv, sv_sim)
        sv_result = sv_sim.run(t_ideal, seed_simulator=req.seed).result()
        sv = sv_result.get_statevector()
        statevector_real = [float(v.real) for v in sv]
        statevector_imag = [float(v.imag) for v in sv]
        probs_ideal = [float(abs(v) ** 2) for v in sv]

        # Measurement counts (with noise)
        nm = _build_noise_model(req.noise)
        meas_qc = _build_circuit(req, measure=True)
        backend_kwargs = {}
        if nm:
            backend_kwargs["noise_model"] = nm
        aer_meas = AerSimulator(**backend_kwargs)
        t_meas = transpile(meas_qc, aer_meas)
        meas_result = aer_meas.run(t_meas, shots=req.shots, seed_simulator=req.seed).result()
        raw_counts = meas_result.get_counts()
        counts = {k: int(v) for k, v in raw_counts.items()}
        counts_normalised = {k: v / req.shots for k, v in counts.items()}

        # Fidelity: compare noisy counts vs ideal probs
        nq = req.num_qubits
        fidelity = 1.0
        if nm:
            noisy_probs: Dict[str, float] = {}
            for k, v in counts.items():
                noisy_probs[k] = v / req.shots
            ideal_prob_dict: Dict[str, float] = {}
            for i, p in enumerate(probs_ideal):
                key = format(i, f"0{nq}b")
                ideal_prob_dict[key] = p
            dot = sum(
                math.sqrt(max(0, noisy_probs.get(k, 0))) * math.sqrt(max(0, ideal_prob_dict.get(k, 0)))
                for k in set(list(noisy_probs.keys()) + list(ideal_prob_dict.keys()))
            )
            fidelity = float(np.clip(dot ** 2, 0.0, 1.0))

        # Circuit ASCII visualization
        circuit_lines = str(ideal_qc.draw(output="text")).split("\n")

        return {
            "statevector_real": statevector_real,
            "statevector_imag": statevector_imag,
            "probabilities": probs_ideal,
            "counts": counts,
            "counts_normalised": counts_normalised,
            "fidelity": fidelity,
            "circuit_lines": circuit_lines,
            "num_qubits": nq,
        }
    except Exception as e:
        raise HTTPException(500, f"Quantum simulation error: {e}")


# ── Advanced Quantum ──────────────────────────────────────────────────────────

class TomographyRequest(BaseModel):
    gate: str = "H"
    angle: float = 0.5
    shots: int = 4096
    seed: Optional[int] = None


class BenchmarkingRequest(BaseModel):
    lengths: List[int] = [2, 4, 8, 16, 32, 48, 64]
    nseeds: int = 16
    shots: int = 4096
    seed: Optional[int] = None


class CalibrateRequest(BaseModel):
    shots: int = 4096
    seed: Optional[int] = None


class FidelityRequest(BaseModel):
    gate: str = "H"
    angle: float = 1.57
    shots: int = 4096
    seed: Optional[int] = None


@app.post("/api/quantum/advanced/tomography")
def quantum_tomography(req: TomographyRequest):
    if not _HAS_QISKIT:
        raise HTTPException(503, "Qiskit not available")
    try:
        qc = QuantumCircuit(1)
        _apply_gate(qc, 0, req.gate, req.angle)
        sim = AerSimulator(method="automatic")

        def measure_basis(basis: str) -> float:
            mc = QuantumCircuit(1, 1)
            mc.compose(qc, inplace=True)
            if basis == "X":
                mc.h(0)
            elif basis == "Y":
                mc.sdg(0)
                mc.h(0)
            mc.measure(0, 0)
            t = transpile(mc, sim)
            counts = sim.run(t, shots=req.shots, seed_simulator=req.seed).result().get_counts()
            p0 = counts.get("0", 0) / max(1, sum(counts.values()))
            return float(2 * p0 - 1)

        ex = measure_basis("X")
        ey = measure_basis("Y")
        ez = measure_basis("Z")
        purity = float(np.clip((ex**2 + ey**2 + ez**2), 0.0, 1.0))
        return {"bloch_x": ex, "bloch_y": ey, "bloch_z": ez, "purity": purity}
    except Exception as e:
        raise HTTPException(500, f"Tomography error: {e}")


@app.post("/api/quantum/advanced/benchmarking")
def quantum_benchmarking(req: BenchmarkingRequest):
    if not _HAS_QISKIT:
        raise HTTPException(503, "Qiskit not available")
    try:
        rng = np.random.default_rng(req.seed)
        sim = AerSimulator(method="automatic")
        survival = []

        for length in req.lengths:
            surv_runs = []
            for seed_i in range(req.nseeds):
                qc = QuantumCircuit(1, 1)
                # Random Clifford sequence
                for _ in range(length):
                    gate_choice = rng.integers(0, 4)
                    if gate_choice == 0:
                        qc.h(0)
                    elif gate_choice == 1:
                        qc.x(0)
                    elif gate_choice == 2:
                        qc.s(0)
                    else:
                        qc.z(0)
                qc.measure(0, 0)
                t = transpile(qc, sim)
                counts = sim.run(t, shots=req.shots, seed_simulator=int(req.seed or 0) + seed_i).result().get_counts()
                p0 = counts.get("0", 0) / max(1, sum(counts.values()))
                surv_runs.append(float(p0))
            survival.append(float(np.mean(surv_runs)))

        # Fit: A * p^m + B
        lengths_arr = np.array(req.lengths, dtype=float)
        surv_arr = np.array(survival, dtype=float)
        A, p_fit, B = 0.5, 0.99, 0.5
        try:
            from scipy.optimize import curve_fit
            def rb_model(m, A_, p_, B_):
                return A_ * p_ ** m + B_
            popt, _ = curve_fit(rb_model, lengths_arr, surv_arr,
                                p0=[0.5, 0.99, 0.25], maxfev=2000,
                                bounds=([0, 0, 0], [1, 1, 1]))
            A, p_fit, B = float(popt[0]), float(popt[1]), float(popt[2])
        except Exception:
            pass

        d = 2  # single qubit → d=2
        epg = float((1 - p_fit) * (1 - 1 / d))
        return {
            "lengths": req.lengths,
            "survival": survival,
            "fit": {"A": A, "p": p_fit, "B": B},
            "EPG": epg,
        }
    except Exception as e:
        raise HTTPException(500, f"Benchmarking error: {e}")


@app.post("/api/quantum/advanced/calibrate")
def quantum_calibrate(req: CalibrateRequest):
    if not _HAS_QISKIT:
        raise HTTPException(503, "Qiskit not available")
    try:
        sim = AerSimulator(method="automatic")
        rng = np.random.default_rng(req.seed)
        results = {}

        params = {
            "depolarizing":      ("H gate", 0.01, 0.05),
            "amplitude_damping": ("T1 decay", 0.005, 0.02),
            "phase_damping":     ("T2 dephasing", 0.01, 0.04),
        }

        for param_name, (gate_label, prior_alpha_raw, prior_beta_raw) in params.items():
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            t = transpile(qc, sim)
            counts = sim.run(t, shots=req.shots, seed_simulator=req.seed).result().get_counts()
            n_0 = counts.get("0", 0)
            n_1 = counts.get("1", 0)

            alpha_post = prior_alpha_raw + n_1
            beta_post = prior_beta_raw + n_0
            mean_val = alpha_post / (alpha_post + beta_post)

            # Credible interval (95%)
            from scipy.stats import beta as beta_dist
            ci_low = float(beta_dist.ppf(0.025, alpha_post, beta_post))
            ci_high = float(beta_dist.ppf(0.975, alpha_post, beta_post))

            results[param_name] = {
                "alpha": float(alpha_post),
                "beta": float(beta_post),
                "mean": float(mean_val),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "gate_label": gate_label,
            }

        return {"posteriors": results, "shots": req.shots}
    except Exception as e:
        raise HTTPException(500, f"Calibration error: {e}")


@app.post("/api/quantum/advanced/fidelity")
def quantum_fidelity(req: FidelityRequest):
    if not _HAS_QISKIT:
        raise HTTPException(503, "Qiskit not available")
    try:
        sim = AerSimulator(method="automatic")
        basis_preps = [QuantumCircuit(1), QuantumCircuit(1), QuantumCircuit(1)]
        basis_preps[1].h(0)
        basis_preps[2].s(0)
        basis_preps[2].h(0)

        fidelities = []
        for prep in basis_preps:
            # Ideal
            ideal = prep.copy()
            _apply_gate(ideal, 0, req.gate, req.angle / math.pi)
            ideal_sv = ideal.copy()
            ideal_sv.save_statevector()
            sv_sim = AerSimulator(method="statevector")
            t_ideal = transpile(ideal_sv, sv_sim)
            ideal_vec = sv_sim.run(t_ideal).result().get_statevector()

            # Noisy (sampled fidelity approximation)
            noisy = prep.copy()
            _apply_gate(noisy, 0, req.gate, req.angle / math.pi)
            noisy.measure_all()
            t_noisy = transpile(noisy, sim)
            counts = sim.run(t_noisy, shots=req.shots, seed_simulator=req.seed).result().get_counts()
            total = sum(counts.values())
            noisy_probs = {k.replace(" ", ""): v / total for k, v in counts.items()}
            ideal_probs = {format(i, f"0{ideal.num_qubits}b"): float(abs(ideal_vec[i])**2)
                          for i in range(len(ideal_vec))}
            fid = sum(
                math.sqrt(max(0, noisy_probs.get(k, 0)) * max(0, ideal_probs.get(k, 0)))
                for k in set(list(noisy_probs) + list(ideal_probs))
            ) ** 2
            fidelities.append(float(np.clip(fid, 0.0, 1.0)))

        avg_fidelity = float(np.mean(fidelities))
        return {"fidelity": avg_fidelity, "per_basis": fidelities, "gate": req.gate}
    except Exception as e:
        raise HTTPException(500, f"Fidelity error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

class FinancialAnalyzeRequest(BaseModel):
    tickers: List[str] = ["SPY", "QQQ", "AAPL"]
    lookback_days: int = Field(365, ge=60, le=2000)
    confidence: float = Field(0.95, ge=0.80, le=0.99)
    simulations: int = Field(50000, ge=1000, le=200000)
    demo_mode: bool = False
    # Optional sentiment multiplier from /api/sentiment/analyze → multiplier field.
    # When provided, stresses var_mc / cvar_mc: stressed = original × multiplier.
    sentiment_multiplier: Optional[float] = None
    # Quantum Amplitude Estimation toggle — requires qiskit-finance installation.
    use_qae: bool = False


def _generate_synthetic_prices(tickers: List[str], days: int) -> "pd.DataFrame":
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    data = {}
    for ticker in tickers:
        seed_val = sum(ord(c) for c in ticker)
        local_rng = np.random.default_rng(seed_val)
        drift = local_rng.uniform(0.0002, 0.0008)
        vol = local_rng.uniform(0.012, 0.025)
        returns = local_rng.normal(drift, vol, days)
        price = 100.0 * np.cumprod(1 + returns)
        data[ticker] = price
    return pd.DataFrame(data, index=dates)


def _monte_carlo_var_cvar(returns_series, horizon: int, simulations: int, confidence: float):
    """Monte Carlo VaR and CVaR."""
    mu = float(returns_series.mean())
    sigma = float(returns_series.std())
    rng = np.random.default_rng(42)
    sim_returns = rng.normal(mu * horizon, sigma * math.sqrt(horizon), simulations)
    alpha = 1.0 - confidence
    var = float(np.percentile(sim_returns, alpha * 100))
    cvar = float(sim_returns[sim_returns <= var].mean())
    return var, cvar


def _qae_var_cvar(returns_series, horizon: int, confidence: float) -> Tuple[float, float, float]:
    """
    Quantum Amplitude Estimation branch for VaR/CVaR.
    Mirrors the Streamlit implementation in qtbn_simulator_clean.py.
    Returns (var, cvar, qae_tail_prob_proxy).
    QAE gives a quantum-estimated tail-probability proxy; VaR/CVaR are then
    derived from the fitted normal model (same as Streamlit behaviour).
    """
    from scipy.stats import norm as _norm
    from qiskit import QuantumCircuit as _QC

    mu = float(returns_series.mean())
    sigma = float(returns_series.std())
    mu_h    = mu    * horizon
    sigma_h = sigma * math.sqrt(horizon)
    alpha   = 1.0 - confidence

    num_q  = 4
    bounds = [mu_h - 3 * sigma_h, mu_h + 3 * sigma_h]
    dist_circ = _NormDist(num_q, mu=mu_h, sigma=sigma_h, bounds=bounds)  # type: ignore

    A = _QC(dist_circ.num_qubits)
    A.compose(dist_circ, inplace=True)

    problem = _EstProb(  # type: ignore
        state_preparation=A,
        objective_qubits=[dist_circ.num_qubits - 1],
        post_processing=lambda a: a,
    )

    ae     = _IAE(epsilon=0.02, alpha=0.05)  # type: ignore
    result = ae.estimate(problem)
    p_tail = float(getattr(result, "estimation", 0.0))

    # VaR / CVaR from the analytic normal (QAE refines the tail-prob estimate)
    var  = float(_norm.ppf(1 - alpha, loc=mu_h, scale=sigma_h))
    cvar = float(mu_h - (sigma_h / alpha) * _norm.pdf(_norm.ppf(alpha)))

    return var, cvar, p_tail


@app.post("/api/financial/analyze")
def financial_analyze(req: FinancialAnalyzeRequest):
    if not _HAS_PANDAS:
        raise HTTPException(503, "pandas not available")
    try:
        tickers = [t.strip().upper() for t in req.tickers if t.strip()]
        if not tickers:
            raise HTTPException(400, "No tickers provided")

        # ── Fetch prices ──────────────────────────────────────────────────────
        prices_df = None
        data_source = "synthetic"

        if not req.demo_mode and _HAS_YF:
            try:
                end = dt.datetime.today()
                start = end - dt.timedelta(days=req.lookback_days + 60)
                raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    prices_df = raw["Close"].dropna(how="all")
                else:
                    prices_df = raw[["Close"]].dropna() if "Close" in raw.columns else raw.dropna()
                    prices_df.columns = tickers[:1]
                prices_df = prices_df.tail(req.lookback_days)
                if prices_df.empty or len(prices_df) < 30:
                    prices_df = None
                else:
                    data_source = "yfinance"
            except Exception:
                prices_df = None

        if prices_df is None:
            prices_df = _generate_synthetic_prices(tickers, req.lookback_days)
            data_source = "synthetic"

        # ── Returns ───────────────────────────────────────────────────────────
        log_rets = _log_return_frame(prices_df)

        # ── Portfolio returns (equal weight) ──────────────────────────────────
        portfolio_returns = log_rets.mean(axis=1)

        # ── Risk metrics (QAE or classical Monte Carlo) ───────────────────────
        qae_used      = False
        qae_tail_prob: Optional[float] = None
        if req.use_qae and _HAS_QAE:
            try:
                var_val, cvar_val, qae_tail_prob = _qae_var_cvar(
                    portfolio_returns, horizon=10, confidence=req.confidence
                )
                qae_used = True
            except Exception:
                # QAE failed — fall back silently to Monte Carlo
                var_val, cvar_val = _monte_carlo_var_cvar(
                    portfolio_returns, horizon=10,
                    simulations=req.simulations, confidence=req.confidence
                )
        else:
            var_val, cvar_val = _monte_carlo_var_cvar(
                portfolio_returns, horizon=10,
                simulations=req.simulations, confidence=req.confidence
            )
        hist_var = float(portfolio_returns.quantile(1 - req.confidence))
        hist_cvar = float(portfolio_returns[portfolio_returns <= hist_var].mean())

        excess = portfolio_returns - 0.04 / 252
        sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 0 else float("nan")
        downside = excess[excess < 0]
        sortino_denom = float(np.sqrt(np.mean(downside**2)) * np.sqrt(252)) if len(downside) > 0 else 0
        sortino = float(np.sqrt(252) * excess.mean() / sortino_denom) if sortino_denom > 0 else float("nan")

        cum = (1 + portfolio_returns).cumprod()
        peak = cum.cummax()
        max_dd = float(((cum - peak) / peak).min())

        ann_vol_rolling = log_rets.rolling(21).std().mean(axis=1) * math.sqrt(252)
        last_ann_vol = float(ann_vol_rolling.dropna().iloc[-1]) if not ann_vol_rolling.dropna().empty else 0.0
        threshold = 0.30
        if last_ann_vol > threshold * 1.25:
            regime = "High Volatility"
        elif last_ann_vol > threshold * 0.75:
            regime = "Medium Volatility"
        else:
            regime = "Low Volatility"

        # ── Serialise price + return data ─────────────────────────────────────
        dates = [str(d.date()) for d in prices_df.index]
        prices_out: Dict[str, List[float]] = {}
        returns_out: Dict[str, List[float]] = {}
        for ticker in prices_df.columns:
            prices_out[ticker] = [_safe_float(v) for v in prices_df[ticker].tolist()]
        for ticker in log_rets.columns:
            returns_out[ticker] = [_safe_float(v) for v in log_rets[ticker].tolist()]

        portfolio_ret_list = [_safe_float(v) for v in portfolio_returns.tolist()]

        # ── Sentiment stress (optional) ───────────────────────────────────────
        sent_mult = req.sentiment_multiplier
        var_mc_stressed   = _safe_float(var_val  * sent_mult) if sent_mult is not None else None
        cvar_mc_stressed  = _safe_float(cvar_val * sent_mult) if sent_mult is not None else None

        return {
            "tickers": list(prices_df.columns),
            "dates": dates,
            "prices": prices_out,
            "returns": returns_out,
            "portfolio_returns": portfolio_ret_list,
            "var_mc": _safe_float(var_val),
            "cvar_mc": _safe_float(cvar_val),
            "var_historical": _safe_float(hist_var),
            "cvar_historical": _safe_float(hist_cvar),
            "sharpe": _safe_float(sharpe),
            "sortino": _safe_float(sortino),
            "max_drawdown": _safe_float(max_dd),
            "annualized_volatility": _safe_float(last_ann_vol),
            "regime": regime,
            "data_source": data_source,
            "sentiment_multiplier": sent_mult,
            "var_mc_stressed": var_mc_stressed,
            "cvar_mc_stressed": cvar_mc_stressed,
            "use_qae": req.use_qae,
            "qae_active": qae_used,
            "qae_available": _HAS_QAE,
            "qae_tail_prob": qae_tail_prob,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Financial analysis error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# QTBN FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

class QTBNForecastRequest(BaseModel):
    prior_regime: str = "calm"
    risk_on_prior: float = Field(0.5, ge=0.0, le=1.0)
    drift_mu: float = 0.08
    horizon_days: int = Field(10, ge=1, le=252)
    steps: int = Field(3, ge=1, le=10)


@app.post("/api/qtbn/forecast")
def qtbn_forecast(req: QTBNForecastRequest):
    try:
        # qtbn_forecast_stub equivalent (pure Python)
        regime_vol = {"calm": 0.12, "stressed": 0.25, "crisis": 0.40}
        vol = regime_vol.get(req.prior_regime.lower(), 0.18)
        h_frac = req.horizon_days / 252.0
        mu_h = req.drift_mu * h_frac
        sigma_h = vol * math.sqrt(h_frac)

        def cdf(x: float) -> float:
            if sigma_h <= 1e-8:
                return 1.0 if x >= mu_h else 0.0
            z = (x - mu_h) / (sigma_h * math.sqrt(2.0))
            return 0.5 * (1.0 + math.erf(z))

        t1, t2, t3 = -2 * sigma_h, -sigma_h, sigma_h
        p_sv = cdf(t1)
        p_l  = cdf(t2) - cdf(t1)
        p_f  = cdf(t3) - cdf(t2)
        p_g  = 1.0 - cdf(t3)

        tilt = req.risk_on_prior - 0.5
        p_g  += 0.40 * tilt
        p_sv -= 0.20 * tilt
        p_l  -= 0.10 * tilt
        p_f  -= 0.10 * tilt

        probs = np.clip([p_g, p_f, p_l, p_sv], 0.0, None)
        total = probs.sum() or 1.0
        probs /= total

        # Regime path using built-in transition matrix
        T_base = np.array([[0.85, 0.13, 0.02], [0.25, 0.55, 0.20], [0.10, 0.25, 0.65]], dtype=float)
        regimes = ["calm", "stressed", "crisis"]
        regime_map = {"calm": 0, "stressed": 1, "crisis": 2}
        start_idx = regime_map.get(req.prior_regime.lower(), 0)
        state = np.full(3, 0.05)
        state[start_idx] = 0.90
        state /= state.sum()

        timeline = []
        p = state.copy()
        drift_path = []
        risk_on_path = []
        drift_by = {"calm": 0.10, "stressed": 0.04, "crisis": -0.05}
        risk_on_by = {"calm": 0.70, "stressed": 0.45, "crisis": 0.20}

        for _ in range(req.steps):
            step_dict = {r: float(p[i]) for i, r in enumerate(regimes)}
            timeline.append(step_dict)
            drift_path.append(float(sum(p[i] * drift_by[r] for i, r in enumerate(regimes))))
            risk_on_path.append(float(sum(p[i] * risk_on_by[r] for i, r in enumerate(regimes))))
            p = p @ T_base

        return {
            "prior_regime": req.prior_regime,
            "horizon_days": req.horizon_days,
            "P_gain": float(probs[0]),
            "P_flat": float(probs[1]),
            "P_loss": float(probs[2]),
            "P_severe_loss": float(probs[3]),
            "regime_timeline": timeline,
            "drift_path": drift_path,
            "risk_on_path": risk_on_path,
        }
    except Exception as e:
        raise HTTPException(500, f"QTBN forecast error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# QAOA PORTFOLIO OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class QAOAOptimizeRequest(BaseModel):
    portfolio: str = "Toy 3-asset tech portfolio"
    depth: int = Field(1, ge=1, le=5)
    shots: int = Field(1024, ge=64, le=8192)
    lam: float = Field(1.0, ge=0.1, le=2.0)
    backend: str = "Classical brute-force"
    regime: Optional[str] = None


class QAOASweepRequest(BaseModel):
    portfolio: str = "Toy 3-asset tech portfolio"
    lam_min: float = 0.1
    lam_max: float = 2.0
    n_points: int = Field(10, ge=3, le=50)


class QAOASaveScenarioRequest(BaseModel):
    name: str
    result: Dict[str, Any]
    portfolio: str
    notes: str = ""


@app.get("/api/qaoa/portfolios")
def qaoa_portfolios():
    if not _HAS_QAOA:
        return {"portfolios": ["Toy 3-asset tech portfolio", "Lachesis benchmark (equities + bond + gold)"],
                "note": "QAOA module unavailable – classical fallback will be used"}
    return {
        "portfolios": ["Toy 3-asset tech portfolio", "Lachesis benchmark (equities + bond + gold)"],
        "toy": _np_to_py(TOY_QAOA_PORTFOLIO),
        "benchmark": _np_to_py(LACHESIS_BENCHMARK_PORTFOLIO),
    }


@app.post("/api/qaoa/optimize")
def qaoa_optimize(req: QAOAOptimizeRequest):
    if not _HAS_QAOA:
        raise HTTPException(503, f"QAOA module unavailable")
    try:
        cfg = get_qaoa_portfolio_config(req.portfolio)
        if req.regime:
            cfg = apply_regime_to_cfg(cfg, req.regime)
        result = run_qaoa_portfolio(cfg, depth=req.depth, shots=req.shots, lam=req.lam, backend=req.backend)
        result["shots"] = req.shots
        log_qaoa_run(result)
        narrative = generate_portfolio_narrative(result, cfg, req.regime)
        return {**_np_to_py(result), "narrative": narrative, "assets": cfg["assets"]}
    except Exception as e:
        raise HTTPException(500, f"QAOA optimization error: {e}")


@app.post("/api/qaoa/sweep")
def qaoa_sweep(req: QAOASweepRequest):
    if not _HAS_QAOA:
        raise HTTPException(503, "QAOA module unavailable")
    try:
        cfg = get_qaoa_portfolio_config(req.portfolio)
        lam_values = np.linspace(req.lam_min, req.lam_max, req.n_points).tolist()
        sweep_data = []
        for lam_val in lam_values:
            result = run_qaoa_portfolio(cfg, depth=1, shots=256, lam=float(lam_val), backend="Classical brute-force")
            sweep_data.append({
                "lam": float(lam_val),
                "expected_return": _safe_float(result.get("expected_return")),
                "risk": _safe_float(result.get("risk")),
                "objective": _safe_float(result.get("objective")),
                "selected_assets": result.get("selected_assets", []),
                "bitstring": result.get("bitstring", ""),
            })
        return {"sweep": sweep_data, "portfolio": req.portfolio}
    except Exception as e:
        raise HTTPException(500, f"QAOA sweep error: {e}")


@app.get("/api/qaoa/scenarios")
def qaoa_get_scenarios():
    if not _HAS_QAOA:
        return {"scenarios": []}
    try:
        scenarios = load_qaoa_scenarios()
        return {"scenarios": _np_to_py(scenarios)}
    except Exception as e:
        raise HTTPException(500, f"Scenario load error: {e}")


@app.post("/api/qaoa/scenarios")
def qaoa_save_scenario(req: QAOASaveScenarioRequest):
    if not _HAS_QAOA:
        raise HTTPException(503, "QAOA module unavailable")
    try:
        scenario = {
            "name": req.name,
            "result": req.result,
            "portfolio": req.portfolio,
            "notes": req.notes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_qaoa_scenario(scenario)
        return {"status": "saved", "name": req.name}
    except Exception as e:
        raise HTTPException(500, f"Scenario save error: {e}")


@app.get("/api/qaoa/log")
def qaoa_get_log():
    if not _HAS_QAOA:
        return {"rows": []}
    try:
        import csv, os
        log_path = str(APP_DIR / "qaoa_runs_log.csv")
        if not os.path.exists(log_path):
            return {"rows": []}
        rows = []
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return {"rows": rows[-50:]}  # latest 50
    except Exception as e:
        raise HTTPException(500, f"Log read error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# VQE RISK GATING
# ═══════════════════════════════════════════════════════════════════════════════

POLICY_LIMITS = {
    "Conservative": {"max_notional_usd": 50_000.0, "max_var_usd": 2_000.0, "max_cvar_usd": 3_500.0, "max_leverage": 1.5},
    "Moderate":     {"max_notional_usd": 250_000.0, "max_var_usd": 10_000.0, "max_cvar_usd": 18_000.0, "max_leverage": 3.0},
    "Aggressive":   {"max_notional_usd": 1_000_000.0, "max_var_usd": 50_000.0, "max_cvar_usd": 90_000.0, "max_leverage": 6.0},
}

TRADE_AUDIT: List[Dict[str, Any]] = []


class VQERiskGateRequest(BaseModel):
    requested_notional_usd: float
    price_usd: float = 100.0
    vol_daily_pct: float = 1.5
    leverage: float = 1.0
    policy: str = "Moderate"


@app.post("/api/vqe/risk-gate")
def vqe_risk_gate(req: VQERiskGateRequest):
    try:
        policy = req.policy if req.policy in POLICY_LIMITS else "Moderate"
        limits = POLICY_LIMITS[policy]

        # Estimate order risk
        vol_daily = req.vol_daily_pct / 100.0
        est_var_usd = req.requested_notional_usd * vol_daily * 1.65
        est_cvar_usd = est_var_usd * 1.30
        leverage_used = req.leverage

        reasons: List[str] = []
        status = "APPROVED"
        final_notional = req.requested_notional_usd

        if req.requested_notional_usd > limits["max_notional_usd"]:
            reasons.append(f"Notional ${req.requested_notional_usd:,.0f} > limit ${limits['max_notional_usd']:,.0f}")
            final_notional = limits["max_notional_usd"]
            status = "PARTIAL"
        if est_var_usd > limits["max_var_usd"]:
            reasons.append(f"Est. VaR ${est_var_usd:,.0f} > limit ${limits['max_var_usd']:,.0f}")
            if status == "APPROVED":
                status = "PARTIAL"
        if est_cvar_usd > limits["max_cvar_usd"]:
            reasons.append(f"Est. CVaR ${est_cvar_usd:,.0f} > limit ${limits['max_cvar_usd']:,.0f}")
            status = "BLOCKED"
        if leverage_used > limits["max_leverage"]:
            reasons.append(f"Leverage {leverage_used:.1f}x > limit {limits['max_leverage']:.1f}x")
            status = "BLOCKED"

        record = {
            "timestamp": dt.datetime.utcnow().isoformat(),
            "policy": policy,
            "requested_notional_usd": req.requested_notional_usd,
            "final_notional_usd": final_notional,
            "est_var_usd": est_var_usd,
            "est_cvar_usd": est_cvar_usd,
            "leverage_used": leverage_used,
            "status": status,
            "reasons": reasons,
            "limits": limits,
        }
        TRADE_AUDIT.append(record)

        return record
    except Exception as e:
        raise HTTPException(500, f"VQE risk gate error: {e}")


@app.get("/api/vqe/audit")
def vqe_audit(limit: int = 20):
    return {"records": TRADE_AUDIT[-limit:], "total": len(TRADE_AUDIT)}


class VQESolveRequest(BaseModel):
    problem: str = "Toy Hamiltonian"
    ansatz_name: str = "RealAmplitudes"
    optimizer_name: str = "COBYLA"
    num_qubits: int = 2
    reps: int = 2
    maxiter: int = 80
    seed: Optional[int] = 42
    pauli_text: str = "ZZ:1, XI:0.4, IX:0.4"
    maxcut_edges_text: str = "0-1:1.0\n1-2:0.8\n0-2:0.6"
    ising_h_text: str = "0.5, -0.5"
    ising_J_text: str = "0 1 1.0"
    backend_choice: str = "Estimator (default)"


@app.post("/api/vqe/solve")
def vqe_solve(req: VQESolveRequest):
    try:
        if not _HAS_VQE:
            raise HTTPException(500, "vqe_tab not available")

        pauli_list, problem_meta = _build_problem_paulis(
            problem=req.problem,
            n=req.num_qubits,
            pauli_text=req.pauli_text,
            maxcut_edges_text=req.maxcut_edges_text,
            maxcut_negative_cost_h=True,
            ising_h_text=req.ising_h_text,
            ising_J_text=req.ising_J_text,
        )

        converged, energy, meta, estimator_name, history = _try_run_real_vqe(
            num_qubits=req.num_qubits,
            pauli_list=pauli_list,
            ansatz_name=req.ansatz_name,
            reps=req.reps,
            optimizer_name=req.optimizer_name,
            maxiter=req.maxiter,
            backend_choice=req.backend_choice,
            seed=req.seed,
        )

        used_fallback = False
        if not converged or not np.isfinite(energy):
            energy, meta, estimator_name, history = _run_toy_energy(seed=req.seed)
            used_fallback = True

        risk_multiplier = float(_energy_to_risk_multiplier(energy))

        return {
            "converged": bool(converged) and not used_fallback,
            "used_fallback": used_fallback,
            "energy": float(energy) if np.isfinite(energy) else None,
            "risk_multiplier": risk_multiplier,
            "estimator": estimator_name,
            "history": history[:200],
            "num_pauli_terms": len(pauli_list),
            "problem_type": problem_meta.get("problem", req.problem),
            "ansatz_desc": f"{req.ansatz_name}({req.num_qubits}q, reps={req.reps})",
            "optimizer_desc": f"{req.optimizer_name}(maxiter={req.maxiter})",
            "num_qubits": req.num_qubits,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"VQE solve error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FORESIGHT SWEEPS
# ═══════════════════════════════════════════════════════════════════════════════

class ForesightSweepRequest(BaseModel):
    shots: int = Field(1024, ge=64, le=8192)
    seeds: List[int] = [17, 42, 99]
    pdep_values: List[float] = [0.0, 0.01, 0.03, 0.05]
    pamp_values: List[float] = [0.0, 0.02]
    circuit: QuantumSimulateRequest = QuantumSimulateRequest()


class ForesightScenario(BaseModel):
    name: str
    data: Dict[str, Any]


SCENARIOS_PATH = APP_DIR / "scenarios.json"


def _load_scenarios() -> Dict[str, Any]:
    try:
        if SCENARIOS_PATH.exists():
            return json.loads(SCENARIOS_PATH.read_text())
        return {}
    except Exception:
        return {}


def _save_scenarios(data: Dict[str, Any]) -> None:
    SCENARIOS_PATH.write_text(json.dumps(data, indent=2, default=str))


@app.post("/api/foresight/sweep")
def foresight_sweep(req: ForesightSweepRequest):
    if not _HAS_QISKIT:
        raise HTTPException(503, "Qiskit not available")
    try:
        results_grid = []
        reference_counts: Optional[Dict[str, float]] = None

        for pdep in req.pdep_values:
            row = []
            for pamp in req.pamp_values:
                run_counts_list: List[Dict[str, int]] = []
                for seed_val in req.seeds:
                    noise = NoiseParams(
                        enable_depolarizing=pdep > 0,
                        depolarizing_prob=pdep,
                        enable_amplitude_damping=pamp > 0,
                        amplitude_damping_prob=pamp,
                    )
                    sim_req = req.circuit.copy()
                    sim_req.noise = noise
                    sim_req.shots = req.shots
                    sim_req.seed = seed_val

                    nm = _build_noise_model(noise)
                    meas_qc = _build_circuit(sim_req, measure=True)
                    be_kwargs = {}
                    if nm:
                        be_kwargs["noise_model"] = nm
                    aer = AerSimulator(**be_kwargs)
                    t = transpile(meas_qc, aer)
                    counts = aer.run(t, shots=req.shots, seed_simulator=seed_val).result().get_counts()
                    run_counts_list.append({k: int(v) for k, v in counts.items()})

                # aggregate across seeds
                all_keys = list({k for c in run_counts_list for k in c})
                agg: Dict[str, float] = {}
                for k in all_keys:
                    vals = [c.get(k, 0) for c in run_counts_list]
                    agg[k] = float(np.mean(vals) / req.shots)

                # reference = first cell (no noise)
                if reference_counts is None:
                    reference_counts = agg.copy()

                # KL divergence from reference
                eps = 1e-12
                kl = 0.0
                for k in set(list(agg.keys()) + list(reference_counts.keys())):
                    p = agg.get(k, eps)
                    q = reference_counts.get(k, eps)
                    if p > eps:
                        kl += p * math.log(p / max(q, eps))

                row.append({
                    "pdep": pdep,
                    "pamp": pamp,
                    "kl_divergence": float(kl),
                    "counts": agg,
                })
            results_grid.append(row)

        return {
            "pdep_values": req.pdep_values,
            "pamp_values": req.pamp_values,
            "grid": results_grid,
        }
    except Exception as e:
        raise HTTPException(500, f"Foresight sweep error: {e}")


@app.get("/api/foresight/scenarios")
def foresight_get_scenarios():
    return {"scenarios": _load_scenarios()}


@app.post("/api/foresight/scenarios")
def foresight_save_scenario(req: ForesightScenario):
    try:
        scenarios = _load_scenarios()
        scenarios[req.name] = {**req.data, "saved_at": dt.datetime.utcnow().isoformat()}
        _save_scenarios(scenarios)
        return {"status": "saved", "name": req.name}
    except Exception as e:
        raise HTTPException(500, f"Scenario save error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentRequest(BaseModel):
    tickers: List[str] = ["AAPL", "MSFT"]
    keywords: List[str] = []
    max_items: int = Field(30, ge=5, le=200)
    provider: str = Field("google_rss", pattern="^(google_rss|perplexity)$")
    perplexity_api_key: Optional[str] = None
    perplexity_model: str = "sonar"


@app.post("/api/sentiment/analyze")
def sentiment_analyze(req: SentimentRequest):
    try:
        tickers = [t.strip().upper() for t in req.tickers if t.strip()]
        if not tickers:
            raise HTTPException(400, "No tickers provided")

        items_out: List[Dict[str, Any]] = []
        headlines: List[str] = []
        scores: List[float] = []
        avg_score = 0.0
        multiplier = 1.0
        provider_label = "Google News RSS + VADER"

        # ── Perplexity branch ─────────────────────────────────────────────────
        if req.provider == "perplexity":
            token = (req.perplexity_api_key or "").strip()
            if not token:
                raise HTTPException(400, "perplexity_api_key is required when provider=perplexity")
            provider_label = f"Perplexity API ({req.perplexity_model})"
            try:
                import requests as _req_lib
            except ImportError:
                raise HTTPException(503, "requests library not available")

            prompt = (
                "You are a financial news sentiment engine.\n"
                f"Tickers: {', '.join(tickers)}\n"
                "Find current, relevant market news and return ONLY valid JSON:\n"
                '{"avg_score": number, "multiplier": number, '
                '"headlines": [{"title": "string", "url": "string", "score": number}]}\n'
                "Rules: avg_score -1..1; multiplier 0.5..1.5; up to 20 headlines; JSON only."
            )
            resp = _req_lib.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "model": req.perplexity_model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": "Return strict JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=60,
            )
            if resp.status_code >= 300:
                raise HTTPException(502, f"Perplexity HTTP {resp.status_code}: {resp.text[:300]}")
            body = resp.json()
            content = ""
            try:
                choices = body.get("choices", [])
                if choices:
                    c = choices[0].get("message", {}).get("content", "")
                    content = c if isinstance(c, str) else "\n".join(
                        p.get("text", "") for p in c if isinstance(p, dict)
                    )
            except Exception:
                content = ""
            if not content:
                raise HTTPException(502, "Perplexity response missing content")
            s = content.strip()
            if s.startswith("```"):
                s = s.strip("`")
                if s.lower().startswith("json"):
                    s = s[4:].lstrip()
            if "{" in s:
                s = s[s.find("{"):s.rfind("}") + 1]
            try:
                parsed = json.loads(s)
            except Exception as e:
                raise HTTPException(502, f"Failed to parse Perplexity JSON: {e}")
            avg_score = float(max(-1.0, min(1.0, parsed.get("avg_score", 0.0))))
            multiplier = float(max(0.5, min(1.5, parsed.get("multiplier", 1.0 - 0.5 * avg_score))))
            for raw in parsed.get("headlines", [])[:req.max_items]:
                title = str(raw.get("title", "")).strip()
                link  = str(raw.get("url", "") or raw.get("link", "")).strip()
                sc    = float(max(-1.0, min(1.0, raw.get("score", avg_score))))
                if not title:
                    continue
                headlines.append(title)
                scores.append(sc)
                items_out.append({"ticker": tickers[0], "title": title, "score": round(sc, 4), "published": "", "link": link})
            # Fallback to citations if model omitted headlines array
            if not headlines:
                for cite in body.get("citations", [])[:req.max_items]:
                    url = str(cite).strip()
                    if not url:
                        continue
                    headlines.append(url)
                    scores.append(avg_score)
                    items_out.append({"ticker": tickers[0], "title": url, "score": round(avg_score, 4), "published": "", "link": url})

        # ── Google News RSS + VADER branch ────────────────────────────────────
        else:
            try:
                import feedparser as _fp  # type: ignore
                _HAS_FEEDPARSER = True
            except ImportError:
                _HAS_FEEDPARSER = False

            sia = None
            _HAS_VADER = False
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer as _SIA  # type: ignore
                import nltk as _nltk  # type: ignore
                try:
                    _nltk.data.find("sentiment/vader_lexicon.zip")
                except LookupError:
                    _nltk.download("vader_lexicon", quiet=True)
                sia = _SIA()
                _HAS_VADER = True
            except Exception:
                pass

            if _HAS_FEEDPARSER:
                per_ticker = max(1, req.max_items // max(len(tickers), 1))
                for ticker in tickers[:10]:
                    feed_url = (
                        f"https://news.google.com/rss/search"
                        f"?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
                    )
                    try:
                        feed = _fp.parse(feed_url)
                        for entry in feed.entries[:per_ticker]:
                            title = getattr(entry, "title", "") or ""
                            link  = getattr(entry, "link",  "") or ""
                            pub   = getattr(entry, "published", "") or ""
                            if not title:
                                continue
                            if _HAS_VADER and sia:
                                sc = float(sia.polarity_scores(title)["compound"])
                            else:
                                tl = title.lower()
                                pos = ["gain","rise","up","rally","bull","growth","profit","beat","surge"]
                                neg = ["fall","drop","down","loss","bear","miss","crash","decline"]
                                raw_sc = sum(1 for w in pos if w in tl) - sum(1 for w in neg if w in tl)
                                sc = float(np.clip(raw_sc / 3.0, -1.0, 1.0))
                            headlines.append(title)
                            scores.append(sc)
                            items_out.append({"ticker": ticker, "title": title, "score": round(sc, 4), "published": pub, "link": link})
                    except Exception:
                        continue

            # Keyword boost
            kw_lower = [k.lower() for k in req.keywords]
            if kw_lower:
                for item in items_out:
                    boost = sum(0.1 for kw in kw_lower if kw in item["title"].lower())
                    item["score"] = float(np.clip(item["score"] + boost, -1.0, 1.0))
                scores = [i["score"] for i in items_out]

            avg_score = float(np.mean(scores)) if scores else 0.0
            multiplier = float(max(0.5, min(1.5, 1.0 - 0.5 * avg_score)))
            provider_label = "Google News RSS + " + ("VADER" if _HAS_VADER else "keyword scoring")

        # ── Synthetic fallback if nothing was fetched ─────────────────────────
        if not items_out:
            rng = np.random.default_rng(int(time.time()) % 10000)
            for ticker in tickers:
                for tmpl in [
                    f"{ticker} reports strong quarterly earnings",
                    f"{ticker} faces headwinds from macro uncertainty",
                ]:
                    s = float(rng.uniform(-0.5, 0.8))
                    items_out.append({"ticker": ticker, "title": tmpl, "score": round(s, 3),
                                      "published": dt.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"), "link": ""})
                    scores.append(s)
            avg_score = float(np.mean(scores)) if scores else 0.0
            multiplier = float(max(0.5, min(1.5, 1.0 - 0.5 * avg_score)))
            provider_label = "synthetic fallback"

        return {
            "tickers": tickers,
            "items": items_out[:req.max_items],
            "headlines": [i["title"] for i in items_out[:req.max_items]],
            "avg_score": round(avg_score, 4),
            "multiplier": round(multiplier, 4),
            "total_items": len(items_out),
            "provider": provider_label,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Sentiment analysis error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM – REDUCED STATES (partial trace per qubit → Bloch vector)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/quantum/reduced-states")
def quantum_reduced_states(req: QuantumSimulateRequest):
    if not _HAS_QISKIT:
        raise HTTPException(503, "Qiskit not available")
    try:
        from qiskit.quantum_info import partial_trace, DensityMatrix
        nq = req.num_qubits
        qc = _build_circuit(req, measure=False)

        noise_model = _build_noise_model(req.noise)
        noise_active = noise_model is not None and len(noise_model.noise_instructions) > 0

        if noise_active:
            # Density-matrix simulator so noise is propagated through ρ
            qc_dm = qc.copy()
            qc_dm.save_density_matrix()
            dm_sim = AerSimulator(method="density_matrix")
            t = transpile(qc_dm, dm_sim)
            job_result = dm_sim.run(t, noise_model=noise_model,
                                    seed_simulator=req.seed).result()
            dm = DensityMatrix(job_result.data()["density_matrix"])
        else:
            # Ideal path: exact statevector → density matrix
            from qiskit.quantum_info import Statevector
            sv = Statevector.from_instruction(qc)
            dm = DensityMatrix(sv)

        reduced = []
        for qi in range(nq):
            keep_qubits = list(range(nq))
            keep_qubits.remove(qi)
            dm_q = partial_trace(dm, keep_qubits)
            rho = np.array(dm_q.data)
            bx = float(2 * rho[0, 1].real)
            by = float(2 * rho[0, 1].imag)
            bz = float((rho[0, 0] - rho[1, 1]).real)
            purity = float(np.real(np.trace(rho @ rho)))
            reduced.append({
                "qubit": qi,
                "bloch_x": bx,
                "bloch_y": by,
                "bloch_z": bz,
                "purity": purity,
                "rho_real": [[float(v.real) for v in row] for row in rho],
                "rho_imag": [[float(v.imag) for v in row] for row in rho],
            })
        return {
            "num_qubits": nq,
            "reduced_states": reduced,
            "noise_applied": noise_active,
        }
    except Exception as e:
        raise HTTPException(500, f"Reduced states error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM – MEASUREMENT (ideal vs noisy counts + TV distance)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/quantum/measurement")
def quantum_measurement(req: QuantumSimulateRequest):
    if not _HAS_QISKIT:
        raise HTTPException(503, "Qiskit not available")
    try:
        nq = req.num_qubits
        # Ideal (no noise)
        ideal_qc = _build_circuit(req, measure=True)
        sv_sim = AerSimulator(method="automatic")
        t_ideal = transpile(ideal_qc, sv_sim)
        ideal_counts_raw = sv_sim.run(t_ideal, shots=req.shots, seed_simulator=req.seed).result().get_counts()
        ideal_counts = {k: int(v) for k, v in ideal_counts_raw.items()}

        # Noisy
        nm = _build_noise_model(req.noise)
        noisy_counts = ideal_counts.copy()
        if nm:
            noisy_qc = _build_circuit(req, measure=True)
            backend_kwargs = {"noise_model": nm}
            aer_noisy = AerSimulator(**backend_kwargs)
            t_noisy = transpile(noisy_qc, aer_noisy)
            noisy_raw = aer_noisy.run(t_noisy, shots=req.shots, seed_simulator=req.seed).result().get_counts()
            noisy_counts = {k: int(v) for k, v in noisy_raw.items()}

        all_keys = sorted(set(list(ideal_counts.keys()) + list(noisy_counts.keys())))
        shots_f = float(req.shots)
        ideal_p = {k: ideal_counts.get(k, 0) / shots_f for k in all_keys}
        noisy_p  = {k: noisy_counts.get(k, 0) / shots_f for k in all_keys}
        tv = float(0.5 * sum(abs(ideal_p.get(k, 0) - noisy_p.get(k, 0)) for k in all_keys))

        return {
            "ideal_counts": ideal_counts,
            "noisy_counts": noisy_counts,
            "ideal_probs": ideal_p,
            "noisy_probs": noisy_p,
            "tv_distance": tv,
            "all_states": all_keys,
            "num_qubits": nq,
        }
    except Exception as e:
        raise HTTPException(500, f"Measurement error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM – PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

PRESET_CIRCUITS = {
    "bell": {
        "label": "Bell State |Φ+⟩",
        "num_qubits": 2,
        "step0": {"q0": "H", "q0_angle": 0.5, "cnot_01": True},
        "step1": {},
        "step2": {},
        "noise": {},
    },
    "dephasing": {
        "label": "Dephasing (H + phase noise)",
        "num_qubits": 1,
        "step0": {"q0": "H", "q0_angle": 0.5},
        "step1": {},
        "step2": {},
        "noise": {"enable_phase_damping": True, "phase_damping_prob": 0.05},
    },
    "amplitude_relaxation": {
        "label": "Amplitude Relaxation (X + T1 decay)",
        "num_qubits": 1,
        "step0": {"q0": "X", "q0_angle": 0.0},
        "step1": {},
        "step2": {},
        "noise": {"enable_amplitude_damping": True, "amplitude_damping_prob": 0.15},
    },
    "ghz_3": {
        "label": "GHZ State (3 qubits)",
        "num_qubits": 3,
        "step0": {"q0": "H", "q0_angle": 0.5, "cnot_01": True, "cnot_12": True},
        "step1": {},
        "step2": {},
        "noise": {},
    },
    "qft": {
        "label": "QFT-inspired (H + RZ)",
        "num_qubits": 2,
        "step0": {"q0": "H", "q0_angle": 0.5},
        "step1": {"q0": "RZ", "q0_angle": 0.25, "q1": "H", "q1_angle": 0.5},
        "step2": {"cnot_01": True},
        "noise": {},
    },
}

@app.get("/api/quantum/presets")
def get_quantum_presets():
    return {"presets": [{"key": k, "label": v["label"]} for k, v in PRESET_CIRCUITS.items()]}


@app.get("/api/quantum/presets/{preset_key}")
def get_quantum_preset(preset_key: str):
    if preset_key not in PRESET_CIRCUITS:
        raise HTTPException(404, f"Preset '{preset_key}' not found")
    return PRESET_CIRCUITS[preset_key]


# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL – INSIDER TRADING / PORTFOLIO DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class InsiderRequest(BaseModel):
    tickers: List[str] = ["AAPL", "MSFT", "NVDA"]
    lookback_days: int = Field(252, ge=30, le=2000)
    portfolio_value: float = Field(100000.0, ge=1.0)
    confidence: float = Field(0.95, ge=0.80, le=0.99)
    simulations: int = Field(10000, ge=1000, le=100000)
    demo_mode: bool = True


@app.post("/api/financial/insider")
def financial_insider(req: InsiderRequest):
    if not _HAS_PANDAS:
        raise HTTPException(503, "pandas not available")
    try:
        tickers = [t.strip().upper() for t in req.tickers if t.strip()]
        if not tickers:
            raise HTTPException(400, "No tickers provided")

        prices_df = None
        data_source = "synthetic"

        if not req.demo_mode and _HAS_YF:
            try:
                end = dt.datetime.today()
                start = end - dt.timedelta(days=req.lookback_days + 30)
                raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    prices_df = raw["Close"].dropna(how="all").tail(req.lookback_days)
                elif "Close" in raw.columns:
                    prices_df = raw[["Close"]].dropna().tail(req.lookback_days)
                    prices_df.columns = tickers[:1]
                if prices_df is not None and (prices_df.empty or len(prices_df) < 20):
                    prices_df = None
                else:
                    data_source = "yfinance"
            except Exception:
                prices_df = None

        if prices_df is None:
            prices_df = _generate_synthetic_prices(tickers, req.lookback_days)
            data_source = "synthetic"

        log_rets = _log_return_frame(prices_df)
        port_rets = log_rets.mean(axis=1)
        port_val = req.portfolio_value

        var_r, cvar_r = _monte_carlo_var_cvar(port_rets, 1, req.simulations, req.confidence)
        var_usd = abs(float(var_r)) * port_val
        cvar_usd = abs(float(cvar_r)) * port_val

        # Per-asset stats
        per_asset = []
        for col in log_rets.columns:
            s = log_rets[col]
            ann_ret = float(s.mean() * 252)
            ann_vol = float(s.std() * math.sqrt(252))
            sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
            cum = (1 + s).cumprod()
            peak = cum.cummax()
            dd = float(((cum - peak) / peak).min())
            last_price = float(prices_df[col].iloc[-1]) if col in prices_df.columns else 0.0
            per_asset.append({
                "ticker": str(col),
                "ann_return_pct": round(ann_ret * 100, 2),
                "ann_vol_pct": round(ann_vol * 100, 2),
                "sharpe": round(sharpe, 3),
                "max_drawdown_pct": round(dd * 100, 2),
                "last_price": round(last_price, 2),
            })

        # Position sizing suggestion (equal weight)
        weight = 1.0 / len(tickers)
        positions = [{"ticker": a["ticker"], "weight_pct": round(weight * 100, 1),
                      "value_usd": round(weight * port_val, 2)} for a in per_asset]

        # Regime
        ann_vol_rolling = log_rets.rolling(21).std().mean(axis=1) * math.sqrt(252)
        last_vol = float(ann_vol_rolling.dropna().iloc[-1]) if not ann_vol_rolling.dropna().empty else 0.0
        regime = "High Volatility" if last_vol > 0.375 else ("Moderate" if last_vol > 0.225 else "Calm")

        return _np_to_py({
            "tickers": tickers,
            "data_source": data_source,
            "portfolio_value": port_val,
            "var_1d_usd": round(var_usd, 2),
            "cvar_1d_usd": round(cvar_usd, 2),
            "regime": regime,
            "current_vol_ann_pct": round(last_vol * 100, 2),
            "per_asset": per_asset,
            "positions": positions,
        })
    except Exception as e:
        raise HTTPException(500, f"Insider analysis error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# LACHESIS GUIDE – AI Narrative (OpenAI if key set, else rule-based)
# ═══════════════════════════════════════════════════════════════════════════════

class LachesisGuideRequest(BaseModel):
    question: str = "What is the current market risk?"
    tickers: List[str] = ["SPY"]
    regime: str = "Unknown"
    var_usd: Optional[float] = None
    cvar_usd: Optional[float] = None
    portfolio_value: Optional[float] = None
    openai_api_key: Optional[str] = None
    language: str = "English"


@app.post("/api/financial/lachesis-guide")
def lachesis_guide(req: LachesisGuideRequest):
    try:
        context_parts = [f"Regime: {req.regime}"]
        if req.tickers:
            context_parts.append(f"Tickers: {', '.join(req.tickers)}")
        if req.var_usd is not None:
            context_parts.append(f"1-day VaR (95%): ${req.var_usd:,.0f}")
        if req.cvar_usd is not None:
            context_parts.append(f"CVaR: ${req.cvar_usd:,.0f}")
        if req.portfolio_value is not None:
            context_parts.append(f"Portfolio value: ${req.portfolio_value:,.0f}")
        context = "; ".join(context_parts)

        # Try OpenAI if key provided
        narrative = None
        if req.openai_api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=req.openai_api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You are Lachesis, a quantum-enhanced financial risk AI. Provide concise, actionable risk guidance. Keep responses to 2-3 paragraphs. Always respond in {req.language}."},
                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {req.question}"},
                    ],
                    max_tokens=400,
                    temperature=0.7,
                )
                narrative = resp.choices[0].message.content
            except Exception:
                narrative = None

        if narrative is None:
            # Rule-based fallback
            q_lower = req.question.lower()
            if "var" in q_lower or "risk" in q_lower:
                narrative = (
                    f"Based on your portfolio context ({context}), the current risk profile suggests "
                    f"{'elevated' if req.regime in ['High Volatility', 'Stress'] else 'moderate'} exposure. "
                    f"{'With VaR of $' + f'{req.var_usd:,.0f}' + ', consider hedging positions.' if req.var_usd else 'Run a financial analysis to compute VaR estimates.'} "
                    f"Lachesis recommends diversification and maintaining a cash buffer of 5-10% during {req.regime} regimes."
                )
            elif "buy" in q_lower or "sell" in q_lower or "trade" in q_lower:
                narrative = (
                    f"Lachesis cannot provide direct buy/sell recommendations. However, given the {req.regime} regime "
                    f"for {', '.join(req.tickers[:3])}, quantitative signals suggest reviewing position sizing. "
                    f"Use the VQE Risk Gate to evaluate individual trade orders against your risk limits."
                )
            else:
                narrative = (
                    f"Lachesis analysis for {', '.join(req.tickers[:3])}: Market regime is currently **{req.regime}**. "
                    f"Quantum-enhanced Bayesian analysis suggests monitoring volatility closely. "
                    f"For deeper insight, run the QTBN forecast and compare against your VaR thresholds."
                )

        return {"narrative": narrative, "context": context, "question": req.question}
    except Exception as e:
        raise HTTPException(500, f"Lachesis guide error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT STUDIO – LLM scenario generation
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATES = {
    "risk_scenario": "Generate a detailed financial risk scenario for {tickers} in a {regime} market. Include probability estimates and suggested hedges.",
    "qtbn_forecast": "Interpret the following QTBN forecast probabilities: P(gain)={p_gain:.1%}, P(flat)={p_flat:.1%}, P(loss)={p_loss:.1%}, P(severe_loss)={p_severe:.1%}. What does this mean for a {portfolio_value} portfolio?",
    "circuit_analysis": "Analyze this quantum circuit configuration: {num_qubits} qubits, gates={gates}, noise={noise}. Explain the quantum state and potential noise effects.",
    "stress_test": "Design a stress test scenario for a portfolio with VaR=${var_usd:,.0f} in a {regime} regime. What macro events should be simulated?",
    "trade_review": "Review this trade: {ticker}, notional=${notional:,.0f}, policy={policy}, status={status}. Provide risk commentary.",
}

class PromptStudioRequest(BaseModel):
    template: str = "risk_scenario"
    variables: Dict[str, Any] = {}
    custom_prompt: Optional[str] = None
    openai_api_key: Optional[str] = None
    max_tokens: int = Field(500, ge=100, le=2000)
    language: str = "English"


@app.get("/api/prompt-studio/templates")
def get_prompt_templates():
    return {"templates": [{"key": k, "template": v} for k, v in PROMPT_TEMPLATES.items()]}


@app.post("/api/prompt-studio/generate")
def prompt_studio_generate(req: PromptStudioRequest):
    try:
        # Build prompt
        if req.custom_prompt:
            prompt_text = req.custom_prompt
        elif req.template in PROMPT_TEMPLATES:
            try:
                prompt_text = PROMPT_TEMPLATES[req.template].format(**req.variables)
            except KeyError as e:
                prompt_text = PROMPT_TEMPLATES[req.template] + f"\n\nVariables: {req.variables}"
        else:
            prompt_text = str(req.variables)

        result_text = None

        # Try OpenAI
        if req.openai_api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=req.openai_api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You are Lachesis, a quantum-enhanced financial analytics AI. Provide precise, data-driven analysis. Always respond in {req.language}."},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_tokens=req.max_tokens,
                    temperature=0.7,
                )
                result_text = resp.choices[0].message.content
            except Exception as oe:
                result_text = f"[OpenAI unavailable: {oe}]\n\nRule-based response: Prompt received for '{req.template}' scenario generation."

        if result_text is None:
            # Rule-based fallback
            result_text = (
                f"**Lachesis Prompt Studio** — Template: `{req.template}`\n\n"
                f"Prompt:\n{prompt_text}\n\n"
                f"_Connect an OpenAI API key in the Admin panel for AI-generated analysis. "
                f"Without a key, Lachesis provides structural templates and quantitative metrics only._"
            )

        return {
            "prompt": prompt_text,
            "result": result_text,
            "template": req.template,
            "tokens_requested": req.max_tokens,
        }
    except Exception as e:
        raise HTTPException(500, f"Prompt studio error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN – API key validation (keys stored client-side, server just validates)
# ═══════════════════════════════════════════════════════════════════════════════

class AdminKeyValidateRequest(BaseModel):
    service: str  # openai | fred | perplexity
    api_key: str


@app.post("/api/admin/validate-key")
def admin_validate_key(req: AdminKeyValidateRequest):
    """Minimal validation — checks key format only (no external calls)."""
    key = req.api_key.strip()
    valid = False
    hint = ""
    if req.service == "openai":
        valid = key.startswith("sk-") and len(key) > 20
        hint = "Should start with 'sk-'"
    elif req.service == "fred":
        valid = len(key) == 32 and key.isalnum()
        hint = "Should be 32 alphanumeric characters"
    elif req.service == "perplexity":
        valid = key.startswith("pplx-") and len(key) > 20
        hint = "Should start with 'pplx-'"
    else:
        valid = len(key) > 10
        hint = "Unknown service"
    return {"service": req.service, "valid": valid, "hint": hint}


# ═══════════════════════════════════════════════════════════════════════════════
# SEC EDGAR — CIK LOOKUP + INSIDER FILINGS (Forms 3, 4, 5)
# Uses the `requests` library (same as qtbn_simulator_clean.py) for reliability.
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_SUBMIT_URL  = "https://data.sec.gov/submissions/CIK{cik}.json"

# 24-hour in-process cache for the full tickers map
_CIK_MAP_CACHE: Optional[Dict] = None


def _edgar_headers(user_agent: str) -> Dict[str, str]:
    ua = (user_agent or "").strip() or "LachesisApp contact@lachesis.local"
    return {"User-Agent": ua, "Accept": "application/json"}


def _fetch_cik_map(user_agent: str) -> Dict:
    global _CIK_MAP_CACHE
    if _CIK_MAP_CACHE is not None:
        return _CIK_MAP_CACHE
    r = _requests.get(_EDGAR_TICKERS_URL, headers=_edgar_headers(user_agent), timeout=30)
    r.raise_for_status()
    _CIK_MAP_CACHE = r.json()
    return _CIK_MAP_CACHE


def _normalize_cik(raw: str) -> Optional[str]:
    import re
    digits = re.sub(r"\D", "", raw)
    return digits.zfill(10) if digits else None


def _lookup_cik_for_ticker(ticker: str, user_agent: str) -> Optional[str]:
    data = _fetch_cik_map(user_agent)
    ticker_u = ticker.strip().upper()
    for entry in (data or {}).values():
        if str(entry.get("ticker", "")).upper() == ticker_u:
            cik_num = entry.get("cik_str")
            if cik_num is None:
                return None
            return str(int(cik_num)).zfill(10)
    return None


class EdgarLoadRequest(BaseModel):
    """Single-call endpoint: provide ticker OR cik (or both). Matches Streamlit UX."""
    ticker: str = ""
    cik: str = ""
    forms: List[str] = ["4"]
    user_agent: str = "LachesisApp contact@lachesis.local"
    max_results: int = Field(50, ge=1, le=200)


class EdgarFiling(BaseModel):
    accession_number: str
    filing_date: str
    form: str
    primary_document: str
    description: str
    filing_url: str


class EdgarLoadResponse(BaseModel):
    ticker: str
    cik: str
    company_name: str
    filings: List[EdgarFiling]
    total_found: int


@app.post("/api/insider/load-filings", response_model=EdgarLoadResponse)
def edgar_load_filings(req: EdgarLoadRequest):
    """
    One-shot endpoint: resolves ticker→CIK (if needed) then fetches filings.
    Accepts ticker OR manual CIK, mirrors qtbn_simulator_clean.py behaviour.
    """
    if not _HAS_REQUESTS:
        raise HTTPException(503, "requests library not available; pip install requests")

    # ── Resolve CIK ──────────────────────────────────────────────────────────
    cik_norm: Optional[str] = None

    if req.cik.strip():
        cik_norm = _normalize_cik(req.cik.strip())

    if not cik_norm and req.ticker.strip():
        try:
            cik_norm = _lookup_cik_for_ticker(req.ticker.strip(), req.user_agent)
        except Exception as e:
            raise HTTPException(502, f"SEC ticker lookup failed: {e}")

    if not cik_norm:
        raise HTTPException(400, "Provide a valid ticker or CIK to load filings.")

    # ── Fetch submissions ─────────────────────────────────────────────────────
    url = _EDGAR_SUBMIT_URL.format(cik=cik_norm)
    try:
        r = _requests.get(url, headers=_edgar_headers(req.user_agent), timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(502, f"EDGAR submissions fetch failed: {e}")

    company_name = data.get("name") or data.get("entityName") or "Unknown"
    recent = (data.get("filings") or {}).get("recent") or {}

    forms_list      = recent.get("form", []) or []
    dates_list      = recent.get("filingDate", []) or []
    accessions_list = recent.get("accessionNumber", []) or []
    primary_doc_list= recent.get("primaryDocument", []) or []
    desc_list       = recent.get("primaryDocDescription", []) or []

    target_forms = {f.strip().upper() for f in req.forms}
    cik_int = int(cik_norm)
    results: List[Dict] = []

    for i, form in enumerate(forms_list):
        if str(form).strip().upper() not in target_forms:
            continue
        acc = accessions_list[i] if i < len(accessions_list) else ""
        acc_clean = acc.replace("-", "")
        primary_doc = primary_doc_list[i] if i < len(primary_doc_list) else ""
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_doc}"
            if primary_doc else
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_norm}&type={form}"
        )
        results.append({
            "accession_number": acc,
            "filing_date": dates_list[i] if i < len(dates_list) else "",
            "form": form,
            "primary_document": primary_doc,
            "description": desc_list[i] if i < len(desc_list) else "",
            "filing_url": filing_url,
        })
        if len(results) >= req.max_results:
            break

    return {
        "ticker": req.ticker.strip().upper(),
        "cik": cik_norm,
        "company_name": company_name,
        "filings": results,
        "total_found": len(results),
    }


# Keep the older two-step endpoints as aliases (backwards compat)
@app.post("/api/insider/lookup-cik")
def edgar_lookup_cik_legacy(ticker: str, user_agent: str = "LachesisApp contact@lachesis.local"):
    if not _HAS_REQUESTS:
        raise HTTPException(503, "requests library not available")
    cik = _lookup_cik_for_ticker(ticker, user_agent)
    if not cik:
        raise HTTPException(404, f"Ticker '{ticker}' not found in SEC EDGAR")
    return {"ticker": ticker.upper(), "cik": cik}


@app.post("/api/insider/filings")
def edgar_filings_legacy(cik: str, forms: str = "3,4,5",
                         user_agent: str = "LachesisApp contact@lachesis.local",
                         max_results: int = 50):
    req = EdgarLoadRequest(cik=cik, forms=forms.split(","), user_agent=user_agent,
                           max_results=max_results)
    return edgar_load_filings(req)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
