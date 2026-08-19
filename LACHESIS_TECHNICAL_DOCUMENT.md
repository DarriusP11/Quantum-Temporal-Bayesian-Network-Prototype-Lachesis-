# Lachesis: Quantum-Financial Analytics Platform
## Technical Reference Document
**Version:** 1.0 — April 2026  
**Audience:** PhD-level quantum computing researcher (IBM Qiskit, Google Cirq, PennyLane)  
**Prepared by:** Lachesis Development Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Quantum Computing Implementation](#3-quantum-computing-implementation)
   - 3a. [Circuit Simulation Engine](#3a-quantum-circuit-simulation-engine)
   - 3b. [QAOA Portfolio Optimization](#3b-qaoa-portfolio-optimization)
   - 3c. [VQE — Variational Quantum Eigensolver](#3c-vqe--variational-quantum-eigensolver)
   - 3d. [Advanced Quantum Diagnostics](#3d-advanced-quantum-diagnostics)
   - 3e. [Quantum Amplitude Estimation for VaR/CVaR](#3e-quantum-amplitude-estimation-for-varcvar)
   - 3f. [Foresight — Noise Sensitivity Sweep](#3f-foresight--noise-sensitivity-sweep)
   - 3g. [QTBN — Quantum Temporal Bayesian Network](#3g-qtbn--quantum-temporal-bayesian-network)
4. [Classical Financial Analytics](#4-classical-financial-analytics)
5. [Tab-by-Tab Reference](#5-tab-by-tab-reference)
6. [Complete API Reference](#6-complete-api-reference)
7. [Global State Architecture](#7-global-state-architecture)
8. [Mathematical Formulations](#8-mathematical-formulations)
9. [Framework Comparison: Qiskit vs. Cirq vs. PennyLane](#9-framework-comparison-qiskit-vs-cirq-vs-pennylane)
10. [Research Extensions & Open Questions](#10-research-extensions--open-questions)

---

## 1. Executive Summary

**Lachesis** is a quantum-enhanced financial risk analytics and portfolio optimization platform. It integrates IBM Qiskit-based quantum algorithms (QAOA, VQE, QAE, state tomography, randomized benchmarking) with classical financial analytics (Monte Carlo VaR/CVaR, regime detection, sentiment analysis, macro stress testing) into a unified interactive application.

### What Problem It Solves

Modern portfolio management faces two compounding challenges:

1. **Combinatorial explosion** — Portfolio selection from N assets requires evaluating 2^N candidate portfolios. For N ≥ 30, classical exhaustive search is intractable.
2. **Tail-risk estimation under non-Gaussian returns** — Standard VaR models assume normality and underestimate extreme losses. Quantum Amplitude Estimation can compute tail probabilities with quadratic speedup over Monte Carlo.

Lachesis addresses both by:
- Using **QAOA** to search the portfolio selection space as a QUBO problem on a quantum circuit
- Offering **QAE** as an optional path for VaR/CVaR estimation
- Providing a **real-time regime-aware framework** (QTBN) to adjust risk expectations dynamically

### Key Technical Differentiators

| Feature | Classical Baseline | Lachesis Quantum Path |
|---------|-------------------|-----------------------|
| Portfolio selection | Markowitz mean-variance, brute-force | QAOA (cost+mixer Hamiltonian, COBYLA optimizer) |
| VaR/CVaR estimation | Monte Carlo (N=50k draws) | Iterative Amplitude Estimation (quadratic speedup) |
| Regime forecasting | Hidden Markov Model | QTBN (quantum-inspired amplitude posterior, QAOA-style MAP) |
| Gate diagnostics | N/A (financial tools) | State tomography, randomized benchmarking, process fidelity |
| Risk gating | Static rule-based limits | VQE energy-derived risk multiplier |

### Deployment URLs
- **Frontend (Vercel):** `lachesisprototype3.vercel.app`
- **Backend (Railway):** `web-production-3d82b.up.railway.app`
- **Supabase project:** `mvprtzaatfbvdxwutrbo`

---

## 2. System Architecture

### 2.1 Technology Stack

#### Backend
| Component | Version | Role |
|-----------|---------|------|
| FastAPI | 0.111.0+ | REST API framework, 45+ endpoints |
| Uvicorn | 0.29.0+ | ASGI server (`0.0.0.0:8000`) |
| Pydantic | 2.0.0+ | Request/response validation |
| Qiskit | 1.0.0+ | Quantum circuit construction & transpilation |
| Qiskit-Aer | 0.14.0+ | AerSimulator: statevector + shot-based measurement |
| Qiskit-Finance | optional | NormalDistribution, IterativeAmplitudeEstimation |
| Qiskit-Optimization | optional | QuadraticProgram, QUBO conversion, MinimumEigenOptimizer |
| yfinance | 0.2.40+ | Live OHLCV price feeds |
| pandas | 2.0.0+ | Time series, log-return computation |
| numpy | 1.26.0+ | Numerical simulation |
| scipy | 1.12.0+ | Statistical distributions, curve fitting |
| fredapi | 0.5.0 | Federal Reserve macroeconomic data |
| NLTK/VADER | — | Sentiment scoring on news headlines |

#### Frontend
| Component | Version | Role |
|-----------|---------|------|
| React | 18.3.1 | SPA UI framework |
| Vite | 8.0.0 | Build tool + dev server |
| TypeScript | 5.8.3 | Type safety across all components |
| Recharts | 2.15.4 | Financial time-series charts |
| Plotly.js | 3.4.0 | 3D surfaces and advanced plots |
| Radix UI + Shadcn | — | Accessible component primitives |
| TailwindCSS | 3.4.17 | Utility-first styling |
| Supabase JS | 2.57.4 | Auth client (JWT) |
| React Query | 5.83.0 | Server state management |

### 2.2 Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    BROWSER (React SPA)                       │
│                                                              │
│  AppContext (global state)                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  num_qubits · shots · seed · gates[3] · noise[4]    │   │
│  │  tickers · portfolio_value · confidence · mc_sims   │   │
│  │  language · regime · sentiment_multiplier           │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │  useAppContext()                   │
│  Dashboard Components ──┘  (all 17 tabs read shared state)  │
│  └─→ lib/api.ts (typed fetch wrapper)                       │
│       VITE_API_URL = https://web-production-3d82b.up.railway.app │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP/JSON  (CORS: allow *)
┌────────────────────────▼─────────────────────────────────────┐
│                FastAPI Backend (api_server.py)               │
│                                                              │
│  Route groups:                                               │
│  /api/quantum/*         → Qiskit simulation                  │
│  /api/financial/*       → yfinance + Monte Carlo             │
│  /api/qtbn/forecast     → QTBN Markov engine                 │
│  /api/qaoa/*            → QAOA portfolio optimization        │
│  /api/vqe/*             → VQE solver + risk gate             │
│  /api/foresight/*       → Noise sensitivity sweep            │
│  /api/sentiment/*       → VADER / Perplexity NLP             │
│  /api/insider/*         → SEC EDGAR                          │
│  /api/fred/*            → FRED macroeconomic data            │
│  /api/prompt-studio/*   → LLM template engine               │
│  /api/search            → SerpAPI web search                 │
│  /api/auth/*            → Supabase admin signup              │
│  /api/health            → Capability flags                   │
│                                                              │
│  Imported modules:                                           │
│  ├── qaoa_scenario1.py   (QAOA portfolio solver)             │
│  ├── vqe_tab.py          (VQE + risk gate logic)             │
│  ├── qtbn_core.py        (QTBN Markov engine)                │
│  └── foresight.py        (noise sweep specs)                 │
└──────────────────────────────────────────────────────────────┘
          │              │             │            │
     Qiskit-Aer      yfinance      FRED API     External APIs
   (local simulation) (Yahoo)   (stlouisfed.org) (OpenAI,
                                                  SerpAPI,
                                                  Perplexity,
                                                  SEC EDGAR,
                                                  ElevenLabs)
```

### 2.3 Authentication Flow

1. User registers/logs in via **Supabase Auth** (email+password, JWT)
2. JWT stored in `localStorage` via `supabase-js` client
3. Backend admin operations (`/api/auth/signup`) use the **service role key** (server-side only, never exposed to browser)
4. Owner-only Admin tab: `OWNER_EMAIL = "darriusperson@gmail.com"` check in `Index.tsx`

### 2.4 Key File Paths

```
/Applications/Quantum Temporal Bayesian Network Prototype/
├── api_server.py                          # FastAPI entry point (2254 lines)
├── qaoa_scenario1.py                      # QAOA implementation (1482 lines)
├── vqe_tab.py                             # VQE + risk gate (2733 lines)
├── qtbn_core.py                           # QTBN Markov engine (77 lines)
├── foresight.py                           # Noise sweep specs (35 lines)
├── sentiment_plugin.py                    # Sentiment Streamlit UI (37 lines)
├── qtbn_simulator_clean.py                # Legacy Streamlit prototype (6706 lines)
├── requirements.txt                       # Full Python dependencies
├── requirements_api.txt                   # Minimal API dependencies
└── tools/Frontend_Loveable/
    └── quantum-foresight-ai-main/
        ├── src/
        │   ├── pages/Index.tsx            # Tab layout, TABS array
        │   ├── components/
        │   │   ├── LachesisAssistant.tsx  # AI copilot
        │   │   ├── FinancialDashboard.tsx
        │   │   ├── QAOADashboard.tsx
        │   │   ├── VQEDashboard.tsx
        │   │   ├── AdvancedQuantumDashboard.tsx
        │   │   ├── ForesightDashboard.tsx
        │   │   └── ... (13 more dashboards)
        │   ├── contexts/AppContext.tsx    # Global state
        │   ├── lib/api.ts                 # Typed API client (539 lines)
        │   ├── lib/qtbn-engine.ts         # Client-side quantum inference
        │   └── types/quantum.ts           # TypeScript type definitions
        ├── .env                           # Local env vars (gitignored)
        └── .env.example                   # Template for deployment
```

---

## 3. Quantum Computing Implementation

> **Note for PhD scholar:** All quantum simulation runs on **Qiskit-Aer** (a classical simulator). No real quantum hardware is currently integrated, though the code is structured to support `qiskit-ibm-runtime` for hardware execution. The QAOA and VQE implementations target NISQ-era devices and should be directly portable to IBM Quantum via `IBMRuntimeService`.

### 3a. Quantum Circuit Simulation Engine

**Endpoint:** `POST /api/quantum/simulate`  
**Source:** `api_server.py` — functions `_build_circuit()`, `_build_noise_model()`, `_apply_gate()`

#### Circuit Construction

The circuit is built from three sequential gate *steps*, each applying one gate per qubit and optional CNOT pairs:

```python
# Pseudocode: _build_circuit(req) in api_server.py
qc = QuantumCircuit(req.num_qubits, req.num_qubits)

for step in [req.step0, req.step1, req.step2]:
    for qubit_idx, gate_cfg in enumerate(step.gates):
        _apply_gate(qc, gate_cfg.name, qubit_idx, gate_cfg.angle)
    if step.cnot_01 and req.num_qubits >= 2:
        qc.cx(0, 1)
    if step.cnot_12 and req.num_qubits >= 3:
        qc.cx(1, 2)
    if step.cnot_23 and req.num_qubits >= 4:
        qc.cx(2, 3)
```

**Supported gates:** `H, X, Y, Z, S, T, RX(θ), RY(θ), RZ(θ), None`

#### Noise Model Construction

```python
# _build_noise_model(req) in api_server.py
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error,
    amplitude_damping_error, phase_damping_error
)

noise_model = NoiseModel()

for qubit_idx, p in enumerate([req.noise.pdep0, req.noise.pdep1, req.noise.pdep2]):
    if req.noise.enable_depolarizing and p > 0:
        err = depolarizing_error(p, num_qubits=1)
        noise_model.add_quantum_error(err, ['h','x','y','z','rx','ry','rz','s','t'], [qubit_idx])

for qubit_idx, p in enumerate([req.noise.pamp0, req.noise.pamp1, req.noise.pamp2]):
    if req.noise.enable_amplitude_damping and p > 0:
        err = amplitude_damping_error(p)
        noise_model.add_quantum_error(err, ['h','x','y','z','rx','ry','rz','s','t'], [qubit_idx])

# Phase damping and CNOT noise follow the same pattern
```

#### Simulation Execution

Two separate Aer backends are used:

```python
# 1. IDEAL — statevector (no noise)
sim_sv = AerSimulator(method="statevector")
qc_sv = qc.copy()
qc_sv.save_statevector()
result_ideal = sim_sv.run(transpile(qc_sv, sim_sv), shots=1).result()
sv = result_ideal.get_statevector()

# 2. NOISY — shot-based measurement
sim_noisy = AerSimulator(method="automatic")
qc_meas = qc.copy()
qc_meas.measure_all()
result_noisy = sim_noisy.run(
    transpile(qc_meas, sim_noisy, basis_gates=noise_model.basis_gates),
    shots=req.shots,
    noise_model=noise_model,
    seed_simulator=req.seed if req.use_seed else None
).result()
counts = result_noisy.get_counts()
```

#### Fidelity Metric

Classical fidelity between ideal and noisy probability distributions (Bhattacharyya-style):

```
F = ( Σ_k √[ p_ideal(k) · p_noisy(k) ] )²
```

This is the standard overlap fidelity between two diagonal density matrices (mixed states in the computational basis).

#### Response Schema

```json
{
  "statevector_real": [float, ...],   // Re(α_k) for each basis state
  "statevector_imag": [float, ...],   // Im(α_k)
  "probabilities":    [float, ...],   // |α_k|² (ideal)
  "counts":           {"00": int, "01": int, ...},  // noisy measurement
  "fidelity":         float,          // F ∈ [0, 1]
  "circuit_lines":    [str, ...]      // ASCII circuit diagram
}
```

---

### 3b. QAOA Portfolio Optimization

**Endpoints:** `POST /api/qaoa/optimize`, `POST /api/qaoa/sweep`  
**Source:** `qaoa_scenario1.py` (imported by `api_server.py`)

#### Problem Formulation

Portfolio selection is framed as a **binary quadratic program**:

```
maximize  J(x) = λ · (μᵀx) − (1−λ) · (xᵀΣx)

subject to  x_i ∈ {0, 1}  ∀ i ∈ {1,...,N}
```

Where:
- `μ ∈ ℝᴺ` — vector of expected annual returns per asset
- `Σ ∈ ℝᴺˣᴺ` — covariance matrix of returns
- `λ ∈ [0.1, 2.0]` — risk-return trade-off (higher λ = risk-tolerant)
- `x_i = 1` means asset `i` is included in portfolio

#### QUBO Conversion (Qiskit)

```python
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

qp = QuadraticProgram()
for i, asset in enumerate(assets):
    qp.binary_var(name=asset)

# Linear terms: λ · μ_i
linear = {assets[i]: lam * mu[i] for i in range(N)}

# Quadratic terms: -(1-λ) · Σ_ij
quadratic = {}
for i in range(N):
    for j in range(i+1, N):
        quadratic[(assets[i], assets[j])] = -2 * (1-lam) * cov[i,j]
    quadratic[(assets[i], assets[i])] = -(1-lam) * cov[i,i]

qp.maximize(linear=linear, quadratic=quadratic)

# Convert to QUBO (Quadratic Unconstrained Binary Optimization)
converter = QuadraticProgramToQubo()
qubo = converter.convert(qp)
```

#### QAOA Circuit Structure

The QAOA ansatz applies `p` alternating layers of cost and mixer operators:

```
|s⟩ = H^⊗N |0⟩^⊗N             (uniform superposition)

For k = 1,...,p:
  e^{-iγ_k H_cost}              (cost unitary — encodes portfolio objective)
  e^{-iβ_k H_mixer}             (mixer unitary — transverse-field X mixer)

Measure in computational basis → decode bitstring → portfolio selection
```

**Cost Hamiltonian** (from QUBO):
```
H_cost = Σ_i c_i · Z_i + Σ_{i<j} Q_{ij} · Z_i ⊗ Z_j
```

**Mixer Hamiltonian** (standard transverse-field):
```
H_mixer = Σ_i X_i
```

#### Qiskit Implementation

```python
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

sampler = Sampler()   # or AerSampler() for Aer backend
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=depth)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qubo)

# result.x → bitstring (selected portfolio)
# result.fval → objective energy (minimized = −J)
```

#### Available Portfolios

**Portfolio 1: Toy 3-asset tech**
```python
assets = ["AAPL", "MSFT", "GOOG"]
mu    = [0.10, 0.12, 0.08]              # annual expected returns
cov   = [[0.04, 0.028, 0.022],
         [0.028, 0.05, 0.024],
         [0.022, 0.024, 0.045]]
```

**Portfolio 2: Lachesis benchmark (equities + bond + gold)**
```python
assets = ["AAPL", "MSFT", "QQQ", "TLT", "GLD"]
mu    = [0.11, 0.12, 0.09, 0.04, 0.06]
# 5×5 covariance matrix with lower equity/bond/gold correlations
```

#### Regime-Aware Adjustments

```python
def apply_regime_to_cfg(cfg, regime):
    if regime == "bull":
        cfg["mu"] = [m + 0.03 for m in cfg["mu"]]     # +3% equity returns
        cfg["cov"] *= 0.9                               # tighter dispersion
    elif regime == "bear":
        cfg["mu"] = [m - 0.04 for m in cfg["mu"]]     # -4% equities
        cfg["cov"] *= 1.3                               # wider dispersion
    elif regime == "shock":
        cfg["cov"] *= 1.8                               # volatility spike only
```

#### Lambda Sweep (Efficient Frontier)

`POST /api/qaoa/sweep` iterates λ across `[lam_min, lam_max]` with `n_points` samples, returning a 2D Pareto frontier of `(expected_return, risk)` pairs. This traces the quantum-optimal efficient frontier.

#### Fallbacks

| Condition | Fallback |
|-----------|---------|
| Qiskit-Optimization unavailable | NumPyMinimumEigensolver (exact diagonalization) |
| NumPy fallback unavailable | Classical brute-force over all 2^N bitstrings |

---

### 3c. VQE — Variational Quantum Eigensolver

**Endpoints:** `POST /api/vqe/solve`, `POST /api/vqe/risk-gate`  
**Source:** `vqe_tab.py`

#### Supported Problem Types

| Problem | Hamiltonian | Financial Interpretation |
|---------|------------|--------------------------|
| Toy Hamiltonian | User-defined Pauli terms | Custom risk factor interaction |
| MaxCut | Graph Laplacian → Ising | Asset correlation network partitioning |
| Ising (h, J) | Σ h_i Z_i + Σ J_ij Z_i Z_j | Spin glass = correlated risk factors |
| Custom Pauli | SparsePauliOp from text | Arbitrary quantum financial models |

#### VQE Algorithm

The variational principle: find circuit parameters `θ` minimizing the Rayleigh quotient:

```
E(θ) = ⟨ψ(θ)| H |ψ(θ)⟩  →  minimize over θ
```

**Ansätze:**

```python
from qiskit.circuit.library import RealAmplitudes, EfficientSU2

# RealAmplitudes: alternating RY + CNOT layers
ansatz = RealAmplitudes(num_qubits=n, reps=reps, entanglement='linear')

# EfficientSU2: alternating RY+RZ (SU(2) rotations) + CNOT
ansatz = EfficientSU2(num_qubits=n, reps=reps)
```

**Expectation value via Estimator primitive:**

```python
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.4)])
estimator = Estimator()

# VQE minimization loop
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=COBYLA(maxiter=maxiter))
result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
# result.eigenvalue → ground state energy E_min
# result.optimal_parameters → θ*
```

#### Energy → Risk Multiplier Mapping

The VQE ground state energy is converted to a dimensionless risk budget multiplier:

```python
def energy_to_risk_multiplier(energy, energy_min_expected, energy_max_expected):
    # Normalize energy to [0, 1]
    norm = (energy - energy_min_expected) / (energy_max_expected - energy_min_expected)
    norm = max(0.0, min(1.0, norm))
    # Map: low energy → high risk budget (allow more); high energy → tight budget
    multiplier = 0.5 + (1.0 - norm)   # range [0.5, 1.5]
    return multiplier
```

#### VQE Risk Gate

The risk gate enforces policy-based trade approval before execution:

```python
POLICY_LIMITS = {
    "Conservative": {
        "max_notional_usd": 50_000,
        "max_var_usd":       2_000,
        "max_cvar_usd":      3_500,
        "max_leverage":      1.5
    },
    "Moderate": {
        "max_notional_usd": 250_000,
        "max_var_usd":      10_000,
        "max_cvar_usd":     18_000,
        "max_leverage":     3.0
    },
    "Aggressive": {
        "max_notional_usd": 1_000_000,
        "max_var_usd":      50_000,
        "max_cvar_usd":     90_000,
        "max_leverage":     6.0
    }
}
```

**Decision logic:**
```
if est_var_usd  > limits.max_var_usd:   → BLOCKED
if est_cvar_usd > limits.max_cvar_usd:  → BLOCKED
if leverage     > limits.max_leverage:  → BLOCKED
if notional     > limits.max_notional:  → PARTIAL (clamped)
else:                                   → APPROVED
```

All risk gate checks are logged to an in-process audit trail with timestamps.

---

### 3d. Advanced Quantum Diagnostics

**Endpoint:** `POST /api/quantum/advanced/{tomography|benchmarking|calibrate|fidelity}`  
**Source:** `api_server.py`

#### State Tomography

Reconstructs the single-qubit Bloch sphere vector `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)` and purity.

**Measurement protocol:**
- **Z-basis:** Direct measurement → `⟨Z⟩ = (N_0 - N_1) / N_total`
- **X-basis:** Apply H then measure → `⟨X⟩ = (N_0 - N_1) / N_total`
- **Y-basis:** Apply S†H then measure → `⟨Y⟩ = (N_0 - N_1) / N_total`

**Purity:**
```
γ = ⟨X⟩² + ⟨Y⟩² + ⟨Z⟩²
ρ = (I + ⟨X⟩X + ⟨Y⟩Y + ⟨Z⟩Z) / 2
purity = Tr(ρ²) = (1 + γ) / 2  →  γ = 1 for pure state, γ = 0 for maximally mixed
```

#### Randomized Benchmarking (RB)

Implements single-qubit Clifford RB to extract error per gate (EPG).

**Protocol:**
1. For each sequence length `m ∈ {2, 4, 8, 16, 32, 48, 64}`:
   - Generate `nseeds` random Clifford sequences of length `m`
   - Append recovery gate (inverse of sequence)
   - Measure survival probability `P(|0⟩)` averaged over seeds
2. Fit survival curve:
   ```
   P(m) = A · p^m + B
   ```
   where `p` is the depolarizing parameter, fitted via `scipy.optimize.curve_fit`.

**Error Per Gate (EPG):**
```
EPG = (1 - p) · (1 - 1/d)   where d = 2 (single qubit Hilbert space dimension)
```

The EPG quantifies the average gate error independent of state preparation and measurement errors (SPAM-free).

#### Bayesian Noise Calibration

Estimates noise parameters using a Bayesian Beta-Binomial model.

**Prior:** Beta(1, 1) (uniform, non-informative)

**Update:** After observing `k` "error" events in `N` shots:
```
Posterior: Beta(α + k, β + N - k)
Mean estimate: (α + k) / (α + β + N)
95% CI: Beta distribution PPF at [0.025, 0.975]
```

Three noise types calibrated simultaneously:
- **Depolarizing** (T0 gate error): measured via H gate failure rate
- **Amplitude damping** (T1 relaxation): measured via excited state population decay
- **Phase damping** (T2 dephasing): measured via coherence decay

#### Gate Process Fidelity

Average gate fidelity measured across three input basis states:

```
F_avg = (1/3) · [F(|0⟩) + F(|+⟩) + F(|+i⟩)]
```

For each basis state `|ψ_in⟩`:
1. Prepare `|ψ_in⟩` (e.g., `|+⟩ = H|0⟩`, `|+i⟩ = S·H|0⟩`)
2. Apply target gate `U`
3. Measure in appropriate basis
4. Compare to ideal: `F(|ψ_in⟩) = |⟨ψ_ideal|ψ_out⟩|²`

---

### 3e. Quantum Amplitude Estimation for VaR/CVaR

**Source:** `api_server.py` — function `_qae_var_cvar()` (optional path in `/api/financial/analyze`)

**Required packages:** `qiskit-finance` (optional; classical fallback if absent)

#### Circuit Construction

```python
from qiskit_finance.circuit.library import NormalDistribution

num_qubits = 4   # 2^4 = 16 discretization points
bounds     = (mu_h - 3*sigma_h, mu_h + 3*sigma_h)

# Quantum state encoding the return distribution
dist_circuit = NormalDistribution(
    num_qubits=num_qubits,
    mu=mu_h,
    sigma=sigma_h**2,
    bounds=bounds
)
```

The `NormalDistribution` circuit prepares:
```
|ψ⟩ = Σ_k √[p(x_k)] |k⟩
```
where `x_k` are discretized return values and `p(x_k)` is the normal PDF evaluated at each point.

#### Amplitude Estimation

```python
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem

problem = EstimationProblem(
    state_preparation=dist_circuit,
    objective_qubits=[num_qubits - 1],  # ancilla qubit for threshold
    grover_operator=...
)

iae = IterativeAmplitudeEstimation(
    epsilon_target=0.02,   # 2% precision
    alpha=0.05             # 95% confidence interval
)
result = iae.estimate(problem)
tail_probability = result.estimation   # P(R ≤ VaR threshold)
```

#### VaR/CVaR from Tail Probability

Once the tail probability `p_tail = P(R ≤ c)` is estimated:
```
VaR_α = Φ⁻¹(1-α) · σ_h + μ_h        (normal quantile formula)
CVaR_α = μ_h - σ_h · φ(Φ⁻¹(α)) / α  (normal CVaR formula)
```
where `Φ` is the standard normal CDF and `φ` is the PDF.

**Theoretical speedup:** Classical Monte Carlo achieves `O(1/ε²)` convergence; QAE achieves `O(1/ε)` — a quadratic speedup in estimation error `ε`.

---

### 3f. Foresight — Noise Sensitivity Sweep

**Endpoints:** `POST /api/foresight/sweep`, `GET/POST /api/foresight/scenarios`  
**Source:** `foresight.py`, `api_server.py`

#### Purpose

Characterizes how circuit measurement statistics degrade as a function of noise parameters. Analogous to a *noise fingerprint* of a quantum circuit.

#### Sweep Protocol

```python
# Grid: all combinations of pdep × pamp
for pdep in [0.0, 0.01, 0.03, 0.05]:
    for pamp in [0.0, 0.02]:
        for seed in seeds:
            # Run circuit with these noise params, req.shots shots
            counts_noisy = run_noisy_circuit(pdep, pamp, seed, shots)
        # Average over seeds → aggregate counts
        counts_avg = aggregate(counts_noisy_list)
        # KL divergence from ideal (no-noise) distribution
        kl = kl_divergence(counts_avg, counts_ideal)
        grid[pdep][pamp] = kl
```

#### KL Divergence

```
KL(p_noisy || p_ideal) = Σ_k p_noisy(k) · ln[ p_noisy(k) / p_ideal(k) ]
```

- `KL = 0` → noisy circuit is indistinguishable from ideal
- `KL → ∞` → severe noise corruption

#### Visualization

Results are rendered as a 2D heatmap with axes:
- X: amplitude damping parameter `p_amp`
- Y: depolarizing parameter `p_dep`
- Color: KL divergence (green=low → red=high)

This identifies the noise parameter regime where the circuit remains trustworthy for financial simulation.

---

### 3g. QTBN — Quantum Temporal Bayesian Network

**Endpoint:** `POST /api/qtbn/forecast`  
**Source:** `qtbn_core.py`, `api_server.py`  
**Client-side:** `src/lib/qtbn-engine.ts`

> **Important technical note for PhD scholar:** The *backend* QTBN is a classical Markov chain model — it does not use Qiskit. The "quantum" designation reflects the quantum-inspired design of the client-side TypeScript engine (`qtbn-engine.ts`), which uses amplitude-based posterior sampling and a QAOA-inspired MAP estimation step. The backend is classical by design to ensure reliable financial-grade regime forecasting; the client-side engine demonstrates how quantum probability concepts can be embedded in browser-side inference. True quantum implementation is listed as a research direction (see Section 10).

#### Backend: Classical Markov Regime Chain

**Regime transition matrix** (calibrated empirically):
```
T = [[0.85, 0.13, 0.02],   # calm → {calm, stressed, crisis}
     [0.25, 0.55, 0.20],   # stressed → {calm, stressed, crisis}
     [0.10, 0.25, 0.65]]   # crisis → {calm, stressed, crisis}
```

**Regime-conditioned volatility:**
```
σ(calm)    = 12%  annualized
σ(stressed) = 25% annualized
σ(crisis)  = 40% annualized
```

**4-bucket outcome probabilities** (at `horizon_days`):

```python
sigma_h = vol[regime] * sqrt(horizon_days / 252)
mu_h    = drift_mu    * (horizon_days / 252)

# Normal distribution thresholds: [-2σ, -σ, σ]
P_gain        = 1 - Φ(σ_h - mu_h)
P_flat        = Φ(σ_h - mu_h) - Φ(-σ_h - mu_h)
P_loss        = Φ(-σ_h - mu_h) - Φ(-2·σ_h - mu_h)
P_severe_loss = Φ(-2·σ_h - mu_h)

# Risk-on tilt (biases toward gain/flat, away from loss)
tilt    = risk_on_prior - 0.5
P_gain        += 0.40 * tilt
P_flat        += 0.10 * tilt
P_loss        -= 0.10 * tilt
P_severe_loss -= 0.40 * tilt
```

**Temporal evolution:**
```python
# Initialize prior regime distribution
p = regime_to_vector(prior_regime)   # e.g., "calm" → [0.9, 0.05, 0.05]

timeline = []
for step in range(n_steps):
    p = p @ T                         # Markov update
    drift_t    = sum(p[i] * mu_by_regime[i]      for i in range(3))
    risk_on_t  = sum(p[i] * risk_on_by_regime[i] for i in range(3))
    timeline.append({"calm": p[0], "stressed": p[1], "crisis": p[2],
                     "drift": drift_t, "risk_on": risk_on_t})
```

#### Client-Side Quantum-Inspired Inference (`qtbn-engine.ts`)

The TypeScript engine implements quantum-inspired Bayesian inference entirely in the browser:

**1. Amplitude-based posterior sampling:**
```typescript
function quantumPosteriorSampling(priors, observations): Record<string, number> {
    // Combine: amplitude = sqrt(prior × observation_weight)
    const amplitudes: Record<string, number> = {};
    for (const state of states) {
        amplitudes[state] = Math.sqrt(priors[state] * (observations[state] ?? 1.0));
    }
    // Normalize: probabilities = |amplitude|²
    const totalNorm = Object.values(amplitudes).reduce((s, a) => s + a * a, 0);
    return Object.fromEntries(
        Object.entries(amplitudes).map(([s, a]) => [s, (a * a) / totalNorm])
    );
}
```

**2. QAOA-inspired MAP estimation:**
```typescript
function quantumMAPEstimation(beliefs): string[] {
    // Cost operator: exponential boost for high-probability states
    const boosted = Object.fromEntries(
        Object.entries(beliefs).map(([s, p]) => [s, Math.exp(2.0 * p)])
    );
    // Mixer: uniform rotation (quantum tunneling analogue)
    const mixedAmplitudes = Object.fromEntries(
        Object.entries(boosted).map(([s, v]) => [
            s, Math.cos(Math.PI / 8) * v + Math.sin(Math.PI / 8) * (1 - v)
        ])
    );
    // Return top-3 MAP states
    return Object.entries(mixedAmplitudes)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 3)
        .map(([s]) => s);
}
```

**Hardcoded financial Bayesian graph (4 nodes):**
- `MarketRegime`: {Bull, Bear, Sideways, Volatile} — prior [0.30, 0.20, 0.35, 0.15]
- `VolatilityLevel`: {Low, Medium, High, Extreme} — conditioned on MarketRegime
- `SentimentScore`: {VeryBearish, Bearish, Neutral, Bullish, VeryBullish} — conditioned on MarketRegime
- `RiskLevel`: {VeryLow, Low, Medium, High, VeryHigh} — conditioned on Volatility + Sentiment

---

## 4. Classical Financial Analytics

### 4a. Financial Risk Engine

**Endpoint:** `POST /api/financial/analyze`  
**Source:** `api_server.py` — functions `_monte_carlo_var_cvar()`, `_compute_risk_metrics()`

#### Log-Return Computation

```
r_t = ln(P_t / P_{t-1})
```

Portfolio return (equal-weight):
```
r_portfolio(t) = (1/N) · Σ_i r_i(t)
```

#### Monte Carlo VaR/CVaR

```python
# Horizon-adjusted parameters
mu_h    = mean(r) * horizon_days
sigma_h = std(r)  * sqrt(horizon_days)

# Simulate N paths
sim_returns = np.random.normal(mu_h, sigma_h, simulations)

# Value at Risk (1-day, 95% confidence)
VaR_95  = np.percentile(sim_returns, (1 - confidence) * 100)

# Conditional Value at Risk (Expected Shortfall)
CVaR_95 = sim_returns[sim_returns <= VaR_95].mean()
```

#### Risk-Adjusted Return Metrics

```
Sharpe  = √252 · (μ_excess) / (σ_excess)
         where μ_excess = mean(r - r_f/252),  r_f = 4% annual

Sortino = √252 · (μ_excess) / (σ_downside)
         where σ_downside = √[ mean( min(r_excess, 0)² ) ]

MaxDD   = min_{t} [ (CumProd_t - CumMax_t) / CumMax_t ]

AnnVol  = std(r_21day_rolling) · √252
```

#### Volatility Regime Classification

```python
if   ann_vol <= 0.225:  regime = "Low Volatility"    # Calm
elif ann_vol <= 0.375:  regime = "Medium Volatility"  # Stressed
else:                   regime = "High Volatility"    # Crisis
```

#### Sentiment Stress

```python
# Sentiment score s ∈ [-1, +1]
multiplier = max(0.5, min(1.5, 1.0 - 0.5 * avg_sentiment_score))

# s = -1 (very bearish) → multiplier = 1.5 (escalate risk by 50%)
# s =  0 (neutral)      → multiplier = 1.0 (no change)
# s = +1 (very bullish) → multiplier = 0.5 (reduce estimated risk by 50%)

var_stressed  = var_mc  * multiplier
cvar_stressed = cvar_mc * multiplier
```

#### Macro Stress (FRED API)

```python
# Fetch from FRED
unemployment_rate = fred.get_series('UNRATE').iloc[-1]   # %
treasury_10y      = fred.get_series('GS10').iloc[-1]      # %

# Construct stress factor
stress_factor = 1.0
if unemployment_rate > 6.0:   stress_factor += 0.15   # recession signal
if treasury_10y > 5.0:         stress_factor += 0.10   # high rate environment
if treasury_10y < 1.0:         stress_factor += 0.05   # deflation signal

var_macro_stressed = var_mc * stress_factor
```

### 4b. Correlation Matrix

Pearson correlation on rolling log-returns, displayed as an interactive heatmap:

```python
corr_matrix = returns_df.corr(method='pearson')   # N×N matrix
```

### 4c. Data Sources

| Source | Type | Access | Fallback |
|--------|------|--------|---------|
| yfinance | Live OHLCV prices | `yfinance.download(tickers)` | Synthetic Gaussian random walk |
| FRED API | Macro indicators (CPI, unemployment, 10Y) | `fredapi.Fred(api_key)` | Hardcoded default values |
| Google News RSS | News headlines | `feedparser.parse(url)` | Empty sentiment (multiplier = 1.0) |
| Perplexity API | LLM-powered financial sentiment | REST API | Fallback to RSS/VADER |
| SEC EDGAR | Insider trading filings | `data.sec.gov/submissions` | Error message |
| OpenAI Vision | Screenshot → portfolio JSON | `gpt-4o-mini` vision endpoint | Error message |

---

## 5. Tab-by-Tab Reference

### Tab Order in Application

```
[1] Lachesis AI       [2] Financial Analytics   [3] Insider Trading
[4] Sentiment         [5] Prompt Studio          [6] Q-TBN
[7] Foresight         [8] Statevector            [9] Reduced States
[10] Measurement      [11] Fidelity & Export     [12] Presets
[13] Present Scenarios [14] Advanced Quantum     [15] Toy QAOA
[16] VQE              [Admin — owner only]
```

---

### Tab 1: Lachesis AI

| | |
|---|---|
| **Icon** | Sparkles |
| **Purpose** | Multi-modal AI financial copilot with voice, web search, screenshot OCR, and full app-state awareness |
| **Engine** | OpenAI GPT-4.1-mini (tool calling: `run_qtbn_analysis`, `search_google`) |
| **Voice** | ElevenLabs TTS, voice ID `VM4OoNVLkEAbLiaL7S14`; OpenAI Whisper for transcription |
| **Screenshot** | Upload brokerage screenshot → GPT-4-Vision → extracted JSON tickers/positions |
| **App awareness** | System prompt includes live tickers, portfolio value, VaR horizon, Monte Carlo sims, noise config, regime |
| **Fallback** | If no OpenAI key: `POST /api/financial/lachesis-guide` (rule-based GPT-4.1-mini with 400 token limit) |
| **Endpoint** | External OpenAI API, `POST /api/financial/extract-screenshot`, `POST /api/qtbn/forecast` |
| **Quantum?** | No (AI/NLP layer) |

---

### Tab 2: Financial Analytics

| | |
|---|---|
| **Icon** | TrendingUp |
| **Purpose** | Portfolio risk analysis: VaR/CVaR, Sharpe/Sortino, correlation matrix, macro and sentiment stress |
| **Primary endpoint** | `POST /api/financial/analyze` |
| **Algorithm** | Monte Carlo (default), optional Quantum Amplitude Estimation |
| **Data** | yfinance live prices or synthetic fallback |
| **Macro inputs** | FRED unemployment + 10Y Treasury yield via `POST /api/fred/macro` |
| **Outputs** | VaR (95%), CVaR, Sharpe, Sortino, Max Drawdown, Annualized Vol, regime badge, price chart, log-returns chart, correlation heatmap, VaR comparison chart |
| **Quantum?** | Optional (QAE path for VaR if `use_qae=True` and qiskit-finance installed) |

---

### Tab 3: Insider Trading

| | |
|---|---|
| **Icon** | Briefcase |
| **Purpose** | SEC EDGAR insider filing browser + per-asset portfolio deep analysis |
| **Filings endpoint** | `POST /api/insider/load-filings` — ticker → CIK lookup → Forms 3, 4, 5 |
| **Portfolio endpoint** | `POST /api/financial/insider` — annual return, vol, Sharpe, max drawdown per asset |
| **Data source** | `data.sec.gov/submissions/{CIK}.json` (24-hour cache) |
| **Outputs** | Filing table (date, form, description, EDGAR link), asset stats table, Sharpe bar chart, equal-weight position bars |
| **Quantum?** | No (classical financial data) |

---

### Tab 4: Sentiment Analysis

| | |
|---|---|
| **Icon** | Newspaper |
| **Purpose** | News sentiment scoring and VaR stress multiplier generation |
| **Endpoint** | `POST /api/sentiment/analyze` |
| **Providers** | Google News RSS + NLTK VADER (free) or Perplexity API (requires key) |
| **VADER scoring** | `-1` (very bearish) to `+1` (very bullish) per headline |
| **Multiplier** | `max(0.5, min(1.5, 1.0 - 0.5 × avg_score))` — feeds into Financial Analytics |
| **Outputs** | Avg sentiment score, VaR multiplier, headline table with per-headline scores |
| **Quantum?** | No (classical NLP) |

---

### Tab 5: Prompt Studio

| | |
|---|---|
| **Icon** | Wand2 |
| **Purpose** | LLM-powered scenario generation via variable-substituted prompt templates |
| **Endpoints** | `GET /api/prompt-studio/templates`, `POST /api/prompt-studio/generate` |
| **Templates** | `risk_scenario`, `qtbn_forecast`, `circuit_analysis`, `stress_test`, `trade_review` |
| **Variables injected** | Tickers, regime, portfolio_value, VaR, CVaR, num_qubits, gates, noise, language |
| **LLM** | OpenAI GPT-4.1-mini (optional; returns template verbatim if no key) |
| **Outputs** | Rendered template (monospace), LLM-generated scenario (Markdown) |
| **Quantum?** | No (AI/NLP layer) |

---

### Tab 6: Q-TBN (Quantum Temporal Bayesian Network)

| | |
|---|---|
| **Icon** | Brain |
| **Purpose** | Multi-step regime forecasting and probability distribution over portfolio outcomes |
| **Endpoint** | `POST /api/qtbn/forecast` |
| **Algorithm** | Classical Markov chain + Normal CDF bucketing (quantum-inspired design) |
| **Inputs** | Prior regime, risk-on prior (%), drift μ (annual %), horizon (days), steps |
| **Outputs** | P(Gain), P(Flat), P(Loss), P(Severe Loss) — pie chart + table; regime timeline line chart; drift path chart |
| **Quantum?** | Quantum-inspired (client-side amplitude posterior); backend is classical |

---

### Tab 7: Foresight

| | |
|---|---|
| **Icon** | Thermometer |
| **Purpose** | Circuit noise characterization via KL-divergence sweep over `(pdep, pamp)` parameter space |
| **Endpoint** | `POST /api/foresight/sweep` |
| **Algorithm** | Qiskit-Aer circuit simulation with grid of noise parameters; KL divergence vs. ideal |
| **Inputs** | Shots, random seeds (comma-separated), depolarizing values, amplitude damping values |
| **Outputs** | 2D KL divergence heatmap (green=trustworthy, red=corrupted), saved scenarios list |
| **Quantum?** | Yes — uses Qiskit-Aer AerSimulator with configurable NoiseModel |

---

### Tab 8: Statevector

| | |
|---|---|
| **Icon** | Atom |
| **Purpose** | Ideal statevector simulation — amplitude and phase visualization of circuit output |
| **Endpoint** | `POST /api/quantum/simulate` |
| **Inputs** | All sidebar quantum controls (qubits, gates, shots, seed, noise) |
| **Outputs** | Probability amplitude bar chart, phase bar chart (color-coded by angle), complex amplitude table `[Re, Im, |α|, |α|², phase°]` |
| **Quantum?** | Yes — Qiskit statevector simulation |

---

### Tab 9: Reduced States

| | |
|---|---|
| **Icon** | Layers |
| **Purpose** | Per-qubit reduced density matrix via partial trace |
| **Endpoint** | `POST /api/quantum/reduced-states` |
| **Algorithm** | `Tr_{all but i}(ρ)` — trace out all qubits except qubit `i` |
| **Outputs** | Bloch vector `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)` and purity for each qubit |
| **Quantum?** | Yes — Qiskit DensityMatrix operations |

---

### Tab 10: Measurement

| | |
|---|---|
| **Icon** | Activity |
| **Purpose** | Side-by-side comparison of ideal vs. noisy measurement outcome distributions |
| **Endpoint** | `POST /api/quantum/measurement` |
| **Metric** | Total variation distance: `TV = (1/2) · Σ_k |p_ideal(k) - p_noisy(k)|` |
| **Outputs** | Grouped bar chart (ideal vs. noisy per basis state), TV distance value |
| **Quantum?** | Yes — Qiskit ideal + noisy circuit simulation |

---

### Tab 11: Fidelity & Export

| | |
|---|---|
| **Icon** | Shield |
| **Purpose** | Circuit fidelity benchmarking + result export |
| **Endpoint** | Fidelity endpoint (Bhattacharyya overlap metric) |
| **Outputs** | Fidelity score F ∈ [0,1], export buttons for simulation results (JSON/CSV) |
| **Quantum?** | Yes — Qiskit simulation |

---

### Tab 12: Presets

| | |
|---|---|
| **Icon** | BookOpen |
| **Purpose** | Load pre-built quantum circuit configurations (Bell state, GHZ, QFT, etc.) |
| **Endpoint** | `GET /api/quantum/presets`, `GET /api/quantum/presets/{key}` |
| **Outputs** | List of available preset names and descriptions; one-click load into AppContext |
| **Quantum?** | Yes — configures quantum sidebar state |

---

### Tab 13: Present Scenarios

| | |
|---|---|
| **Icon** | BarChart2 |
| **Purpose** | Curated market scenario library for structured risk analysis |
| **Endpoint** | `GET /api/foresight/scenarios` |
| **Outputs** | Scenario cards with saved sweep results, timestamps, KL divergence summaries |
| **Quantum?** | Indirect (references saved quantum noise sweep results) |

---

### Tab 14: Advanced Quantum

| | |
|---|---|
| **Icon** | Gauge |
| **Purpose** | Full quantum gate diagnostics suite: tomography, benchmarking, calibration, fidelity |
| **Sub-tabs** | State Tomography, Randomized Benchmarking, Bayesian Noise Calibration, Process Fidelity |
| **Endpoints** | `POST /api/quantum/advanced/{tomography\|benchmarking\|calibrate\|fidelity}` |
| **Key outputs** | Bloch sphere coordinates, EPG (error per gate), Beta posterior credible intervals, per-basis gate fidelity |
| **Quantum?** | Yes — all Qiskit-based |

---

### Tab 15: Toy QAOA

| | |
|---|---|
| **Icon** | Zap |
| **Purpose** | QAOA portfolio optimization with interactive λ sweep and efficient frontier visualization |
| **Endpoints** | `POST /api/qaoa/optimize`, `POST /api/qaoa/sweep`, `GET/POST /api/qaoa/scenarios`, `GET /api/qaoa/log` |
| **Controls** | Portfolio, backend (Qiskit/Aer/classical), regime overlay, depth p, λ slider, shots |
| **Outputs** | Bitstring visualization, selected assets, expected return + risk, QAOA energy, Lachesis narrative, λ-sweep frontier chart, run log |
| **Quantum?** | Yes — QAOA with Qiskit primitives (classical brute-force fallback) |

---

### Tab 16: VQE

| | |
|---|---|
| **Icon** | LineChart |
| **Purpose** | Dual function: (A) Pre-trade risk gate for position approval; (B) VQE solver for quantum Hamiltonians |
| **Risk gate endpoint** | `POST /api/vqe/risk-gate` |
| **VQE solver endpoint** | `POST /api/vqe/solve` |
| **Problems supported** | Toy Hamiltonian, MaxCut, Ising (h/J), Custom Pauli string |
| **Ansätze** | RealAmplitudes, EfficientSU2, TwoLocal |
| **Outputs** | APPROVED/PARTIAL/BLOCKED status + risk limits table; VQE convergence chart; ground state energy |
| **Quantum?** | Yes — VQE with Qiskit Estimator primitive |

---

### Admin Tab (Owner-Only)

| | |
|---|---|
| **Icon** | KeyRound |
| **Visibility** | Only shown when `user.email === "darriusperson@gmail.com"` |
| **Purpose** | API key validation, system health check, capability flags |
| **Endpoint** | `POST /api/admin/validate-key`, `GET /api/health` |
| **Outputs** | Key format validity, system capability matrix (Qiskit ✓/✗, yfinance ✓/✗, FRED ✓/✗, etc.) |

---

## 6. Complete API Reference

### Quantum Simulation (9 endpoints)

| Method | Endpoint | Purpose | Qiskit Used |
|--------|----------|---------|------------|
| POST | `/api/quantum/simulate` | Statevector + noisy measurement | QuantumCircuit, AerSimulator, NoiseModel |
| POST | `/api/quantum/reduced-states` | Per-qubit density matrices | DensityMatrix, partial_trace |
| POST | `/api/quantum/measurement` | Ideal vs. noisy count comparison | AerSimulator ×2 |
| POST | `/api/quantum/advanced/tomography` | Bloch sphere reconstruction | QuantumCircuit, Sampler |
| POST | `/api/quantum/advanced/benchmarking` | Clifford RB, EPG | QuantumCircuit, scipy.curve_fit |
| POST | `/api/quantum/advanced/calibrate` | Bayesian Beta posterior | Sampler, scipy.stats.beta |
| POST | `/api/quantum/advanced/fidelity` | Gate process fidelity | AerSimulator, 3 bases |
| GET | `/api/quantum/presets` | Available preset circuits | — |
| GET | `/api/quantum/presets/{key}` | Load specific preset | — |

### Financial Analytics (5 endpoints)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/financial/analyze` | VaR/CVaR/Sharpe + optional QAE |
| POST | `/api/financial/insider` | Per-asset stats + equal-weight VaR |
| POST | `/api/financial/extract-screenshot` | Vision → JSON portfolio |
| POST | `/api/financial/lachesis-guide` | AI risk narrative (rule-based) |
| POST | `/api/fred/macro` | FRED CPI + unemployment + 10Y |

### QTBN (1 endpoint)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/qtbn/forecast` | Markov regime evolution + outcome probabilities |

### QAOA Portfolio (6 endpoints)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/qaoa/portfolios` | Available portfolio configs |
| POST | `/api/qaoa/optimize` | Single QAOA optimization run |
| POST | `/api/qaoa/sweep` | λ parameter sweep (efficient frontier) |
| GET | `/api/qaoa/scenarios` | Fetch saved scenarios |
| POST | `/api/qaoa/scenarios` | Save scenario |
| GET | `/api/qaoa/log` | Audit trail (last 50 runs) |

### VQE (3 endpoints)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/vqe/risk-gate` | Trade approval decision |
| GET | `/api/vqe/audit` | Risk gate history (last 20) |
| POST | `/api/vqe/solve` | VQE Hamiltonian minimization |

### Foresight (3 endpoints)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/foresight/sweep` | Noise param grid + KL divergence |
| GET | `/api/foresight/scenarios` | Saved sweep results |
| POST | `/api/foresight/scenarios` | Save sweep result |

### Other (12 endpoints)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/sentiment/analyze` | News sentiment scoring |
| GET | `/api/prompt-studio/templates` | LLM template list |
| POST | `/api/prompt-studio/generate` | Fill + execute template |
| POST | `/api/insider/load-filings` | SEC EDGAR Forms 3/4/5 |
| POST | `/api/insider/lookup-cik` | Ticker → CIK (legacy) |
| POST | `/api/insider/filings` | CIK → filings (legacy) |
| POST | `/api/search` | SerpAPI web search |
| POST | `/api/auth/signup` | Admin user creation |
| GET | `/api/health` | System status + capabilities |
| POST | `/api/admin/validate-key` | API key format check |

---

## 7. Global State Architecture

### AppContext (`src/contexts/AppContext.tsx`)

All 17 dashboard components read from a single React context, ensuring consistent quantum and financial parameters across the entire application:

```typescript
interface AppState {
  // ── Quantum Circuit ──────────────────────────────────────
  num_qubits:  number;             // 1–4
  shots:       number;             // 64–32768
  use_seed:    boolean;
  seed_val:    number;

  // 3 sequential gate steps, each configuring N qubits + CNOTs
  step0: GateStepConfig;           // { gates: [{name, angle}], cnot_01, cnot_12, cnot_23 }
  step1: GateStepConfig;
  step2: GateStepConfig;

  // ── Noise Model ───────────────────────────────────────────
  noise: {
    enable_depolarizing:      boolean;
    pdep0, pdep1, pdep2:      number;  // per-qubit depolarizing p ∈ [0, 0.2]
    enable_amplitude_damping: boolean;
    pamp0, pamp1, pamp2:      number;  // per-qubit T1 relaxation p
    enable_phase_damping:     boolean;
    pph0, pph1, pph2:         number;  // per-qubit T2 dephasing p
    enable_cnot_noise:        boolean;
    pcnot0, pcnot1, pcnot2:   number;  // CNOT pair depolarizing p
  };

  // ── Financial Configuration ───────────────────────────────
  finance: {
    tickers:              string;   // comma-separated, e.g. "SPY,QQQ,AAPL"
    lookback_days:        number;   // 30–2000
    portfolio_value:      number;   // USD
    confidence_level:     number;   // 0.80–0.99
    var_horizon:          number;   // days (e.g. 10)
    mc_sims:              number;   // Monte Carlo draws
    volatility_threshold: number;
    apply_macro_stress:   boolean;
    demo_mode:            boolean;  // synthetic data if true
    per_share:            boolean;
    show_position:        boolean;
  };

  // ── Global ────────────────────────────────────────────────
  language: string;                // One of 60+ supported languages
}
```

### Sidebar Controls → Circuit Parameters

```
Sidebar Control                AppState Field            Backend Usage
───────────────────────────────────────────────────────────────────────
Qubits slider (1–4)        →   num_qubits            →   QuantumCircuit(n)
Shots slider (128–16384)   →   shots                 →   AerSimulator shots
Fixed Seed toggle + value  →   use_seed, seed_val    →   seed_simulator=
Gate step rows (×3)        →   step0/1/2.gates       →   _apply_gate()
CNOT checkboxes            →   step*.cnot_01/12/23   →   qc.cx()
Depolarizing sliders       →   noise.pdep0/1/2       →   depolarizing_error(p)
Amplitude damping sliders  →   noise.pamp0/1/2       →   amplitude_damping_error(p)
Phase damping sliders      →   noise.pph0/1/2        →   phase_damping_error(p)
```

---

## 8. Mathematical Formulations

### 8.1 QAOA Objective

```
Maximize  J(x) = λ · μᵀx  −  (1−λ) · xᵀΣx
          x ∈ {0,1}^N

QUBO form (minimization):
Minimize  E(x) = −J(x) = −λ · μᵀx  +  (1−λ) · xᵀΣx

Hamiltonian encoding (Z-basis, x_i = (1−Z_i)/2):
H_cost = Σ_i c_i · Z_i + Σ_{i<j} Q_{ij} · Z_i⊗Z_j + const
```

### 8.2 VaR and CVaR (Monte Carlo)

```
Given portfolio returns R ~ N(μ_h, σ_h²):

VaR_α  = Φ⁻¹(1−α) · σ_h + μ_h        [classical quantile]
CVaR_α = μ_h − σ_h · φ(Φ⁻¹(α)) / α   [classical conditional expectation]

QAE path:
P(R ≤ threshold) ≈ IterativeAmplitudeEstimation(ε=0.02)
Speedup: O(1/ε²) → O(1/ε)
```

### 8.3 VQE Energy Minimization

```
Ground state problem:  min_θ ⟨ψ(θ)| H |ψ(θ)⟩

Ansatz:  |ψ(θ)⟩ = U_n(θ_n) · ... · U_1(θ_1) · |0⟩^⊗n
where each U_k is a parametrized rotation layer

COBYLA update:  θ_{k+1} = θ_k + α_k · ∇E(θ_k)  (derivative-free)

Convergence criterion:  |E(θ_k) − E(θ_{k-1})| < 10⁻⁶
```

### 8.4 QTBN Regime Evolution

```
State vector:  p(t) ∈ Δ²  [probability simplex over 3 regimes]
Update:        p(t+1) = p(t) @ T

T = [[0.85, 0.13, 0.02],
     [0.25, 0.55, 0.20],
     [0.10, 0.25, 0.65]]

Expected drift at step t:
μ(t) = Σ_i p_i(t) · μ_regime[i]
```

### 8.5 Sharpe and Sortino Ratios

```
Sharpe  = √252 · (μ_excess) / σ_excess
Sortino = √252 · (μ_excess) / σ_downside

where:
  μ_excess    = mean(r_t − r_f/252)
  σ_excess    = std(r_t  − r_f/252)
  σ_downside  = √[ mean( (min(r_excess, 0))² ) ]
  r_f         = 0.04  (4% annual risk-free rate)
```

### 8.6 Quantum Fidelity (Measurement)

```
F(p_ideal, p_noisy) = ( Σ_k √[ p_ideal(k) · p_noisy(k) ] )²

Equivalently:  F = ||√p_ideal||·||√p_noisy||·cos(angle)  [Bhattacharyya coefficient]
Range:  F ∈ [0, 1],  F=1 iff p_ideal = p_noisy
```

### 8.7 KL Divergence (Foresight)

```
KL(p_noisy || p_ideal) = Σ_k p_noisy(k) · ln[ p_noisy(k) / p_ideal(k) ]

Properties:
  KL ≥ 0  (Gibbs' inequality)
  KL = 0  iff  p_noisy = p_ideal
  KL is asymmetric:  KL(p||q) ≠ KL(q||p)
```

### 8.8 Randomized Benchmarking EPG

```
Survival model:  P(m) = A · p^m + B

where:
  m = sequence length (number of Clifford gates)
  p = depolarizing parameter ∈ [0, 1]
  A, B = SPAM (state preparation and measurement) error parameters

Error per gate:
  EPG = (1 − p) · (1 − 1/d)
  d = 2  for single-qubit (Hilbert space dimension)
```

### 8.9 Bayesian Noise Calibration

```
Prior:     α₀ = β₀ = 1  (Beta(1,1) = Uniform[0,1])
Update:    α = α₀ + k_err,  β = β₀ + (N − k_err)
           where k_err = observed error events,  N = total shots

Posterior: p_noise | data ~ Beta(α, β)
Mean:      E[p] = α / (α + β)
95% CI:    [Beta.ppf(0.025, α, β),  Beta.ppf(0.975, α, β)]
```

---

## 9. Framework Comparison: Qiskit vs. Cirq vs. PennyLane

### What Lachesis Uses

Lachesis is built on **IBM Qiskit 1.0+** with **Qiskit-Aer 0.14.0+**. The specific APIs used are:

```python
# Core
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error

# Optimization (optional)
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA, VQE
from qiskit_algorithms.optimizers import COBYLA

# Finance (optional)
from qiskit_finance.circuit.library import NormalDistribution
from qiskit_algorithms import IterativeAmplitudeEstimation

# Primitives (Qiskit 1.0 unified interface)
from qiskit.primitives import Sampler, Estimator
```

### Equivalent Implementations in Other Frameworks

#### QAOA in Google Cirq

```python
import cirq
import numpy as np

# Qubits
qubits = cirq.LineQubit.range(N)

# QAOA circuit
def qaoa_circuit(gamma, beta, reps):
    circuit = cirq.Circuit()
    circuit += cirq.H.on_each(*qubits)   # Hadamard layer
    for _ in range(reps):
        # Cost layer
        for i, j in edges:
            circuit += cirq.ZZPowGate(exponent=gamma/np.pi).on(qubits[i], qubits[j])
        # Mixer layer
        circuit += [cirq.rx(2*beta).on(q) for q in qubits]
    circuit += cirq.measure(*qubits)
    return circuit

simulator = cirq.Simulator()
result = simulator.simulate(qaoa_circuit(gamma, beta, reps=2))
```

**Key difference from Qiskit:** Cirq uses `ZZPowGate` directly; Qiskit-Optimization abstracts the QUBO → Hamiltonian → circuit conversion automatically via `MinimumEigenOptimizer`. In Cirq, the researcher must manually encode the cost function into circuit gates.

#### VQE in PennyLane

```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def vqe_circuit(params, hamiltonian):
    # RealAmplitudes-style ansatz
    for layer in range(reps):
        for i in range(n_qubits):
            qml.RY(params[layer][i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
    return qml.expval(hamiltonian)

# Hamiltonian
coeffs = [1.0, 0.4]
obs    = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0)]
H      = qml.Hamiltonian(coeffs, obs)

# Optimization
opt    = qml.GradientDescentOptimizer(stepsize=0.1)
params = np.random.randn(reps, n_qubits)

for step in range(200):
    params, energy = opt.step_and_cost(lambda p: vqe_circuit(p, H), params)
```

**Key difference from Qiskit:** PennyLane uses auto-differentiation (parameter-shift rule or backpropagation) for gradient-based optimizers; Qiskit's `COBYLA` is derivative-free. PennyLane's differentiable programming model is generally more compatible with JAX/PyTorch workflows.

### Framework Trade-Off Summary

| Aspect | Qiskit | Cirq | PennyLane |
|--------|--------|------|-----------|
| **Primary use** | Gate-level simulation + NISQ algorithms | Google hardware focus | Hybrid QML |
| **High-level algorithms** | ✅ Built-in QAOA, VQE, QAE | ❌ Manual construction | ✅ Built-in `qml.VQE` |
| **Noise modeling** | ✅ Rich NoiseModel API (Aer) | ✅ `cirq.ConstantQubitNoiseModel` | ⚠️ Limited (device-specific) |
| **Financial extensions** | ✅ qiskit-finance (NormalDistribution, QAE) | ❌ None | ❌ None |
| **Hardware backends** | IBM Quantum (IBMQ/Runtime) | Google Quantum AI | 20+ backends |
| **Gradient support** | ⚠️ Derivative-free (COBYLA, SPSA) | ⚠️ Manual | ✅ Auto-diff (JAX, Torch) |
| **Primitives API** | ✅ Unified Sampler/Estimator (v1.0+) | ❌ | ✅ `@qml.qnode` |
| **Portability to real HW** | ✅ `IBMRuntimeService` | ✅ `cirq.google` | ✅ `qml.device("qiskit.ibmq")` |

**Recommendation for PhD research:** Lachesis uses Qiskit because of `qiskit-finance` and `qiskit-optimization` — the only framework with built-in quantum amplitude estimation for financial distributions and QUBO converters. For a PennyLane port, the gradient-based VQE path would be improved; for a Cirq port, hardware-level gate compilation would be more transparent.

---

## 10. Research Extensions & Open Questions

### 10.1 Warm-Start QAOA

**Current:** QAOA parameters `(γ, β)` are initialized randomly.  
**Extension:** Initialize with parameters derived from semidefinite programming (SDP) relaxation of the QUBO, achieving better approximation ratio from fewer layers. Reference: Egger et al. (2021), "Warm-starting quantum optimization".

```python
# Proposed warm-start: classical SDP → initial rotation angles
sdp_solution = solve_sdp_relaxation(cov_matrix, mu_vector)
gamma_init, beta_init = sdp_to_qaoa_angles(sdp_solution)
qaoa = QAOA(initial_point=[gamma_init, beta_init], ...)
```

### 10.2 Hardware-Efficient Ansätze for NISQ Devices

**Current:** `RealAmplitudes` / `EfficientSU2` — general-purpose, may have high two-qubit gate depth.  
**Extension:** Design ansatz matching the native gate set and connectivity graph of the target QPU (e.g., IBM Eagle topology for 127-qubit devices). Use `PassManager` in Qiskit 1.0 for transpilation-aware ansatz design.

### 10.3 Quantum Error Mitigation

Lachesis currently runs on the Aer simulator without error mitigation. For real hardware execution, the following techniques should be layered:

- **Zero-Noise Extrapolation (ZNE):** Run circuit at multiple noise scales (1×, 2×, 3×) and extrapolate to zero noise. Compatible with Qiskit's `ZNE` transpiler pass.
- **Probabilistic Error Cancellation (PEC):** Represent ideal gate as quasi-probability distribution over noisy operations. Computationally expensive but asymptotically unbiased.
- **Readout Error Mitigation:** Calibrate measurement errors with a confusion matrix; invert at post-processing time. Low overhead, easy to implement.

### 10.4 True Quantum Bayesian Network

**Current QTBN:** Classical Markov chain with quantum-inspired amplitude posterior in TypeScript.

**Research direction:** Implement a genuinely quantum Bayesian network where:
1. Regime states are encoded in quantum superposition
2. Conditional probability tables are represented as quantum state preparations
3. Bayesian belief propagation is performed via quantum circuit inference (Tucci, 1995; Low et al., 2014)
4. Transition matrix estimation uses VQE on a parameterized density matrix circuit

This would make QTBN the first truly quantum component of the financial pipeline.

### 10.5 Scalability Beyond 4 Qubits

**Current limitation:** AerSimulator statevector requires O(2^N) memory. At N=30 assets, statevector simulation requires 8 GB RAM; N=40 requires 8 TB — infeasible.

**Path forward:**
- **Matrix Product State (MPS) simulator:** `AerSimulator(method="matrix_product_state")` — efficient for low-entanglement circuits, practical to N≈50 with bond dimension limits
- **Real quantum hardware:** IBM Quantum 127-qubit Eagle processor via `qiskit-ibm-runtime` — limited by coherence time and gate fidelity
- **Variational representation:** Replace exact QUBO with variational Gibbs state — approximate but scalable

### 10.6 Classical Benchmark Comparisons

To validate quantum advantage claims, Lachesis should be benchmarked against:

| Algorithm | Implementation | Comparison metric |
|-----------|---------------|-------------------|
| Gurobi MILP | `gurobipy` | Time to optimal, solution quality |
| Simulated Annealing | `scipy.optimize.dual_annealing` | Approximation ratio |
| Genetic Algorithm | `pymoo` | Solution diversity |
| CPLEX | IBM Decision Optimization | Global optimum guarantee |
| Classical SDP relaxation | `cvxpy` | Relaxation bound tightness |

### 10.7 Multi-Objective Portfolio Optimization

**Current:** Single-objective QUBO with λ trade-off parameter.  
**Extension:** Multi-objective QAOA with Pareto front sampling — optimize `(return, risk, liquidity, ESG_score)` simultaneously using a vectorized objective encoding. Research frontier: Coelho et al. (2022).

### 10.8 Live IBM Quantum Integration

The QAOA and VQE implementations are structured to support `IBMRuntimeService`. The required change:

```python
# Current (Aer simulator)
from qiskit.primitives import Sampler, Estimator

# IBM Quantum hardware
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session

service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
backend = service.least_busy(operational=True, simulator=False)

with Session(backend=backend) as session:
    sampler   = Sampler(session=session)
    estimator = Estimator(session=session)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=depth)
    result = MinimumEigenOptimizer(qaoa).solve(qubo)
```

No changes required to `QuadraticProgram`, QUBO conversion, or portfolio narrative generation — the Primitives API (`Sampler`/`Estimator`) is hardware-agnostic by design.

---

## Appendix A: Environment Setup

### Local Development

```bash
# Backend
cd "/Applications/Quantum Temporal Bayesian Network Prototype"
pip3 install -r requirements.txt
uvicorn api_server:app --reload --port 8000

# Frontend
cd tools/Frontend_Loveable/quantum-foresight-ai-main
cp .env.example .env       # fill in VITE_SUPABASE_URL, VITE_SUPABASE_PUBLISHABLE_KEY
npm install
npm run dev                 # http://localhost:5173
```

### Required Environment Variables

**Backend (`.env` at project root):**
```
SUPABASE_URL=https://mvprtzaatfbvdxwutrbo.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service_role_jwt>
OPENAI_API_KEY=sk-...      (optional — fallback to rule-based guide)
FRED_API_KEY=...            (optional — fallback to hardcoded macro values)
SERPAPI_KEY=...             (optional — web search disabled if absent)
ELEVENLABS_API_KEY=...      (optional — TTS disabled if absent)
```

**Frontend (`.env`):**
```
VITE_SUPABASE_URL=https://mvprtzaatfbvdxwutrbo.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=<anon_jwt>
VITE_API_URL=http://localhost:8000    (or Railway URL for production)
```

### Optional Quantum Packages

```bash
pip3 install qiskit-optimization   # QAOA portfolio optimization
pip3 install qiskit-finance        # QAE for VaR/CVaR
pip3 install qiskit-ibm-runtime    # IBM Quantum hardware access
```

Lachesis detects these at startup via `try/except ImportError` and sets capability flags (`_HAS_QAOA`, `_HAS_QAE`, `_HAS_IBMQ`) visible at `GET /api/health`.

---

## Appendix B: Key Source Locations for Deep Review

| Area | File | Key Lines |
|------|------|-----------|
| Circuit construction + noise | `api_server.py` | `_build_circuit()`, `_build_noise_model()` |
| QAE for VaR/CVaR | `api_server.py` | `_qae_var_cvar()` |
| QTBN Markov engine | `api_server.py` | `/api/qtbn/forecast` handler (~line 875) |
| QAOA full implementation | `qaoa_scenario1.py` | `run_qaoa_portfolio()`, `apply_regime_to_cfg()` |
| VQE solver | `vqe_tab.py` | `_try_run_real_vqe()` |
| Risk gate decision logic | `vqe_tab.py` | `apply_risk_gates()`, `POLICY_LIMITS` |
| Risk multiplier from energy | `vqe_tab.py` | `build_scaled_risk_limits()` |
| KL divergence + sweep | `api_server.py` | `/api/foresight/sweep` handler |
| Client-side QTBN inference | `src/lib/qtbn-engine.ts` | `performInference()`, `quantumPosteriorSampling()` |
| Global state definition | `src/contexts/AppContext.tsx` | Full file (234 lines) |
| Typed API client | `src/lib/api.ts` | Full file (539 lines) |

---

*Document generated April 2026. For questions on Qiskit-specific implementation details, refer to the Qiskit 1.0 documentation and the source files listed in Appendix B.*
