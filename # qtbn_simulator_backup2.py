# qtbn_simulator_clean.py
# Q-TBN Simulator (Qiskit 2.x) + Finance (MC VaR/CVaR, regime) + FRED macro stress (with graceful fallback)
# + Full Foresight/Sweeps manager (A/B compare, CSV I/O, scenario cloning, repo scenarios)
# + LACHESIS AI Guide (OpenAI integration)
# + ADVANCED QUANTUM tab (1q tomography, process fidelity proxy, Quantum Volume proxy, RB)
# + Enhanced with sticky config, dependency checks, calibration snapshots, and improved error handling
# + Quantum depth: Bayesian noise calibration, randomized benchmarking, process proxy, noise-aware suggestions,
#   scenario-to-circuit library
# + NEW: Quantum Amplitude Estimation hook for VaR/CVaR (guarded/optional)

from __future__ import annotations
import io, os, json, math, hashlib, time, csv, sys
import re
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import traceback
import logging
from functools import lru_cache
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _prefer_browser_app():
    """Ensure Streamlit opens in Google Chrome (preferred) with sensible macOS fallbacks to avoid Safari issues."""
    if os.environ.get("BROWSER"):
        return
    if sys.platform != "darwin":
        return

    def has_app(app_name: str) -> bool:
        exec_path = Path(f"/Applications/{app_name}.app/Contents/MacOS/{app_name}")
        return exec_path.exists()

    preferred_app = os.environ.get("QTBN_PREFERRED_BROWSER_APP", "Google Chrome")
    candidates = [preferred_app, "Firefox", "Atlas"]
    for app in candidates:
        if has_app(app):
            os.environ["BROWSER"] = f'open -a "{app}"'
            logger.info("Configured Streamlit to auto-launch in %s.", app)
            return

_prefer_browser_app()

def _patch_streamlit_for_safari():
    """Patch Streamlit's frontend bundle to avoid regex features unsupported in older Safari builds."""
    if os.environ.get("QTBN_SKIP_SAFARI_PATCH"):
        return
    try:
        import streamlit as _st
    except Exception:
        return

    js_dir = Path(_st.__file__).resolve().parent / "static" / "static" / "js"
    if not js_dir.exists():
        return

    single_caps_marker = 'const handlePreserveConsecutiveUppercase$1='
    single_caps_patch = (
        'const handlePreserveConsecutiveUppercase$1=(t,g)=>{'
        'if(typeof t!="string")return t;'
        'const isUpper=c=>c>="A"&&c<="Z";'
        'const isLower=c=>c>="a"&&c<="z";'
        'const isAlphaNum=c=>isUpper(c)||isLower(c)||(c>="0"&&c<="9");'
        'const chars=Array.from(t);'
        'for(let i=0;i<chars.length;i++){const ch=chars[i];'
        'if(isAlphaNum(ch)){const prev=i>0?chars[i-1]:"";'
        'const next=i<chars.length-1?chars[i+1]:"";'
        'if(!isAlphaNum(prev)&&!isAlphaNum(next))chars[i]=ch.toLowerCase();}}'
        'const normalized=chars.join("");'
        'let out="";'
        'let idx=0;'
        'while(idx<normalized.length){const ch=normalized[idx];'
        'if(isUpper(ch)){const prev=idx>0?normalized[idx-1]:"";'
        'let end=idx;'
        'while(end<normalized.length&&isUpper(normalized[end]))end++;'
        'const segment=normalized.slice(idx,end);'
        'if((prev===""||!isUpper(prev))&&end<normalized.length&&isUpper(normalized[end])){'
        'let lower=end+1;'
        'while(lower<normalized.length&&isLower(normalized[lower]))lower++;'
        'if(lower>end+1){'
        'out+=segment+g+normalized.slice(end,lower).toLowerCase();'
        'idx=lower;'
        'continue;}}'
        'out+=segment;'
        'idx=end;}else{out+=ch;idx++;}}'
        'return out;'
        '};/*QTBN_SAFARI_PATCH*/'
    )

    email_marker = 'function transformGfmAutolinkLiterals'
    email_patch = (
        'function transformGfmAutolinkLiterals(t){'
        'findAndReplace(t,[[/(https?:\\/\\/|www(?=\\.))([-\\.\\w]+)([^ \\t\\r\\n]*)/gi,findUrl]],'
        '{ignore:["link","linkReference"]});'
        '}/*QTBN_SAFARI_PATCH*/'
    )

    for js_bundle in sorted(js_dir.glob("index.*.js")):
        try:
            src = js_bundle.read_text()
        except Exception:
            continue

        if "QTBN_SAFARI_PATCH" in src:
            return

        updated = False
        start_uc = src.find(single_caps_marker)
        if start_uc != -1:
            end_uc = src.find('function decamelize$2', start_uc)
            if end_uc != -1 and 'QTBN_SAFARI_PATCH' not in src[start_uc:end_uc]:
                src = src[:start_uc] + single_caps_patch + src[end_uc:]
                updated = True

        start_email = src.find(email_marker)
        if start_email != -1:
            end_email = src.find('function findUrl', start_email)
            if end_email != -1 and 'QTBN_SAFARI_PATCH' not in src[start_email:end_email]:
                src = src[:start_email] + email_patch + src[end_email:]
                updated = True

        if updated:
            try:
                js_bundle.write_text(src)
                logger.info("Patched %s for Safari compatibility", js_bundle.name)
            except Exception as e:
                logger.warning("Failed to write Safari patch to %s: %s", js_bundle, e)
            return

_patch_streamlit_for_safari()

# ----- Optional NLP deps -----
try:
    import feedparser
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    logger.info("feedparser and nltk imported successfully")
except ImportError as e:
    feedparser = None
    SentimentIntensityAnalyzer = None
    logger.warning(f"feedparser or nltk not available: {e}. Sentiment analysis will be disabled.")

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    logger.info("plotly imported successfully")
except ImportError as e:
    px = go = make_subplots = None
    logger.warning(f"plotly not available: {e}. Charts will be disabled.")

from scipy.stats import norm
from scipy.cluster.vq import kmeans2
from scipy.optimize import least_squares

# PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    logger.info("reportlab imported successfully")
except ImportError as e:
    SimpleDocTemplate = None
    Paragraph = Spacer = Image = Preformatted = None
    getSampleStyleSheet = None
    inch = None
    logger.warning(f"reportlab not available: {e}. Executive report export will be disabled.")

# TOML for settings persistence
try:
    import toml
    logger.info("toml imported successfully")
except ImportError as e:
    toml = None
    logger.warning(f"toml not available: {e}. Settings persistence will be disabled.")

# ---------- Optional external deps (handled gracefully) ----------
try:
    import yfinance as yf
    logger.info("yfinance imported successfully")
except Exception as e:
    yf = None
    logger.warning(f"yfinance not available: {e}")

try:
    from fredapi import Fred
    logger.info("fredapi imported successfully")
except Exception as e:
    Fred = None
    logger.warning(f"fredapi not available: {e}")

try:
    import openai
    logger.info("openai imported successfully")
except ImportError as e:
    openai = None
    logger.warning(f"openai not available: {e}")

try:
    import cvxpy as cp
    logger.info("cvxpy imported successfully")
except ImportError as e:
    cp = None
    logger.warning(f"cvxpy not available: {e}. Portfolio optimization will be disabled.")

# ---------- Qiskit imports (fixed & guarded) ----------
# Core (strict) — if this fails, we truly cannot run the quantum parts.
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        depolarizing_error, amplitude_damping_error, phase_damping_error, NoiseModel
    )
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import (
        Statevector, DensityMatrix, partial_trace, Pauli, state_fidelity
    )
    HAVE_QISKIT_CORE = True
except ImportError as e:
    HAVE_QISKIT_CORE = False
    interpreter = sys.executable or "python3"
    st.set_page_config(page_title="Q-TBN Simulator + Finance/Macro", layout="wide")
    st.error(f"Qiskit core missing: {e}")
    st.markdown("Install Qiskit core into the SAME Python environment that runs Streamlit:")
    st.code(f'"{interpreter}" -m pip install qiskit qiskit-aer', language="bash")
    st.stop()

# ---------- QAE/Finance optional imports (robust shim) ----------
HAVE_QAE = True
_QAE_ERRORS: Dict[str, str] = {}
IterativeAmplitudeEstimation = None  # type: ignore
NormalDistribution = None  # type: ignore
EstimationProblem = None  # type: ignore

# Iterative Amplitude Estimation (new then legacy)
try:
    from qiskit_algorithms.amplitude_estimators import IterativeAmplitudeEstimation  # type: ignore
except Exception as e1:
    try:
        from qiskit.algorithms.amplitude_estimators import IterativeAmplitudeEstimation  # type: ignore
    except Exception as e2:
        HAVE_QAE = False
        _QAE_ERRORS["IterativeAmplitudeEstimation"] = f"{e1!r} | {e2!r}"

# NormalDistribution (moved in finance)
try:
    from qiskit_finance.circuit.library import NormalDistribution  # type: ignore
except Exception as e1:
    try:
        from qiskit_finance.circuit.library.probability_distributions import NormalDistribution  # type: ignore
    except Exception as e2:
        HAVE_QAE = False
        _QAE_ERRORS["NormalDistribution"] = f"{e1!r} | {e2!r}"

# EstimationProblem (new then legacy)
try:
    from qiskit_algorithms import EstimationProblem  # type: ignore
except Exception as e1:
    try:
        from qiskit.algorithms import EstimationProblem  # type: ignore
    except Exception as e2:
        HAVE_QAE = False
        _QAE_ERRORS["EstimationProblem"] = f"{e1!r} | {e2!r}"

# -------------------------------------------------------------------
# Streamlit config
# -------------------------------------------------------------------
st.set_page_config(page_title="Q-TBN Simulator + Finance/Macro", layout="wide")
st.markdown("""
<script>
if (navigator.userAgent.indexOf("Safari") !== -1 && navigator.userAgent.indexOf("Chrome") === -1) {
    alert("This app has known compatibility issues in Safari (SyntaxError in regex). Please open in Chrome: http://localhost:8620");
}
</script>
""", unsafe_allow_html=True)

st.caption(
    "Repo-backed scenarios, statevector, noisy counts, fidelity, sweeps, foresight + "
    "market VaR/CVaR & regime with FRED macro stress (auto fallback)."
)

# --- Small runtime diagnostics to confirm interpreter & versions ---
with st.expander("🛠 Runtime diagnostics", expanded=False):
    import platform
    diag = {
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import qiskit, qiskit_aer
        diag.update({
            "qiskit_version": getattr(qiskit, "__qiskit_version__", {}),
            "aer_version": getattr(qiskit_aer, "__version__", "unknown"),
        })
    except Exception as e:
        diag["qiskit_diag_error"] = str(e)
    try:
        import qiskit_algorithms, qiskit_finance
        diag.update({
            "algorithms_version": getattr(qiskit_algorithms, "__version__", "unknown"),
            "finance_version": getattr(qiskit_finance, "__version__", "unknown"),
        })
    except Exception as e:
        diag["qae_finance_diag_error"] = str(e)
    st.write(diag)
    if not HAVE_QAE:
        st.warning("QAE components not available (QAE toggle will be disabled).")
        if _QAE_ERRORS:
            st.code(json.dumps(_QAE_ERRORS, indent=2), language="json")
            st.caption("Above shows which QAE imports failed and why.")

# -------------------------------------------------------------------
# Utility: session helpers
# -------------------------------------------------------------------
def ss_get(k, default=None):
    """Safe session state getter with automatic initialization"""
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state.get(k, default)

def ss_set(k, v):
    st.session_state[k] = v

def _safe_render_template(tmpl: str, values: dict) -> str:
    """
    Lightweight {var} renderer. If a {var} is missing, it stays as {var} so
    the user sees what wasn't filled instead of crashing.
    """
    def repl(m):
        key = m.group(1)
        return str(values.get(key, "{"+key+"}"))
    return re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, tmpl)

def _studio_collect_context_from_app(var_map: dict) -> dict:
    """
    Pulls values out of Streamlit session state using the mapping in var_map.
    Example: {"alpha":"confidence_level"} -> {"alpha": 0.95}
    Falls back to the raw string if key not in session.
    """
    ctx = {}
    for k, state_key in (var_map or {}).items():
        ctx[k] = st.session_state.get(state_key, ss_get(state_key))
    return ctx

def _studio_run_openai_chat(messages: list, model: str, temperature: float, max_tokens: int, api_key: str) -> tuple[str, dict]:
    """
    Try both legacy (openai.ChatCompletion) and new SDK styles gracefully.
    Returns (text, raw_meta). If OpenAI is unavailable or errors, return a stub.
    """
    if openai is None or not api_key:
        # Offline fallback: just echo the last user message with a stub.
        user_last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[LLM offline] Echo:\n{user_last}", {"offline": True}

    try:
        # Legacy-style usage
        openai.api_key = api_key
        if hasattr(openai, "ChatCompletion"):
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            text = resp["choices"][0]["message"]["content"]
            return text, {"provider": "openai_legacy", "raw": resp}
        # New style
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            return text, {"provider": "openai_v1", "raw": resp.model_dump()}
        except Exception as e2:
            return f"[LLM error] {e2}", {"error": str(e2)}
    except Exception as e:
        return f"[LLM error] {e}", {"error": str(e)}

def _studio_export_store_json(store: dict) -> bytes:
    try:
        return json.dumps(store, indent=2).encode()
    except Exception:
        return json.dumps({}, indent=2).encode()

def _studio_import_store_json(file_bytes: bytes) -> dict:
    try:
        data = json.loads(file_bytes.decode("utf-8"))
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}

# Ensure some global defaults exist
ss_get("num_qubits", 1)
ss_get("shots", 2048)
ss_get("use_seed", True)
ss_get("seed_val", 17)
ss_get("enable_dep", False)
ss_get("enable_amp", False)
ss_get("enable_phs", False)
ss_get("enable_cnot_noise", False)
ss_get("risk_free_rate", 0.04)
ss_get("confidence_level", 0.95)
ss_get("portfolio_value", 1_000_000.0)
ss_get("volatility_threshold", 0.3)
ss_get("lookback_days", 365)
ss_get("var_horizon", 10)
ss_get("mc_sims", 50000)
ss_get("DEMO_MODE", False)
ss_get("use_calibrated_noise", False)
ss_get("feedback_list", [])  # For storing feedback
ss_get("fred_api_key", "")   # Default to empty string to avoid AttributeError
ss_get("use_qae", False)     # New for QAE toggle
ss_get("tickers", "AAPL,MSFT,SPY")
ss_get("apply_macro_stress", False)
ss_get("macro_lookback_days", 365)
ss_get("sentiment_multiplier", 1.0)
ss_get("preloaded_demo", False)
ss_get("openai_api_key", "")
ss_get("prompt_studio_store", {
    # example starter template so the tab isn’t empty
    "Quick Explain": {
        "description": "Short explanation of the current QTBN noise settings’ effect on VaR.",
        "system": "You are Lachesis, a precise, neutral explainer.",
        "template": "Explain succinctly how the current noise settings (dep={dep}, amp={amp}, phs={phs}, cnot={cnot}) and gates influence VaR at confidence {alpha} over a horizon of {h} days.",
        "variables": {"dep": "pdep1", "amp": "pamp1", "phs": "pph1", "cnot": "enable_cnot_noise", "alpha": "confidence_level", "h": "var_horizon"},
        "few_shots": [
            {"role": "user", "content": "Explain impact of high depolarizing noise on VaR."},
            {"role": "assistant", "content": "Higher depolarizing noise randomizes outcomes, widens return dispersion, and typically pushes VaR and CVaR more negative at fixed confidence."}
        ]
    }
})

# -------------------------------------------------------------------
# Settings persistence
# -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / ".qtbn.toml"

def load_persistent_settings():
    """Load settings from .qtbn.toml if available"""
    if toml is None:
        return
    if CONFIG_PATH.exists():
        try:
            settings = toml.load(CONFIG_PATH)
            for key, value in settings.get("settings", {}).items():
                ss_set(key, value)
            logger.info(f"Loaded {len(settings.get('settings', {}))} settings from {CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

def save_persistent_settings():
    """Save current settings to .qtbn.toml"""
    if toml is None:
        return False, "toml package not installed"
    try:
        settings_to_save = {}
        persistent_keys = [
            "num_qubits", "shots", "use_seed", "seed_val",
            "enable_dep", "enable_amp", "enable_phs", "enable_cnot_noise",
            "g0_q0", "a0_q0", "g0_q1", "a0_q1", "cnot0",
            "g1_q0", "a1_q0", "g1_q1", "a1_q1", "cnot1",
            "g2_q0", "a2_q0", "g2_q1", "a2_q1", "cnot2",
            "pdep0", "pdep1", "pdep2",
            "pamp0", "pamp1", "pamp2",
            "pph0", "pph1", "pph2",
            "pcnot0", "pcnot1", "pcnot2",
            "risk_free_rate", "confidence_level", "portfolio_value",
            "volatility_threshold", "tickers", "lookback_days", "var_horizon", "mc_sims",
            "fred_api_key", "macro_lookback_days", "apply_macro_stress",
            "lachesis_mode", "openai_api_key", "DEMO_MODE",
            "use_calibrated_noise", "sentiment_multiplier"
        ]
        for key in persistent_keys:
            if key in st.session_state:
                settings_to_save[key] = st.session_state[key]
        with open(CONFIG_PATH, 'w') as f:
            toml.dump({"settings": settings_to_save}, f)
        logger.info(f"Saved {len(settings_to_save)} settings to {CONFIG_PATH}")
        return True, f"Settings saved to {CONFIG_PATH.name}"
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return False, f"Save failed: {e}"

# -------------------------------------------------------------------
# Dependency checking
# -------------------------------------------------------------------
def check_dependencies():
    """Check availability of key dependencies"""
    deps = {
        "yfinance": yf is not None,
        "fredapi": Fred is not None,
        "openai": openai is not None,
        "cvxpy": cp is not None,
        "reportlab": SimpleDocTemplate is not None,
        "toml": toml is not None
    }
    return deps

# -------------------------------------------------------------------
# Calibration snapshot management
# -------------------------------------------------------------------
if "calibration_snapshots" not in st.session_state:
    st.session_state.calibration_snapshots = []
if "current_calibration" not in st.session_state:
    st.session_state.current_calibration = None
if "disk_scenarios" not in st.session_state:
    st.session_state.disk_scenarios = {}
if "custom_scenarios" not in st.session_state:
    st.session_state.custom_scenarios = {}
if "foresight_sweeps" not in st.session_state:
    st.session_state.foresight_sweeps = {}

def save_calibration_snapshot(params, confidence, seed):
    """Save calibration parameters as a snapshot"""
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot = {
        "timestamp": timestamp,
        "params": params.tolist() if isinstance(params, np.ndarray) else params,
        "confidence": float(confidence),
        "seed": seed
    }
    st.session_state.calibration_snapshots.append(snapshot)
    st.session_state.calibration_snapshots = st.session_state.calibration_snapshots[-10:]

# -------------------------------------------------------------------
# Quantum helper utilities
# -------------------------------------------------------------------
QC_REGISTRY: Dict[str, QuantumCircuit] = {}

def _register_qc(key: str, qc: QuantumCircuit):
    QC_REGISTRY[key] = qc

def _qc_key(qc: QuantumCircuit) -> str:
    key = hashlib.sha256(str(qc).encode()).hexdigest()
    _register_qc(key, qc)
    return key

def _to_qpy(qc: QuantumCircuit) -> bytes:
    return str(qc).encode()

@lru_cache(maxsize=8)
def get_simulator(method: str = "automatic", seed: int | None = None):
    """Create an AerSimulator with optional noise model from session toggles."""
    enable_dep = ss_get("enable_dep", False)
    enable_amp = ss_get("enable_amp", False)
    enable_phs = ss_get("enable_phs", False)
    enable_cnot_noise = ss_get("enable_cnot_noise", False)
    use_cal = ss_get("use_calibrated_noise", False)

    noise_model = None
    if enable_dep or enable_amp or enable_phs or enable_cnot_noise or use_cal:
        noise_model = NoiseModel()
        pdep = [ss_get(f"pdep{i}", 0.0) for i in range(3)]
        pamp = [ss_get(f"pamp{i}", 0.0) for i in range(3)]
        pphs = [ss_get(f"pph{i}", 0.0) for i in range(3)]
        pcnot = [ss_get(f"pcnot{i}", 0.0) for i in range(3)]

        if use_cal and st.session_state.current_calibration:
            try:
                p_cal, g_cal, l_cal = st.session_state.current_calibration["params"]
                pdep = [p_cal] * 3
                pamp = [g_cal] * 3
                pphs = [l_cal] * 3
            except Exception:
                pass

        nq = int(ss_get("num_qubits", 1) or 1)
        for step in range(3):
            if enable_dep and pdep[step] > 0:
                de = depolarizing_error(pdep[step], 1)
                for q in range(nq):
                    noise_model.add_all_qubit_quantum_error(de, ['id', 'x', 'rx', 'ry', 'rz', 'h'])
            if enable_amp and pamp[step] > 0:
                ae = amplitude_damping_error(pamp[step])
                for q in range(nq):
                    noise_model.add_all_qubit_quantum_error(ae, ['id', 'x', 'rx', 'ry', 'rz', 'h'])
            if enable_phs and pphs[step] > 0:
                pe = phase_damping_error(pphs[step])
                for q in range(nq):
                    noise_model.add_all_qubit_quantum_error(pe, ['id', 'x', 'rx', 'ry', 'rz', 'h'])
            if enable_cnot_noise and pcnot[step] > 0:
                cde = depolarizing_error(pcnot[step], 2)
                noise_model.add_all_qubit_quantum_error(cde, ['cx'])

    sim_kwargs = {}
    if noise_model is not None:
        sim_kwargs["noise_model"] = noise_model
    if method != "automatic":
        sim_kwargs["method"] = method

    sim = AerSimulator(**sim_kwargs)
    if seed is not None:
        sim.set_options(seed_simulator=seed)
    return sim

@lru_cache(maxsize=64)
def cached_transpile(backend_key: str, circ_key: str, _qpy_bytes: bytes, opt_level: int = 1):
    backend = get_simulator()
    qc = QC_REGISTRY.get(circ_key)
    if qc is None:
        qc = build_unitary_circuit()
    return transpile(qc, backend=backend, optimization_level=opt_level)

# --------- counts runner (hashable inputs only) ----------
@lru_cache(maxsize=64)
def run_counts_cached(backend_key: str, circ_key: str, _qpy_bytes: bytes, shots: int, seed: int | None):
    """
    Cached counts runner that ONLY takes hashable inputs.
    Looks up the circuit from QC_REGISTRY using circ_key, and transpiles via cached_transpile.
    """
    backend = get_simulator(seed=seed)
    tqc = cached_transpile(backend_key, circ_key, _qpy_bytes, 1)
    res = backend.run(tqc, shots=int(shots or 1024)).result()
    counts = res.get_counts(0)
    norm = {}
    for k, v in counts.items():
        key = k.replace(' ', '')
        norm[key] = int(v)
    return norm

def apply_gate(qc: QuantumCircuit, q: int, gate: str, angle: float = 0.0):
    g = (gate or "None").upper()
    if g == "H": qc.h(q)
    elif g == "X": qc.x(q)
    elif g == "Y": qc.y(q)
    elif g == "Z": qc.z(q)
    elif g == "RX": qc.rx(float(angle), q)
    elif g == "RY": qc.ry(float(angle), q)
    elif g == "RZ": qc.rz(float(angle), q)
    elif g == "S": qc.s(q)
    elif g == "T": qc.t(q)
    else: pass

def build_unitary_circuit() -> QuantumCircuit:
    nq = int(ss_get("num_qubits", 1) or 1)
    qc = QuantumCircuit(nq, nq)

    # Step 0
    apply_gate(qc, 0, ss_get("g0_q0", "H"), float(ss_get("a0_q0", 0.5) or 0.5) * math.pi)
    if nq > 1:
        apply_gate(qc, 1, ss_get("g0_q1", "None"), float(ss_get("a0_q1", 0.0) or 0.0) * math.pi)
    if ss_get("cnot0", False) and nq > 1:
        qc.cx(0, 1)
    qc.barrier()

    # Step 1
    apply_gate(qc, 0, ss_get("g1_q0", "None"), float(ss_get("a1_q0", 0.0) or 0.0) * math.pi)
    if nq > 1:
        apply_gate(qc, 1, ss_get("g1_q1", "None"), float(ss_get("a1_q1", 0.0) or 0.0) * math.pi)
    if ss_get("cnot1", False) and nq > 1:
        qc.cx(0, 1)
    qc.barrier()

    # Step 2
    apply_gate(qc, 0, ss_get("g2_q0", "None"), float(ss_get("a2_q0", 0.0) or 0.0) * math.pi)
    if nq > 1:
        apply_gate(qc, 1, ss_get("g2_q1", "None"), float(ss_get("a2_q1", 0.0) or 0.0) * math.pi)
    if ss_get("cnot2", False) and nq > 1:
        qc.cx(0, 1)
    qc.barrier()

    return qc

def build_measure_circuit_with_noise() -> QuantumCircuit:
    nq = int(ss_get("num_qubits", 1) or 1)
    qc = QuantumCircuit(nq, nq)
    qc.compose(build_unitary_circuit(), inplace=True)
    qc.barrier()
    qc.measure(range(nq), range(nq))
    return qc

def state_tomography_1q(base_qc: QuantumCircuit, shots: int = 4096, seed: Optional[int] = None) -> Tuple[float, float, float]:
    def measure_in_basis(prep: QuantumCircuit, basis: str):
        nq = 1
        qc = QuantumCircuit(nq, nq)
        qc.compose(prep, inplace=True)
        if basis == "X":
            qc.h(0)
        elif basis == "Y":
            qc.sdg(0); qc.h(0)
        qc.measure(0, 0)
        tqc = transpile(qc, get_simulator(seed=seed))
        res = get_simulator(seed=seed).run(tqc, shots=shots).result().get_counts(0)
        p0 = res.get('0', 0) / max(1, sum(res.values()))
        return 2 * p0 - 1

    ex = measure_in_basis(base_qc, "X")
    ey = measure_in_basis(base_qc, "Y")
    ez = measure_in_basis(base_qc, "Z")
    return float(ex), float(ey), float(ez)

def process_fidelity_basic(gate: str, angle: float = 1.57, shots: int = 4096, seed: Optional[int] = None) -> float:
    basis_preps = []
    p0 = QuantumCircuit(1); basis_preps.append(p0)
    p_plus = QuantumCircuit(1); p_plus.h(0); basis_preps.append(p_plus)
    p_iplus = QuantumCircuit(1); p_iplus.s(0); p_iplus.h(0); basis_preps.append(p_iplus)

    def apply_gate_choice(qc: QuantumCircuit):
        g = gate.upper()
        if g == "H": qc.h(0)
        elif g == "X": qc.x(0)
        elif g == "Z": qc.z(0)
        elif g == "RX": qc.rx(angle, 0)
        elif g == "RY": qc.ry(angle, 0)

    Fs = []
    for prep in basis_preps:
        ideal = QuantumCircuit(1)
        ideal.compose(prep, inplace=True); apply_gate_choice(ideal)
        noisy = ideal.copy()
        sv_ideal = Statevector.from_instruction(ideal)
        dm_sim = get_simulator(method="density_matrix", seed=seed)
        tqc = transpile(noisy, dm_sim)
        res = dm_sim.run(tqc).result()
        try:
            noisy_dm = DensityMatrix(res.data(0)['density_matrix'])
        except Exception:
            noisy_dm = DensityMatrix(res.get_density_matrix(0))
        Fs.append(state_fidelity(sv_ideal, noisy_dm))
    return float(np.mean(Fs))

def quantum_volume_proxy(qc: QuantumCircuit) -> Dict[str, float]:
    width = qc.num_qubits
    depth = 0
    for inst, qargs, cargs in qc.data:
        if inst.name not in ("measure", "barrier"):
            depth += 1
    depth = max(1, depth)
    qv_proxy = 2 ** min(width, depth)
    return {"width": float(width), "depth": float(depth), "QV_proxy": float(qv_proxy)}

def auto_calibrate_noise(shots: int = 4096) -> Tuple[Optional[np.ndarray], float]:
    try:
        qc = QuantumCircuit(1); qc.h(0)
        ex, ey, ez = state_tomography_1q(qc, shots=shots, seed=ss_get("seed_val", 17) if ss_get("use_seed", True) else None)
        p = max(0.0, min(0.2, (1.0 - abs(ex)) * 0.2))
        gamma = max(0.0, min(0.3, abs(ez) * 0.3))
        lamb = max(0.0, min(0.3, abs(ey) * 0.3))
        params = np.array([p, gamma, lamb], dtype=float)
        confidence = float(max(0.0, 1.0 - (abs(1.0 - abs(ex)) + abs(ey) + abs(ez)) / 3.0))
        save_calibration_snapshot(params, confidence, ss_get("seed_val", 17))
        return params, confidence
    except Exception as e:
        logger.error(f"Auto calibration failed: {e}")
        return None, 0.0

def tvdist(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys)

def kldiv(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p) | set(q)
    out = 0.0
    for k in keys:
        pk = max(float(p.get(k, 0.0)), eps)
        qk = max(float(q.get(k, 0.0)), eps)
        out += pk * math.log(pk / qk)
    return float(out)

SAMPLE_SCENARIOS = {
    "Balanced-1q": {"keys": ["0", "1"], "p": {"0": 0.5, "1": 0.5}, "note": "Uniform single-qubit"},
    "Bell-like-2q": {"keys": ["00", "01", "10", "11"], "p": {"00": 0.5, "11": 0.5, "01": 0.0, "10": 0.0}, "note": "Idealized Bell"},
    "Bias-1q": {"keys": ["0", "1"], "p": {"0": 0.7, "1": 0.3}, "note": "Biased outcome"},
    "Noise-robust": {"keys": ["00", "01", "10", "11"], "p": {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}, "note": "Max entropy"}
}

def load_disk_scenarios(path: Path = APP_DIR / "scenarios.json"):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load scenarios.json: {e}")
    return {}

def save_disk_scenarios(data: dict, path: Path = APP_DIR / "scenarios.json"):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True, f"Saved {len(data)} scenarios to {path.name}"
    except Exception as e:
        return False, f"Save failed: {e}"

def get_app_context() -> dict:
    return {
        "num_qubits": ss_get("num_qubits", 1),
        "noise": {
            "dep": ss_get("enable_dep", False),
            "amp": ss_get("enable_amp", False),
            "phs": ss_get("enable_phs", False),
            "cnot": ss_get("enable_cnot_noise", False),
        },
        "market": {
            "confidence": ss_get("confidence_level", 0.95),
            "vol_threshold": ss_get("volatility_threshold", 0.3),
        }
    }

def ask_lachesis(prompt: str, context: dict) -> str:
    # Placeholder / local fallback summary
    return (
        "Markets exhibit regime-dependent volatility. Your circuit settings control superposition and entanglement, "
        "while the noise toggles approximate realistic decoherence. Use the optimizer with turnover penalties and "
        "sector caps to achieve institutional-grade portfolios, and stress-test via macro inputs to assess tail risk."
    )

# -------------------------------------------------------------------
# Quantum depth additions
# -------------------------------------------------------------------
@dataclass
class BetaPosterior:
    alpha: float
    beta: float
    mean: float
    ci_low: float
    ci_high: float

def _beta_ci(a: float, b: float, q: float = 0.95):
    from scipy.stats import beta as beta_dist
    lo = beta_dist.ppf((1-q)/2, a, b)
    hi = beta_dist.ppf(1-(1-q)/2, a, b)
    return float(lo), float(hi)

def _estimate_channel_rate_from_experiment(kind: str, shots: int, seed: int | None) -> tuple[int, int]:
    shots = int(shots)
    seed = int(seed) if seed is not None else None
    sim = get_simulator(seed=seed)

    if kind == "depolarizing":
        qc = QuantumCircuit(1, 1)
        qc.h(0); qc.h(0)
        qc.measure(0, 0)
        basis_key = '0'
    elif kind == "amplitude":
        qc = QuantumCircuit(1, 1)
        qc.x(0); qc.barrier(); qc.measure(0, 0)
        basis_key = '0'
    elif kind == "phase":
        qc = QuantumCircuit(1, 1)
        qc.h(0); qc.barrier(); qc.h(0); qc.measure(0, 0)
        basis_key = '0'
    else:
        raise ValueError("Unknown kind")

    tqc = transpile(qc, sim)
    counts = sim.run(tqc, shots=shots).result().get_counts(0)
    succ = int(counts.get(basis_key, 0))
    fail = shots - succ
    return succ, fail

def bayesian_calibrate_noise(shots: int = 4096, priors: dict | None = None, seed: int | None = None, cred: float = 0.95):
    priors = priors or {"p": (1.0, 1.0), "gamma": (1.0, 1.0), "lambda": (1.0, 1.0)}

    succ_p, fail_p = _estimate_channel_rate_from_experiment("depolarizing", shots, seed)
    a_p0, b_p0 = priors["p"]; a_p = a_p0 + fail_p; b_p = b_p0 + succ_p
    mean_p = a_p / (a_p + b_p); lo_p, hi_p = _beta_ci(a_p, b_p, q=cred)

    succ_g, fail_g = _estimate_channel_rate_from_experiment("amplitude", shots, seed)
    a_g0, b_g0 = priors["gamma"]; a_g = a_g0 + succ_g; b_g = b_g0 + fail_g
    mean_g = a_g / (a_g + b_g); lo_g, hi_g = _beta_ci(a_g, b_g, q=cred)

    succ_l, fail_l = _estimate_channel_rate_from_experiment("phase", shots, seed)
    a_l0, b_l0 = priors["lambda"]; a_l = a_l0 + fail_l; b_l = b_l0 + succ_l
    mean_l = a_l / (a_l + b_l); lo_l, hi_l = _beta_ci(a_l, b_l, q=cred)

    post = {
        "p": BetaPosterior(a_p, b_p, float(mean_p), float(lo_p), float(hi_p)),
        "gamma": BetaPosterior(a_g, b_g, float(mean_g), float(lo_g), float(hi_g)),
        "lambda": BetaPosterior(a_l, b_l, float(mean_l), float(lo_l), float(hi_l)),
    }
    summary = (
        f"p ~ Beta({a_p:.1f},{b_p:.1f}) mean={mean_p:.4f} CI[{lo_p:.4f},{hi_p:.4f}] | "
        f"γ ~ Beta({a_g:.1f},{b_g:.1f}) mean={mean_g:.4f} CI[{lo_g:.4f},{hi_g:.4f}] | "
        f"λ ~ Beta({a_l:.1f},{b_l:.1f}) mean={mean_l:.4f} CI[{lo_l:.4f},{hi_l:.4f}]"
    )
    return post, summary

SINGLE_Q_CLIFFORD = [
    ["I"], ["X"], ["Y"], ["Z"], ["H"], ["S"], ["SDG"],
    ["HX"], ["HY"], ["HZ"], ["HS"], ["HSDG"],
    ["SX"], ["SY"], ["SZ"], ["SH"], ["S2"],
]

def _apply_named(qc: QuantumCircuit, name: str, q: int = 0):
    if name == "I": qc.id(q)
    elif name == "X": qc.x(q)
    elif name == "Y": qc.y(q)
    elif name == "Z": qc.z(q)
    elif name == "H": qc.h(q)
    elif name == "S": qc.s(q)
    elif name == "SDG": qc.sdg(q)
    elif name == "S2": qc.s(q); qc.s(q)
    elif name == "HX": qc.h(q); qc.x(q)
    elif name == "HY": qc.h(q); qc.y(q)
    elif name == "HZ": qc.h(q); qc.z(q)
    elif name == "HS": qc.h(q); qc.s(q)
    elif name == "HSDG": qc.h(q); qc.sdg(q)
    else: pass

def _compose_sequence(seq: list[str]) -> QuantumCircuit:
    qc = QuantumCircuit(1)
    for name in seq:
        _apply_named(qc, name, 0)
    return qc

def _inverse_sequence(seq: list[str]) -> list[str]:
    inv = []
    for name in reversed(seq):
        if name in ["I", "X", "Y", "Z", "H"]:
            inv.append(name)
        elif name == "S": inv.append("SDG")
        elif name == "SDG": inv.append("S")
        elif name == "S2": inv.append("S2")
        elif name == "HX": inv.append("X"); inv.append("H")
        elif name == "HY": inv.append("Y"); inv.append("H")
        elif name == "HZ": inv.append("Z"); inv.append("H")
        elif name == "HS": inv.append("SDG"); inv.append("H")
        elif name == "HSDG": inv.append("S"); inv.append("H")
        else: inv.append("I")
    return inv

def randomized_benchmarking_1q(lengths: list[int] = [2,4,8,16,32,48,64], nseeds: int = 16, shots: int = 4096, seed: int | None = None):
    rng = np.random.default_rng(seed or 123)
    sim = get_simulator(seed=seed)
    surv = []
    for m in lengths:
        probs = []
        for _ in range(nseeds):
            seq = []
            for _k in range(m):
                seq += rng.choice(SINGLE_Q_CLIFFORD)
            inv = _inverse_sequence(seq)
            qc = QuantumCircuit(1, 1)
            for name in seq: _apply_named(qc, name, 0)
            for name in inv: _apply_named(qc, name, 0)
            qc.measure(0, 0)
            tqc = transpile(qc, sim)
            counts = sim.run(tqc, shots=shots).result().get_counts(0)
            p0 = counts.get('0', 0) / max(1, sum(counts.values()))
            probs.append(p0)
        surv.append(float(np.mean(probs)))

    xs = np.array(lengths, float)
    ys = np.array(surv, float)

    def model(params, x):
        A, p, B = params
        return A * (p ** x) + B

    def resid(params):
        return model(params, xs) - ys

    p0 = np.array([0.5, 0.99, 0.5])
    fitted = least_squares(resid, p0, bounds=([0, 0.8, 0.0], [1.0, 1.0, 1.0]))
    A, p, B = fitted.x
    epg = max(0.0, min(1.0, (1 - p) / 2.0))
    return {"lengths": list(map(int, lengths)), "survival": [float(v) for v in surv],
            "fit": {"A": float(A), "p": float(p), "B": float(B)}, "EPG": float(epg)}

def process_tomography_proxy_1q(circuit_builder: callable, seed: int | None = None):
    out = {}
    prep_z = QuantumCircuit(1)
    prep_x = QuantumCircuit(1); prep_x.h(0)
    prep_y = QuantumCircuit(1); prep_y.s(0); prep_y.h(0)
    preps = {"Z": prep_z, "X": prep_x, "Y": prep_y}
    for name, prep in preps.items():
        qc = QuantumCircuit(1)
        qc.compose(prep, inplace=True)
        qc.compose(circuit_builder(), inplace=True)
        sv = Statevector.from_instruction(qc)
        dm = DensityMatrix(sv)
        Xp, Yp, Zp = Pauli("X"), Pauli("Y"), Pauli("Z")
        ex = float(np.real(np.trace(dm.data @ Xp.to_matrix())))
        ey = float(np.real(np.trace(dm.data @ Yp.to_matrix())))
        ez = float(np.real(np.trace(dm.data @ Zp.to_matrix())))
        out[name] = (ex, ey, ez)
    return out

def noise_aware_suggestions() -> list[str]:
    s = []
    p = float(ss_get("pdep1", ss_get("pdep0", 0.0)))
    g = float(ss_get("pamp1", ss_get("pamp0", 0.0)))
    l = float(ss_get("pph1", ss_get("pph0", 0.0)))
    if g > 0.05:
        s.append("High amplitude damping: avoid long excited-state dwell near end. Move RX at T2 earlier or replace with RY at T0.")
    if l > 0.05:
        s.append("Notable dephasing: prefer Z-commuting gates later; bunch phase-sensitive operations earlier.")
    if p > 0.02:
        s.append("Elevated depolarizing: reduce total depth and CNOT usage; consolidate single-qubit rotations where possible.")
    if ss_get("num_qubits", 1) > 1 and ss_get("cnot0", False) and ss_get("enable_cnot_noise", False):
        s.append("Two-qubit noise on CNOT: consider echo sequences or reducing entangling steps.")
    if not s:
        s.append("Current settings look balanced. Minor gains possible from gate reordering to reduce idle time.")
    return s

SCENARIO_LIBRARY = {
    "Bull Market (entangled)": {
        "num_qubits": 2,
        "g0_q0": "H", "a0_q0": 0.5, "g0_q1": "None", "a0_q1": 0.0, "cnot0": True,
        "g1_q0": "None", "a1_q0": 0.0, "g1_q1": "None", "a1_q1": 0.0, "cnot1": False,
        "g2_q0": "RZ", "a2_q0": 0.25, "g2_q1": "RY", "a2_q1": 0.25, "cnot2": False,
        "enable_dep": True, "enable_amp": False, "enable_phs": False, "enable_cnot_noise": True,
        "pdep0": 0.01, "pdep1": 0.015, "pdep2": 0.015,
        "pcnot0": 0.02, "pcnot1": 0.02, "pcnot2": 0.02
    },
    "Bear Market (dephasing)": {
        "num_qubits": 1,
        "g0_q0": "H", "a0_q0": 0.5, "g1_q0": "RZ", "a1_q0": 0.4, "g2_q0": "RZ", "a2_q0": 0.2,
        "enable_dep": False, "enable_amp": False, "enable_phs": True,
        "pph0": 0.04, "pph1": 0.05, "pph2": 0.05
    },
    "Crisis (high damping)": {
        "num_qubits": 1,
        "g0_q0": "X", "a0_q0": 0.0, "g1_q0": "RX", "a1_q0": 0.3, "g2_q0": "None", "a2_q0": 0.0,
        "enable_amp": True, "pamp0": 0.15, "pamp1": 0.20, "pamp2": 0.20
    }
}

def apply_scenario_preset(name: str) -> bool:
    cfg = SCENARIO_LIBRARY.get(name)
    if not cfg:
        return False
    for k, v in cfg.items():
        ss_set(k, v)
    return True

# -------------------------------------------------------------------
# Synthetic Data Generators (for Demo Mode and fallbacks)
# -------------------------------------------------------------------
def _synthetic_prices(tickers, days=365, seed=42):
    try:
        rng = np.random.default_rng(seed)
        dates = pd.date_range(end=dt.date.today(), periods=days)
        data = {}
        if len(tickers) == 1:
            start = 100 + 50 * rng.random()
            drift = 0.0005 * rng.choice([-1, 1]) + 0.001 * rng.random() * rng.choice([-1, 1])
            vol = 0.03 + 0.05 * rng.random()
            close = [start]
            for _ in range(days - 1):
                close.append(close[-1] * np.exp(drift + vol * rng.standard_normal()))
            open_prices = [start] + close[:-1]
            high = [max(o, c) + vol * close[-1] * rng.random() for o, c in zip(open_prices, close)]
            low = [min(o, c) - vol * close[-1] * rng.random() for o, c in zip(open_prices, close)]
            adj_close = close
            volume = [int(1e7 + 1e7 * rng.random()) for _ in range(days)]
            data = {
                'Open': open_prices,
                'High': high,
                'Low': low,
                'Close': close,
                'Adj Close': adj_close,
                'Volume': volume
            }
            return pd.DataFrame(data, index=dates)
        else:
            for t in tickers:
                start = 100 + 50 * rng.random()
                drift = 0.0005 * rng.choice([-1, 1]) + 0.001 * rng.random() * rng.choice([-1, 1])
                vol = 0.03 + 0.05 * rng.random()
                series = [start]
                for _ in range(days - 1):
                    series.append(series[-1] * np.exp(drift + vol * rng.standard_normal()))
                data[t] = series
            return pd.DataFrame(data, index=dates)
    except Exception as e:
        logger.error(f"Synthetic price generation failed: {e}")
        return None

def _synthetic_macro(days=365, seed=11):
    try:
        rng = np.random.default_rng(seed)
        dates = pd.date_range(end=dt.date.today(), periods=days)
        cpi = np.cumsum(0.02/252 + 0.001 * rng.normal(size=days))
        unemp = 4.5 + 0.7 * np.sin(np.linspace(0, 2*np.pi, days)) + 0.1 * rng.normal(size=days)
        ffr = 4.75 + 0.25 * np.sin(np.linspace(0, 2*np.pi, days)) + 0.05 * rng.normal(size=days)
        return pd.DataFrame({
            "CPIAUCSL": cpi,
            "UNRATE": unemp,
            "DGS10": ffr
        }, index=dates)
    except Exception as e:
        logger.error(f"Synthetic macro generation failed: {e}")
        return None

# -------------------------------------------------------------------
# Finance + Macro engine
# -------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def fetch_market_data(tickers: str, lookback_days: int) -> Optional[pd.DataFrame]:
    if ss_get("DEMO_MODE", False):
        tick_list = [t.strip() for t in tickers.split(",") if t.strip()]
        result = _synthetic_prices(tick_list, lookback_days)
        if result is not None:
            logger.info(f"Using synthetic data for demo mode with {len(tick_list)} tickers")
        return result

    if yf is None:
        error_msg = "yfinance is not installed. Install it with 'pip install yfinance' or use Demo Mode."
        logger.error(error_msg)
        st.error(error_msg)
        return None
    try:
        tick_list = [t.strip() for t in tickers.split(",") if t.strip()]
        if not tick_list:
            raise ValueError("No tickers provided.")
        logger.info(f"Fetching market data for {tick_list} with {lookback_days} days lookback")
        df = yf.download(tick_list, period=f"{lookback_days}d", progress=False)
        df = df.dropna(how="all").dropna(axis=0)
        logger.info(f"Successfully fetched market data: {df.shape}")
        return df
    except Exception as e:
        error_msg = f"Market data fetch failed: {e}"
        logger.error(error_msg, exc_info=True)
        st.error(f"{error_msg}. Falling back to synthetic data.")
        tick_list = [t.strip() for t in tickers.split(",") if t.strip()]
        result = _synthetic_prices(tick_list, lookback_days)
        if result is not None:
            logger.info("Using synthetic data as fallback")
        return result

@st.cache_data(ttl=3600)
def log_return_frame(data: pd.DataFrame) -> pd.DataFrame:
    try:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
        else:
            prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
        return np.log(prices / prices.shift(1)).dropna(how="any")
    except Exception as e:
        logger.error(f"Log return calculation failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def monte_carlo_var_cvar(
    data: pd.DataFrame, horizon_days: int, sims: int, alpha: float, use_quantum: bool = False
) -> Tuple[float, float]:
    """Simulate horizon-sum log returns for an equal-weighted basket -> VaR & CVaR (return space). With optional QAE."""
    try:
        rets = log_return_frame(data)
        if rets.empty:
            logger.warning("Empty returns DataFrame in VaR calculation")
            return float("nan"), float("nan")

        basket = rets.mean(axis=1)
        mu, sigma = basket.mean(), basket.std()

        if use_quantum and HAVE_QAE:
            # QAE branch (illustrative)
            mu_h = float(mu) * horizon_days
            sigma_h = float(sigma) * math.sqrt(horizon_days)
            num_qubits = 4
            bounds = [mu_h - 3*sigma_h, mu_h + 3*sigma_h]
            dist = NormalDistribution(num_qubits, mu=mu_h, sigma=sigma_h, bounds=bounds)
            from qiskit.circuit import QuantumCircuit
            A = QuantumCircuit(dist.num_qubits)
            A.compose(dist, inplace=True)
            objective_qubits = [dist.num_qubits - 1]
            problem = EstimationProblem(
                state_preparation=A,
                objective_qubits=objective_qubits,
                post_processing=lambda a: a
            )
            ae = IterativeAmplitudeEstimation(epsilon=0.02, alpha=0.05)
            result = ae.estimate(problem)
            p_tail = float(getattr(result, "estimation", 0.0))
            var = norm.ppf(1 - alpha, loc=mu_h, scale=sigma_h)
            cvar = mu_h - (sigma_h / alpha) * norm.pdf(norm.ppf(alpha))
            logger.info(f"QAE tail-prob proxy ≈ {p_tail:.4f}")
        else:
            # Classical MC
            progress_text = f"Running {sims:,} Monte Carlo simulations..."
            progress_bar = st.progress(0)
            chunk_size = min(10000, max(1, sims // 10))
            chunks = sims // chunk_size
            remainder = sims % chunk_size
            draws = np.array([])

            for i in range(chunks + 1):
                current_size = chunk_size if i < chunks else remainder
                if current_size == 0:
                    continue
                chunk_draws = np.random.normal(mu, sigma, size=(current_size, int(horizon_days))).sum(axis=1)
                draws = np.concatenate([draws, chunk_draws])
                progress = (i + 1) / (chunks + 1)
                progress_bar.progress(progress)

            var = np.percentile(draws, (1.0 - alpha) * 100.0)
            cvar = draws[draws <= var].mean() if np.isfinite(var) else float("nan")

        logger.info(f"Calculated VaR: {var:.4f}, CVaR: {cvar:.4f}")
        return float(var), float(cvar)
    except Exception as e:
        logger.error(f"VaR/CVaR calculation failed: {e}")
        return float("nan"), float("nan")
    finally:
        if not (use_quantum and HAVE_QAE):
            try:
                progress_bar.empty()
            except Exception:
                pass

def detect_regime(data: pd.DataFrame, ann_threshold: float) -> str:
    try:
        rets = log_return_frame(data)
        if rets.empty:
            return "Unknown"
        roll = rets.rolling(21).std().mean(axis=1)
        ann = roll * math.sqrt(252.0)
        last = float(ann.dropna().iloc[-1]) if not ann.dropna().empty else float("nan")
        if not np.isfinite(last):
            return "Unknown"
        if last > ann_threshold * 1.25:
            return "High Volatility"
        if last > ann_threshold * 0.75:
            return "Medium Volatility"
        return "Low Volatility"
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        return "Unknown"

def _fred_get_latest(f: "Fred", series_id: str) -> Optional[float]:
    try:
        s = f.get_series_latest_release(series_id)
        if s is None or len(s) == 0:
            return None
        return float(pd.Series(s).dropna().iloc[-1])
    except Exception:
        return None

def fetch_fred_bundle(api_key: str) -> Optional[Dict[str, float]]:
    if ss_get("DEMO_MODE", False):
        return {
            "CPI": 305.0 + 15 * np.random.random(),
            "Unemployment": 3.8 + 2.4 * np.random.random(),
            "10Y Yield": 3.9 + 0.9 * np.random.random()
        }
    if Fred is None or not api_key:
        return None
    try:
        fred = Fred(api_key=api_key)
    except Exception as e:
        logger.error(f"FRED initialization failed: {e}")
        return None

    try:
        bundle = {
            "CPI": _fred_get_latest(fred, "CPIAUCSL"),
            "Unemployment": _fred_get_latest(fred, "UNRATE"),
            "10Y Yield": _fred_get_latest(fred, "DGS10"),
        }
        if all(v is None for v in bundle.values()):
            return None
        return bundle
    except Exception as e:
        error_msg = f"FRED data fetch failed: {e}"
        logger.error(error_msg, exc_info=True)
        return None

def simulate_macro_bundle(data: pd.DataFrame) -> Dict[str, float]:
    rets = log_return_frame(data)
    vol = float(rets.std().mean()) if not rets.empty else 0.01
    ann_vol = vol * math.sqrt(252.0)
    if ann_vol < 0.18:
        regime = "calm"; unemp = 3.8; ten_y = 3.90; cpi = 305.0
    elif ann_vol < 0.30:
        regime = "medium"; unemp = 4.6; ten_y = 4.30; cpi = 312.0
    else:
        regime = "stressed"; unemp = 6.2; ten_y = 4.80; cpi = 320.0
    return {"CPI": cpi, "Unemployment": unemp, "10Y Yield": ten_y,
            "_regime_hint": regime, "_ann_vol": ann_vol}

def macro_stress_multiplier(macro: Dict[str, float]) -> float:
    u = float(macro.get("Unemployment") or 4.0)
    y10 = float(macro.get("10Y Yield") or 4.0)
    k1, k2 = 0.40, 0.20
    stress = 1.0 + k1 * ((u - 4.0) / 4.0) + k2 * ((y10 - 4.0) / 4.0)
    return float(np.clip(stress, 0.8, 1.8))

# -------------------------------------------------------------------
# Enhanced Financial Visualization & Metrics
# -------------------------------------------------------------------
def create_comprehensive_financial_charts(data: pd.DataFrame, returns: pd.DataFrame) -> "go.Figure | None":
    if px is None or go is None or make_subplots is None:
        st.warning("Plotly not available. Charts are disabled.")
        return None
    try:
        st.subheader("Comprehensive Financial Analysis")
        col1, col2 = st.columns(2)

        # Price history
        with col1:
            st.subheader("Price History")
            if not isinstance(data.columns, pd.MultiIndex) and all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True, vertical_spacing=0.03,
                    subplot_titles=('Price', 'Volume'),
                    row_heights=[0.7, 0.3]
                )
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='OHLC'
                ), row=1, col=1)
                if 'Volume' in data.columns:
                    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)
                fig.update_layout(xaxis_title='Date', yaxis_title='Price', height=400, showlegend=False)
                fig.update_xaxes(rangeslider_visible=False)
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
                else:
                    prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
                fig = go.Figure()
                for col in prices.columns:
                    fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode='lines', name=col))
                    ma = prices[col].rolling(50).mean()
                    fig.add_trace(go.Scatter(x=prices.index, y=ma, mode='lines', name=f'{col} 50-MA', line=dict(dash='dot')))
                fig.update_layout(xaxis_title='Date', yaxis_title='Price', height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Returns distribution
        with col2:
            if not returns.empty:
                returns_mean = returns.mean(axis=1)
                st.subheader("Returns Distribution")
                fig_dist = px.histogram(returns_mean, nbins=50) if px is not None else None
                if fig_dist:
                    x = np.linspace(returns_mean.min(), returns_mean.max(), 100)
                    fig_dist.add_trace(go.Scatter(
                        x=x, y=norm.pdf(x, returns_mean.mean(), returns_mean.std()),
                        mode='lines', name='Normal Fit'
                    ))
                    fig_dist.update_layout(height=300)
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.write(returns_mean.describe())

        # Rolling volatility
        if not returns.empty:
            rolling_vol = returns.rolling(21).std() * np.sqrt(252)
            st.subheader("21-Day Rolling Volatility (Annualized)")
            fig_vol = px.line(rolling_vol) if px is not None else None
            if fig_vol:
                fig_vol.add_hline(y=ss_get("volatility_threshold", 0.3), line_dash="dash")
                fig_vol.update_layout(xaxis_title='Date', yaxis_title='Value', height=300)
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.write(rolling_vol.describe())

        # Correlation
        if len(returns.columns) > 1:
            corr = returns.corr()
            st.subheader("Correlation Matrix")
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu') if px is not None else None
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.write(corr)

        return fig
    except Exception as e:
        logger.error(f"Financial chart creation failed: {e}")
        st.error(f"Chart error: {str(e)}")
        return None

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    try:
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    except:
        return float("nan")

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.04, target_return: float = 0.0) -> float:
    try:
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < target_return]
        downside_risk = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        return np.sqrt(252) * excess_returns.mean() / downside_risk if downside_risk != 0 else float("nan")
    except:
        return float("nan")

def calculate_max_drawdown(prices: pd.Series) -> float:
    try:
        cumulative_returns = prices / prices.iloc[0]
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    except:
        return float("nan")

def calculate_historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    try:
        return returns.quantile(alpha)
    except:
        return float("nan")

def calculate_expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    try:
        var = returns.quantile(alpha)
        return returns[returns <= var].mean()
    except:
        return float("nan")

def compute_advanced_financial_metrics(data: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, float]:
    try:
        if returns.empty:
            return {}
        basket_returns = returns.mean(axis=1)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
        else:
            prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
        metrics = {
            "sharpe_ratio": calculate_sharpe_ratio(basket_returns),
            "sortino_ratio": calculate_sortino_ratio(basket_returns),
            "max_drawdown": calculate_max_drawdown(prices.mean(axis=1)),
            "value_at_risk_historical": calculate_historical_var(basket_returns),
            "expected_shortfall": calculate_expected_shortfall(basket_returns),
            "skewness": basket_returns.skew(),
            "kurtosis": basket_returns.kurtosis()
        }
        return metrics
    except Exception as e:
        logger.error(f"Advanced metrics calculation failed: {e}")
        return {}

# -------------------------------------------------------------------
# Portfolio Optimizer with enhancements
# -------------------------------------------------------------------
def regime_aware_portfolio_optimizer(
    returns: pd.DataFrame,
    objective: str = "CVaR",
    long_only: bool = True,
    max_weight: float = 0.3,
    include_stress: bool = False,
    stress_factor: float = 1.0,
    num_scenarios: int = 1000,
    previous_weights: np.ndarray = None,
    turnover_penalty: float = 0.0,
    sector_mapping: Dict[str, str] = None,
    max_sector_weight: float = 0.5,
    use_regime_conditional: bool = False,
    views: Dict = None,
    alpha: float = None
):
    try:
        if cp is None:
            st.warning("cvxpy not available. Install with 'pip install cvxpy' for portfolio optimization.")
            return None
        n_assets = returns.shape[1]
        tickers = returns.columns.to_list()

        if use_regime_conditional and len(returns) >= 10:
            _, labels = kmeans2(returns.values, 3, minit='points')
            cov_matrices = []
            probs = []
            for regime in range(3):
                mask = (labels == regime)
                regime_rets = returns.iloc[mask]
                if len(regime_rets) > 1:
                    cov = regime_rets.cov().values
                    cov_matrices.append(cov)
                    probs.append(len(regime_rets) / len(returns))
                else:
                    cov_matrices.append(np.eye(n_assets))
                    probs.append(0)
            cov = sum(p * c for p, c in zip(probs, cov_matrices))
        else:
            cov = returns.cov().values

        mu = returns.mean().values

        # Black-Litterman style views (optional)
        if views:
            P = np.zeros((len(views), n_assets))
            Q = np.zeros(len(views))
            for i, (ticker, view) in enumerate(views.items()):
                if ticker in tickers:
                    P[i, tickers.index(ticker)] = 1.0
                    Q[i] = float(view)
            tau = 0.025
            Omega = np.eye(len(Q)) * (np.mean(np.diag(cov)) if np.isfinite(np.mean(np.diag(cov))) else 1.0) * tau
            mu_bl = mu + tau * cov @ P.T @ np.linalg.inv(P @ (tau * cov) @ P.T + Omega) @ (Q - P @ mu)
            mu = mu_bl

        if objective == "CVaR":
            n = len(returns)
            num_scenarios = int(num_scenarios or 1000)
            scenarios = np.zeros((num_scenarios, n_assets))
            for i in range(num_scenarios):
                sample_idx = np.random.choice(range(n), size=max(10, n // 4), replace=True)
                scenarios[i] = returns.iloc[sample_idx].mean()
            if include_stress:
                scenarios *= float(stress_factor)

        w = cp.Variable(n_assets)
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints += [w >= 0]
        constraints += [w <= max_weight]

        if sector_mapping:
            sectors = {}
            for i, t in enumerate(tickers):
                sector = sector_mapping.get(t, 'Other')
                sectors.setdefault(sector, []).append(i)
            for sector_indices in sectors.values():
                constraints += [cp.sum(w[sector_indices]) <= max_sector_weight]

        turnover = cp.norm1(w - previous_weights) if previous_weights is not None else 0

        if objective == "CVaR":
            alpha_val = float(alpha if alpha is not None else ss_get("confidence_level", 0.95))
            eta = cp.Variable()
            z = cp.Variable(num_scenarios, nonneg=True)
            risk = eta + (1 / ((1 - alpha_val) * num_scenarios)) * cp.sum(z)
            for i in range(num_scenarios):
                constraints += [z[i] >= -scenarios[i] @ w - eta]
            obj = -mu @ w + turnover_penalty * turnover + risk
            prob = cp.Problem(cp.Minimize(obj), constraints)
        else:
            risk = cp.quad_form(w, cov)
            obj = -mu @ w + turnover_penalty * turnover + risk
            prob = cp.Problem(cp.Minimize(obj), constraints)

        prob.solve()
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            st.warning("Optimization failed. Using equal weights.")
            return np.ones(n_assets) / n_assets
        return w.value
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        st.error(f"Optimization failed: {e}")
        return None

def plot_efficient_frontier(returns: pd.DataFrame, long_only: bool = True, max_weight: float = 0.3, num_points: int = 20):
    try:
        if cp is None:
            return None
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        ef_returns = []
        ef_risks = []
        min_ret = float(np.min(mu))
        max_ret = float(np.max(mu))
        target_returns = np.linspace(min_ret, max_ret, num_points)
        for target_ret in target_returns:
            w = cp.Variable(n_assets)
            constraints = [cp.sum(w) == 1, w <= max_weight]
            if long_only:
                constraints += [w >= 0]
            constraints += [mu @ w >= target_ret]
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints)
            prob.solve()
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                ef_returns.append(target_ret)
                wr = np.array(w.value).ravel()
                ef_risks.append(float(np.sqrt(wr @ cov @ wr)))
        fig = px.scatter(x=ef_risks, y=ef_returns, title='Efficient Frontier') if px is not None else None
        if fig:
            fig.update_layout(xaxis_title='Risk (Std Dev)', yaxis_title='Return')
        return fig
    except Exception as e:
        logger.error(f"Efficient frontier plot failed: {e}")
        return None

def stress_test_portfolio(returns: pd.DataFrame, weights: np.ndarray, stress_factor: float = 1.2) -> Dict[str, float]:
    try:
        portfolio_returns = returns @ weights
        var = np.percentile(portfolio_returns, 5)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        stressed_returns = portfolio_returns * stress_factor
        stressed_var = np.percentile(stressed_returns, 5)
        stressed_cvar = stressed_returns[stressed_returns <= stressed_var].mean()
        return {
            "base_var": float(var), "base_cvar": float(cvar),
            "stressed_var": float(stressed_var), "stressed_cvar": float(stressed_cvar)
        }
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        return {}

# -------------------------------------------------------------------
# Backtest and rebalance simulator
# -------------------------------------------------------------------
def backtest_rebalance(prices: pd.DataFrame, benchmark_ticker: str = "SPY", slippage: float = 0.001, fees: float = 0.0005, rebalance_freq: int = 1):
    try:
        benchmark_prices = fetch_market_data(benchmark_ticker, len(prices))
        if benchmark_prices is None:
            st.warning("Failed to fetch benchmark data. Backtest without benchmark.")
            benchmark_returns = pd.Series(0, index=prices.index)
        else:
            benchmark_returns = log_return_frame(benchmark_prices).mean(axis=1)

        monthly_prices = prices.resample('ME').last()
        monthly_returns = log_return_frame(monthly_prices)

        portfolio_value = 1.0
        portfolio_values = [portfolio_value]
        rolling_var = []
        drawdowns = []
        previous_w = np.ones(len(prices.columns)) / len(prices.columns)

        for i in range(rebalance_freq, len(monthly_returns), rebalance_freq):
            historical_returns = monthly_returns.iloc[max(0, i-12):i]
            w = regime_aware_portfolio_optimizer(historical_returns, previous_weights=previous_w, alpha=ss_get("confidence_level", 0.95))
            if w is None:
                w = previous_w

            turnover = np.sum(np.abs(w - previous_w))
            cost = turnover * (slippage + fees)
            portfolio_value *= (1 - cost)

            forward_returns = monthly_returns.iloc[i:i+rebalance_freq]
            period_return = (forward_returns @ w).sum()
            portfolio_value *= np.exp(period_return)
            portfolio_values.append(portfolio_value)

            if len(portfolio_values) > 1:
                port_returns = np.diff(np.log(portfolio_values))
                rv = np.percentile(port_returns[-min(12, len(port_returns)):], 5)
                rolling_var.append(rv)
                current_dd = 1 - portfolio_values[-1] / max(portfolio_values)
                drawdowns.append(current_dd)

            previous_w = w

        fig = px.line(title='Backtest Performance') if px is not None else None
        if fig:
            fig.add_trace(go.Scatter(x=monthly_prices.index[::rebalance_freq][:len(portfolio_values)], y=portfolio_values,
                                     mode='lines', name='Portfolio'))
            bench = benchmark_returns.resample('ME').sum()
            if len(bench) >= len(portfolio_values):
                benchmark_value = np.cumprod(np.exp(bench))[:len(portfolio_values)]
                fig.add_trace(go.Scatter(
                    x=monthly_prices.index[::rebalance_freq][:len(portfolio_values)],
                    y=benchmark_value / benchmark_value[0],
                    mode='lines', name='Benchmark'
                ))
            fig.update_layout(yaxis_title='Normalized Value')
        return fig, rolling_var, drawdowns
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return None, [], []

# -------------------------------------------------------------------
# Executive Report Generation
# -------------------------------------------------------------------
def generate_executive_report(theme: str = "light"):
    try:
        if SimpleDocTemplate is None:
            st.warning("reportlab not available. Install with 'pip install reportlab' for report export.")
            return None

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        flowables = []

        flowables.append(Paragraph("Quantum Finance Executive Report", styles['Title']))
        flowables.append(Spacer(1, 0.2*inch))

        circuit_ascii = str(unitary_qc.draw(output="text"))
        flowables.append(Paragraph("Circuit Snapshot", styles['Heading2']))
        flowables.append(Preformatted(circuit_ascii, styles['Code']))
        flowables.append(Spacer(1, 0.2*inch))

        ideal_sv = Statevector.from_instruction(unitary_qc); ideal_dm = DensityMatrix(ideal_sv)
        noisy_unitary = build_unitary_circuit()
        sim_dm = get_simulator(method="density_matrix")
        noisy_unitary.save_density_matrix()
        tqc = transpile(noisy_unitary, sim_dm)
        res = sim_dm.run(tqc).result()
        try:
            noisy_dm = DensityMatrix(res.data(0)['density_matrix'])
        except Exception:
            noisy_dm = DensityMatrix(res.get_density_matrix(0))
        F_global = state_fidelity(ideal_dm, noisy_dm)
        flowables.append(Paragraph(f"Global Fidelity: {F_global:.4f}", styles['Normal']))
        flowables.append(Spacer(1, 0.2*inch))

        data = ss_get("market_data")
        if data is not None:
            regime = detect_regime(data, ss_get("volatility_threshold", 0.3))
            flowables.append(Paragraph(f"Market Regime: {regime}", styles['Normal']))
            flowables.append(Spacer(1, 0.2*inch))

            alpha = ss_get("confidence_level", 0.95)
            var_h = ss_get("var_horizon", 10)
            sims = ss_get("mc_sims", 50000)
            var_r, cvar_r = monte_carlo_var_cvar(data, var_h, sims, alpha, use_quantum=False)
            flowables.append(Paragraph(f"VaR: {var_r:.4f}", styles['Normal']))
            flowables.append(Paragraph(f"CVaR: {cvar_r:.4f}", styles['Normal']))
            flowables.append(Spacer(1, 0.2*inch))

            macro = ss_get("macro_bundle")
            if macro:
                stress = macro_stress_multiplier(macro)
                svar = var_r * stress
                scvar = cvar_r * stress
                flowables.append(Paragraph(f"Stressed VaR: {svar:.4f}", styles['Normal']))
                flowables.append(Paragraph(f"Stressed CVaR: {scvar:.4f}", styles['Normal']))
                flowables.append(Spacer(1, 0.2*inch))

        if data is not None:
            returns = log_return_frame(data)
            chart_fig = create_comprehensive_financial_charts(data, returns)
            if chart_fig:
                chart_buffer = io.BytesIO()
                try:
                    chart_fig.write_image(chart_buffer, format="png")
                    chart_buffer.seek(0)
                    flowables.append(Image(chart_buffer, width=6*inch, height=4*inch))
                    flowables.append(Spacer(1, 0.2*inch))
                except Exception:
                    pass

        context = get_app_context()
        explanation = ask_lachesis("Provide a short executive summary of the current simulation results.", context)
        flowables.append(Paragraph("Lachesis Explanation", styles['Heading2']))
        flowables.append(Paragraph(explanation, styles['Normal']))

        doc.build(flowables)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        st.error(f"Report generation failed: {e}")
        return None

# -------------------------------------------------------------------
# Sentiment Analysis Function
# -------------------------------------------------------------------
def analyze_sentiment(tickers: List[str]) -> Dict:
    if feedparser is None or SentimentIntensityAnalyzer is None:
        return {"error": "Sentiment analysis dependencies not available.", "multiplier": 1.0}
    sia = SentimentIntensityAnalyzer()
    headlines = []
    for ticker in tickers:
        url = f"https://news.google.com/rss/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            headlines.append(entry.title)
    if not headlines:
        return {"error": "No headlines found.", "multiplier": 1.0}
    scores = [sia.polarity_scores(h)['compound'] for h in headlines]
    avg_score = sum(scores) / len(scores)
    multiplier = 1 - 0.5 * avg_score
    multiplier = max(0.5, min(1.5, multiplier))
    return {
        "headlines": headlines,
        "scores": scores,
        "avg_score": avg_score,
        "multiplier": multiplier
    }

# -------------------------------------------------------------------
# Sidebar (Quantum controls + Market data)
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("Quantum")
    st.number_input("Qubits", 1, 2, key="num_qubits")
    st.number_input("Shots", 128, 200000, key="shots", step=128)
    st.checkbox("Use fixed seed", key="use_seed")
    st.number_input("Seed value", 0, 10_000, key="seed_val")

    st.markdown("**Gates (T0/T1/T2)**")
    gate_choices = ["None", "H", "X", "Y", "Z", "RX", "RY", "RZ", "S", "T"]

    # T0
    st.selectbox("T0 q0 gate", gate_choices, index=gate_choices.index(ss_get("g0_q0", "H")), key="g0_q0")
    st.slider("T0 q0 angle (π)", 0.0, 1.0, float(ss_get("a0_q0", 0.5)), 0.01, key="a0_q0")
    if ss_get("num_qubits", 1) > 1:
        st.selectbox("T0 q1 gate", gate_choices, index=gate_choices.index(ss_get("g0_q1", "None")), key="g0_q1")
        st.slider("T0 q1 angle (π)", 0.0, 1.0, float(ss_get("a0_q1", 0.0)), 0.01, key="a0_q1")
        st.checkbox("T0 CX(0,1)", key="cnot0")

    # T1
    st.selectbox("T1 q0 gate", gate_choices, index=gate_choices.index(ss_get("g1_q0", "None")), key="g1_q0")
    st.slider("T1 q0 angle (π)", 0.0, 1.0, float(ss_get("a1_q0", 0.0)), 0.01, key="a1_q0")
    if ss_get("num_qubits", 1) > 1:
        st.selectbox("T1 q1 gate", gate_choices, index=gate_choices.index(ss_get("g1_q1", "None")), key="g1_q1")
        st.slider("T1 q1 angle (π)", 0.0, 1.0, float(ss_get("a1_q1", 0.0)), 0.01, key="a1_q1")
        st.checkbox("T1 CX(0,1)", key="cnot1")

    # T2
    st.selectbox("T2 q0 gate", gate_choices, index=gate_choices.index(ss_get("g2_q0", "None")), key="g2_q0")
    st.slider("T2 q0 angle (π)", 0.0, 1.0, float(ss_get("a2_q0", 0.0)), 0.01, key="a2_q0")
    if ss_get("num_qubits", 1) > 1:
        st.selectbox("T2 q1 gate", gate_choices, index=gate_choices.index(ss_get("g2_q1", "None")), key="g2_q1")
        st.slider("T2 q1 angle (π)", 0.0, 1.0, float(ss_get("a2_q1", 0.0)), 0.01, key="a2_q1")
        st.checkbox("T2 CX(0,1)", key="cnot2")

    st.markdown("---")
    st.subheader("Noise")
    st.checkbox("Enable depolarizing", key="enable_dep")
    st.checkbox("Enable amplitude damping", key="enable_amp")
    st.checkbox("Enable phase damping", key="enable_phs")
    st.checkbox("Enable CNOT depolarizing", key="enable_cnot_noise")

    cdep0, cdep1, cdep2 = st.columns(3)
    with cdep0: st.number_input("p_dep T0", 0.0, 1.0, key="pdep0", step=0.005, format="%.3f")
    with cdep1: st.number_input("p_dep T1", 0.0, 1.0, key="pdep1", step=0.005, format="%.3f")
    with cdep2: st.number_input("p_dep T2", 0.0, 1.0, key="pdep2", step=0.005, format="%.3f")

    camp0, camp1, camp2 = st.columns(3)
    with camp0: st.number_input("γ T0", 0.0, 1.0, key="pamp0", step=0.005, format="%.3f")
    with camp1: st.number_input("γ T1", 0.0, 1.0, key="pamp1", step=0.005, format="%.3f")
    with camp2: st.number_input("γ T2", 0.0, 1.0, key="pamp2", step=0.005, format="%.3f")

    cph0, cph1, cph2 = st.columns(3)
    with cph0: st.number_input("λ T0", 0.0, 1.0, key="pph0", step=0.005, format="%.3f")
    with cph1: st.number_input("λ T1", 0.0, 1.0, key="pph1", step=0.005, format="%.3f")
    with cph2: st.number_input("λ T2", 0.0, 1.0, key="pph2", step=0.005, format="%.3f")

    if ss_get("num_qubits", 1) > 1:
        cc0, cc1, cc2 = st.columns(3)
        with cc0: st.number_input("p_CX T0", 0.0, 1.0, key="pcnot0", step=0.005, format="%.3f")
        with cc1: st.number_input("p_CX T1", 0.0, 1.0, key="pcnot1", step=0.005, format="%.3f")
        with cc2: st.number_input("p_CX T2", 0.0, 1.0, key="pcnot2", step=0.005, format="%.3f")

    st.checkbox("Use last calibration as noise", key="use_calibrated_noise")
    if st.button("Auto-calibrate from quick RB/Tomo"):
        params, conf = auto_calibrate_noise()
        if params is not None:
            st.success(f"Calibrated: p={params[0]:.3f}, γ={params[1]:.3f}, λ={params[2]:.3f} (conf {conf:.2f})")
            st.session_state.current_calibration = {"params": params, "confidence": conf}
            ss_set("use_calibrated_noise", True)

    st.markdown("---")
    st.subheader("Finance / Data")
    st.text_input("Tickers (comma)", key="tickers")
    st.number_input("Lookback days", 30, 2000, key="lookback_days")
    if st.button("Fetch Market Data"):
        data = fetch_market_data(st.session_state.tickers, st.session_state.lookback_days)
        if data is not None:
            st.session_state.market_data = data
            st.success("Market data ready.")

    st.number_input("Portfolio value ($)", 1_000, 10_000_000_000, key="portfolio_value", step=10_000)
    st.slider("Confidence (alpha)", 0.80, 0.99, key="confidence_level")
    st.number_input("Horizon (days)", 1, 60, key="var_horizon")
    st.number_input("MC simulations", 1000, 1_000_000, key="mc_sims", step=1000)
    st.slider("Volatility threshold (ann.)", 0.05, 1.0, key="volatility_threshold")
    st.checkbox("Apply macro stress", key="apply_macro_stress")
    st.text_input("FRED API Key (optional)", key="fred_api_key")
    st.checkbox("Demo Mode (synthetic data, safer fallbacks)", key="DEMO_MODE")

    st.markdown("---")
    st.subheader("Persistence")
    if st.button("Save settings"):
        ok, msg = save_persistent_settings()
        st.info(msg)
    if st.button("Load settings"):
        load_persistent_settings()
        st.success("Loaded .qtbn.toml (if present).")

# -------------------------------------------------------------------
# Header + Tabs
# -------------------------------------------------------------------
DEMO_MODE = ss_get("DEMO_MODE", False)
unitary_qc = build_unitary_circuit()

st.text("Circuit (ASCII, no measurement)")
st.code(str(unitary_qc.draw(output="text")), language="text")
st.caption("Legend: q[i] quantum wires; c[i] classical; barriers enforce step order; measures map q→c.")

tab_sv, tab_red, tab_meas, tab_fid, tab_presets, tab_present, tab_fx, tab_fin, tab_guide, tab_advanced_q, tab_sentiment, tab_prompt = st.tabs(
    [
        "Statevector", "Reduced States", "Measurement", "Fidelity & Export",
        "Presets", "Present Scenarios", "Foresight", "Financial Analysis",
        "Lachesis Guide", "Advanced Quantum", "Sentiment Analysis", "Prompt Studio"
    ]
)

# -- Statevector
with tab_sv:
    st.text("Statevector (before measurement)")
    if st.button("Run Statevector") or (DEMO_MODE and st.session_state.get("preloaded_demo")):
        try:
            sv = Statevector.from_instruction(unitary_qc).data
            if st.session_state.num_qubits == 1:
                alpha_, beta_ = complex(sv[0]), complex(sv[1])
                p0 = alpha_.real**2 + alpha_.imag**2
                p1 = beta_.real**2 + beta_.imag**2
                st.write({"|0>": round(p0, 6), "|1>": round(p1, 6)})
            else:
                probs = []
                labels = [f"|{i:0{st.session_state.num_qubits}b}>" for i in range(2**st.session_state.num_qubits)]
                for i, _lab in enumerate(labels):
                    amp = complex(sv[i]); probs.append(amp.real**2 + amp.imag**2)
                fig_prob = px.bar(y=probs, title="State Probabilities") if px is not None else None
                if fig_prob:
                    st.plotly_chart(fig_prob)
                else:
                    st.write(probs)
        except Exception as e:
            st.error(f"Statevector failed: {e}")

# -- Reduced states (2q)
with tab_red:
    if st.session_state.num_qubits == 1:
        st.info("Switch to 2 qubits to view reduced states.")
    else:
        if st.button("Compute Reduced States") or (DEMO_MODE and st.session_state.get("preloaded_demo")):
            try:
                noisy_unitary = build_unitary_circuit()
                sim_dm = get_simulator(
                    method="density_matrix",
                    seed=st.session_state.seed_val if st.session_state.use_seed else None
                )
                noisy_unitary.save_density_matrix()
                tqc = transpile(noisy_unitary, sim_dm)
                res = sim_dm.run(tqc).result()
                try:
                    noisy_dm = DensityMatrix(res.data(0)['density_matrix'])
                except Exception:
                    noisy_dm = DensityMatrix(res.get_density_matrix(0))

                rho_q0 = partial_trace(noisy_dm, [1])
                rho_q1 = partial_trace(noisy_dm, [0])
                Xp, Yp, Zp = Pauli("X"), Pauli("Y"), Pauli("Z")

                def bloch(rho):
                    ex = float(np.real(np.trace(rho.data @ Xp.to_matrix())))
                    ey = float(np.real(np.trace(rho.data @ Yp.to_matrix())))
                    ez = float(np.real(np.trace(rho.data @ Zp.to_matrix())))
                    return ex, ey, ez

                b0 = bloch(rho_q0)
                b1 = bloch(rho_q1)
                st.write({"q0 Bloch": b0, "q1 Bloch": b1})

                fig_bloch = go.Figure() if go is not None else None
                if fig_bloch:
                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                    x = np.cos(u)*np.sin(v)
                    y = np.sin(u)*np.sin(v)
                    z = np.cos(v)
                    fig_bloch.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.2, name='q0 Sphere'))
                    fig_bloch.add_trace(go.Scatter3d(
                        x=[0, b0[0]], y=[0, b0[1]], z=[0, b0[2]],
                        mode='lines', name='q0 Vector'
                    ))
                    fig_bloch.add_trace(go.Surface(x=x+2, y=y, z=z, opacity=0.2, name='q1 Sphere'))
                    fig_bloch.add_trace(go.Scatter3d(
                        x=[2, 2+b1[0]], y=[0, b1[1]], z=[0, b1[2]],
                        mode='lines', name='q1 Vector'
                    ))
                    fig_bloch.update_layout(title='Bloch Spheres',
                        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
                    st.plotly_chart(fig_bloch)
                else:
                    st.write("Bloch spheres visualization unavailable (plotly not installed)")
            except Exception as e:
                st.error(f"Reduced states failed: {e}")

# -- Measurement
with tab_meas:
    st.text("Measurement (counts) — ideal vs noisy")
    if st.button("Run Counts") or (DEMO_MODE and st.session_state.get("preloaded_demo")):
        try:
            nq = st.session_state.num_qubits
            qc_i = QuantumCircuit(nq, nq)
            qc_i.compose(build_unitary_circuit(), inplace=True)
            qc_i.measure(range(nq), range(nq))

            qc_n = build_measure_circuit_with_noise()

            key_i = _qc_key(qc_i); key_n = _qc_key(qc_n)
            _ = cached_transpile("default", key_i, _to_qpy(qc_i), 1)
            _ = cached_transpile("default", key_n, _to_qpy(qc_n), 1)
            seed = st.session_state.seed_val if st.session_state.use_seed else None

            counts_i = run_counts_cached("default", key_i, _to_qpy(qc_i), st.session_state.shots, seed)
            counts_n = run_counts_cached("default", key_n, _to_qpy(qc_n), st.session_state.shots, seed)

            keys = ["0", "1"] if st.session_state.num_qubits == 1 else [f"{i:0{nq}b}" for i in range(2**nq)]
            N1 = sum(counts_i.values()) or 1
            N2 = sum(counts_n.values()) or 1
            pi = {k: counts_i.get(k, 0)/N1 for k in keys}
            pn = {k: counts_n.get(k, 0)/N2 for k in keys}
            tv = tvdist(pi, pn)
            st.text(f"Robustness (1 − TV): {max(0.0, 1.0-tv):.4f}")

            fig_counts = go.Figure() if go is not None else None
            if fig_counts:
                fig_counts.add_trace(go.Bar(x=keys, y=[counts_i.get(k, 0) for k in keys], name="Ideal"))
                fig_counts.add_trace(go.Bar(x=keys, y=[counts_n.get(k, 0) for k in keys], name="Noisy"))
                fig_counts.update_layout(title="Measurement Counts", barmode='group')
                st.plotly_chart(fig_counts)
            else:
                st.write("Ideal counts:", counts_i)
                st.write("Noisy counts:", counts_n)
        except Exception as e:
            st.error(f"Counts run failed: {e}")

# -- Fidelity
with tab_fid:
    st.text("Global fidelity (ideal vs noisy)")
    if st.button("Compute Fidelity") or (DEMO_MODE and st.session_state.get("preloaded_demo")):
        try:
            ideal_sv = Statevector.from_instruction(build_unitary_circuit()); ideal_dm = DensityMatrix(ideal_sv)
            noisy_unitary = build_unitary_circuit()
            sim_dm = get_simulator(
                method="density_matrix",
                seed=st.session_state.seed_val if st.session_state.use_seed else None
            )
            noisy_unitary.save_density_matrix()
            tqc = transpile(noisy_unitary, sim_dm)
            res = sim_dm.run(tqc).result()
            try:
                noisy_dm = DensityMatrix(res.data(0)['density_matrix'])
            except Exception:
                noisy_dm = DensityMatrix(res.get_density_matrix(0))
            F_global = state_fidelity(ideal_dm, noisy_dm)
            st.text(f"Global fidelity: {F_global:.6f}")
            st.info("Neurosymbolic Explanation: Quantum sampling provides parallel exploration, "
                    "reducing computation time compared to classical methods for large state spaces.")
        except Exception as e:
            st.error(f"Fidelity failed: {e}")
    st.markdown("---")
    if st.button("Download Executive Report PDF"):
        pdf_bytes = generate_executive_report()
        if pdf_bytes:
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name="qtbn_executive_report.pdf",
                mime="application/pdf"
            )

# -- Presets
def apply_preset_bell():
    ss_set("num_qubits", 2)
    ss_set("g0_q0", "H"); ss_set("a0_q0", 0.5)
    ss_set("g0_q1", "None"); ss_set("a0_q1", 0.0); ss_set("cnot0", True)
    ss_set("g1_q0", "None"); ss_set("a1_q0", 0.0)
    ss_set("g1_q1", "None"); ss_set("a1_q1", 0.0); ss_set("cnot1", False)
    ss_set("g2_q0", "None"); ss_set("a2_q0", 0.0)
    ss_set("g2_q1", "None"); ss_set("a2_q1", 0.0); ss_set("cnot2", False)
    ss_set("enable_dep", True); ss_set("enable_amp", False)
    ss_set("enable_phs", False); ss_set("enable_cnot_noise", True)
    ss_set("pdep0", 0.01); ss_set("pdep1", 0.02); ss_set("pdep2", 0.02)
    ss_set("pcnot0", 0.02); st.rerun()

def apply_preset_dephase():
    ss_set("num_qubits", 1)
    ss_set("g0_q0", "H"); ss_set("a0_q0", 0.5)
    ss_set("g1_q0", "None"); ss_set("a1_q0", 0.0)
    ss_set("g2_q0", "None"); ss_set("a2_q0", 0.0)
    ss_set("enable_dep", False); ss_set("enable_amp", False); ss_set("enable_phs", True)
    ss_set("pdep0", 0.00); ss_set("pdep1", 0.00); ss_set("pdep2", 0.00); st.rerun()

def apply_preset_amp():
    ss_set("num_qubits", 1)
    ss_set("g0_q0", "X"); ss_set("a0_q0", 0.0)
    ss_set("enable_amp", True); ss_set("pamp1", 0.20); ss_set("pamp2", 0.20); st.rerun()

with tab_presets:
    c1, c2, c3 = st.columns(3)
    c1.button("Bell prep (H→CX)", on_click=apply_preset_bell)
    c2.button("Dephasing stress", on_click=apply_preset_dephase)
    c3.button("Amplitude relaxation", on_click=apply_preset_amp)

    st.markdown("---")
    st.subheader("Scenario → Circuit")
    scn = st.selectbox("Select scenario", list(SCENARIO_LIBRARY.keys()))
    if st.button("Apply Scenario"):
        if apply_scenario_preset(scn):
            st.success(f"Applied scenario: {scn}")
            st.rerun()
        else:
            st.error("Scenario not found.")

# -- Present Scenarios
with tab_present:
    st.caption("Quick check: build & compare ideal vs noisy distributions with TV robustness.")
    if st.button("Analyze current scenario") or (DEMO_MODE and st.session_state.get("preloaded_demo")):
        try:
            nq = st.session_state.num_qubits
            qc_i = QuantumCircuit(nq, nq)
            qc_i.compose(build_unitary_circuit(), inplace=True)
            qc_i.measure(range(nq), range(nq))
            qc_n = build_measure_circuit_with_noise()
            key_i = _qc_key(qc_i); key_n = _qc_key(qc_n)
            _ = cached_transpile("default", key_i, _to_qpy(qc_i), 1)
            _ = cached_transpile("default", key_n, _to_qpy(qc_n), 1)
            seed = st.session_state.seed_val if st.session_state.use_seed else None
            ri = run_counts_cached("default", key_i, _to_qpy(qc_i), st.session_state.shots, seed)
            rn = run_counts_cached("default", key_n, _to_qpy(qc_n), st.session_state.shots, seed)

            keys = ["0", "1"] if st.session_state.num_qubits == 1 else ["00", "01", "10", "11"]
            N1 = sum(ri.values()) or 1; N2 = sum(rn.values()) or 1
            pi = {k: ri.get(k, 0)/N1 for k in keys}
            pn = {k: rn.get(k, 0)/N2 for k in keys}
            tv = tvdist(pi, pn)
            st.text(f"Robustness (1 − TV): {max(0.0, 1.0-tv):.4f}")

            fig_counts = go.Figure() if go is not None else None
            if fig_counts:
                fig_counts.add_trace(go.Bar(x=keys, y=[ri.get(k, 0) for k in keys], name="Ideal"))
                fig_counts.add_trace(go.Bar(x=keys, y=[rn.get(k, 0) for k in keys], name="Noisy"))
                fig_counts.update_layout(title="Measurement Counts", barmode='group')
                st.plotly_chart(fig_counts)
            else:
                st.write("Ideal counts:", ri)
                st.write("Noisy counts:", rn)
        except Exception as e:
            st.error(f"Present scenarios failed: {e}")

# -- Foresight (sweeps manager)
with tab_fx:
    st.subheader("Analytical Foresight — scenario ranking + sweeps")
    keys = ["0", "1"] if st.session_state.num_qubits == 1 else ["00", "01", "10", "11"]

    def combined_scenarios_for_keys(keys_):
        out = {k: {**v} for k, v in SAMPLE_SCENARIOS.items() if v["keys"] == keys_}
        for name, meta in st.session_state.get("disk_scenarios", {}).items():
            if meta.get("keys") == keys_:
                out[name] = {**meta}
        for name, meta in st.session_state.get("custom_scenarios", {}).items():
            if meta.get("keys") == keys_:
                out[name] = {**meta}
        for name, meta in out.items():
            if "impact" not in meta:
                meta["impact"] = 1.0
        return out

    ACTIVE_SCENARIOS = combined_scenarios_for_keys(keys)

    st.markdown("### Current baseline (multi-seed)")
    seeds_str = st.text_input("Seeds (comma)", "11,17,29")
    base_seeds = [int(s.strip()) for s in seeds_str.split(",") if s.strip().isdigit()]
    baseline_shots = int(min(1024, max(256, st.session_state.shots // 2 or 256)))

    def run_noisy_p(seed):
        qc_n = build_measure_circuit_with_noise()
        key_n = _qc_key(qc_n)
        _ = cached_transpile("default", key_n, _to_qpy(qc_n), 1)
        counts = run_counts_cached("default", key_n, _to_qpy(qc_n), baseline_shots, seed)
        N = sum(counts.values()) or 1
        return {k: counts.get(k, 0)/N for k in keys}

    try:
        samples = [run_noisy_p(sd) for sd in (base_seeds or [None])]
        current_p = {k: float(np.mean([s.get(k, 0.0) for s in samples])) for k in keys}
        stds = {k: float(np.std([s.get(k, 0.0) for s in samples])) for k in keys}
        avg_std = float(np.mean(list(stds.values()))) if stds else 0.0
        uncertainty = max(0.0, min(1.0, avg_std/0.15))
        st.write({"mean_p": current_p, "avg_std": round(avg_std, 6), "uncertainty_penalty": round(uncertainty, 3)})
    except Exception as e:
        current_p = {k: 1.0/len(keys) for k in keys}
        uncertainty = 0.5
        st.info(f"Baseline failed; using uniform fallback. ({e})")

    st.markdown("### Closest scenarios")
    ranked = []
    for name, meta in ACTIVE_SCENARIOS.items():
        tgt = meta["p"]
        ranked.append((name, kldiv(current_p, tgt), tvdist(current_p, tgt), meta.get("note", "")))
    ranked.sort(key=lambda x: (x[1], x[2]))

    if ranked:
        st.markdown(f"**Suggested scenario:** `{ranked[0][0]}` — KL≈`{ranked[0][1]:.4f}`, TV≈`{ranked[0][2]:.4f}`")
        topN = ranked[:3]
        st.dataframe(
            {"Scenario": [n for n, _, _, _ in topN],
             "KL": [round(kl, 6) for _, kl, _, _ in topN],
             "TV": [round(tv, 6) for _, _, tv, _ in topN],
             "Note": [note for _, _, _, note in topN]},
            use_container_width=True
        )

    valid_names = [n for n, _, _, _ in ranked] or list(ACTIVE_SCENARIOS.keys())
    sc_name = st.selectbox("Scenario to compare against", valid_names, index=0 if valid_names else None, key="sc_choice")
    scenario_p = ACTIVE_SCENARIOS.get(sc_name, {}).get("p") if sc_name else None
    scenario_note = ACTIVE_SCENARIOS.get(sc_name, {}).get("note", "") if sc_name else ""

    st.markdown("---")
    st.subheader("Sweep (what-if)")
    chan = st.selectbox(
        "Channel to sweep",
        ["None (no sweep)", "Depolarizing (p)", "Amplitude damping (γ)", "Phase damping (λ)"] +
        (["CNOT depolarizing (p2)"] if st.session_state.num_qubits == 2 else [])
    )
    step_label = st.selectbox("Temporal step", ["T0", "T1", "T2"], index=1)
    step_idx = {"T0": 0, "T1": 1, "T2": 2}[step_label]

    c1, c2, c3 = st.columns(3)
    with c1: v_start = st.number_input("Start", 0.0, 0.5, 0.0, 0.01)
    with c2: v_end = st.number_input("End", 0.0, 0.5, 0.2, 0.01)
    with c3: n_pts = st.number_input("Points", 3, 61, 11, 1)

    seeds_fx = [int(s.strip()) for s in st.text_input("Seeds (comma)", "7,13,23").split(",") if s.strip().isdigit()]
    shots_fx = st.number_input("Shots (per point)", 128, 8192,
                               min(2048, max(512, st.session_state.shots)), 128)

    def run_counts_noisy_once(seed_val):
        qc_n = build_measure_circuit_with_noise()
        key_n = _qc_key(qc_n)
        _ = cached_transpile("default", key_n, _to_qpy(qc_n), 1)
        return run_counts_cached("default", key_n, _to_qpy(qc_n), int(shots_fx), seed_val)

    def agg_counts(list_of_counts):
        tot = {k: 0 for k in keys}
        for c in list_of_counts:
            for k in keys:
                tot[k] += c.get(k, 0)
        N = sum(tot.values()) or 1
        return {k: tot[k]/N for k in keys}

    if st.button("Run sweep"):
        if chan == "None (no sweep)":
            st.info("Choose a channel to sweep.")
        else:
            try:
                X = np.linspace(v_start, v_end, int(n_pts)).tolist()
                series = {k: [] for k in keys}
                originals = {
                    "pdep": [ss_get("pdep0", 0.0), ss_get("pdep1", 0.0), ss_get("pdep2", 0.0)],
                    "pamp": [ss_get("pamp0", 0.0), ss_get("pamp1", 0.0), ss_get("pamp2", 0.0)],
                    "pph":  [ss_get("pph0", 0.0), ss_get("pph1", 0.0), ss_get("pph2", 0.0)],
                    "pcnot": [ss_get("pcnot0", 0.0), ss_get("pcnot1", 0.0), ss_get("pcnot2", 0.0)]
                }

                def set_step_param(kind, idx, val):
                    if kind == "pdep": ss_set(f"pdep{idx}", val)
                    elif kind == "pamp": ss_set(f"pamp{idx}", val)
                    elif kind == "pph": ss_set(f"pph{idx}", val)
                    elif kind == "pcnot": ss_set(f"pcnot{idx}", val)

                if chan == "Depolarizing (p)":
                    param = "pdep"; ss_set("enable_dep", True)
                elif chan == "Amplitude damping (γ)":
                    param = "pamp"; ss_set("enable_amp", True)
                elif chan == "Phase damping (λ)":
                    param = "pph"; ss_set("enable_phs", True)
                elif chan == "CNOT depolarizing (p2)":
                    param = "pcnot"; ss_set("enable_cnot_noise", True)

                sweep_progress = st.progress(0)
                status_text = st.empty()
                for i, v in enumerate(X):
                    status_text.text(f"Running point {i+1}/{len(X)}: {param} = {v:.3f}")
                    set_step_param(param, step_idx, float(v))
                    bag = []
                    for sd in (seeds_fx or [None]):
                        bag.append(run_counts_noisy_once(sd))
                    agg = agg_counts(bag)
                    for k in keys:
                        series[k].append(agg.get(k, 0.0))
                    sweep_progress.progress((i + 1) / len(X))

                # Restore original params
                ss_set("pdep0", originals["pdep"][0]); ss_set("pdep1", originals["pdep"][1]); ss_set("pdep2", originals["pdep"][2])
                ss_set("pamp0", originals["pamp"][0]); ss_set("pamp1", originals["pamp"][1]); ss_set("pamp2", originals["pamp"][2])
                ss_set("pph0", originals["pph"][0]); ss_set("pph1", originals["pph"][1]); ss_set("pph2", originals["pph"][2])
                ss_set("pcnot0", originals["pcnot"][0]); ss_set("pcnot1", originals["pcnot"][1]); ss_set("pcnot2", originals["pcnot"][2])

                st.subheader(f"Sweep results: {chan} @ {step_label}")
                data = {"x": X}; [data.setdefault(k, series[k]) for k in keys]
                fig_sweep = px.line(data, x="x", y=list(keys), title="Sweep Results") if px is not None else None
                if fig_sweep:
                    st.plotly_chart(fig_sweep)
                else:
                    st.write(data)

                current_end = {k: series[k][-1] for k in keys}
                if scenario_p:
                    D = kldiv(current_end, scenario_p)
                    TV = tvdist(current_end, scenario_p)
                    st.text(f"Scenario: {sc_name} — {scenario_note}")
                    st.text(f"KL(current || scenario) at end of sweep: {D:.6f} | TV: {TV:.6f}")

                    fig_end = px.bar(x=keys, y=[current_end.get(k, 0.0) for k in keys], title="Current (end of sweep)") if px is not None else None
                    fig_scenario = px.bar(x=keys, y=[scenario_p.get(k, 0.0) for k in keys], title="Scenario target") if px is not None else None
                    c1, c2 = st.columns(2)
                    with c1:
                        if fig_end: st.plotly_chart(fig_end)
                        else: st.write(current_end)
                    with c2:
                        if fig_scenario: st.plotly_chart(fig_scenario)
                        else: st.write(scenario_p)

                ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                sweep_name_default = f"{chan}@{step_label}_{ts}"
                sweep_name = st.text_input("Name this sweep (for save & compare)", sweep_name_default, key=f"sweep_name_{ts}")
                df = pd.DataFrame({"x": X})
                for k in keys:
                    df[k] = series[k]
                meta = {"qubits": 1 if keys == ["0", "1"] else 2, "keys": keys, "channel": chan, "step": step_label,
                        "start": float(v_start), "end": float(v_end), "points": int(n_pts),
                        "shots_per_point": int(shots_fx), "seeds": seeds_fx, "scenario": sc_name}
                csv_buf = io.StringIO()
                csv_buf.write("# meta=" + json.dumps(meta) + "\n")
                df.to_csv(csv_buf, index=False)
                st.download_button("⬇️ Download sweep CSV", data=csv_buf.getvalue().encode(),
                                   file_name=f"{sweep_name}.csv", mime="text/csv")
                st.session_state.foresight_sweeps[sweep_name] = {
                    "meta": meta, "x": X, "series": series,
                    "df": df.to_dict(orient="list"), "saved_at": ts
                }
                st.success(f"Sweep saved in memory as: {sweep_name}")
            except Exception as e:
                st.error(f"Sweep failed: {e}")
            finally:
                try:
                    sweep_progress.empty(); status_text.empty()
                except:
                    pass

    st.markdown("---")
    st.subheader("Manage sweeps (Load CSV & Compare)")
    up_csv = st.file_uploader("Load a sweep CSV", type=["csv"])
    if up_csv is not None:
        try:
            up_csv.seek(0)
            first = up_csv.readline().decode("utf-8")
            meta = json.loads(first[len("# meta="):].strip()) if first.startswith("# meta=") else {}
            up_csv.seek(0)
            df_loaded = pd.read_csv(up_csv, skiprows=1 if first.startswith("#") else 0)
            if "x" not in df_loaded.columns:
                st.error("CSV missing 'x' column.")
            else:
                loaded_keys = [c for c in df_loaded.columns if c != "x"]
                sweep_name = meta.get("name") or f"loaded_{int(time.time())}"
                st.session_state.foresight_sweeps[sweep_name] = {
                    "meta": {**meta, "keys": loaded_keys},
                    "x": df_loaded["x"].tolist(),
                    "series": {k: df_loaded[k].tolist() for k in loaded_keys},
                    "df": df_loaded.to_dict(orient="list"),
                    "saved_at": dt.datetime.now().strftime("%Y%m%d%H%M%S")
                }
                st.success(f"Loaded sweep '{sweep_name}' into memory.")
        except Exception as e:
            st.error(f"Load failed: {e}")

    if "foresight_sweeps" not in st.session_state:
        st.session_state.foresight_sweeps = {}
    sweep_names = sorted(list(st.session_state.foresight_sweeps.keys()))
    if len(sweep_names) >= 2:
        cols = st.columns(2)
        with cols[0]:
            a_name = st.selectbox("Sweep A", sweep_names, key="cmp_a")
        with cols[1]:
            b_name = st.selectbox("Sweep B", sweep_names, index=min(1, len(sweep_names)-1), key="cmp_b")

        if a_name != b_name:
            A = st.session_state.foresight_sweeps[a_name]
            B = st.session_state.foresight_sweeps[b_name]
            keysA = list(A["series"].keys()); keysB = list(B["series"].keys())
            if set(keysA) != set(keysB):
                st.warning("Sweeps have different outcome keys; cannot compare directly.")
            else:
                XA, XB = A["x"], B["x"]
                keys_cmp = keysA
                grids_match = (len(XA) == len(XB)) and all(abs(a-b) < 1e-9 for a, b in zip(XA, XB))

                def dist_at_index(i):
                    p = {k: A["series"][k][i] for k in keys_cmp}
                    q = {k: B["series"][k][i] for k in keys_cmp}
                    return kldiv(p, q), tvdist(p, q)

                st.markdown(f"**A:** {a_name} | **B:** {b_name}")
                KLs, TVs = [], []
                if grids_match:
                    for i in range(len(XA)):
                        kl, tv = dist_at_index(i)
                        KLs.append(kl); TVs.append(tv)
                    fig_dist = px.line(
                        x=XA, y=[KLs, TVs],
                        labels={'value': 'Distance', 'variable': 'Metric'},
                        title="Distance (pointwise over sweep X)"
                    ) if px is not None else None
                    if fig_dist:
                        fig_dist.data[0].name = 'KL(A||B)'
                        fig_dist.data[1].name = 'TV(A,B)'
                        st.plotly_chart(fig_dist)
                    else:
                        st.write("Distances not visualized (plotly unavailable)")
                else:
                    st.info("X-grids differ; computing only endpoint distances.")

                pA_end = {k: A["series"][k][-1] for k in keys_cmp}
                pB_end = {k: B["series"][k][-1] for k in keys_cmp}
                KL_end = kldiv(pA_end, pB_end); TV_end = tvdist(pA_end, pB_end)
                st.text(f"Endpoint distances — KL(A||B): {KL_end:.6f} | TV(A,B): {TV_end:.6f}")

                fig_a_end = px.bar(x=keys_cmp, y=[pA_end.get(k, 0.0) for k in keys_cmp], title="Sweep A (end)") if px is not None else None
                fig_b_end = px.bar(x=keys_cmp, y=[pB_end.get(k, 0.0) for k in keys_cmp], title="Sweep B (end)") if px is not None else None
                c1, c2 = st.columns(2)
                with c1:
                    if fig_a_end: st.plotly_chart(fig_a_end)
                    else: st.write(pA_end)
                with c2:
                    if fig_b_end: st.plotly_chart(fig_b_end)
                    else: st.write(pB_end)

                st.markdown("---")
                st.subheader("Clone endpoint to a new scenario")
                cc1, cc2 = st.columns(2)
                with cc1:
                    newA = st.text_input("Name for scenario from A endpoint", f"{a_name}_END_SCN")
                    if st.button("Clone A endpoint → scenario"):
                        st.session_state.custom_scenarios[newA] = {
                            "keys": keys_cmp, "p": {k: float(pA_end[k]) for k in keys_cmp},
                            "note": f"Cloned from sweep '{a_name}' endpoint", "impact": 1.0
                        }
                        st.success(f"Added custom scenario: {newA}")
                with cc2:
                    newB = st.text_input("Name for scenario from B endpoint", f"{b_name}_END_SCN")
                    if st.button("Clone B endpoint → scenario"):
                        st.session_state.custom_scenarios[newB] = {
                            "keys": keys_cmp, "p": {k: float(pB_end[k]) for k in keys_cmp},
                            "note": f"Cloned from sweep '{b_name}' endpoint", "impact": 1.0
                        }
                        st.success(f"Added custom scenario: {newB}")

                st.caption("Custom scenarios appear automatically and can be saved to disk.")
    else:
        st.info("Run a sweep (and/or load a CSV) to enable A/B comparison.")

    st.markdown("---")
    st.subheader("Repo scenarios (scenarios.json)")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Reload scenarios.json"):
            st.session_state.disk_scenarios = load_disk_scenarios()
            st.success("Reloaded disk scenarios."); st.rerun()
    with c2:
        if st.button("💾 Save custom scenarios → scenarios.json"):
            merged = dict(st.session_state.disk_scenarios)
            merged.update(st.session_state.custom_scenarios)
            ok, msg = save_disk_scenarios(merged)
            if ok:
                st.success(msg)
                st.session_state.disk_scenarios = load_disk_scenarios()
            else:
                st.error(msg)

    if st.checkbox("Show current disk scenarios"):
        st.json(st.session_state.disk_scenarios)

# -- Financial Analysis
with tab_fin:
    st.subheader("Quantum Financial Analysis (Monte-Carlo VaR/CVaR + Macro stress)")
    st.checkbox("Use Quantum Amplitude Estimation", value=ss_get("use_qae", False), key="use_qae")
    if st.session_state.use_qae and not HAVE_QAE:
        st.warning("QAE components not available. Install/upgrade qiskit-algorithms and qiskit-finance, or turn QAE off.")
        st.session_state.use_qae = False

    data = ss_get("market_data")
    if data is None:
        st.info("Use the sidebar to **Fetch Market Data** first.")
    else:
        if isinstance(data.columns, pd.MultiIndex):
            n_series = len(data.columns.levels[1])
        else:
            n_series = len(data.columns)
        st.success(f"Data ready: {len(data)} trading days · {n_series} series")
        returns = log_return_frame(data)
        create_comprehensive_financial_charts(data, returns)
        advanced_metrics = compute_advanced_financial_metrics(data, returns)

        alpha = st.session_state.confidence_level
        var_h = st.session_state.var_horizon
        sims = st.session_state.mc_sims
        pv = st.session_state.portfolio_value
        threshold = st.session_state.volatility_threshold

        var_r, cvar_r = monte_carlo_var_cvar(data, var_h, sims, alpha, use_quantum=st.session_state.use_qae)

        # Sentiment multiplier
        sentiment_mult = ss_get("sentiment_multiplier", 1.0)
        var_r *= sentiment_mult
        cvar_r *= sentiment_mult

        regime = detect_regime(data, threshold)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Regime", regime)
            if advanced_metrics:
                st.metric("Sharpe Ratio", f"{advanced_metrics.get('sharpe_ratio', 0):.2f}")
                st.metric("Max Drawdown", f"{advanced_metrics.get('max_drawdown', 0)*100:.1f}%")
        with col2:
            st.metric("MC VaR (return)", f"{var_r:.4f}", f"{int(alpha*100)}%, {var_h}d")
            if advanced_metrics:
                st.metric("Sortino Ratio", f"{advanced_metrics.get('sortino_ratio', 0):.2f}")
                st.metric("Skewness", f"{advanced_metrics.get('skewness', 0):.2f}")
        with col3:
            st.metric("MC CVaR (return)", f"{cvar_r:.4f}", "Tail-average")
            if advanced_metrics:
                st.metric("Historical VaR", f"{advanced_metrics.get('value_at_risk_historical', 0):.4f}")
                st.metric("Kurtosis", f"{advanced_metrics.get('kurtosis', 0):.2f}")

        st.subheader("Portfolio Impact")
        d1, d2 = st.columns(2)
        d1.metric("Dollar VaR", f"${pv * abs(var_r):,.2f}")
        d2.metric("Dollar CVaR", f"${pv * abs(cvar_r):,.2f}")

        st.markdown("---")
        st.subheader("Macro Stress")
        macro = st.session_state.get("macro_bundle")
        if macro is None:
            live = fetch_fred_bundle(ss_get("fred_api_key", ""))
            macro = live if live is not None else simulate_macro_bundle(data)
            if live is None:
                st.warning("FRED unavailable – using simulated macro from market volatility.")
            st.session_state.macro_bundle = macro

        macro_col1, macro_col2, macro_col3 = st.columns(3)
        with macro_col1:
            st.metric("CPI", f"{macro.get('CPI', 0):.1f}")
        with macro_col2:
            st.metric("Unemployment", f"{macro.get('Unemployment', 0):.1f}%")
        with macro_col3:
            st.metric("10Y Yield", f"{macro.get('10Y Yield', 0):.2f}%")

        if st.session_state.apply_macro_stress:
            stress = macro_stress_multiplier(macro)
            svar, scvar = var_r * stress, cvar_r * stress
            st.subheader("Macro-Stressed Results")
            s1, s2, s3 = st.columns(3)
            s1.metric("Stress factor", f"{stress:.3f}")
            s2.metric("Stressed VaR (ret.)", f"{svar:.4f}")
            s3.metric("Stressed CVaR (ret.)", f"{scvar:.4f}")

            sd1, sd2 = st.columns(2)
            sd1.metric("Stressed Dollar VaR", f"${pv * abs(svar):,.2f}")
            sd2.metric("Stressed Dollar CVaR", f"${pv * abs(scvar):,.2f}")

            fig_stress = go.Figure() if go is not None else None
            if fig_stress:
                categories = ['VaR', 'CVaR']
                fig_stress.add_trace(go.Bar(x=categories, y=[var_r, cvar_r], name='Base'))
                fig_stress.add_trace(go.Bar(x=categories, y=[svar, scvar], name='Stressed'))
                fig_stress.update_layout(
                    barmode='group',
                    title='Base vs Stressed VaR/CVaR',
                    yaxis_title='Return'
                )
                st.plotly_chart(fig_stress, use_container_width=True)

        st.markdown("---")
        st.subheader("Backtest & Efficient Frontier (optional)")
        if st.button("Run 5-year Backtest (monthly, rebalance=1)"):
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
                else:
                    prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
                back_fig, rolling_var, drawdowns = backtest_rebalance(prices)
                if back_fig:
                    st.plotly_chart(back_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Backtest error: {e}")

        if st.button("Show Efficient Frontier (static)"):
            try:
                ef_fig = plot_efficient_frontier(returns)
                if ef_fig:
                    st.plotly_chart(ef_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Efficient frontier error: {e}")

# -- Lachesis Guide
with tab_guide:
    st.subheader("Lachesis Guide — Explain This Configuration")
    st.markdown(
        "Use this panel to have Lachesis (or the local fallback) explain what your current **quantum** "
        "and **market** settings imply."
    )
    openai_key = st.text_input("OpenAI API Key (optional, for live LLM calls)", type="password", value=ss_get("openai_api_key", ""))
    ss_set("openai_api_key", openai_key)

    prompt = st.text_area(
        "Question for Lachesis",
        "Explain how the current circuit noise and market settings affect tail risk and regime.",
        height=120
    )

    if st.button("Ask Lachesis"):
        ctx = get_app_context()
        sys_msg = {
            "role": "system",
            "content": "You are Lachesis, an AI specializing in quantum-financial hybrid reasoning. "
                       "Explain clearly, concretely, and without hype."
        }
        user_msg = {"role": "user", "content": f"Context: {json.dumps(ctx)}\n\nQuestion: {prompt}"}
        text, meta = _studio_run_openai_chat(
            [sys_msg, user_msg],
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=600,
            api_key=openai_key
        )
        st.markdown("**Response:**")
        st.write(text)
        if meta.get("offline"):
            st.caption("Using local fallback (OpenAI not configured).")

    st.markdown("---")
    st.subheader("Noise-Aware Suggestions")
    for s in noise_aware_suggestions():
        st.markdown(f"- {s}")

# -- Advanced Quantum
with tab_advanced_q:
    st.subheader("Advanced Quantum Diagnostics")
    colA, colB = st.columns(2)
    # State tomography
    with colA:
        st.subheader("State Tomography (1-qubit)")
        tomo_shots = st.number_input("Tomography shots", 512, 32768, 4096, 512, key="tomo_shots")
        qc_tomo = QuantumCircuit(1)
        apply_gate(qc_tomo, 0, ss_get("g0_q0","H"), ss_get("a0_q0",0.5))
        apply_gate(qc_tomo, 0, ss_get("g1_q0","None"), ss_get("a1_q0",0.0))
        apply_gate(qc_tomo, 0, ss_get("g2_q0","RX"), ss_get("a2_q0",2.0))
        if st.button("Run state tomography"):
            ex, ey, ez = state_tomography_1q(qc_tomo, shots=int(tomo_shots),
                                             seed=ss_get("seed_val") if ss_get("use_seed", True) else None)
            st.write({"⟨X⟩": round(ex, 4), "⟨Y⟩": round(ey, 4), "⟨Z⟩": round(ez, 4)})
            # Plot Bloch sphere with Plotly
            fig_bloch = go.Figure() if go is not None else None
            if fig_bloch:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = np.cos(u)*np.sin(v)
                y = np.sin(u)*np.sin(v)
                z = np.cos(v)
                fig_bloch.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.2))
                fig_bloch.add_trace(go.Scatter3d(x=[0, ex], y=[0, ey], z=[0, ez], mode='lines'))
                fig_bloch.update_layout(title="Reconstructed Bloch vector", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
                st.plotly_chart(fig_bloch, use_container_width=True)
            else:
                st.write("Bloch sphere visualization unavailable (plotly not installed)")
    # Process fidelity proxy
    with colB:
        st.subheader("Process Fidelity (1-qubit, basic)")
        gate_choice = st.selectbox("Gate", ["H","X","Z","RX","RY"], index=0, key="proc_gate")
        gate_angle = st.slider("Angle (for RX/RY)", 0.0, 3.14, 1.57, 0.01, key="proc_angle")
        proc_shots = st.number_input("Shots", 512, 32768, 4096, 512, key="proc_shots")
        if st.button("Estimate gate fidelity"):
            F = process_fidelity_basic(
                gate_choice, angle=float(gate_angle), shots=int(proc_shots),
                seed=ss_get("seed_val") if ss_get("use_seed", True) else None
            )
            st.metric("Average state fidelity", f"{F:.4f}")
    st.markdown("---")
    st.subheader("Quantum Volume (proxy)")
    qv_info = quantum_volume_proxy(unitary_qc)
    c1, c2, c3 = st.columns(3)
    c1.metric("Width (qubits)", f"{int(qv_info['width']) if np.isfinite(qv_info['width']) else '—'}")
    c2.metric("Depth (no meas.)", f"{int(qv_info['depth']) if np.isfinite(qv_info['depth']) else '—'}")
    c3.metric("QV proxy", f"{qv_info['QV_proxy']:.0f}" if np.isfinite(qv_info["QV_proxy"]) else "—")
    st.caption("This is a quick educational proxy for Quantum Volume; it is not IBM’s full QV benchmark.")
    st.markdown("---")
    st.subheader("Auto-Calibrate Noise (LSQ)")
    cal_shots = st.number_input("Calibration shots", 512, 32768, 4096, 512)
    if st.button("Calibrate Noise"):
        if st.session_state.num_qubits != 1:
            st.info("Calibration for 1 qubit only.")
        else:
            with st.spinner("Calibrating..."):
                fitted, confidence = auto_calibrate_noise(cal_shots)
                if fitted is not None:
                    st.table({"Parameter": ["Depolarizing p", "Amplitude γ", "Phase λ"],
                              "Fitted": [round(fitted[0],4), round(fitted[1],4), round(fitted[2],4)],
                              "Confidence": [round(confidence,4)]*3})
                    st.success("Calibration completed. Toggle 'Use calibrated noise' in sidebar to apply.")
                else:
                    st.error("Calibration failed. Check console for details.")
    # Bayesian calibration
    st.markdown("---")
    st.subheader("Bayesian Noise Calibration")
    bay_shots = st.number_input("Bayesian calibration shots", 512, 32768, 4096, 512, key="bay_shots")
    priors_in = st.text_input("Priors (JSON, keys p,gamma,lambda as [a,b])", value='{"p":[1,1],"gamma":[1,1],"lambda":[1,1]}')
    if st.button("Run Bayesian Calibration"):
        try:
            pri = json.loads(priors_in)
            pri = {k: (float(v[0]), float(v[1])) for k,v in pri.items()}
        except Exception:
            pri = None
        post, summary = bayesian_calibrate_noise(shots=int(bay_shots), priors=pri,
                                                seed=ss_get("seed_val") if ss_get("use_seed", True) else None)
        st.write(summary)
        st.table({
            "Param": ["p","gamma","lambda"],
            "Mean": [round(post["p"].mean,4), round(post["gamma"].mean,4), round(post["lambda"].mean,4)],
            "CI95% low": [round(post["p"].ci_low,4), round(post["gamma"].ci_low,4), round(post["lambda"].ci_low,4)],
            "CI95% high": [round(post["p"].ci_high,4), round(post["gamma"].ci_high,4), round(post["lambda"].ci_high,4)],
        })
    # Calibration snapshots
    snapshots = ss_get("calibration_snapshots", [])
    if snapshots:
        st.markdown("---")
        st.subheader("Calibration Snapshots")
        snapshot_options = [f"{s['timestamp']} (conf: {s['confidence']:.3f})" for s in snapshots]
        selected_snapshot = st.selectbox("Select calibration snapshot", snapshot_options)
        if st.button("Apply Selected Snapshot"):
            idx = snapshot_options.index(selected_snapshot)
            snapshot = snapshots[idx]
            st.session_state.current_calibration = snapshot
            # Apply the parameters
            p, gamma, lambda_p = snapshot["params"]
            for s in range(3):
                ss_set(f"pdep{s}", p)
                ss_set(f"pamp{s}", gamma)
                ss_set(f"pph{s}", lambda_p)
            st.success("Calibration snapshot applied")
            st.rerun()
    if st.session_state.current_calibration:
        st.info(f"Current calibration: {st.session_state.current_calibration['timestamp']}")
        if st.button("Clear Current Calibration"):
            st.session_state.current_calibration = None
            st.rerun()
    # RB
    st.markdown("---")
    st.subheader("Randomized Benchmarking (1‑qubit)")
    rb_lengths = st.text_input("Sequence lengths (comma)", "2,4,8,16,32,48,64")
    rb_nseeds = st.number_input("Seeds per length", 1, 64, 16, 1)
    rb_shots = st.number_input("Shots per sequence", 256, 32768, 4096, 256)
    if st.button("Run RB"):
        lens = [int(x.strip()) for x in rb_lengths.split(",") if x.strip().isdigit()]
        res = randomized_benchmarking_1q(lens, int(rb_nseeds), int(rb_shots),
                                         seed=ss_get("seed_val") if ss_get("use_seed", True) else None)
        st.write({"fit": res["fit"], "EPG": res["EPG"]})
        fig_rb = px.scatter(x=res["lengths"], y=res["survival"], title="Randomized Benchmarking") if px is not None else None
        if fig_rb:
            A,p,B = res["fit"]["A"], res["fit"]["p"], res["fit"]["B"]
            xs = np.array(res["lengths"], float)
            fig_rb.add_trace(go.Scatter(x=xs, y=A*(p**xs)+B, mode='lines', name="fit"))
            fig_rb.update_layout(xaxis_title="Sequence length m", yaxis_title="Ground-state survival")
            st.plotly_chart(fig_rb)
        else:
            st.write(res)
    # Process tomography proxy
    st.markdown("---")
    st.subheader("Process Tomography (proxy)")
    if st.button("Run process proxy"):
        mapping = process_tomography_proxy_1q(lambda: build_unitary_circuit(),
                                              seed=ss_get("seed_val") if ss_get("use_seed", True) else None)
        st.write({"Input-axis → Output Bloch": {k: tuple(round(x,4) for x in v) for k,v in mapping.items()}})
    # Noise-aware suggestions
    st.markdown("---")
    st.subheader("Noise-aware Circuit Suggestions")
    if st.button("Suggest improvements"):
        tips = noise_aware_suggestions()
        for t in tips:
            st.write(f"- {t}")

# -- Sentiment Analysis
with tab_sentiment:
    st.subheader("Market Sentiment Analysis (News Headlines)")
    st.markdown("Pull Google News RSS for your tickers and derive a sentiment-based **VaR multiplier**.")

    tickers_str = st.text_input("Tickers for sentiment (comma)", ss_get("tickers", "AAPL,MSFT,SPY"))
    if st.button("Analyze Sentiment"):
        tick_list = [t.strip() for t in tickers_str.split(",") if t.strip()]
        res = analyze_sentiment(tick_list)
        if "error" in res and res["error"]:
            st.error(res["error"])
        else:
            st.write(f"Average compound score: {res['avg_score']:.3f}")
            st.write(f"Suggested stress multiplier: {res['multiplier']:.3f}")
            st.session_state.sentiment_multiplier = float(res["multiplier"])
            st.caption("This multiplier will be applied inside the **Financial Analysis** tab VaR/CVaR calculations.")

            if res.get("headlines"):
                st.markdown("**Sample Headlines:**")
                for h, s in list(zip(res["headlines"], res["scores"]))[:20]:
                    st.write(f"- ({s:+.3f}) {h}")

    st.markdown("---")
    st.subheader("Override Multiplier Manually")
    manual_mult = st.slider("Manual sentiment multiplier", 0.5, 1.5, float(ss_get("sentiment_multiplier", 1.0)), 0.01)
    if st.button("Apply manual multiplier"):
        ss_set("sentiment_multiplier", float(manual_mult))
        st.success(f"Sentiment multiplier set to {manual_mult:.3f}")

# -- Prompt Studio
with tab_prompt:
    st.subheader("Lachesis — LLM Prompt Studio")
    # Left: template manager; Right: run panel
    colL, colR = st.columns([0.55, 0.45])
    store: dict = ss_get("prompt_studio_store", {})
    if not isinstance(store, dict):
        store = {}
        ss_set("prompt_studio_store", store)
    with colL:
        st.markdown("### Templates")
        # List existing templates
        tmpl_names = sorted(store.keys())
        sel = st.selectbox("Select template", options=(["<new>"] + tmpl_names))
        if sel == "<new>":
            name = st.text_input("New template name", value="My Template")
            desc = st.text_input("Description", value="")
            sys_prompt = st.text_area("System prompt", value="You are Lachesis, a precise, neutral explainer.")
            tmpl_text = st.text_area(
                "User prompt template",
                value="Explain succinctly how the current noise settings (dep={dep}, amp={amp}, phs={phs}, cnot={cnot}) and gates influence VaR at confidence {alpha} over {h} days."
            )
            st.caption("Use {var} placeholders. Example: {alpha}, {h}.")
            st.markdown("**Variable mapping (JSON)** → maps template variables to session keys")
            var_json_default = json.dumps({"dep":"pdep1", "amp":"pamp1", "phs":"pph1", "cnot":"enable_cnot_noise", "alpha":"confidence_level", "h":"var_horizon"}, indent=2)
            var_json = st.text_area("Variables JSON", value=var_json_default, height=120)
            # few-shots editor
            st.markdown("**Few-shot examples (optional)**")
            fs_user = st.text_area("Few-shot USER example", value="", height=80)
            fs_assistant = st.text_area("Few-shot ASSISTANT example", value="", height=80)
            if st.button("Save template"):
                try:
                    var_map = json.loads(var_json) if var_json.strip() else {}
                    few_shots = []
                    if fs_user.strip():
                        few_shots.append({"role":"user","content":fs_user.strip()})
                    if fs_assistant.strip():
                        few_shots.append({"role":"assistant","content":fs_assistant.strip()})
                    store[name] = {
                        "description": desc,
                        "system": sys_prompt,
                        "template": tmpl_text,
                        "variables": var_map,
                        "few_shots": few_shots
                    }
                    ss_set("prompt_studio_store", store)
                    st.success(f"Saved template: {name}")
                except Exception as e:
                    st.error(f"Failed to save template: {e}")
            if st.button("Clear form"):
                st.rerun()
        else:
            # Edit existing template
            tpl = store.get(sel, {})
            name = st.text_input("Template name", value=sel, disabled=True)
            desc = st.text_input("Description", value=tpl.get("description",""))
            sys_prompt = st.text_area("System prompt", value=tpl.get("system",""))
            tmpl_text = st.text_area("User prompt template", value=tpl.get("template",""))
            var_json = st.text_area("Variables JSON", value=json.dumps(tpl.get("variables",{}), indent=2), height=120)
            # few-shots display/edit (simple)
            fs_list = tpl.get("few_shots", [])
            st.markdown("**Few-shot examples (optional)**")
            fs_user = st.text_area("Few-shot USER example", value=(fs_list[0]["content"] if fs_list and fs_list[0]["role"]=="user" else ""), height=80)
            fs_assistant = st.text_area("Few-shot ASSISTANT example", value=(fs_list[1]["content"] if len(fs_list)>1 and fs_list[1]["role"]=="assistant" else ""), height=80)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Update"):
                    try:
                        var_map = json.loads(var_json) if var_json.strip() else {}
                        few_shots = []
                        if fs_user.strip():
                            few_shots.append({"role":"user","content":fs_user.strip()})
                        if fs_assistant.strip():
                            few_shots.append({"role":"assistant","content":fs_assistant.strip()})
                        store[name] = {
                            "description": desc,
                            "system": sys_prompt,
                            "template": tmpl_text,
                            "variables": var_map,
                            "few_shots": few_shots
                        }
                        ss_set("prompt_studio_store", store)
                        st.success("Updated.")
                    except Exception as e:
                        st.error(f"Invalid variables JSON: {e}")
            with c2:
                if st.button("Duplicate"):
                    new_name = f"{name} Copy"
                    i = 2
                    while new_name in store:
                        new_name = f"{name} Copy {i}"; i += 1
                    store[new_name] = dict(store[name])
                    ss_set("prompt_studio_store", store)
                    st.success(f"Duplicated as '{new_name}'")
                    st.rerun()
            with c3:
                if st.button("Delete", type="secondary"):
                    if name in store:
                        del store[name]
                        ss_set("prompt_studio_store", store)
                        st.warning(f"Deleted '{name}'")
                        st.rerun()
            st.markdown("---")
            # Export / Import
            exp = _studio_export_store_json(store)
            st.download_button("⬇️ Export all templates (JSON)", data=exp, data_type="application/json", file_name="lachesis_prompt_studio.json", mime="application/json")
            up = st.file_uploader("Import templates JSON", type=["json"])
            if up is not None:
                data = _studio_import_store_json(up.read())
                if data:
                    # merge (overwrite by name)
                    store.update(data)
                    ss_set("prompt_studio_store", store)
                    st.success(f"Imported {len(data)} templates.")
                    st.rerun()
    with colR:
        st.markdown("### Run")
        api_key = st.text_input("OpenAI API Key (sk-…)", type="password", value=ss_get("openai_api_key",""))
        if api_key != ss_get("openai_api_key",""):
            ss_set("openai_api_key", api_key)
        model = st.text_input("Model", value="gpt-4o-mini")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.6, 0.05)
        max_tokens = st.number_input("Max tokens", 16, 4096, 600, 16)
        if not store:
            st.info("Create or select a template on the left.")
        else:
            use_name = st.selectbox("Template to run", options=sorted(store.keys()), index=0)
            tpl = store[use_name]
            # Pull app-context values for variables
            app_ctx = _studio_collect_context_from_app(tpl.get("variables", {}))
            st.markdown("**Override variables (JSON)** (optional)")
            st.caption("You can override the auto-collected values here at run time.")
            override_json = st.text_area("Overrides JSON", value="", placeholder='{"alpha":0.9}')
            overrides = {}
            if override_json.strip():
                try:
                    overrides = json.loads(override_json)
                except Exception as e:
                    st.warning(f"Ignoring override JSON error: {e}")
            vals = {**app_ctx, **overrides}
            # Show the rendered user prompt
            rendered = _safe_render_template(tpl.get("template",""), vals)
            st.markdown("**Rendered USER prompt**")
            st.code(rendered)
            if st.button("Run with Lachesis"):
                try:
                    messages = []
                    sys_text = tpl.get("system", "").strip()
                    if sys_text:
                        messages.append({"role":"system","content":sys_text})
                    # few-shots if any
                    for fs in tpl.get("few_shots", []):
                        if fs and "role" in fs and "content" in fs:
                            messages.append({"role": fs["role"], "content": fs["content"]})
                    # user message
                    messages.append({"role":"user","content":rendered})
                    with st.spinner("Querying model..."):
                        text, meta = _studio_run_openai_chat(messages, model, temperature, max_tokens, ss_get("openai_api_key",""))
                    st.markdown("**Lachesis response**")
                    st.write(text)
                    with st.expander("Raw meta"):
                        st.write(meta)
                except Exception as e:
                    st.error(f"Lachesis run failed: {e}")
        st.markdown("---")
        st.caption("Tip: Your template variables can pull from any session key (e.g., confidence_level, var_horizon, enable_cnot_noise, g0_q0, etc.).")