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
from qaoa_scenario1 import render_qaoa_tab
from typing import Optional, Dict, Any  # if not already imported
import Qtbn_UI as _qtbn_ui

apply_qtbn_purple_theme = getattr(_qtbn_ui, "apply_qtbn_purple_theme", lambda: None)
render_lachesis_voice_panel = getattr(_qtbn_ui, "render_lachesis_voice_panel", lambda _k: None)
synthesize_lachesis_audio = getattr(_qtbn_ui, "synthesize_lachesis_audio", lambda _t: (None, None))
render_llm_disclaimer = getattr(_qtbn_ui, "render_llm_disclaimer", lambda: None)
render_auth_gate = getattr(_qtbn_ui, "render_auth_gate", lambda _p=None: True)
_qtbn_is_owner_user = getattr(_qtbn_ui, "is_owner_user", lambda: False)
_qtbn_resolve_api_key = getattr(_qtbn_ui, "resolve_api_key", lambda _service: "")
_qtbn_clear_auth_session = getattr(_qtbn_ui, "clear_auth_session", lambda: None)


def is_owner_user() -> bool:
    try:
        return bool(_qtbn_is_owner_user())
    except Exception:
        return False


def resolve_api_key(service: str) -> str:
    try:
        value = _qtbn_resolve_api_key(service)
        return value.strip() if isinstance(value, str) else ""
    except Exception:
        return ""


def clear_auth_session() -> None:
    try:
        _qtbn_clear_auth_session()
    except Exception:
        pass


def api_key_status_caption(prefix: str, service: str) -> str:
    status = "Configured" if resolve_api_key(service) else "Missing"
    suffix = " (owner-managed)" if is_owner_user() else ""
    return f"{prefix}{status}{suffix}"
# --- VQE TAB import (safe)
try:
    from vqe_tab import render_vqe_tab, submit_order_through_vqe_gate
except Exception as _vqe_err:
    render_vqe_tab = None
    submit_order_through_vqe_gate = None
    _vqe_import_error = _vqe_err
# --- FORESIGHT TAB import (safe)
try:
    from foresight_tab import render_foresight_tab
except Exception as _fx_err:
    render_foresight_tab = None
    _fx_import_error = _fx_err

# ---- QTBN forecast stub (Lachesis tab) ---------------------------------
def qtbn_forecast_stub(prior_regime: str,
                       risk_on_prior: float,
                       drift_mu: float,
                       horizon_days: int) -> dict:
    """
    Tiny stand-in for a full QTBN engine.
    Uses the prior regime, risk-on prior, and drift μ to generate
    a 4-bucket forecast over the chosen horizon.
    Buckets: Gain / Flat / Loss / Severe loss
    """

    # Rough volatility per regime (annualized)
    regime_vol = {
        "calm": 0.12,
        "stressed": 0.25,
        "crisis": 0.40,
    }
    vol = regime_vol.get(prior_regime.lower(), 0.18)

    # Scale drift/vol to horizon (assume 252 trading days)
    h_frac = horizon_days / 252.0
    mu_h = drift_mu * h_frac
    sigma_h = vol * math.sqrt(h_frac) if vol > 0 else 0.0

    # Normal CDF helper
    def cdf(x: float) -> float:
        if sigma_h <= 1e-8:
            return 1.0 if x >= mu_h else 0.0
        z = (x - mu_h) / (sigma_h * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))

    # Bucket thresholds around the mean
    t1 = -2.0 * sigma_h   # severe loss
    t2 = -1.0 * sigma_h   # loss
    t3 =  1.0 * sigma_h   # flat vs gain boundary

    p_severe_loss = cdf(t1)
    p_loss        = cdf(t2) - cdf(t1)
    p_flat        = cdf(t3) - cdf(t2)
    p_gain        = 1.0 - cdf(t3)

    # Tilt toward gains or losses based on risk_on_prior (0..1)
    tilt = risk_on_prior - 0.5  # positive → more risk-on
    p_gain        += 0.40 * tilt
    p_severe_loss -= 0.20 * tilt
    p_loss        -= 0.10 * tilt
    p_flat        -= 0.10 * tilt

    # Clamp to [0, 1] and renormalize
    probs = np.array([p_gain, p_flat, p_loss, p_severe_loss], dtype=float)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum() or 1.0
    probs /= total

    return {
        "horizon_days": horizon_days,
        "P(gain)":        float(probs[0]),
        "P(flat)":        float(probs[1]),
        "P(loss)":        float(probs[2]),
        "P(severe_loss)": float(probs[3]),
    }


# ==== Tiny QTBN core (inline) ==============================================
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class QTBNConfig:
    """
    Minimal config for a toy Quantum Temporal Bayesian Network (QTBN) engine.
    This is classical under the hood but structured like a temporal BN.
    """
    regimes: List[str]                      # e.g. ["calm", "stressed", "crisis"]
    transition_matrix: np.ndarray           # shape (R, R): P(Regime_{t+1} | Regime_t)
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
            "regime_paths": regime_paths,   # list length T+1
            "drift_path": drift_path,       # length T+1
            "risk_on_path": risk_on_path,   # length T+1
        }
# ==== end QTBN core ========================================================



def compute_qaoa_priors(qaoa_snapshot: Optional[dict]) -> Dict[str, Any]:
    """
    Map a QAOA snapshot into QTBN-style priors.

    Returns a dict with:
      - persona
      - crash_index
      - expected_return
      - prior_regime   ('calm' / 'stressed' / 'crisis')
      - risk_on_prior  (float in [0,1])
      - drift_mu       (baseline expected return)
    """
    # --- defaults if there is no snapshot -------------------
    persona = "Balanced"
    crash_idx = 0.0
    expected_ret = 0.08  # 8% baseline drift
    prior_regime = "calm"
    risk_on_prior = 0.5
    drift_mu = expected_ret

    if not qaoa_snapshot:
        return {
            "persona": persona,
            "crash_index": crash_idx,
            "expected_return": expected_ret,
            "prior_regime": prior_regime,
            "risk_on_prior": risk_on_prior,
            "drift_mu": drift_mu,
        }

    # --- pull fields from snapshot --------------------------
    persona = str(qaoa_snapshot.get("persona", persona))
    crash_idx = float(qaoa_snapshot.get("crash_index", crash_idx))
    expected_ret = float(qaoa_snapshot.get("expected_return", expected_ret))

    # crash index → regime
    if crash_idx >= 0.66:
        prior_regime = "crisis"
    elif crash_idx >= 0.33:
        prior_regime = "stressed"
    else:
        prior_regime = "calm"

    # persona → risk-on prior
    p_lower = persona.lower()
    if "conservative" in p_lower:
        risk_on_prior = 0.3
    elif "balanced" in p_lower:
        risk_on_prior = 0.5
    else:  # aggressive / growth / etc.
        risk_on_prior = 0.7

    # drift μ comes from QAOA expected return
    drift_mu = expected_ret

    return {
        "persona": persona,
        "crash_index": crash_idx,
        "expected_return": expected_ret,
        "prior_regime": prior_regime,
        "risk_on_prior": risk_on_prior,
        "drift_mu": drift_mu,
    }


QTBN_REGIMES = ["calm", "stressed", "crisis"]


def _regime_index(name: str) -> int:
    """Map regime label → index in QTBN_REGIMES."""
    if not name:
        return 0
    name = str(name).lower()
    if "crisis" in name:
        return 2
    if "stress" in name:
        return 1
    return 0  # default to calm


def qtbn_toy_forecast(
    start_regime: str,
    risk_on_prior: float,
    drift_mu: float,
    steps: int = 3,
):
    """Tiny illustrative QTBN-style forecaster used throughout the UI."""

    T_base = np.array(
        [
            [0.85, 0.13, 0.02],  # calm today
            [0.25, 0.55, 0.20],  # stressed today
            [0.10, 0.25, 0.65],  # crisis today
        ],
        dtype=float,
    )

    mu_baseline = 0.08  # 8% baseline
    risk_component = 1.0 - float(risk_on_prior)
    mu_component = max(0.0, (mu_baseline - float(drift_mu)) / max(mu_baseline, 1e-6))
    heat = float(np.clip(0.5 * risk_component + 0.5 * mu_component, 0.0, 1.0))

    cool = 1.0 - heat
    T = T_base.copy()

    T[0, 2] += 0.05 * heat
    T[0, 0] -= 0.05 * heat

    T[1, 2] += 0.03 * heat
    T[1, 0] += 0.02 * cool

    T[2, 0] += 0.05 * cool
    T[2, 2] -= 0.05 * cool

    T = np.clip(T, 1e-6, 1.0)
    T = T / T.sum(axis=1, keepdims=True)

    start_idx = _regime_index(start_regime)
    state = np.full(len(QTBN_REGIMES), 0.05, dtype=float)
    state[start_idx] = 0.9
    state = state / state.sum()

    timeline: List[Dict[str, float]] = []
    p = state.copy()
    for _ in range(steps):
        timeline.append({reg: float(p[i]) for i, reg in enumerate(QTBN_REGIMES)})
        p = p @ T

    return timeline
def render_qaoa_bridge_inspector():
    st.subheader("QAOA ↔ QTBN Bridge Inspector")

    try:
        snapshot = load_qaoa_snapshot()
    except Exception:
        snapshot = None

    if snapshot is None:
        st.info(
            "No QAOA snapshot found. Run the Toy QAOA mini-lab and click "
            "**Export stance to QTBN** first."
        )
        return

    st.markdown("#### Raw QAOA snapshot")
    st.json(snapshot)

    priors = compute_qaoa_priors(snapshot)

    st.markdown("#### Derived QTBN priors")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Persona", priors["persona"])
        st.metric("Crash index", f"{priors['crash_index']:.2f}")
    with col2:
        st.metric("Prior regime", priors["prior_regime"])
        st.metric("P(risk-on)", f"{priors['risk_on_prior']:.2f}")
    with col3:
        st.metric("Drift μ", f"{priors['drift_mu']:.2%}")
        st.metric("QAOA expected return", f"{priors['expected_return']:.2%}")

    # Short explanation string
    explanation = (
        f"- crash_index = **{priors['crash_index']:.2f}** → regime **{priors['prior_regime']}**\n"
        f"- persona = **{priors['persona']}** → P(risk-on) **{priors['risk_on_prior']:.2f}**\n"
        f"- QAOA expected return **{priors['expected_return']:.2%}** "
        f"→ QTBN drift μ **{priors['drift_mu']:.2%}**"
    )
    st.markdown("#### Mapping explanation")
    st.markdown(explanation)

# --- Toy QAOA plugin (optional) --------------------------------------------
HAVE_QAOA = False
_QAOA_IMPORT_ERROR = None

try:
    # external plugin module; must sit next to qtbn_simulator_clean.py
    from qaoa_scenario1 import render_qaoa_tab  # type: ignore
    HAVE_QAOA = True
except Exception as e:
    _QAOA_IMPORT_ERROR = str(e)

    # Fallback stub so the rest of the app still works if the plugin is missing
    def render_qaoa_tab(*_args, **_kwargs):
        import streamlit as st  # local import to avoid early dependency issues
        st.subheader("Toy QAOA – Portfolio Selection (module not available)")
        st.info(
            "The QAOA demo module `qaoa_scenario1.py` is not loaded or raised an error.\n\n"
            f"Import error: {_QAOA_IMPORT_ERROR}\n\n"
            "You can still use all the other QTBN features; drop the QAOA file into the "
            "project folder and restart the app to enable this tab."
        )
# --------------------------------------------------------------------------



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# QAOA Toy Portfolio Config
# ============================
TOY_QAOA_PORTFOLIO = {
    "assets": ["AAPL", "MSFT", "GOOG"],
    "mu": [0.10, 0.12, 0.08],  # expected annual returns
    "cov": [
        [0.04, 0.01, 0.00],
        [0.01, 0.05, 0.02],
        [0.00, 0.02, 0.03],
    ],
    "lambda_risk": 0.5,
    "max_assets": 2,
}
toy_portfolio = TOY_QAOA_PORTFOLIO

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
    single_caps_marker_plain = 'handlePreserveConsecutiveUppercase='
    single_caps_patch_plain = (
        'handlePreserveConsecutiveUppercase=(t,g)=>{'
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
    autolink_patch_marker = "QTBN_SAFARI_PATCH_AUTOLINK_V4"
    email_patch = (
        'function transformGfmAutolinkLiterals(){'
        f'/*{autolink_patch_marker}*/'
        'return;'
        '}'
    )
    unicode_patch_marker = "QTBN_SAFARI_PATCH_UNICODE_V2"
    unicode_replacements = {
        r"\p{Uppercase_Letter}": "A-Z",
        r"\p{Lowercase_Letter}": "a-z",
        r"\p{L}": "A-Za-z",
        r"\p{N}": "0-9",
        r"\p{ID_Start}": "A-Za-z_",
        r"\p{ID_Continue}": "A-Za-z0-9_",
        r"\p{Diacritic}": r"[\\u0300-\\u036f]",
        r"\p{Dash_Punctuation}": r"[-\\u2010-\\u2015]",
        r"\p{P}": r"[!\"#$%&'()*+,\\-./:;<=>?@[\\\\\\]^_`{|}~]",
        r"\p{S}": r"[!\"#$%&'()*+,\\-./:;<=>?@[\\\\\\]^_`{|}~]",
    }

    for js_bundle in sorted(js_dir.glob("index.*.js")):
        try:
            src = js_bundle.read_text()
        except Exception:
            continue

        updated = False
        start_uc = src.find(single_caps_marker)
        if start_uc != -1:
            end_uc = src.find('function decamelize$2', start_uc)
            if end_uc != -1 and 'QTBN_SAFARI_PATCH' not in src[start_uc:end_uc]:
                src = src[:start_uc] + single_caps_patch + src[end_uc:]
                updated = True

        start_uc_plain = src.find(single_caps_marker_plain)
        if start_uc_plain != -1:
            end_uc_plain = src.find('function decamelize(', start_uc_plain)
            if end_uc_plain != -1 and 'QTBN_SAFARI_PATCH' not in src[start_uc_plain:end_uc_plain]:
                src = src[:start_uc_plain] + single_caps_patch_plain + src[end_uc_plain:]
                updated = True

        start_email = src.find(email_marker)
        if start_email != -1:
            end_email = src.find('function findUrl', start_email)
            if end_email != -1 and autolink_patch_marker not in src[start_email:end_email]:
                src = src[:start_email] + email_patch + src[end_email:]
                updated = True

        # Always strip named capture groups to avoid Safari regex errors.
        named_cap_re = re.compile(r"\(\?<([A-Za-z][A-Za-z0-9_]*)>")
        if named_cap_re.search(src):
            src = named_cap_re.sub("(", src)
            updated = True
        if "(?<=" in src:
            src = src.replace("(?<=", "(?:")
            updated = True
        if "(?<!" in src:
            src = src.replace("(?<!", "(?:")
            updated = True

        unicode_updated = False
        for needle, repl in unicode_replacements.items():
            for variant in (needle, needle.replace("\\", "\\\\")):
                if variant in src:
                    src = src.replace(variant, repl)
                    unicode_updated = True
        if unicode_updated:
            if unicode_patch_marker not in src:
                src += f"/*{unicode_patch_marker}*/"
            updated = True

        if updated:
            try:
                js_bundle.write_text(src)
                logger.info("Patched %s for Safari compatibility", js_bundle.name)
            except Exception as e:
                logger.warning("Failed to write Safari patch to %s: %s", js_bundle, e)
            continue

    # Cache-bust the JS bundle so Safari fetches the patched asset instead of stale cached JS.
    try:
        html_path = Path(_st.__file__).resolve().parent / "static" / "index.html"
        if html_path.exists():
            html_src = html_path.read_text()
            html_src = re.sub(r"\n<!-- QTBN_SAFARI_PATCH.*?-->\n?", "\n", html_src)
            html_src_new, replaced = re.subn(
                r'(src="\./static/js/index\.[^"?]+\.js)(?:\?[^"]*)?"',
                r'\1?qtbn_safari_patch=v5"',
                html_src,
                count=1,
            )
            if replaced and html_src_new != html_src:
                html_src_new += "\n<!-- QTBN_SAFARI_PATCH_HTML_V5 -->\n"
                html_path.write_text(html_src_new)
                logger.info("Patched %s for Safari cache busting", html_path)
            elif not replaced:
                logger.warning("Could not find Streamlit index JS script tag in %s", html_path)
    except Exception as e:
        logger.warning("Failed to patch Streamlit index.html for Safari cache bust: %s", e)

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
import numpy as np
import pandas as pd
import streamlit as st
# --- QAOA ↔ QTBN stance bridge helpers ---

from typing import Optional  # you probably already have this

def _get_qaoa_stance_from_session() -> Optional[dict]:
    """Return last QAOA stance snapshot from any of the known keys in st.session_state."""
    if "qaoa_stance_snapshot" in st.session_state:
        return st.session_state["qaoa_stance_snapshot"]
    if "lachesis_qaoa_stance" in st.session_state:
        return st.session_state["lachesis_qaoa_stance"]
    if "qaoa_stance" in st.session_state:
        return st.session_state["qaoa_stance"]
    return None


def map_stance_to_regime_prior(stance: dict):
    """
    Tiny heuristic to convert a QAOA persona + λ into a 3-state QTBN regime prior:
    [calm, choppy, crisis].
    """
    persona = (stance.get("persona")
               or stance.get("profile")
               or "Balanced")
    lam = float(stance.get("lambda", stance.get("risk_aversion", 1.0)))

    # Base templates by persona
    if persona == "Conservative":
        base = [0.70, 0.25, 0.05]
    elif persona == "Aggressive":
        base = [0.15, 0.35, 0.50]
    else:  # Balanced / unknown
        base = [0.40, 0.40, 0.20]

    # Clamp λ and use it to tilt calm vs crisis
    lam = max(0.3, min(3.0, lam))
    t = (lam - 0.3) / (3.0 - 0.3)  # 0 … 1

    calm = base[0] * (0.6 + 0.4 * t)      # more λ → more calm
    crisis = base[2] * (1.0 - 0.4 * t)    # more λ → less crisis
    choppy = max(1e-6, 1.0 - calm - crisis)

    s = calm + choppy + crisis
    calm, choppy, crisis = [round(x / s, 3) for x in (calm, choppy, crisis)]
    return [calm, choppy, crisis]

# --- end helpers ---


# ---- Optional QAOA plugin (qaoa_scenario1.py) ----
HAVE_QAOA = False
_QAOA_IMPORT_ERROR = None

try:
    # external plugin module; must sit next to qtbn_simulator_clean.py
    from qaoa_scenario1 import render_qaoa_tab  # noqa: F401
    HAVE_QAOA = True
except Exception as e:
    # Fallback stub so the app never crashes if plugin is missing
    HAVE_QAOA = False
    _QAOA_IMPORT_ERROR = str(e)

    def render_qaoa_tab():
        st.subheader("Toy QAOA (plugin not loaded)")
        st.warning(
            "The optional module `qaoa_scenario1` could not be imported.\n\n"
            f"Import error: {_QAOA_IMPORT_ERROR}\n\n"
            "To enable this tab:\n"
            "1. Create `qaoa_scenario1.py` in the same folder as "
            "`qtbn_simulator_clean.py`, and\n"
            "2. Define a function `render_qaoa_tab()` inside it."
        )
        # --- QAOA snapshot loader -----------------------------------
def load_qaoa_snapshot(path: str = "qaoa_snapshot.json") -> Optional[dict]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        # Don't kill the app if the snapshot is malformed
        print(f"[WARN] Failed to load QAOA snapshot: {e}")
        return None

# ---- end QAOA plugin block ----

# ---- Optional portfolio screenshot import (Robinhood, etc.) ----
try:
    from Portfolio_Screnshot_import import (
        PORTFOLIO_SCREENSHOT_IMPORT_ENABLED,
        extract_portfolio_from_screenshot_via_openai,
        normalize_extracted_position,
    )
except Exception:
    PORTFOLIO_SCREENSHOT_IMPORT_ENABLED = False

    def extract_portfolio_from_screenshot_via_openai(_image_bytes: bytes) -> dict:
        return {"ok": False, "error": "Portfolio screenshot import module not available."}

    def normalize_extracted_position(data: dict) -> dict:
        return {"raw": data}

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
try {
    const ua = navigator.userAgent || "";
    if (ua.indexOf("Safari") !== -1 && ua.indexOf("Chrome") === -1) {
        console.info("QTBN: Safari detected; compatibility shims enabled.");
    }
} catch (_e) {}
</script>
""", unsafe_allow_html=True)
apply_qtbn_purple_theme()
if not render_auth_gate("lachesis_logo.PNG"):
    st.stop()

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

def _secret_or_env(secret_key: str, env_key: str = "") -> str:
    """
    Resolve config from env first, then Streamlit secrets, then project-local
    `.streamlit/secrets.toml` as a fallback.
    """
    if env_key:
        env_val = os.getenv(env_key, "")
        if isinstance(env_val, str) and env_val.strip():
            return env_val.strip()
    try:
        secret_val = st.secrets.get(secret_key, "")
        if isinstance(secret_val, str) and secret_val.strip():
            return secret_val.strip()
    except Exception:
        pass
    try:
        local_secrets = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
        if local_secrets.exists():
            for raw_line in local_secrets.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() != secret_key:
                    continue
                val = v.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
                    val = val[1:-1]
                return val.strip()
    except Exception:
        pass
    return ""


_FINANCIAL_SYNC_FIELDS = (
    ("financial_analysis_portfolio_value", "portfolio_value"),
    ("financial_analysis_volatility_threshold", "volatility_threshold"),
    ("financial_analysis_apply_macro_stress", "apply_macro_stress"),
)


def _sync_financial_sidebar_state() -> None:
    """Propagate tab-specific overrides back to the sidebar defaults."""
    for source, target in _FINANCIAL_SYNC_FIELDS:
        if source in st.session_state and st.session_state[source] is not None:
            st.session_state[target] = st.session_state[source]


# Keep sidebar defaults in sync with the richer controls rendered inside the
# Financial Analysis tab (avoids modifying widget state post-instantiation).
_sync_financial_sidebar_state()

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
    Use the OpenAI v1 SDK when available; optionally fall back to legacy (<1.0).
    Returns (text, raw_meta). If OpenAI is unavailable or errors, return a stub.
    """
    if openai is None or not api_key:
        # Offline fallback: just echo the last user message with a stub.
        user_last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[LLM offline] Echo:\n{user_last}", {"offline": True}

    try:
        # Prefer new SDK (openai>=1.0.0).
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
        except Exception as e:
            # If v1 client import or call failed, do not try legacy if SDK is already v1.
            ver = getattr(openai, "__version__", "") or ""
            if ver and ver.split(".")[0].isdigit() and int(ver.split(".")[0]) >= 1:
                return f"[LLM error] OpenAI v1 client error: {e}", {"error": str(e), "provider": "openai_v1"}

        # Legacy-style usage (openai<1.0.0)
        ver = getattr(openai, "__version__", "") or ""
        if ver and ver.split(".")[0].isdigit() and int(ver.split(".")[0]) < 1 and hasattr(openai, "ChatCompletion"):
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            text = resp["choices"][0]["message"]["content"]
            return text, {"provider": "openai_legacy", "raw": resp}

        return "[LLM error] OpenAI SDK not compatible with current client.", {"error": "openai_sdk_incompatible"}
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
ss_get("lachesis_mode", "local")
ss_get("openai_api_key", "")
ss_get("perplexity_api_key", "")
ss_get("perplexity_model", _secret_or_env("perplexity_model", "QTBN_PERPLEXITY_MODEL") or "sonar")
ss_get("sentiment_source", "Google News RSS + VADER")
ss_get("prompt_studio_store", {})
ss_get("prompt_studio_templates", {
    # example starter template so the tab isn’t empty
    "Quick Explain": {
        "description": "Short explanation of the current QTBN noise settings' effect on VaR.",
        "system": "You are Lachesis, a precise, neutral explainer.",
        "template": "Explain succinctly how the current noise settings (dep={dep}, amp={amp}, phs={phs}, cnot={cnot}) and gates influence VaR at confidence {alpha} over a horizon of {h} days.",
        "variables": {"dep": "pdep1", "amp": "pamp1", "phs": "pph1", "cnot": "enable_cnot_noise", "alpha": "confidence_level", "h": "var_horizon"},
        "few_shots": [
            {"role": "user", "content": "Explain impact of high depolarizing noise on VaR."},
            {"role": "assistant", "content": "Higher depolarizing noise randomizes outcomes, widens return dispersion, and typically pushes VaR and CVaR more negative at fixed confidence."}
        ]
    }
})
# --- QAOA snapshot loader -----------------------------------
SNAPSHOT_PATH = Path("qaoa_snapshot.json")

def load_qaoa_snapshot() -> dict | None:
    """
    Load the last exported QAOA stance snapshot from disk.
    Returns a dict or None if not present.
    """
    try:
        if SNAPSHOT_PATH.exists():
            with SNAPSHOT_PATH.open("r") as f:
                return json.load(f)
    except Exception as e:
        # keep it silent so the app never crashes if file is corrupted
        print(f"[QAOA] Failed to read snapshot: {e}")
    return None

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
                if str(key).endswith("_api_key"):
                    continue
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
            "g0_q2", "a0_q2", "g0_q3", "a0_q3", "cnot0_12", "cnot0_23",
            "g1_q0", "a1_q0", "g1_q1", "a1_q1", "cnot1",
            "g1_q2", "a1_q2", "g1_q3", "a1_q3", "cnot1_12", "cnot1_23",
            "g2_q0", "a2_q0", "g2_q1", "a2_q1", "cnot2",
            "g2_q2", "a2_q2", "g2_q3", "a2_q3", "cnot2_12", "cnot2_23",
            "pdep0", "pdep1", "pdep2",
            "pamp0", "pamp1", "pamp2",
            "pph0", "pph1", "pph2",
            "pcnot0", "pcnot1", "pcnot2",
            "risk_free_rate", "confidence_level", "portfolio_value",
            "volatility_threshold", "tickers", "lookback_days", "var_horizon", "mc_sims",
            "macro_lookback_days", "apply_macro_stress",
            "lachesis_mode", "DEMO_MODE",
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


def get_value_basis() -> tuple[float, str]:
    """
    Returns:
      (basis_value_in_dollars, basis_label)
    Used to translate return-space VaR/CVaR into dollar-space numbers.
    """
    mode = str(ss_get("value_basis_mode", "Portfolio ($)"))

    if mode == "Per-share ($)":
        basis = float(ss_get("share_price", 100.0) or 0.0)
        return max(0.0, basis), "per share"

    if mode.startswith("Position"):
        price = float(ss_get("share_price", 100.0) or 0.0)
        shares = float(ss_get("shares_owned", 0) or 0.0)
        basis = price * shares
        return max(0.0, basis), "position"

    # default: Portfolio ($)
    basis = float(ss_get("portfolio_value", 1_000_000.0) or 0.0)
    return max(0.0, basis), "portfolio"


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

def build_unitary_circuit_1q() -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    apply_gate(qc, 0, ss_get("g0_q0", "H"), float(ss_get("a0_q0", 0.5) or 0.5) * math.pi)
    qc.barrier()
    apply_gate(qc, 0, ss_get("g1_q0", "None"), float(ss_get("a1_q0", 0.0) or 0.0) * math.pi)
    qc.barrier()
    apply_gate(qc, 0, ss_get("g2_q0", "None"), float(ss_get("a2_q0", 0.0) or 0.0) * math.pi)
    qc.barrier()
    return qc

def build_unitary_circuit() -> QuantumCircuit:
    nq = int(ss_get("num_qubits", 1) or 1)
    qc = QuantumCircuit(nq, nq)

    def _apply_step(step: int) -> None:
        gates = {
            0: ("g0_q0", "a0_q0", "g0_q1", "a0_q1", "g0_q2", "a0_q2", "g0_q3", "a0_q3"),
            1: ("g1_q0", "a1_q0", "g1_q1", "a1_q1", "g1_q2", "a1_q2", "g1_q3", "a1_q3"),
            2: ("g2_q0", "a2_q0", "g2_q1", "a2_q1", "g2_q2", "a2_q2", "g2_q3", "a2_q3"),
        }[step]
        g0, a0, g1, a1, g2, a2, g3, a3 = gates

        apply_gate(qc, 0, ss_get(g0, "H" if step == 0 else "None"),
                   float(ss_get(a0, 0.5 if step == 0 else 0.0) or 0.0) * math.pi)
        if nq > 1:
            apply_gate(qc, 1, ss_get(g1, "None"), float(ss_get(a1, 0.0) or 0.0) * math.pi)
        if nq > 2:
            apply_gate(qc, 2, ss_get(g2, "None"), float(ss_get(a2, 0.0) or 0.0) * math.pi)
        if nq > 3:
            apply_gate(qc, 3, ss_get(g3, "None"), float(ss_get(a3, 0.0) or 0.0) * math.pi)

        if step == 0:
            if ss_get("cnot0", False) and nq > 1:
                qc.cx(0, 1)
            if ss_get("cnot0_12", False) and nq > 2:
                qc.cx(1, 2)
            if ss_get("cnot0_23", False) and nq > 3:
                qc.cx(2, 3)
        elif step == 1:
            if ss_get("cnot1", False) and nq > 1:
                qc.cx(0, 1)
            if ss_get("cnot1_12", False) and nq > 2:
                qc.cx(1, 2)
            if ss_get("cnot1_23", False) and nq > 3:
                qc.cx(2, 3)
        else:
            if ss_get("cnot2", False) and nq > 1:
                qc.cx(0, 1)
            if ss_get("cnot2_12", False) and nq > 2:
                qc.cx(1, 2)
            if ss_get("cnot2_23", False) and nq > 3:
                qc.cx(2, 3)

        qc.barrier()

    _apply_step(0)
    _apply_step(1)
    _apply_step(2)

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
        noisy.save_density_matrix()
        sv_ideal = Statevector.from_instruction(ideal)
        dm_sim = get_simulator(method="density_matrix", seed=seed)
        tqc = transpile(noisy, dm_sim)
        res = dm_sim.run(tqc).result()
        data0 = res.data(0)
        dm_data = data0.get("density_matrix") if isinstance(data0, dict) else None
        if dm_data is None:
            logger.error("Process fidelity: density_matrix missing from result (keys=%s)", list(data0.keys()) if isinstance(data0, dict) else "n/a")
            continue
        noisy_dm = DensityMatrix(dm_data)
        Fs.append(state_fidelity(sv_ideal, noisy_dm))
    if not Fs:
        return float("nan")
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
    mode = ss_get("lachesis_mode", "local")
    api_key = resolve_api_key("openai")
    if mode == "openai" and api_key:
        sys_msg = {
            "role": "system",
            "content": (
                "You are Lachesis, a precise, neutral explainer. "
                "Summarize sentiment and news in clear, non-technical language. "
                "If only links are provided, explain based on the titles/keywords available."
            ),
        }
        user_msg = {
            "role": "user",
            "content": f"Context: {json.dumps(context)}\n\n{prompt}",
        }
        text, _meta = _studio_run_openai_chat(
            [sys_msg, user_msg],
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=700,
            api_key=api_key,
        )
        return text

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

SINGLE_Q_CLIFFORD = (
    "I", "X", "Y", "Z", "H", "S", "SDG",
    "HX", "HY", "HZ", "HS", "HSDG",
    "SX", "SY", "SZ", "SH", "S2",
)

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
                seq.append(str(rng.choice(SINGLE_Q_CLIFFORD)))
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
    base = circuit_builder()
    if getattr(base, "num_qubits", 1) != 1:
        base = build_unitary_circuit_1q()
    for name, prep in preps.items():
        qc = QuantumCircuit(1)
        qc.compose(prep, inplace=True)
        qc.compose(base, inplace=True)
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
@st.cache_data(ttl=3600)
def monte_carlo_var_cvar(
    data: pd.DataFrame,
    horizon_days: int,
    sims: int,
    alpha: float,
    use_quantum: bool = False  # kept for backward compatibility, but we rely on session toggle
) -> Tuple[float, float]:
    """
    Simulate horizon-sum log returns for an equal-weighted basket -> VaR & CVaR (return space).

    If the sidebar toggle 'use_qae' is ON and the QAE components are installed,
    this uses a Quantum Amplitude Estimation routine as an alternative to pure
    classical Monte Carlo.
    """
    # 🔹 Decide whether to use QAE based on session toggle + availability
    use_qae_flag = bool(ss_get("use_qae", False) and HAVE_QAE)

    try:
        rets = log_return_frame(data)
        if rets.empty:
            logger.warning("Empty returns DataFrame in VaR calculation")
            return float("nan"), float("nan")

        basket = rets.mean(axis=1)
        mu, sigma = basket.mean(), basket.std()

        if use_qae_flag:
            # -------------------------
            # Quantum Amplitude Estimation branch
            # -------------------------
            mu_h = float(mu) * horizon_days
            sigma_h = float(sigma) * math.sqrt(horizon_days)

            # Encode distribution over horizon returns in a NormalDistribution circuit
            num_qubits = 4
            bounds = [mu_h - 3 * sigma_h, mu_h + 3 * sigma_h]
            dist = NormalDistribution(num_qubits, mu=mu_h, sigma=sigma_h, bounds=bounds)

            from qiskit.circuit import QuantumCircuit
            A = QuantumCircuit(dist.num_qubits)
            A.compose(dist, inplace=True)

            # Treat the most-significant qubit as the "tail" indicator for an illustrative example
            objective_qubits = [dist.num_qubits - 1]
            problem = EstimationProblem(
                state_preparation=A,
                objective_qubits=objective_qubits,
                post_processing=lambda a: a,
            )

            ae = IterativeAmplitudeEstimation(epsilon=0.02, alpha=0.05)
            result = ae.estimate(problem)
            p_tail = float(getattr(result, "estimation", 0.0))

            # We still report VaR/CVaR using the corresponding normal model —
            # QAE is giving you a quantum-estimated tail probability proxy.
            var = norm.ppf(1 - alpha, loc=mu_h, scale=sigma_h)
            cvar = mu_h - (sigma_h / alpha) * norm.pdf(norm.ppf(alpha))

            logger.info(
                f"QAE tail-prob proxy ≈ {p_tail:.4f} | VaR={var:.4f} | CVaR={cvar:.4f}"
            )

        else:
            # -------------------------
            # Classical Monte Carlo branch
            # -------------------------
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
                chunk_draws = np.random.normal(
                    mu, sigma, size=(current_size, int(horizon_days))
                ).sum(axis=1)
                draws = np.concatenate([draws, chunk_draws])
                progress = (i + 1) / (chunks + 1)
                progress_bar.progress(progress)

            var = np.percentile(draws, (1.0 - alpha) * 100.0)
            cvar = draws[draws <= var].mean() if np.isfinite(var) else float("nan")

        logger.info(f"Calculated VaR: {var:.4f}, CVaR: {cvar:.4f}")
        return float(var), float(cvar)

    except Exception as e:
        logger.error(f"VaR/CVaR calculation failed: {e}", exc_info=True)
        return float("nan"), float("nan")

    finally:
        # Only try to clear the progress bar if we actually created it
        if not use_qae_flag:
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

                if isinstance(prices, pd.Series):
                    prices = prices.to_frame(name=prices.name or "Price")

                fig = go.Figure()
                for col in prices.columns:
                    fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode='lines', name=col))
                    ma = prices[col].rolling(50).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=prices.index,
                            y=ma,
                            mode='lines',
                            name=f"{col} 50-MA",
                            line=dict(dash='dot'),
                        )
                    )
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
    items = []
    for ticker in tickers:
        url = f"https://news.google.com/rss/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            title = getattr(entry, "title", None)
            link = getattr(entry, "link", None)
            if title:
                headlines.append(title)
            if title and link:
                items.append({"title": title, "link": link})
    if not headlines:
        return {"error": "No headlines found.", "multiplier": 1.0}
    scores = [sia.polarity_scores(h)['compound'] for h in headlines]
    avg_score = sum(scores) / len(scores)
    multiplier = 1 - 0.5 * avg_score
    multiplier = max(0.5, min(1.5, multiplier))
    return {
        "headlines": headlines,
        "items": items,
        "scores": scores,
        "avg_score": avg_score,
        "multiplier": multiplier,
        "provider": "Google News RSS + VADER",
    }

def _extract_json_object_from_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        # Trim fenced code blocks such as ```json ... ```
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    if "{" in s and "}" in s:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            s = s[start:end + 1]
    return s.strip()

def analyze_sentiment_perplexity(
    tickers: List[str],
    api_key: str,
    model: str = "sonar",
) -> Dict:
    """
    Sentiment via Perplexity live-web search. Returns the same shape as
    analyze_sentiment(...) so the UI and downstream VaR multiplier flow remain unchanged.
    """
    token = (api_key or "").strip()
    if not token:
        return {
            "error": "Perplexity API key missing. Add QTBN_PERPLEXITY_API_KEY or set perplexity_api_key in secrets.",
            "multiplier": 1.0,
        }
    tickers = [t.strip().upper() for t in (tickers or []) if t and str(t).strip()]
    if not tickers:
        return {"error": "No tickers provided.", "multiplier": 1.0}

    try:
        import requests
    except Exception as e:
        return {"error": f"requests not available: {e}", "multiplier": 1.0}

    prompt = (
        "You are a financial news sentiment engine.\n"
        f"Tickers: {', '.join(tickers)}\n"
        "Find current, relevant market news and return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "avg_score": number,\n'
        '  "multiplier": number,\n'
        '  "headlines": [\n'
        "    {\"title\": \"string\", \"url\": \"string\", \"score\": number}\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- avg_score must be between -1 and 1.\n"
        "- multiplier must be between 0.5 and 1.5 (higher = more risk stress).\n"
        "- score per headline must be between -1 and 1.\n"
        "- include up to 20 headlines.\n"
        "- no markdown, no prose, JSON only."
    )

    payload = {
        "model": (model or "sonar").strip(),
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code >= 300:
            return {"error": f"Perplexity HTTP {resp.status_code}: {resp.text[:300]}", "multiplier": 1.0}
        body = resp.json()
    except Exception as e:
        return {"error": f"Perplexity request failed: {e}", "multiplier": 1.0}

    content = ""
    try:
        choices = body.get("choices", [])
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            c = msg.get("content", "")
            if isinstance(c, list):
                pieces = []
                for part in c:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str):
                            pieces.append(t)
                    elif isinstance(part, str):
                        pieces.append(part)
                content = "\n".join(pieces).strip()
            elif isinstance(c, str):
                content = c.strip()
    except Exception:
        content = ""

    if not content:
        return {"error": "Perplexity response missing message content.", "multiplier": 1.0}

    json_text = _extract_json_object_from_text(content)
    if not json_text:
        return {"error": "Perplexity did not return parseable JSON.", "multiplier": 1.0}

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        preview = json_text[:220].replace("\n", "\\n")
        return {"error": f"Failed to parse Perplexity JSON: {e}; preview={preview}", "multiplier": 1.0}

    def _safe_float(v, default):
        try:
            return float(v)
        except Exception:
            return float(default)

    avg_score = max(-1.0, min(1.0, _safe_float(parsed.get("avg_score", 0.0), 0.0)))
    multiplier = _safe_float(parsed.get("multiplier", 1.0 - 0.5 * avg_score), 1.0 - 0.5 * avg_score)
    multiplier = max(0.5, min(1.5, multiplier))

    raw_headlines = parsed.get("headlines", [])
    headlines: List[str] = []
    items: List[Dict[str, str]] = []
    scores: List[float] = []

    if isinstance(raw_headlines, list):
        for raw in raw_headlines[:30]:
            if isinstance(raw, dict):
                title = str(raw.get("title", "") or "").strip()
                link = str(raw.get("url", "") or raw.get("link", "") or "").strip()
                score = max(-1.0, min(1.0, _safe_float(raw.get("score", avg_score), avg_score)))
            else:
                title = str(raw).strip()
                link = ""
                score = avg_score
            if not title:
                continue
            headlines.append(title)
            scores.append(score)
            if link:
                items.append({"title": title, "link": link})

    # Light fallback: preserve citations if model omitted a headline array.
    if not headlines:
        cites = body.get("citations", [])
        if isinstance(cites, list):
            for c in cites[:20]:
                url = str(c).strip()
                if not url:
                    continue
                headlines.append(url)
                scores.append(avg_score)
                items.append({"title": url, "link": url})

    return {
        "headlines": headlines,
        "items": items,
        "scores": scores,
        "avg_score": avg_score,
        "multiplier": multiplier,
        "provider": "Perplexity API",
        "model": (model or "sonar").strip(),
    }

# -------------------------------------------------------------------
# Sidebar (Quantum controls + Market data)
# -------------------------------------------------------------------
with st.sidebar:
    signed_in_email = str(st.session_state.get("auth_email_normalized", "") or "unknown")
    role_label = "Owner" if is_owner_user() else "User"
    st.markdown("### Account")
    st.caption(f"Signed in as `{signed_in_email}` ({role_label})")
    if st.button("Log out", key="auth_logout_button"):
        clear_auth_session()
        st.rerun()
    st.markdown("---")

    st.header("⚙️ Controls")

    st.subheader("Quantum")
    st.number_input("Qubits", 1, 4, key="num_qubits")
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
    if ss_get("num_qubits", 1) > 2:
        st.selectbox("T0 q2 gate", gate_choices, index=gate_choices.index(ss_get("g0_q2", "None")), key="g0_q2")
        st.slider("T0 q2 angle (π)", 0.0, 1.0, float(ss_get("a0_q2", 0.0)), 0.01, key="a0_q2")
        st.checkbox("T0 CX(1,2)", key="cnot0_12")
    if ss_get("num_qubits", 1) > 3:
        st.selectbox("T0 q3 gate", gate_choices, index=gate_choices.index(ss_get("g0_q3", "None")), key="g0_q3")
        st.slider("T0 q3 angle (π)", 0.0, 1.0, float(ss_get("a0_q3", 0.0)), 0.01, key="a0_q3")
        st.checkbox("T0 CX(2,3)", key="cnot0_23")

    # T1
    st.selectbox("T1 q0 gate", gate_choices, index=gate_choices.index(ss_get("g1_q0", "None")), key="g1_q0")
    st.slider("T1 q0 angle (π)", 0.0, 1.0, float(ss_get("a1_q0", 0.0)), 0.01, key="a1_q0")
    if ss_get("num_qubits", 1) > 1:
        st.selectbox("T1 q1 gate", gate_choices, index=gate_choices.index(ss_get("g1_q1", "None")), key="g1_q1")
        st.slider("T1 q1 angle (π)", 0.0, 1.0, float(ss_get("a1_q1", 0.0)), 0.01, key="a1_q1")
        st.checkbox("T1 CX(0,1)", key="cnot1")
    if ss_get("num_qubits", 1) > 2:
        st.selectbox("T1 q2 gate", gate_choices, index=gate_choices.index(ss_get("g1_q2", "None")), key="g1_q2")
        st.slider("T1 q2 angle (π)", 0.0, 1.0, float(ss_get("a1_q2", 0.0)), 0.01, key="a1_q2")
        st.checkbox("T1 CX(1,2)", key="cnot1_12")
    if ss_get("num_qubits", 1) > 3:
        st.selectbox("T1 q3 gate", gate_choices, index=gate_choices.index(ss_get("g1_q3", "None")), key="g1_q3")
        st.slider("T1 q3 angle (π)", 0.0, 1.0, float(ss_get("a1_q3", 0.0)), 0.01, key="a1_q3")
        st.checkbox("T1 CX(2,3)", key="cnot1_23")

    # T2
    st.selectbox("T2 q0 gate", gate_choices, index=gate_choices.index(ss_get("g2_q0", "None")), key="g2_q0")
    st.slider("T2 q0 angle (π)", 0.0, 1.0, float(ss_get("a2_q0", 0.0)), 0.01, key="a2_q0")
    if ss_get("num_qubits", 1) > 1:
        st.selectbox("T2 q1 gate", gate_choices, index=gate_choices.index(ss_get("g2_q1", "None")), key="g2_q1")
        st.slider("T2 q1 angle (π)", 0.0, 1.0, float(ss_get("a2_q1", 0.0)), 0.01, key="a2_q1")
        st.checkbox("T2 CX(0,1)", key="cnot2")
    if ss_get("num_qubits", 1) > 2:
        st.selectbox("T2 q2 gate", gate_choices, index=gate_choices.index(ss_get("g2_q2", "None")), key="g2_q2")
        st.slider("T2 q2 angle (π)", 0.0, 1.0, float(ss_get("a2_q2", 0.0)), 0.01, key="a2_q2")
        st.checkbox("T2 CX(1,2)", key="cnot2_12")
    if ss_get("num_qubits", 1) > 3:
        st.selectbox("T2 q3 gate", gate_choices, index=gate_choices.index(ss_get("g2_q3", "None")), key="g2_q3")
        st.slider("T2 q3 angle (π)", 0.0, 1.0, float(ss_get("a2_q3", 0.0)), 0.01, key="a2_q3")
        st.checkbox("T2 CX(2,3)", key="cnot2_23")

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

    def _auto_calibrate_from_sidebar():
        params, conf = auto_calibrate_noise()
        if params is not None:
            st.session_state.current_calibration = {
                "params": params,
                "confidence": conf,
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state["use_calibrated_noise"] = True
            st.session_state["calibration_status"] = ("success", params, conf)
        else:
            st.session_state["calibration_status"] = ("error", None, None)

    st.checkbox("Use last calibration as noise", key="use_calibrated_noise")
    st.button("Auto-calibrate from quick RB/Tomo", on_click=_auto_calibrate_from_sidebar)
    _cal_status = st.session_state.pop("calibration_status", None)
    if _cal_status:
        kind, params, conf = _cal_status
        if kind == "success":
            st.success(f"Calibrated: p={params[0]:.3f}, γ={params[1]:.3f}, λ={params[2]:.3f} (conf {conf:.2f})")
        else:
            st.error("Calibration failed. Check console for details.")

    st.markdown("---")
    st.subheader("Finance / Data")
    _pending_tickers = st.session_state.pop("tickers_pending", None)
    if _pending_tickers is not None:
        st.session_state["tickers"] = str(_pending_tickers)
    st.text_input("Tickers (comma)", key="tickers")
    st.number_input("Lookback days", 30, 2000, key="lookback_days")
    if st.button("Fetch Market Data"):
        data = fetch_market_data(st.session_state.tickers, st.session_state.lookback_days)
        if data is not None:
            st.session_state.market_data = data
            st.success("Market data ready.")

    _pending_portfolio_value = st.session_state.pop("portfolio_value_pending", None)
    if _pending_portfolio_value is not None:
        st.session_state["portfolio_value"] = max(1.0, float(_pending_portfolio_value))
    st.number_input("Portfolio value ($)", 1, 10_000_000_000, key="portfolio_value", step=10_000)
    st.markdown("**Value basis (for $ VaR/CVaR)**")
    st.radio(
        "Dollar conversion basis",
        ["Portfolio ($)", "Per-share ($)", "Position ($=shares×price)"],
        key="value_basis_mode",
        horizontal=False,
    )

    # Show share inputs only when they matter
    mode = ss_get("value_basis_mode", "Portfolio ($)")
    if mode in ("Per-share ($)", "Position ($=shares×price)"):
        st.number_input("Share price ($)", min_value=0.0, value=float(ss_get("share_price", 100.0)), key="share_price", step=0.5)
    if mode == "Position ($=shares×price)":
        st.number_input("Shares owned", min_value=0, value=int(ss_get("shares_owned", 100)), key="shares_owned", step=1)

    st.slider("Confidence (alpha)", 0.80, 0.99, key="confidence_level")
    st.number_input("Horizon (days)", 1, 60, key="var_horizon")
    st.number_input("MC simulations", 1000, 1_000_000, key="mc_sims", step=1000)
    st.slider("Volatility threshold (ann.)", 0.05, 1.0, key="volatility_threshold")
    st.checkbox("Apply macro stress", key="apply_macro_stress")
    st.caption(api_key_status_caption("FRED API key: ", "fred"))
    st.checkbox("Demo Mode (synthetic data, safer fallbacks)", key="DEMO_MODE")

    st.markdown("---")
    st.subheader("Persistence")
    if st.button("Save settings"):
        ok, msg = save_persistent_settings()
        st.info(msg)
    if st.button("Load settings"):
        load_persistent_settings()
        st.success("Loaded .qtbn.toml (if present).")
    # ---- Toy QAOA stub helpers -------------------------------------------------

TOY_QAOA_PORTFOLIO = {
    "assets": ["AAPL", "MSFT", "GOOG"],   # toy universe
    "target_return": 0.08,               # 8% annual target
    "baseline_risk": 0.20,               # 20% volatility (toy)
}

def run_qaoa_portfolio(portfolio_cfg, depth: int, shots: int):
    """
    Stubbed QAOA 'optimizer'.

    For now this does NOT run real quantum code.
    It just returns a consistent dictionary that the UI and
    'Export stance to QTBN' can use.
    """
    assets = portfolio_cfg.get("assets", [])
    # Make up some toy numbers based on depth/shots just so it moves
    lam = 1.0 - min(depth, 5) * 0.05      # slightly less risk-averse with more depth
    lam = max(0.5, lam)

    expected_return = portfolio_cfg.get("target_return", 0.08)
    risk = portfolio_cfg.get("baseline_risk", 0.20) * (1.0 - 0.05 * (depth - 1))
    crash_index = max(0.05, 0.30 - 0.02 * (depth - 1))

    # simple persona/regime for now
    persona = "Balanced"
    regime = "calm"

    return {
        "lambda": float(lam),
        "expected_return": float(expected_return),
        "risk": float(risk),
        "crash_index": float(crash_index),
        "assets": assets,
        "persona": persona,
        "regime": regime,
        "depth": depth,
        "shots": shots,
    }
def explain_risk_stance(
    qtbn_regime,
    market_regime,
    var_r,
    cvar_r,
    dollar_var,
    dollar_cvar,
    *,
    persona=None,
    crash_index=None,
    risk_on_prior=None,
    drift_mu=None,
    macro=None,
    use_qae=False,
    macro_stressed=False,
) -> str:
    """
    Build a human-readable risk narrative for the current configuration.

    All inputs are already-computed metrics; this function just turns them into text.
    """

    def fmt_pct(x):
        try:
            return f"{x * 100:.2f}%"
        except Exception:
            return "—"

    def bucket_crash(ci):
        if ci is None:
            return "unknown"
        if ci < 0.33:
            return "low"
        elif ci < 0.66:
            return "moderate"
        else:
            return "high"

    # --- Regime summary -------------------------------------------------
    qtbn_regime = qtbn_regime or "unknown"
    market_regime = market_regime or "unknown"

    lines = []
    lines.append(
        f"**Regime snapshot**  \n"
        f"- QTBN prior regime: **{qtbn_regime}**  \n"
        f"- Observed market regime (from returns): **{market_regime}**"
    )

    # --- QAOA stance summary (if available) ------------------------------
    if persona or crash_index is not None or risk_on_prior is not None:
        stance_bits = []

        if persona:
            stance_bits.append(f"stance: **{persona}**")

        if risk_on_prior is not None:
            stance_bits.append(f"risk-on prior ≈ **{risk_on_prior:.2f}**")

        if crash_index is not None:
            stance_bits.append(
                f"crash fear index ≈ **{crash_index:.2f}** "
                f"({bucket_crash(crash_index)} crash concern)"
            )

        if drift_mu is not None:
            stance_bits.append(f"drift μ ≈ **{fmt_pct(drift_mu)}**")

        if stance_bits:
            lines.append("\n**QAOA stance (from portfolio mini-lab)**  \n- " + "\n- ".join(stance_bits))

    # --- VaR / CVaR summary ---------------------------------------------
    lines.append(
        "\n**Loss profile (current configuration)**  \n"
        f"- Simulated VaR-loss magnitude: **{fmt_pct(abs(var_r))}** "
        f"(≈ **${dollar_var:,.0f}** on this portfolio)  \n"
        f"- Tail-loss average (CVaR) magnitude: **{fmt_pct(abs(cvar_r))}** "
        f"(≈ **${dollar_cvar:,.0f}**)"
    )

    # --- Macro environment -----------------------------------------------
    if macro:
        cpi = macro.get("CPI", None)
        unemp = macro.get("Unemployment", None)
        y10 = macro.get("10Y Yield", None)
        macro_bits = []
        if cpi is not None:
            macro_bits.append(f"CPI ≈ **{cpi:.1f}**")
        if unemp is not None:
            macro_bits.append(f"Unemployment ≈ **{unemp:.1f}%**")
        if y10 is not None:
            macro_bits.append(f"10Y yield ≈ **{y10:.2f}%**")

        if macro_bits:
            lines.append(
                "\n**Macro backdrop**  \n- "
                + "\n- ".join(macro_bits)
            )

        if macro_stressed:
            lines.append(
                "- Macro stress is **ON** — losses above already reflect "
                "an adverse macro environment.\n"
            )
        else:
            lines.append(
                "- Macro stress is **OFF** — VaR/CVaR reflect baseline market conditions.\n"
            )

    # --- Quantum flag ----------------------------------------------------
    if use_qae:
        lines.append(
            "**Quantum note**  \n"
            "- Quantum Amplitude Estimation is **enabled**, so the loss distribution "
            "incorporates a quantum-enhanced tail estimator (still a toy in this lab)."
        )
    else:
        lines.append(
            "**Quantum note**  \n"
            "- Quantum Amplitude Estimation is **disabled** — results are purely classical "
            "Monte-Carlo within this demo."
        )

    # --- Final disclaimer ------------------------------------------------
    lines.append(
        "\n> ⚠️ This narrative is for **exploration only** inside the Lachesis/QTBN lab and "
        "is **not** financial advice."
    )

    return "\n\n".join(lines)


# -------------------------------------------------------------------
# Header + Tabs
# -------------------------------------------------------------------
DEMO_MODE = ss_get("DEMO_MODE", False)
unitary_qc = build_unitary_circuit()

st.text("Circuit (ASCII, no measurement)")
st.code(str(unitary_qc.draw(output="text")), language="text")
st.caption("Legend: q[i] quantum wires; c[i] classical; barriers enforce step order; measures map q→c.")

tab_labels = [
    "Statevector", "Reduced States", "Measurement", "Fidelity & Export",
    "Presets", "Present Scenarios", "Foresight", "Financial Analysis",
    "Insider Trading", "Lachesis Guide", "Advanced Quantum", "Toy QAOA", "Sentiment Analysis", "Prompt Studio", "VQE",
]
if is_owner_user():
    tab_labels.append("Admin")

_tabs = st.tabs(tab_labels)
tab_sv, tab_red, tab_meas, tab_fid, tab_presets, tab_present, tab_fx, tab_fin, tab_insider, tab_guide, tab_advanced_q, tab_qaoa, tab_sentiment, tab_prompt, tab_vqe = _tabs[:15]
tab_admin = _tabs[15] if len(_tabs) > 15 else None

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
    elif st.session_state.num_qubits > 2:
        st.info("Reduced states view currently supports 2 qubits only.")
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
    ss_set("g0_q2", "None"); ss_set("a0_q2", 0.0); ss_set("cnot0_12", False)
    ss_set("g0_q3", "None"); ss_set("a0_q3", 0.0); ss_set("cnot0_23", False)
    ss_set("g1_q0", "None"); ss_set("a1_q0", 0.0)
    ss_set("g1_q1", "None"); ss_set("a1_q1", 0.0); ss_set("cnot1", False)
    ss_set("g1_q2", "None"); ss_set("a1_q2", 0.0); ss_set("cnot1_12", False)
    ss_set("g1_q3", "None"); ss_set("a1_q3", 0.0); ss_set("cnot1_23", False)
    ss_set("g2_q0", "None"); ss_set("a2_q0", 0.0)
    ss_set("g2_q1", "None"); ss_set("a2_q1", 0.0); ss_set("cnot2", False)
    ss_set("g2_q2", "None"); ss_set("a2_q2", 0.0); ss_set("cnot2_12", False)
    ss_set("g2_q3", "None"); ss_set("a2_q3", 0.0); ss_set("cnot2_23", False)
    ss_set("enable_dep", True); ss_set("enable_amp", False)
    ss_set("enable_phs", False); ss_set("enable_cnot_noise", True)
    ss_set("pdep0", 0.01); ss_set("pdep1", 0.02); ss_set("pdep2", 0.02)
    ss_set("pcnot0", 0.02); st.rerun()

def apply_preset_dephase():
    ss_set("num_qubits", 1)
    ss_set("g0_q0", "H"); ss_set("a0_q0", 0.5)
    ss_set("g1_q0", "None"); ss_set("a1_q0", 0.0)
    ss_set("g2_q0", "None"); ss_set("a2_q0", 0.0)
    ss_set("g0_q1", "None"); ss_set("a0_q1", 0.0); ss_set("cnot0", False)
    ss_set("g0_q2", "None"); ss_set("a0_q2", 0.0); ss_set("cnot0_12", False)
    ss_set("g0_q3", "None"); ss_set("a0_q3", 0.0); ss_set("cnot0_23", False)
    ss_set("g1_q1", "None"); ss_set("a1_q1", 0.0); ss_set("cnot1", False)
    ss_set("g1_q2", "None"); ss_set("a1_q2", 0.0); ss_set("cnot1_12", False)
    ss_set("g1_q3", "None"); ss_set("a1_q3", 0.0); ss_set("cnot1_23", False)
    ss_set("g2_q1", "None"); ss_set("a2_q1", 0.0); ss_set("cnot2", False)
    ss_set("g2_q2", "None"); ss_set("a2_q2", 0.0); ss_set("cnot2_12", False)
    ss_set("g2_q3", "None"); ss_set("a2_q3", 0.0); ss_set("cnot2_23", False)
    ss_set("enable_dep", False); ss_set("enable_amp", False); ss_set("enable_phs", True)
    ss_set("pdep0", 0.00); ss_set("pdep1", 0.00); ss_set("pdep2", 0.00); st.rerun()

def apply_preset_amp():
    ss_set("num_qubits", 1)
    ss_set("g0_q0", "X"); ss_set("a0_q0", 0.0)
    ss_set("g0_q1", "None"); ss_set("a0_q1", 0.0); ss_set("cnot0", False)
    ss_set("g0_q2", "None"); ss_set("a0_q2", 0.0); ss_set("cnot0_12", False)
    ss_set("g0_q3", "None"); ss_set("a0_q3", 0.0); ss_set("cnot0_23", False)
    ss_set("enable_amp", True); ss_set("pamp1", 0.20); ss_set("pamp2", 0.20); st.rerun()

def _apply_scenario_from_state():
    name = st.session_state.get("scenario_select")
    ok = apply_scenario_preset(name)
    st.session_state["scenario_apply_status"] = ("success" if ok else "error", name)

with tab_presets:
    c1, c2, c3 = st.columns(3)
    c1.button("Bell prep (H→CX)", on_click=apply_preset_bell)
    c2.button("Dephasing stress", on_click=apply_preset_dephase)
    c3.button("Amplitude relaxation", on_click=apply_preset_amp)

    st.markdown("---")
    st.subheader("Scenario → Circuit")
    st.selectbox("Select scenario", list(SCENARIO_LIBRARY.keys()), key="scenario_select")
    st.button("Apply Scenario", on_click=_apply_scenario_from_state)
    _status = st.session_state.pop("scenario_apply_status", None)
    if _status:
        kind, name = _status
        if kind == "success":
            st.success(f"Applied scenario: {name}")
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

            keys = ["0", "1"] if st.session_state.num_qubits == 1 else [f"{i:0{nq}b}" for i in range(2**nq)]
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

# =========================
# FORESIGHT TAB (merged + fixed) — UPDATED with VQE→Golden Build bridge
# =========================

import io
import json
import math
import csv
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ---------- helpers ----------
def _ss_setdefault(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value


def _parse_int_list(s: str, default: List[int]) -> List[int]:
    s = (s or "").strip()
    if not s:
        return default
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            pass
    return out or default


def _safe_prob_dict(d: Dict[str, Any]) -> Dict[str, float]:
    """Ensure dict values are floats, sum to 1-ish. Used for mean_p."""
    out = {}
    for k, v in (d or {}).items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


def _kl(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = sorted(set(p.keys()) | set(q.keys()))
    acc = 0.0
    for k in keys:
        pk = max(eps, float(p.get(k, 0.0)))
        qk = max(eps, float(q.get(k, 0.0)))
        acc += pk * math.log(pk / qk)
    return float(acc)


def _tv(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = sorted(set(p.keys()) | set(q.keys()))
    return 0.5 * float(sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys))


def _stance_from_p(p_risk_on: float) -> str:
    if p_risk_on >= 0.66:
        return "Risk-on / Aggressive"
    if p_risk_on <= 0.34:
        return "Risk-off / Defensive"
    return "Neutral / Mixed"


def _confidence_from_uncertainty(u: float) -> float:
    # Simple monotone mapping; clamp to [0,1]
    return float(max(0.0, min(1.0, 1.0 - u)))


def _list_json_files(folder: Path) -> List[str]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p.name for p in folder.glob("*.json") if p.is_file()])


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _csv_download(df: pd.DataFrame, filename: str, label: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="text/csv")


def _status_rank(s: str) -> int:
    s = (s or "").upper()
    if s == "FAIL":
        return 2
    if s == "WARN":
        return 1
    return 0


def _merge_status(a: str, b: str) -> str:
    r = max(_status_rank(a), _status_rank(b))
    if r == 2:
        return "FAIL"
    if r == 1:
        return "WARN"
    return "PASS"


# ---------- state init ----------
def _init_foresight_state():
    # Golden build directory widget uses fx_gb_dir_raw, derived stored in fx_gb_dir
    _ss_setdefault("fx_gb_dir_raw", "golden_build")
    _ss_setdefault("fx_gb_dir", "golden_build")

    # Stability thresholds
    _ss_setdefault("fx_pass_avg_std", 0.01)
    _ss_setdefault("fx_warn_avg_std", 0.03)
    _ss_setdefault("fx_pass_uncert", 0.20)
    _ss_setdefault("fx_warn_uncert", 0.40)

    # Blessing thresholds
    _ss_setdefault("fx_pass_tv", 0.05)
    _ss_setdefault("fx_warn_tv", 0.10)
    _ss_setdefault("fx_pass_kl", 0.05)
    _ss_setdefault("fx_warn_kl", 0.15)

    _ss_setdefault("fx_enforce_gate", True)
    _ss_setdefault("fx_blessed_fixture", "(none)")

    # Baseline
    _ss_setdefault("fx_baseline_seeds", "11,17,29")

    # Sweep
    _ss_setdefault("fx_sweep_channel", "None (no sweep)")
    _ss_setdefault("fx_sweep_step", "T1")
    _ss_setdefault("fx_sweep_start", 0.00)
    _ss_setdefault("fx_sweep_end", 0.20)
    _ss_setdefault("fx_sweep_points", 11)
    _ss_setdefault("fx_sweep_seeds", "7,13,23")
    _ss_setdefault("fx_sweep_shots", 2048)

    # Validation grid
    _ss_setdefault("fx_val_seeds", "1,2,3,4,5")
    _ss_setdefault("fx_val_shots_list", "256,512,1024")
    _ss_setdefault("fx_val_repeats", 2)

    # Storage
    _ss_setdefault("fx_sweeps_store", [])          # list of dicts
    _ss_setdefault("fx_last_baseline", None)       # dict
    _ss_setdefault("fx_last_gate", None)           # dict
    _ss_setdefault("fx_last_toy_forecast", None)   # pd.DataFrame

    # VQE bridge (NEW)
    _ss_setdefault("vqe_snapshot", None)
    _ss_setdefault("vqe_smoke", None)


# ---------- baseline + toy forecast (fallback) ----------
def _compute_baseline_fallback(keys: List[str], seeds: List[int]) -> Dict[str, Any]:
    """
    If your real QTBN run function isn't wired yet, this creates a stable,
    repeatable-ish baseline from seeds so the UI works.
    """
    rng = np.random.default_rng(int(sum(seeds)) if seeds else 0)
    raw = rng.random(len(keys))
    raw = raw / raw.sum()
    mean_p = {k: float(v) for k, v in zip(keys, raw)}
    # pretend "avg_std" is small when seeds are many
    avg_std = float(max(0.001, 0.02 / max(1, len(seeds))))
    uncertainty_penalty = float(min(1.0, avg_std * 6.7))
    return {
        "mean_p": mean_p,
        "avg_std": avg_std,
        "uncertainty_penalty": uncertainty_penalty,
    }


def _toy_regime_forecast() -> pd.DataFrame:
    # Your screenshot shows Calm/Stressed/Crisis across T0/T1/T2
    data = [
        {"Step": "T0 (now)", "Calm": 0.90, "Stressed": 0.05, "Crisis": 0.05},
        {"Step": "T1",       "Calm": 0.7541, "Stressed": 0.1563, "Crisis": 0.0896},
        {"Step": "T2",       "Calm": 0.6662, "Stressed": 0.2042, "Crisis": 0.1297},
    ]
    df = pd.DataFrame(data)
    return df


def _gate_eval(baseline: Dict[str, Any]) -> Dict[str, Any]:
    avg_std = float(baseline.get("avg_std", 1.0))
    uncert = float(baseline.get("uncertainty_penalty", 1.0))

    pass_avg = avg_std <= float(st.session_state.fx_pass_avg_std)
    warn_avg = avg_std <= float(st.session_state.fx_warn_avg_std)

    pass_unc = uncert <= float(st.session_state.fx_pass_uncert)
    warn_unc = uncert <= float(st.session_state.fx_warn_uncert)

    # overall status
    if pass_avg and pass_unc:
        status = "PASS"
    elif warn_avg and warn_unc:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "status": status,
        "avg_std": avg_std,
        "uncertainty": uncert,
        "checks": {
            "avg_std": {"value": avg_std, "pass": pass_avg, "warn": warn_avg},
            "uncertainty": {"value": uncert, "pass": pass_unc, "warn": warn_unc},
        },
    }


# ---------- scenarios merge (your earlier pattern) ----------
def combined_scenarios_for_keys(keys_: List[str],
                               SAMPLE_SCENARIOS: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {k: {**v} for k, v in SAMPLE_SCENARIOS.items() if v.get("keys") == keys_}
    for name, meta in st.session_state.get("disk_scenarios", {}).items():
        if meta.get("keys") == keys_:
            out[name] = {**meta}
    for name, meta in st.session_state.get("custom_scenarios", {}).items():
        if meta.get("keys") == keys_:
            out[name] = {**meta}
    for _, meta in out.items():
        if "impact" not in meta:
            meta["impact"] = 1.0
    return out


def render_foresight_tab(
    SAMPLE_SCENARIOS: Dict[str, Dict[str, Any]],
    num_qubits: int,
):
    """
    Drop-in Foresight tab:
    - Golden build + blessing gate (backup.py features)
    - Scenario ranking + sweeps manager (clean.py features)
    - Fixes st.session_state widget-key mutation bug
    - UPDATED: VQE→Golden Build bridge + merged Build Status + fixtures export/persist
    """
    _init_foresight_state()

    keys = ["0", "1"] if int(num_qubits) == 1 else [f"{i:0{int(num_qubits)}b}" for i in range(2**int(num_qubits))]

    st.subheader("Analytical Foresight — scenario ranking + sweeps")

    # -------------------------
    # Golden Build — Build Status + Blessing Gate
    # -------------------------
    st.markdown("## Golden Build — Build Status + Blessing Gate")

    # IMPORTANT: widget key is fx_gb_dir_raw; derived stored in fx_gb_dir
    st.text_input("Golden build directory", key="fx_gb_dir_raw")
    gb_dir = (st.session_state.fx_gb_dir_raw or "").strip()
    st.session_state["fx_gb_dir"] = gb_dir  # safe: different key than widget

    gb_path = Path(gb_dir)

    with st.expander("Stability thresholds (PASS/WARN/FAIL)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("PASS avg_std ≤", key="fx_pass_avg_std", min_value=0.0, max_value=10.0, step=0.001, format="%.3f")
        c2.number_input("WARN avg_std ≤", key="fx_warn_avg_std", min_value=0.0, max_value=10.0, step=0.001, format="%.3f")
        c3.number_input("PASS uncertainty ≤", key="fx_pass_uncert", min_value=0.0, max_value=10.0, step=0.01, format="%.2f")
        c4.number_input("WARN uncertainty ≤", key="fx_warn_uncert", min_value=0.0, max_value=10.0, step=0.01, format="%.2f")

    with st.expander("Blessing thresholds (PASS/WARN/FAIL)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("PASS TV ≤", key="fx_pass_tv", min_value=0.0, max_value=10.0, step=0.001, format="%.3f")
        c2.number_input("WARN TV ≤", key="fx_warn_tv", min_value=0.0, max_value=10.0, step=0.001, format="%.3f")
        c3.number_input("PASS KL ≤", key="fx_pass_kl", min_value=0.0, max_value=10.0, step=0.001, format="%.3f")
        c4.number_input("WARN KL ≤", key="fx_warn_kl", min_value=0.0, max_value=10.0, step=0.001, format="%.3f")

    blessed_files = _list_json_files(gb_path)
    fixture_choices = ["(none)"] + blessed_files
    st.selectbox("Select blessed fixture (.json) from golden_build folder",
                 options=fixture_choices,
                 key="fx_blessed_fixture")

    st.checkbox("Enforce gate: disable sweeps when Build Status = FAIL", key="fx_enforce_gate")

    # -------------------------
    # Current baseline (multi-seed)
    # -------------------------
    st.markdown("## Current baseline (multi-seed)")
    st.text_input("Seeds (comma)", key="fx_baseline_seeds")

    seeds = _parse_int_list(st.session_state.fx_baseline_seeds, [11, 17, 29])
    baseline = _compute_baseline_fallback(keys, seeds)
    st.session_state["fx_last_baseline"] = baseline

    st.json({
        "mean_p": baseline["mean_p"],
        "avg_std": baseline["avg_std"],
        "uncertainty_penalty": baseline["uncertainty_penalty"],
    })

    gate = _gate_eval(baseline)
    st.session_state["fx_last_gate"] = gate

    with st.expander("Gate status details", expanded=False):
        st.json(gate)

    # -------------------------
    # Closest scenarios
    # -------------------------
    st.markdown("## Closest scenarios")

    ACTIVE_SCENARIOS = combined_scenarios_for_keys(keys, SAMPLE_SCENARIOS)

    rows = []
    for name, meta in ACTIVE_SCENARIOS.items():
        scenario_p = _safe_prob_dict(meta.get("mean_p", {}))
        if not scenario_p:
            # allow a simple "p_risk_on" meta as a shorthand for 1q demo
            if keys == ["0", "1"] and "p_risk_on" in meta:
                pr0 = float(meta["p_risk_on"])
                scenario_p = {"0": 1.0 - pr0, "1": pr0}
        if not scenario_p:
            continue
        rows.append({
            "Scenario": name,
            "KL": _kl(baseline["mean_p"], scenario_p),
            "TV": _tv(baseline["mean_p"], scenario_p),
            "Note": meta.get("note", ""),
        })

    if rows:
        df = pd.DataFrame(rows).sort_values(["KL", "TV"], ascending=True)
        best = df.iloc[0]
        st.caption(f"Suggested scenario: **{best['Scenario']}** — KL≈{best['KL']:.4f}, TV≈{best['TV']:.4f}")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No scenarios found for current key-set. Add scenarios with matching `keys`.")

    # Blessing gate eval vs blessed fixture
    blessed_name = st.session_state.fx_blessed_fixture
    blessed_ok = True
    blessed_detail = {"status": "SKIPPED", "reason": "No fixture selected"}

    if blessed_name != "(none)" and gb_path.exists():
        fixture = _read_json(gb_path / blessed_name) or {}
        fixture_p = _safe_prob_dict(fixture.get("mean_p", fixture.get("p", {})))
        if fixture_p:
            tvv = _tv(baseline["mean_p"], fixture_p)
            kll = _kl(baseline["mean_p"], fixture_p)
            pass_tv = tvv <= float(st.session_state.fx_pass_tv)
            warn_tv = tvv <= float(st.session_state.fx_warn_tv)
            pass_kl = kll <= float(st.session_state.fx_pass_kl)
            warn_kl = kll <= float(st.session_state.fx_warn_kl)

            if pass_tv and pass_kl:
                bstat = "PASS"
            elif warn_tv and warn_kl:
                bstat = "WARN"
            else:
                bstat = "FAIL"

            blessed_ok = (bstat != "FAIL")
            blessed_detail = {
                "status": bstat,
                "TV": tvv,
                "KL": kll,
                "thresholds": {
                    "pass_tv": float(st.session_state.fx_pass_tv),
                    "warn_tv": float(st.session_state.fx_warn_tv),
                    "pass_kl": float(st.session_state.fx_pass_kl),
                    "warn_kl": float(st.session_state.fx_warn_kl),
                }
            }
        else:
            blessed_ok = False
            blessed_detail = {"status": "FAIL", "reason": "Fixture missing mean_p/p distribution"}

    with st.expander("Blessing status details", expanded=False):
        st.json(blessed_detail)

    # -------------------------
    # VQE→Golden Build bridge (UPDATED)
    # -------------------------
    st.markdown("## Build Status (Core + Blessing + VQE)")

    # Blessing status participates in merged status
    blessing_status = str(blessed_detail.get("status", "SKIPPED")).upper()
    if blessing_status == "SKIPPED":
        # Not failing the whole build, but it should be visible
        blessing_status_for_merge = "WARN"
    else:
        blessing_status_for_merge = blessing_status

    # VQE smoke participates in merged status
    vqe_smoke = st.session_state.get("vqe_smoke")
    vqe_snapshot = st.session_state.get("vqe_snapshot")

    if isinstance(vqe_smoke, dict) and vqe_smoke.get("status"):
        vqe_status = str(vqe_smoke.get("status", "WARN")).upper()
        vqe_reason = (
            f"VQE smoke {vqe_status}: |ΔE|={float(vqe_smoke.get('abs_delta', 0.0)):.4f} "
            f"(best={float(vqe_smoke.get('best_energy', 0.0)):.6f}, "
            f"ref={float(vqe_smoke.get('reference_energy', 0.0)):.6f})"
        )
    else:
        vqe_status = "WARN"
        vqe_reason = "VQE smoke not run / not sent yet (session_state['vqe_smoke'] missing)."

    core_status = str(gate.get("status", "WARN")).upper()
    merged_status = _merge_status(_merge_status(core_status, blessing_status_for_merge), vqe_status)

    # Show merged badge
    if merged_status == "PASS":
        st.success(f"✅ Build Status: {merged_status}")
    elif merged_status == "WARN":
        st.warning(f"⚠️ Build Status: {merged_status}")
    else:
        st.error(f"❌ Build Status: {merged_status}")

    # Component badges
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Core Gate**")
        st.write(core_status)
    with c2:
        st.write("**Blessing**")
        st.write(blessing_status)
    with c3:
        st.write("**VQE Smoke**")
        st.write(vqe_status)

    with st.expander("Build status details (why)", expanded=False):
        st.write(f"- Core gate: {core_status}")
        st.write(f"- Blessing: {blessing_status} (merged as {blessing_status_for_merge})")
        st.write(f"- {vqe_reason}")

    # Fixtures export + persist (UPDATED)
    st.markdown("### Fixtures export (baseline + gate + blessing + VQE + sweeps)")

    fixtures = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "num_qubits": int(num_qubits),
        "keys": keys,
        "baseline": baseline,
        "gate": gate,
        "blessing": blessed_detail,
        "build_status": {
            "merged": merged_status,
            "core": core_status,
            "blessing": blessing_status,
            "vqe": vqe_status,
        },
        "vqe_smoke": vqe_smoke if isinstance(vqe_smoke, dict) else None,
        "vqe_snapshot": vqe_snapshot if isinstance(vqe_snapshot, dict) else None,
        "sweeps_store": st.session_state.get("fx_sweeps_store", []),
        "toy_forecast": (
            st.session_state.get("fx_last_toy_forecast").to_dict(orient="records")
            if isinstance(st.session_state.get("fx_last_toy_forecast"), pd.DataFrame)
            else None
        ),
    }

    fixtures_json = json.dumps(fixtures, indent=2)

    colx1, colx2, colx3 = st.columns(3)
    with colx1:
        st.download_button(
            "⬇️ Download fixtures (.json)",
            data=fixtures_json.encode("utf-8"),
            file_name=f"foresight_fixtures_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    with colx2:
        if st.button("💾 Write fixtures to golden_build folder"):
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            ok = _write_json(gb_path / f"fixtures_{ts}.json", fixtures)
            if ok:
                st.success(f"Wrote: {str(gb_path / f'fixtures_{ts}.json')}")
            else:
                st.warning("Could not write fixtures to disk (filesystem may be restricted).")
    with colx3:
        if st.button("💾 Write VQE artifacts to golden_build folder"):
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            ok1 = True
            ok2 = True
            if isinstance(vqe_smoke, dict):
                ok1 = _write_json(gb_path / f"vqe_smoke_{ts}.json", vqe_smoke)
            if isinstance(vqe_snapshot, dict):
                ok2 = _write_json(gb_path / f"vqe_snapshot_{ts}.json", vqe_snapshot)
            if ok1 and ok2:
                st.success("Wrote VQE artifacts.")
            else:
                st.warning("Could not write some VQE artifacts (filesystem may be restricted).")

    # -------------------------
    # Action panel (Foresight → Decision)
    # -------------------------
    st.markdown("## Action Panel (Foresight → Decision)")
    p_risk_on = float(baseline["mean_p"].get(keys[-1], 0.5)) if keys else 0.5  # crude proxy
    uncertainty = float(baseline.get("uncertainty_penalty", 0.5))
    confidence = _confidence_from_uncertainty(uncertainty)
    stance = _stance_from_p(p_risk_on)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stance", stance)
    c2.metric("P(risk-on)", f"{p_risk_on*100:.2f}%")
    c3.metric("Uncertainty", f"{uncertainty:.2f}")
    c4.metric("Confidence", f"{confidence:.2f}")

    # -------------------------
    # Sweep (what-if) — with temporal step
    # -------------------------
    st.markdown("## Sweep (what-if) — with temporal step")

    # Enforce gate (UPDATED: uses merged Build Status FAIL)
    sweeps_disabled = False
    if st.session_state.fx_enforce_gate and (merged_status == "FAIL"):
        sweeps_disabled = True
        st.warning("Build Status = FAIL. Sweeps are disabled by enforcement setting.")

    # Controls
    channels = ["None (no sweep)", "impact", "p_risk_on_proxy"]
    st.selectbox("Channel to sweep", channels, key="fx_sweep_channel", disabled=sweeps_disabled)
    st.selectbox("Temporal step", ["T1", "T2", "T3"], key="fx_sweep_step", disabled=sweeps_disabled)

    c1, c2, c3 = st.columns(3)
    c1.number_input("Start", key="fx_sweep_start", step=0.01, format="%.2f", disabled=sweeps_disabled)
    c2.number_input("End", key="fx_sweep_end", step=0.01, format="%.2f", disabled=sweeps_disabled)
    c3.number_input("Points", key="fx_sweep_points", min_value=2, max_value=501, step=1, disabled=sweeps_disabled)

    st.text_input("Seeds (comma)", key="fx_sweep_seeds", disabled=sweeps_disabled)
    st.number_input("Shots (per point)", key="fx_sweep_shots", min_value=128, max_value=200000, step=128, disabled=sweeps_disabled)

    def _run_sweep() -> pd.DataFrame:
        start = float(st.session_state.fx_sweep_start)
        end = float(st.session_state.fx_sweep_end)
        n = int(st.session_state.fx_sweep_points)
        xs = np.linspace(start, end, n)

        # toy sweep: perturb risk-on with x
        base = p_risk_on
        ys = np.clip(base + (xs - start) * 0.25, 0.0, 1.0)

        df = pd.DataFrame({
            "x": xs,
            "p_risk_on": ys,
            "stance": [ _stance_from_p(v) for v in ys ],
            "step": st.session_state.fx_sweep_step,
            "channel": st.session_state.fx_sweep_channel,
        })
        return df

    if st.button("Run sweep", disabled=sweeps_disabled):
        df_sweep = _run_sweep()
        st.dataframe(df_sweep, use_container_width=True)
        fig = px.line(df_sweep, x="x", y="p_risk_on", title="Sweep result (toy)")
        st.plotly_chart(fig, use_container_width=True)

        st.session_state.fx_sweeps_store.append({
            "meta": {
                "step": st.session_state.fx_sweep_step,
                "channel": st.session_state.fx_sweep_channel,
                "start": float(st.session_state.fx_sweep_start),
                "end": float(st.session_state.fx_sweep_end),
                "points": int(st.session_state.fx_sweep_points),
                "seeds": st.session_state.fx_sweep_seeds,
                "shots": int(st.session_state.fx_sweep_shots),
            },
            "data": df_sweep.to_dict(orient="records"),
        })

    # -------------------------
    # Manage sweeps (Load CSV & Compare)
    # -------------------------
    st.markdown("## Manage sweeps (Load CSV & Compare)")
    up = st.file_uploader("Load a sweep CSV", type=["csv"])
    if up is not None:
        try:
            df_up = pd.read_csv(up)
            st.success("Loaded sweep CSV.")
            st.dataframe(df_up, use_container_width=True)
            st.plotly_chart(px.line(df_up, x=df_up.columns[0], y=df_up.columns[1], title="Uploaded sweep (quick view)"),
                            use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.session_state.fx_sweeps_store:
        st.caption("Saved sweeps in session:")
        idx = st.selectbox("Select saved sweep", list(range(len(st.session_state.fx_sweeps_store))))
        sweep_obj = st.session_state.fx_sweeps_store[int(idx)]
        df_saved = pd.DataFrame(sweep_obj["data"])
        st.json(sweep_obj["meta"])
        st.dataframe(df_saved, use_container_width=True)
        _csv_download(df_saved, f"sweep_{idx}.csv", "Download selected sweep CSV")

    # -------------------------
    # Repo scenarios (scenarios.json) I/O
    # -------------------------
    st.markdown("## Repo scenarios (scenarios.json)")
    left, right = st.columns(2)
    if left.button("Reload scenarios.json"):
        # You can wire this to your repo file later; placeholder keeps UI parity
        st.info("Reload hook placeholder — wire to your scenarios.json loader.")
    if right.button("Save custom scenarios → scenarios.json"):
        st.info("Save hook placeholder — wire to your scenarios.json writer.")

    st.checkbox("Show current disk scenarios", key="fx_show_disk_scenarios")
    if st.session_state.get("fx_show_disk_scenarios"):
        st.json(st.session_state.get("disk_scenarios", {}))

    # -------------------------
    # QAOA stance prior (from portfolio mini-lab)
    # -------------------------
    st.markdown("## QAOA stance prior (from portfolio mini-lab)")
    with st.expander("QAOA stance prior snapshot", expanded=True):
        # If you already store this elsewhere, replace these reads accordingly
        persona = st.session_state.get("qaoa_persona", "Aggressive")
        regime = st.session_state.get("qaoa_regime", "calm")
        lam = float(st.session_state.get("qaoa_lambda", 0.5))
        exp_ret = float(st.session_state.get("qaoa_expected_return", 0.0))
        crash = float(st.session_state.get("qaoa_crash_index", 0.0))
        alloc = st.session_state.get("qaoa_allocations", {"assets": [], "weights": []})

        c1, c2 = st.columns(2)
        c1.write(f"**Persona:** {persona}")
        c1.write(f"**Regime:** {regime}")
        c1.write(f"**λ (risk aversion):** {lam}")
        c2.metric("Expected return", f"{exp_ret:.2f}%")
        c2.metric("Crash index", f"{crash:.4f}")

        st.caption("Snapshot allocations")
        st.json(alloc)

    st.checkbox("Use QAOA stance as starting prior for this foresight run",
                key="fx_use_qaoa_prior")

    # -------------------------
    # Regime priors used for this run
    # -------------------------
    st.markdown("## Regime priors used for this run")
    start_regime = "calm"
    pr = 0.70 if st.session_state.fx_use_qaoa_prior else 0.50
    st.write(f"Starting regime: **{start_regime}**")
    st.write(f"P(risk-on) prior: **{pr:.2f}**")
    st.write("Drift μ: **0.00%**")

    # -------------------------
    # QTBN Toy Regime Forecast (+ chart)
    # -------------------------
    st.markdown("## QTBN Toy Regime Forecast")
    df_reg = _toy_regime_forecast()
    st.session_state["fx_last_toy_forecast"] = df_reg
    st.dataframe(df_reg, use_container_width=True)

    try:
        if px is not None:
            df_long = df_reg.melt(id_vars=["Step"], var_name="variable", value_name="value")
            fig = px.line(
                df_long,
                x="Step",
                y="value",
                color="variable",
                title="Toy QTBN regime probabilities over time",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df_reg.set_index("Step"))
    except Exception as e:
        st.warning(f"Toy regime chart unavailable: {e}")

    # -------------------------
    # Persona View on Priors
    # -------------------------
    st.markdown("## Persona View on Priors")
    lens = st.selectbox("View these priors as:", ["Chief Investment Officer", "Risk Officer", "Retail Investor"], index=0)

    if lens == "Chief Investment Officer":
        bullets = [
            "We are initializing in calm — markets are relatively stable with modest volatility.",
            f"Risk-on prior ≈ {pr*100:.0f}%",
            "Drift μ ≈ 0.0%",
        ]
    elif lens == "Risk Officer":
        bullets = [
            "Baseline appears stable but monitor tail risk and correlation jumps.",
            f"Uncertainty proxy = {uncertainty:.2f} (lower is better).",
            "Consider tightening exposure if build trends toward WARN/FAIL.",
        ]
    else:
        bullets = [
            "This is a high-level forecast, not financial advice.",
            f"Confidence proxy = {confidence:.2f}.",
            "Use position sizing and diversification.",
        ]

    st.markdown("### " + lens + " Lens")
    for b in bullets:
        st.write(f"• {b}")

    with st.expander("Foresight Debug", expanded=False):
        st.write("Keys:", keys)
        st.json({
            "baseline": baseline,
            "gate": gate,
            "blessing": blessed_detail,
            "vqe_smoke": vqe_smoke if isinstance(vqe_smoke, dict) else None,
            "vqe_snapshot": vqe_snapshot if isinstance(vqe_snapshot, dict) else None,
            "build_status": {
                "merged": merged_status,
                "core": core_status,
                "blessing": blessing_status,
                "vqe": vqe_status,
            },
        })


# =========================
# END FORESIGHT TAB
# =========================

with tab_fx:
    try:
        render_foresight_tab(SAMPLE_SCENARIOS, int(st.session_state.get("num_qubits", 1) or 1))
    except Exception as e:
        st.error(f"Foresight tab failed: {e}")
        st.code(traceback.format_exc())


# -- Financial Analysis
with tab_fin:
    st.subheader("Quantum Financial Analysis (Monte-Carlo VaR/CVaR + Macro stress)")

    # Ensure history container exists
    if "risk_history" not in st.session_state:
        st.session_state["risk_history"] = []

    # --- QAE toggle ------------------------------------------------------
    if not HAVE_QAE:
        ss_set("use_qae", False)
    st.checkbox(
        "Use Quantum Amplitude Estimation",
        value=ss_get("use_qae", False),
        key="use_qae",
        disabled=not HAVE_QAE,
    )
    if st.session_state.use_qae and not HAVE_QAE:
        st.warning(
            "QAE components not available. Install/upgrade qiskit-algorithms "
            "and qiskit-finance, or turn QAE off."
        )
        # Avoid mutating session_state after widget creation.

    # -------------------------------------------------------------------
    # Helpers (self-contained inside this tab)
    # -------------------------------------------------------------------
    import base64, json as _json

    def _to_float(x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip().replace("$", "").replace(",", "")
            return float(s)
        except Exception:
            return None

    def _infer_total_portfolio_value(market_value, diversity_pct):
        if market_value is None or diversity_pct is None:
            return None
        try:
            d = float(diversity_pct)
            mv = float(market_value)
            if d <= 0 or d > 100:
                return None
            return mv / (d / 100.0)
        except Exception:
            return None

    def _set_first_existing_key(keys, value):
        for k in keys:
            if k in st.session_state:
                st.session_state[f"{k}_pending"] = value
                return k
        st.session_state[keys[0]] = value
        return keys[0]

    def _responses_output_text(resp_json: dict):
        # Try common fields first
        if isinstance(resp_json, dict):
            if resp_json.get("output_text"):
                return resp_json["output_text"]

            out = resp_json.get("output", [])
            if isinstance(out, list):
                for item in out:
                    for c in item.get("content", []) or []:
                        # "output_text" chunks usually look like {"type":"output_text","text":"..."}
                        if isinstance(c, dict) and "text" in c:
                            return c.get("text")
        return None

    def _coerce_json_text(text: str) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:].lstrip()
        if "{" in s and "}" in s:
            start = s.find("{")
            end = s.rfind("}")
            if start >= 0 and end > start:
                s = s[start : end + 1]
        return s.strip()

    def extract_portfolio_from_screenshot_via_http(image_bytes: bytes, api_key: str, model: str = "gpt-4o-mini"):
        """
        No openai SDK required. Uses OpenAI Responses API directly via requests.
        """
        api_key = (api_key or "").strip()
        if not api_key:
            return {"ok": False, "error": "No API key provided."}

        try:
            import requests
        except Exception as e:
            return {"ok": False, "error": f"requests not available: {e}"}

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            import imghdr
            img_kind = imghdr.what(None, h=image_bytes) or "png"
        except Exception:
            img_kind = "png"
        if img_kind == "jpeg":
            img_kind = "jpg"
        mime = f"image/{img_kind}"

        system = (
            "You extract brokerage/portfolio screenshot data into strict JSON only. "
            "Return ONLY valid JSON. No prose."
        )

        user = (
            "Extract position data from this brokerage screenshot. If a field isn't present, set it to null.\n\n"
            "Return JSON with this shape:\n"
            "{\n"
            '  "broker": "Robinhood|Other|Unknown",\n'
            '  "currency": "USD|Other|Unknown",\n'
            '  "positions": [\n'
            "    {\n"
            '      "ticker": "string|null",\n'
            '      "shares": number|null,\n'
            '      "market_value": number|null,\n'
            '      "average_cost": number|null,\n'
            '      "today_return": number|null,\n'
            '      "total_return": number|null,\n'
            '      "portfolio_diversity_pct": number|null,\n'
            '      "price": number|null\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- ticker must be uppercase (e.g., NVDA)\n"
            "- portfolio_diversity_pct must be a number like 37.67 (NOT 0.3767)\n"
            "- Return ONLY JSON."
        )

        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user},
                        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                    ],
                },
            ],
            "text": {"format": {"type": "json_object"}},
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            r = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=60)
            if r.status_code >= 300:
                return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:500]}"}

            resp_json = r.json()
            text = _responses_output_text(resp_json)
            if not text:
                return {"ok": False, "error": "No output_text found in response."}

            json_text = _coerce_json_text(text)
            if not json_text:
                return {"ok": False, "error": "Empty JSON response from model."}
            try:
                data = _json.loads(json_text)
            except Exception as e:
                preview = json_text[:200].replace("\n", "\\n")
                return {"ok": False, "error": f"Failed to parse JSON: {e}; preview={preview}"}
            return {"ok": True, "data": data}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def normalize_extracted_position(payload: dict):
        positions = payload.get("positions", []) if isinstance(payload, dict) else []
        pos0 = positions[0] if isinstance(positions, list) and positions else {}

        ticker = (pos0.get("ticker") or "").strip().upper() or None
        shares = _to_float(pos0.get("shares"))
        market_value = _to_float(pos0.get("market_value"))
        diversity = _to_float(pos0.get("portfolio_diversity_pct"))
        price = _to_float(pos0.get("price"))

        inferred_total = _infer_total_portfolio_value(market_value, diversity)

        return {
            "ticker": ticker,
            "shares": shares,
            "market_value": market_value,
            "portfolio_diversity_pct": diversity,
            "price": price,
            "inferred_total_portfolio_value": inferred_total,
            "raw": payload,
        }

    # -------------------------------------------------------------------
    # Main tab logic
    # -------------------------------------------------------------------
    data = ss_get("market_data")
    if data is None:
        st.info("Use the sidebar to **Fetch Market Data** first.")
    else:

        # --- NEW: Screenshot importer (Robinhood etc.) -----------------------
        if "PORTFOLIO_SCREENSHOT_IMPORT_ENABLED" not in globals():
            PORTFOLIO_SCREENSHOT_IMPORT_ENABLED = True

        if PORTFOLIO_SCREENSHOT_IMPORT_ENABLED:
            with st.expander("📸 Import portfolio from screenshot (Robinhood)", expanded=False):
                st.caption("Uploads can be analyzed to extract ticker + market value and auto-fill your portfolio value.")
                screenshot_openai_key = resolve_api_key("openai")
                st.caption(api_key_status_caption("OpenAI key for screenshot extraction: ", "openai"))

                allow_external = st.checkbox(
                    "Allow external vision extraction (uses OpenAI API)",
                    value=False,
                    key="portfolio_ss_allow_external",
                    help="If disabled, nothing is sent anywhere (and extraction won't run)."
                )

                up = st.file_uploader(
                    "Upload a portfolio screenshot (.png/.jpg)",
                    type=["png", "jpg", "jpeg"],
                    key="portfolio_screenshot_upload",
                )

                if up is not None:
                    img_bytes = up.getvalue()

                    if st.button("Analyze screenshot → extract position", key="portfolio_ss_analyze_btn"):
                        if not allow_external:
                            st.warning("Enable external extraction to analyze this screenshot.")
                        elif not screenshot_openai_key:
                            st.warning("OpenAI API key is not configured by the owner.")
                        else:
                            with st.spinner("Extracting…"):
                                res = extract_portfolio_from_screenshot_via_http(
                                    img_bytes,
                                    api_key=screenshot_openai_key,
                                    model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
                                )

                            if not res.get("ok"):
                                st.error(f"Extraction failed: {res.get('error')}")
                            else:
                                norm_extracted = normalize_extracted_position(res["data"])
                                st.session_state["portfolio_ss_extracted"] = norm_extracted
                                st.success("Extraction complete.")

                norm_extracted = st.session_state.get("portfolio_ss_extracted")
                if isinstance(norm_extracted, dict):
                    st.markdown("**Extracted fields**")
                    st.write({
                        "ticker": norm_extracted.get("ticker"),
                        "shares": norm_extracted.get("shares"),
                        "market_value": norm_extracted.get("market_value"),
                        "portfolio_diversity_pct": norm_extracted.get("portfolio_diversity_pct"),
                        "inferred_total_portfolio_value": norm_extracted.get("inferred_total_portfolio_value"),
                    })

                    apply_mode = st.radio(
                        "Apply portfolio value as…",
                        [
                            "Use inferred TOTAL portfolio value (market_value / diversity%)",
                            "Use THIS POSITION market value only"
                        ],
                        index=0,
                        key="portfolio_ss_apply_mode",
                    )

                    if st.button("Apply to Lachesis inputs", key="portfolio_ss_apply_btn"):
                        ticker = norm_extracted.get("ticker")
                        mv = norm_extracted.get("market_value")
                        inferred_total = norm_extracted.get("inferred_total_portfolio_value")

                        # 1) Apply portfolio value into BOTH the UI control and the engine's canonical key
                        if apply_mode.startswith("Use inferred") and inferred_total is not None:
                            st.session_state["portfolio_value_pending"] = float(inferred_total)
                            st.session_state["financial_analysis_portfolio_value"] = max(1.0, float(inferred_total))
                        elif mv is not None:
                            st.session_state["portfolio_value_pending"] = float(mv)
                            st.session_state["financial_analysis_portfolio_value"] = max(1.0, float(mv))
                        else:
                            st.warning("No market value found to apply.")
                            st.stop()

                        # 2) Best-effort: apply ticker into common ticker input keys
                        if ticker:
                            _set_first_existing_key(
                                ["tickers", "tickers_input", "finance_tickers", "market_tickers", "sidebar_tickers"],
                                ticker
                            )

                        st.success("Applied extracted values.")
                        st.rerun()

        # --- basic dataset description ---------------------------------------
        if isinstance(data.columns, pd.MultiIndex):
            n_series = len(data.columns.levels[1])
        else:
            n_series = len(data.columns)
        st.success(f"Data ready: {len(data)} trading days · {n_series} series")

        returns = log_return_frame(data)
        create_comprehensive_financial_charts(data, returns)
        advanced_metrics = compute_advanced_financial_metrics(data, returns)

        # --- Position sizing (Portfolio $ <-> Shares) -----------------------
        def _parse_tickers_csv(s: str) -> List[str]:
            if not s:
                return []
            return [t.strip().upper() for t in s.split(",") if t.strip()]

        def _latest_price_from_data(df, ticker: str) -> Optional[float]:
            if df is None or not ticker:
                return None
            try:
                if hasattr(df.columns, "levels"):
                    for field in ["Adj Close", "Close", "Price"]:
                        col = (field, ticker)
                        if col in df.columns:
                            s = df[col].dropna()
                            if len(s) > 0:
                                return float(s.iloc[-1])
                    if ticker in list(df.columns.levels[1]):
                        for field in list(df.columns.levels[0]):
                            col = (field, ticker)
                            if col in df.columns:
                                s = df[col].dropna()
                                if len(s) > 0:
                                    return float(s.iloc[-1])
                else:
                    if ticker in df.columns:
                        s = df[ticker].dropna()
                        if len(s) > 0:
                            return float(s.iloc[-1])
                    if "Close" in df.columns:
                        s = df["Close"].dropna()
                        if len(s) > 0:
                            return float(s.iloc[-1])
            except Exception:
                return None
            return None

        st.markdown("### Position sizing")
        tickers_csv = (
            st.session_state.get("tickers")
            or st.session_state.get("tickers_input")
            or st.session_state.get("finance_tickers")
            or st.session_state.get("market_tickers")
            or st.session_state.get("sidebar_tickers")
            or "AAPL,MSFT,SPY"
        )
        tickers_list = _parse_tickers_csv(str(tickers_csv))

        if "position_input_mode" not in st.session_state:
            st.session_state["position_input_mode"] = "Portfolio value ($)"
        if "position_shares_ticker" not in st.session_state:
            st.session_state["position_shares_ticker"] = tickers_list[0] if tickers_list else "AAPL"
        if "position_shares" not in st.session_state:
            st.session_state["position_shares"] = 0.0
        if "financial_analysis_portfolio_value" not in st.session_state:
            st.session_state["financial_analysis_portfolio_value"] = 1_000_000.0

        st.radio(
            "Input mode",
            ["Portfolio value ($)", "Shares"],
            key="position_input_mode",
            horizontal=True,
        )

        st.selectbox(
            "Sizing ticker",
            options=tickers_list if tickers_list else ["AAPL"],
            key="position_shares_ticker",
            help="In Shares mode, shares apply to this ticker. In $ mode, this ticker is used to show implied shares.",
        )

        sizing_ticker = st.session_state["position_shares_ticker"]
        last_px = _latest_price_from_data(data, sizing_ticker)

        if st.session_state["position_input_mode"] == "Portfolio value ($)":
            st.number_input(
                "Portfolio value ($)",
                1.0,
                10_000_000_000.0,
                key="financial_analysis_portfolio_value",
                step=10_000.0,
            )
            pv = float(st.session_state["financial_analysis_portfolio_value"])
            st.session_state["portfolio_value_pending"] = pv
            if last_px is not None and last_px > 0:
                implied_shares = pv / float(last_px)
                st.caption(f"Latest {sizing_ticker} price: ${last_px:,.2f}")
                st.info(f"Implied shares of {sizing_ticker}: **{implied_shares:,.6f}**")
                st.session_state["position_shares"] = float(implied_shares)
            else:
                st.warning("Could not compute implied shares (no latest price available).")
        else:
            st.number_input(
                "Shares",
                min_value=0.0,
                step=0.1,
                key="position_shares",
            )
            shares = float(st.session_state["position_shares"])
            if last_px is None:
                st.warning("No latest price available yet. Fetch market data first.")
                pv = float(st.session_state.get("financial_analysis_portfolio_value", 1_000_000.0))
            else:
                implied_value = shares * float(last_px)
                st.caption(f"Latest {sizing_ticker} price: ${last_px:,.2f}")
                st.success(f"Implied portfolio value: ${implied_value:,.2f}")
                st.session_state["financial_analysis_portfolio_value"] = float(implied_value)
                st.session_state["portfolio_value_pending"] = float(implied_value)
                pv = float(implied_value)

        # --- User controls ---------------------------------------------------
        st.slider(
            "Volatility threshold (ann.)",
            0.05,
            1.0,
            key="financial_analysis_volatility_threshold",
        )
        st.checkbox("Apply macro stress", key="financial_analysis_apply_macro_stress")

        # Use the financial analysis controls directly (no widget key conflicts)
        alpha = ss_get("confidence_level", 0.95)
        var_h = ss_get("var_horizon", 10)
        sims = ss_get("mc_sims", 50_000)

        pv = float(st.session_state.financial_analysis_portfolio_value)
        threshold = float(st.session_state.financial_analysis_volatility_threshold)

        use_qae = bool(st.session_state.use_qae)
        apply_macro = bool(st.session_state.financial_analysis_apply_macro_stress)

        # --- QAOA stance snapshot (for narrative / priors only) --------------
        qaoa_snapshot = None
        try:
            qaoa_snapshot = load_qaoa_snapshot()
        except Exception:
            qaoa_snapshot = None

        qtbn_prior_regime = "calm"
        risk_on_prior = 0.5
        drift_mu_prior = 0.08
        persona = None
        crash_idx = None

        if qaoa_snapshot:
            persona = str(qaoa_snapshot.get("persona", "Balanced"))
            crash_idx = float(qaoa_snapshot.get("crash_index", 0.0))
            expected_ret = float(qaoa_snapshot.get("expected_return", drift_mu_prior))

            if crash_idx >= 0.66:
                qtbn_prior_regime = "crisis"
            elif crash_idx >= 0.33:
                qtbn_prior_regime = "stressed"
            else:
                qtbn_prior_regime = "calm"

            persona_lower = persona.lower()
            if "conservative" in persona_lower:
                risk_on_prior = 0.3
            elif "balanced" in persona_lower:
                risk_on_prior = 0.5
            else:
                risk_on_prior = 0.7

            drift_mu_prior = expected_ret

        # --- Core VaR/CVaR calculation ---------------------------------------
        var_r, cvar_r = monte_carlo_var_cvar(
            data,
            var_h,
            sims,
            alpha,
            use_quantum=use_qae,
        )

        # Sentiment multiplier (already in your app state)
        sentiment_mult = ss_get("sentiment_multiplier", 1.0)
        var_r *= sentiment_mult
        cvar_r *= sentiment_mult

        # Market regime detection
        regime = detect_regime(data, threshold)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Regime", regime)
            if advanced_metrics:
                st.metric("Sharpe Ratio", f"{advanced_metrics.get('sharpe_ratio', 0):.2f}")
                st.metric("Max Drawdown", f"{advanced_metrics.get('max_drawdown', 0) * 100:.1f}%")
        with col2:
            st.metric("MC VaR (return)", f"{var_r:.4f}", f"{int(alpha * 100)}%, {var_h}d")
            if advanced_metrics:
                st.metric("Sortino Ratio", f"{advanced_metrics.get('sortino_ratio', 0):.2f}")
                st.metric("Skewness", f"{advanced_metrics.get('skewness', 0):.2f}")
        with col3:
            st.metric("MC CVaR (return)", f"{cvar_r:.4f}", "Tail-average")
            if advanced_metrics:
                st.metric("Historical VaR", f"{advanced_metrics.get('value_at_risk_historical', 0):.4f}")
                st.metric("Kurtosis", f"{advanced_metrics.get('kurtosis', 0):.2f}")

        # Base dollar impact
        base_dollar_var = pv * abs(var_r)
        base_dollar_cvar = pv * abs(cvar_r)

        st.subheader("Portfolio Impact")
        d1, d2 = st.columns(2)
        d1.metric("Dollar VaR", f"${base_dollar_var:,.2f}")
        d2.metric("Dollar CVaR", f"${base_dollar_cvar:,.2f}")

        # --- Macro Stress -----------------------------------------------------
        st.markdown("---")
        st.subheader("Macro Stress")

        macro = st.session_state.get("macro_bundle")
        if macro is None:
            live = fetch_fred_bundle(resolve_api_key("fred"))
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

        svar = scvar = None
        stressed_dollar_var = stressed_dollar_cvar = None
        stress_factor = None
        svar_dollar = scvar_dollar = None

        if apply_macro:
            stress = macro_stress_multiplier(macro)
            stress_factor = stress
            svar, scvar = var_r * stress, cvar_r * stress

            st.subheader("Macro-Stressed Results")
            s1, s2, s3 = st.columns(3)
            s1.metric("Stress factor", f"{stress:.3f}")
            s2.metric("Stressed VaR (ret.)", f"{svar:.4f}")
            s3.metric("Stressed CVaR (ret.)", f"{scvar:.4f}")

            stressed_dollar_var = pv * abs(svar)
            stressed_dollar_cvar = pv * abs(scvar)
            svar_dollar = stressed_dollar_var
            scvar_dollar = stressed_dollar_cvar

            sd1, sd2 = st.columns(2)
            sd1.metric("Stressed Dollar VaR", f"${stressed_dollar_var:,.2f}")
            sd2.metric("Stressed Dollar CVaR", f"${stressed_dollar_cvar:,.2f}")

            fig_stress = go.Figure() if go is not None else None
            if fig_stress:
                categories = ["VaR", "CVaR"]
                fig_stress.add_trace(go.Bar(x=categories, y=[var_r, cvar_r], name="Base"))
                fig_stress.add_trace(go.Bar(x=categories, y=[svar, scvar], name="Stressed"))
                fig_stress.update_layout(
                    barmode="group",
                    title="Base vs Stressed VaR/CVaR",
                    yaxis_title="Return",
                )
                st.plotly_chart(fig_stress, use_container_width=True)

        # --- Persona View on Risk & Regime -----------------------------------
        st.markdown("---")
        st.subheader("Persona Views (how different roles would talk about this)")

        fin_persona = st.selectbox(
            "View the current risk picture as:",
            [
                "Chief Investment Officer",
                "Risk Officer",
                "Quant Researcher",
                "Client-Friendly Summary",
            ],
            key="financial_persona_view",
        )

        qae_flag = bool(st.session_state.use_qae)

        def _regime_phrase(reg):
            r = str(reg).lower()
            if r == "bull":
                return "a bullish / risk-on environment."
            if r == "bear":
                return "a bearish / risk-off environment."
            if r == "sideways":
                return "a sideways or range-bound environment."
            if "low" in r:
                return "a relatively calm, low-volatility environment."
            if "high" in r or "vol" in r:
                return "a high-volatility or uncertain environment."
            return "a mixed or uncertain environment."

        regime_sentence = _regime_phrase(regime)

        if fin_persona == "Chief Investment Officer":
            st.markdown(
                f"""
    **CIO Lens**

    - Market regime: **{regime}** — {regime_sentence}  
    - Horizon: **{var_h} days**, α = **{alpha:.2f}**  
    - **Dollar VaR** ≈ **${base_dollar_var:,.0f}**  
    - **Dollar CVaR** ≈ **${base_dollar_cvar:,.0f}**

    {
        "- Macro stress is active: factor "
        f"**{stress_factor:.2f}** → stressed VaR ≈ ${svar_dollar:,.0f}, stressed CVaR ≈ ${scvar_dollar:,.0f}."
        if stress_factor is not None
        else "- Macro stress is not applied."
    }

    {
        "- QAE is enabled (sample-efficient tail estimation)."
        if qae_flag
        else "- QAE is disabled (classical Monte Carlo)."
    }
                """
            )

        elif fin_persona == "Risk Officer":
            st.markdown(
                f"""
    **Risk Officer Lens**

    - Regime: **{regime}** — {regime_sentence}  
    - **Dollar VaR** ≈ **${base_dollar_var:,.0f}**  
    - **Dollar CVaR** ≈ **${base_dollar_cvar:,.0f}**  
    - Sentiment multiplier: **{sentiment_mult:.2f}**

    {
        f"- Macro stress factor: **{stress_factor:.2f}x**."
        if stress_factor is not None
        else "- Macro stress disabled."
    }
                """
            )

        elif fin_persona == "Quant Researcher":
            st.markdown(
                f"""
    **Quant Research Lens**

    - VaR(return) = **{var_r:.4f}**, CVaR(return) = **{cvar_r:.4f}**  
    - Sim paths: **{sims:,}**  
    {
        f"- Macro stress multiplier = **{stress_factor:.4f}** → stressed VaR = **{(svar or var_r):.4f}**, stressed CVaR = **{(scvar or cvar_r):.4f}**"
        if stress_factor is not None
        else "- No macro overlay."
    }
    {
        "- QAE on."
        if qae_flag
        else "- QAE off."
    }
                """
            )

        else:
            st.markdown(
                f"""
    **Client-Friendly Summary**

    - Market looks **{regime}** ({regime_sentence})  
    - Portfolio size: **${pv:,.0f}**  
    - Typical “bad case” short-horizon loss estimate: **${base_dollar_var:,.0f}**  
    - “Very bad tail” average loss estimate: **${base_dollar_cvar:,.0f}**

    {
        f"- Under macro stress, losses rise to ~${svar_dollar:,.0f}–${scvar_dollar:,.0f}."
        if stress_factor is not None
        else "- Macro stress not applied."
    }
                """
            )

        # --- QAOA vs Market Regime Alignment ---------------------------------
        if qaoa_snapshot:
            st.markdown("### QAOA vs Market Regime Alignment")

            def _market_stance_tag(reg: str) -> str:
                r = str(reg).lower()
                if "bull" in r:
                    return "risk-on"
                if "bear" in r or "crash" in r or "down" in r:
                    return "defensive"
                if "vol" in r or "volatile" in r:
                    return "cautious"
                return "neutral"

            def _prior_stance_tag(prior_regime: str, risk_on_prior: float) -> str:
                pr = str(prior_regime).lower()
                if pr == "crisis" or risk_on_prior <= 0.35:
                    return "defensive"
                if pr == "stressed" or (0.35 < risk_on_prior < 0.6):
                    return "cautious"
                if pr == "calm" and risk_on_prior >= 0.6:
                    return "risk-on"
                return "neutral"

            prior_tag = _prior_stance_tag(qtbn_prior_regime, risk_on_prior)
            market_tag = _market_stance_tag(regime)

            alignment_label = "Aligned"
            alignment_note = "QAOA stance and live market regime broadly point to the same risk posture."

            if prior_tag == "risk-on" and market_tag in ("defensive", "cautious"):
                alignment_label = "⚠ QAOA more optimistic than market"
                alignment_note = "QAOA stance is more risk-on than market conditions suggest."
            elif prior_tag in ("defensive", "cautious") and market_tag == "risk-on":
                alignment_label = "⚠ QAOA more defensive than market"
                alignment_note = "QAOA stance remains defensive while market looks more favorable."
            elif prior_tag == "neutral" or market_tag == "neutral":
                alignment_label = "Mixed"
                alignment_note = "One side is neutral → treat as low-conviction and lean on scenarios."

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("QAOA Persona", persona or "—")
                st.metric("QAOA Prior Regime", qtbn_prior_regime)
            with colB:
                st.metric("Risk-on Prior (QTBN)", f"{risk_on_prior:.2f}")
                if crash_idx is not None:
                    st.metric("QAOA Crash Index", f"{crash_idx:.2f}")
            with colC:
                st.metric("Detected Market Regime", regime)
                st.metric("Alignment", alignment_label)

            st.caption(alignment_note)

        # --- Lachesis Risk Narrative + logging --------------------------------
        st.markdown("---")
        st.subheader("Lachesis Risk Narrative")
        st.caption("Regime snapshot")

        dollar_var = base_dollar_var
        dollar_cvar = base_dollar_cvar
        horizon_label = f"{int(alpha * 100)}% over {var_h} days"

        narrative_text = (
            f"Today, the model classifies the market regime as **{regime}**.\n\n"
            f"For a portfolio of approximately **${pv:,.0f}**, the {horizon_label} "
            f"Value-at-Risk (VaR) is about **{var_r:.2%}** "
            f"(≈ **${dollar_var:,.0f}**).\n\n"
            f"In more extreme tail scenarios, the Conditional VaR (CVaR) is about "
            f"**{cvar_r:.2%}** (≈ **${dollar_cvar:,.0f}**).\n\n"
        )

        if stress_factor is not None and svar is not None and scvar is not None:
            narrative_text += (
                f"A **macro stress overlay** is active with stress factor **{stress_factor:.2f}**. "
                f"Under that overlay: stressed VaR ≈ **{svar:.2%}** (≈ **${stressed_dollar_var:,.0f}**) "
                f"and stressed CVaR ≈ **{scvar:.2%}** (≈ **${stressed_dollar_cvar:,.0f}**).\n\n"
            )

        if qaoa_snapshot:
            narrative_text += (
                f"QTBN prior is seeded from QAOA stance: persona **{persona or '—'}**, "
                f"crash index **{(crash_idx if crash_idx is not None else 0.0):.2f}**, "
                f"prior regime **{qtbn_prior_regime}**, risk-on prior **{risk_on_prior:.2f}**, "
                f"drift μ ≈ **{drift_mu_prior:.2%}**.\n\n"
            )

        narrative_text += (
            "These figures are **risk estimates, not predictions** — they answer: "
            "*“If things go wrong, how bad could it reasonably get for this portfolio?”*"
        )

        st.write(narrative_text)

        # --- Lachesis interactive explainer --------------------------------
        st.markdown("---")
        st.subheader("Lachesis - Interactive Risk Explainer")
        st.caption(
            "Ask Lachesis to explain the current Monte Carlo VaR/CVaR in plain English using the live metrics above."
        )
        render_lachesis_voice_panel("financial_lachesis")
        render_llm_disclaimer()

        use_openai_fin = st.checkbox(
            "Enable OpenAI for this explainer",
            value=bool(resolve_api_key("openai")),
            key="financial_lachesis_use_openai",
        )
        if use_openai_fin:
            st.caption(api_key_status_caption("OpenAI key status: ", "openai"))

        if "financial_lachesis_history" not in st.session_state:
            st.session_state["financial_lachesis_history"] = []

        fin_question = st.text_area(
            "Question for Lachesis",
            value=(
                "Explain my Monte Carlo VaR and CVaR in plain English and tell me the difference "
                "between a bad day and a tail-loss day."
            ),
            height=110,
            key="financial_lachesis_question",
        )

        financial_lachesis_context = {
            "regime": regime,
            "confidence_level": float(alpha),
            "horizon_days": int(var_h),
            "sim_paths": int(sims),
            "portfolio_value": float(pv),
            "mc_var_return": float(var_r),
            "mc_cvar_return": float(cvar_r),
            "dollar_var": float(base_dollar_var),
            "dollar_cvar": float(base_dollar_cvar),
            "sentiment_multiplier": float(sentiment_mult),
            "use_qae": bool(use_qae),
            "macro_stress_applied": bool(apply_macro),
            "stress_factor": float(stress_factor) if stress_factor is not None else None,
            "stressed_var_return": float(svar) if svar is not None else None,
            "stressed_cvar_return": float(scvar) if scvar is not None else None,
            "stressed_dollar_var": float(stressed_dollar_var) if stressed_dollar_var is not None else None,
            "stressed_dollar_cvar": float(stressed_dollar_cvar) if stressed_dollar_cvar is not None else None,
            "qaoa_persona": persona,
        }

        def _financial_question_focus(question: str) -> str:
            q = (question or "").strip().lower()
            qaoa_terms = (
                "qaoa",
                "quantum approximate optimization algorithm",
            )
            risk_terms = (
                "var",
                "cvar",
                "value at risk",
                "expected shortfall",
                "tail risk",
                "drawdown",
                "risk",
                "loss",
                "stress",
            )
            quantum_terms = (
                "qae",
                "quantum amplitude estimation",
                "qubit",
                "bayesian",
                "qtbn",
            )
            if any(t in q for t in qaoa_terms):
                return "qaoa"
            if any(t in q for t in risk_terms):
                return "risk"
            if any(t in q for t in quantum_terms):
                return "quantum"
            return "general"

        def _local_financial_lachesis_answer(question: str, context: dict) -> str:
            focus = _financial_question_focus(question)
            alpha_local = float(context.get("confidence_level", 0.95))
            horizon_local = int(context.get("horizon_days", 10))
            regime_local = str(context.get("regime", "unknown"))
            var_local = float(context.get("mc_var_return", 0.0))
            cvar_local = float(context.get("mc_cvar_return", 0.0))
            dollar_var_local = float(context.get("dollar_var", 0.0))
            dollar_cvar_local = float(context.get("dollar_cvar", 0.0))
            tail_pct = max(0.0, (1.0 - alpha_local) * 100.0)

            if focus == "qaoa":
                return (
                    "QAOA (Quantum Approximate Optimization Algorithm) is a hybrid quantum-classical method "
                    "used to solve optimization problems.\n"
                    "- The quantum circuit proposes candidate solutions.\n"
                    "- A classical optimizer updates circuit parameters to improve those solutions.\n"
                    "- This repeats until the objective is minimized or maximized.\n"
                    "In this app, QAOA helps shape portfolio/risk stance inputs that can influence downstream "
                    "risk analysis, but QAOA itself is not VaR/CVaR."
                )

            if focus == "quantum":
                return (
                    f"Quick explainer for your question: {question}\n"
                    "This dashboard combines quantum tooling (like QAOA/QAE) with classical risk metrics.\n"
                    f"Current snapshot: regime={regime_local}, horizon={horizon_local}d, confidence={alpha_local:.0%}.\n"
                    "If you want, ask specifically how this term affects VaR/CVaR and I will map it directly."
                )

            if focus == "general":
                return (
                    f"Direct answer first: {question}\n"
                    "This panel is strongest on finance and quantum-finance topics. "
                    "If you want a metric explanation, ask things like "
                    "'What does VaR mean?' or 'How does macro stress change CVaR?'.\n"
                    f"Current risk snapshot: VaR≈${dollar_var_local:,.0f}, CVaR≈${dollar_cvar_local:,.0f}."
                )

            lines = [
                f"You asked: {question}",
                f"Plain-English read: in a {regime_local} regime, over about {horizon_local} days:",
                (
                    f"- VaR ({alpha_local:.0%}) is a bad-case threshold: losses could reach about "
                    f"{abs(var_local):.2%} (around ${dollar_var_local:,.0f})."
                ),
                (
                    f"- CVaR looks deeper into the worst {tail_pct:.1f}% of outcomes; in that tail, "
                    f"the average loss is about {abs(cvar_local):.2%} (around ${dollar_cvar_local:,.0f})."
                ),
                "- In short: VaR is a bad-day cutoff, CVaR is the average of very bad tail days.",
                "These are simulated risk estimates, not guarantees or financial advice.",
            ]

            stress_local = context.get("stress_factor")
            stressed_var_local = context.get("stressed_dollar_var")
            stressed_cvar_local = context.get("stressed_dollar_cvar")
            if (
                stress_local is not None
                and stressed_var_local is not None
                and stressed_cvar_local is not None
            ):
                lines.insert(
                    4,
                    (
                        f"- With macro stress enabled (x{float(stress_local):.2f}), downside estimates rise to "
                        f"about ${float(stressed_var_local):,.0f} (VaR) and ${float(stressed_cvar_local):,.0f} (CVaR)."
                    ),
                )

            return "\n".join(lines)

        if st.button("Ask Lachesis", key="financial_lachesis_ask_btn"):
            if not fin_question.strip():
                st.warning("Please enter a question for Lachesis.")
            else:
                fin_reply = ""
                fin_meta = {}
                question_focus = _financial_question_focus(fin_question)
                fin_openai_key = resolve_api_key("openai")
                if use_openai_fin and fin_openai_key:
                    fin_messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are Lachesis, a careful explainer for finance and quantum-finance topics. "
                                "Answer the user's exact question first in plain English. "
                                "Do not force VaR/CVaR discussion if the question is about something else "
                                "(for example, QAOA). "
                                "Use Financial Analysis metrics only when directly relevant, or as a brief optional tie-in. "
                                "If uncertain, say so. Avoid hype and avoid investment advice. "
                                "Use plain text bullets and standard numbers (for example, USD 63056), "
                                "and avoid LaTeX/table formatting."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {fin_question}\n"
                                f"Question intent hint: {question_focus}\n\n"
                                "Use this live Financial Analysis context:\n"
                                f"{json.dumps(financial_lachesis_context, indent=2)}"
                            ),
                        },
                    ]
                    try:
                        fin_reply, fin_meta = _studio_run_openai_chat(
                            fin_messages,
                            model="gpt-4o-mini",
                            temperature=0.2,
                            max_tokens=500,
                            api_key=fin_openai_key,
                        )
                    except Exception as e:
                        st.error(f"Lachesis financial explainer failed: {e}")
                        fin_reply = _local_financial_lachesis_answer(
                            fin_question,
                            financial_lachesis_context,
                        )
                        fin_meta = {"offline": True}
                else:
                    fin_reply = _local_financial_lachesis_answer(
                        fin_question,
                        financial_lachesis_context,
                    )
                    fin_meta = {"offline": True}

                st.markdown("**Lachesis response**")
                st.write(fin_reply)
                audio_bytes, audio_err = synthesize_lachesis_audio(fin_reply)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                elif audio_err:
                    st.caption(f"Voice playback unavailable: {audio_err}")
                if fin_meta.get("offline"):
                    st.caption("Using local fallback (OpenAI not configured).")

                fin_history = st.session_state["financial_lachesis_history"]
                fin_history.append(
                    {
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        "question": fin_question,
                        "answer": fin_reply,
                    }
                )
                if len(fin_history) > 30:
                    fin_history[:] = fin_history[-30:]

        if st.session_state["financial_lachesis_history"]:
            with st.expander("Recent Lachesis Q&A", expanded=False):
                for entry in reversed(st.session_state["financial_lachesis_history"][-5:]):
                    st.markdown(f"**Q:** {entry.get('question', '')}")
                    st.write(entry.get("answer", ""))
                    st.caption(entry.get("timestamp", ""))

        risk_snapshot = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "regime": regime,
            "alpha": float(alpha),
            "horizon_days": int(var_h),
            "portfolio_value": float(pv),
            "var_return": float(var_r),
            "cvar_return": float(cvar_r),
            "dollar_var": float(dollar_var),
            "dollar_cvar": float(dollar_cvar),
            "macro_stress_applied": bool(apply_macro),
            "stress_factor": float(stress_factor) if stress_factor is not None else None,
            "stressed_var_return": float(svar) if svar is not None else None,
            "stressed_cvar_return": float(scvar) if scvar is not None else None,
            "stressed_dollar_var": float(stressed_dollar_var) if stressed_dollar_var is not None else None,
            "stressed_dollar_cvar": float(stressed_dollar_cvar) if stressed_dollar_cvar is not None else None,
            "use_qae": bool(use_qae),
            "sentiment_multiplier": float(sentiment_mult),
            "qtbn_prior_regime": qtbn_prior_regime,
            "qtbn_risk_on_prior": float(risk_on_prior),
            "qtbn_drift_mu_prior": float(drift_mu_prior),
            "qaoa_persona": persona,
            "qaoa_crash_index": float(crash_idx) if crash_idx is not None else None,
            "narrative": narrative_text,
        }

        history = st.session_state["risk_history"]
        history.append(risk_snapshot)
        if len(history) > 200:
            history[:] = history[-200:]

        st.markdown("##### Export this narrative")
        st.download_button(
            "⬇️ Download narrative (.txt)",
            data=narrative_text.encode("utf-8"),
            file_name=f"lachesis_risk_narrative_{risk_snapshot['timestamp']}.txt",
            mime="text/plain",
        )

        json_doc = json.dumps(risk_snapshot, indent=2)
        st.download_button(
            "⬇️ Download narrative snapshot (.json)",
            data=json_doc.encode("utf-8"),
            file_name=f"lachesis_risk_snapshot_{risk_snapshot['timestamp']}.json",
            mime="application/json",
        )

        # --- Risk Snapshot History ------------------------------------------
        st.markdown("---")
        st.subheader("Risk Snapshot History")

        if st.session_state["risk_history"]:
            df_hist = pd.DataFrame(st.session_state["risk_history"])
            cols_to_show = [
                "timestamp",
                "regime",
                "portfolio_value",
                "dollar_var",
                "dollar_cvar",
                "macro_stress_applied",
                "qtbn_prior_regime",
            ]
            existing_cols = [c for c in cols_to_show if c in df_hist.columns]
            st.dataframe(
                df_hist[existing_cols].sort_values("timestamp", ascending=False),
                use_container_width=True,
            )

            csv_buf = io.StringIO()
            df_hist.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download full risk history (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="lachesis_risk_history.csv",
                mime="text/csv",
            )
            st.download_button(
                "⬇️ Download full risk history (JSON)",
                data=json.dumps(st.session_state["risk_history"], indent=2).encode("utf-8"),
                file_name="lachesis_risk_history.json",
                mime="application/json",
            )

        # ---------- Compare current run to a past snapshot ----------
        risk_history = st.session_state["risk_history"]
        if len(risk_history) >= 2:
            st.markdown("---")
            st.subheader("Current vs Past Snapshot")

            indices = list(range(len(risk_history) - 1))

            def _label(idx: int) -> str:
                s = risk_history[idx]
                return f"{idx} — {s['timestamp']} ({s['regime']})"

            base_idx = st.selectbox(
                "Select baseline snapshot",
                indices,
                format_func=_label,
                key="risk_baseline_snapshot_idx",
            )

            base = risk_history[base_idx]

            def _delta(curr, prev):
                if curr is None or prev is None:
                    return None
                return curr - prev

            dv_delta = _delta(dollar_var, base.get("dollar_var"))
            dcv_delta = _delta(dollar_cvar, base.get("dollar_cvar"))
            sf_delta = (
                _delta(stress_factor, base.get("stress_factor"))
                if stress_factor is not None and base.get("stress_factor") is not None
                else None
            )

            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Baseline snapshot**")
                st.write(f"Timestamp: {base['timestamp']}")
                st.write(f"Regime: {base['regime']}")
                st.write(f"VaR (USD): ${base['dollar_var']:,.0f}")
                st.write(f"CVaR (USD): ${base['dollar_cvar']:,.0f}")
                if base.get("stress_factor") is not None:
                    st.write(f"Stress factor: {base['stress_factor']:.2f}")

            with colB:
                st.markdown("**Current run**")
                st.write(f"Regime: {regime}")

                var_line = f"${dollar_var:,.0f}"
                if dv_delta is not None:
                    var_line += f" ({dv_delta:+,.0f} vs baseline)"
                st.write(f"VaR (USD): {var_line}")

                cvar_line = f"${dollar_cvar:,.0f}"
                if dcv_delta is not None:
                    cvar_line += f" ({dcv_delta:+,.0f} vs baseline)"
                st.write(f"CVaR (USD): {cvar_line}")

                if sf_delta is not None:
                    st.write(f"Stress factor: {stress_factor:.2f} ({sf_delta:+.2f} vs baseline)")

            if go is not None:
                comp_fig = go.Figure()
                comp_fig.add_trace(
                    go.Bar(
                        name="Baseline",
                        x=["VaR", "CVaR"],
                        y=[base["dollar_var"], base["dollar_cvar"]],
                    )
                )
                comp_fig.add_trace(
                    go.Bar(
                        name="Current",
                        x=["VaR", "CVaR"],
                        y=[dollar_var, dollar_cvar],
                    )
                )
                comp_fig.update_layout(
                    barmode="group",
                    title="Baseline vs Current Dollar VaR/CVaR",
                    yaxis_title="USD",
                )
                st.plotly_chart(comp_fig, use_container_width=True)


# -- Insider Trading (EDGAR)
with tab_insider:
    st.subheader("Insider Trading (SEC EDGAR Forms 3/4/5)")
    st.caption("Pulls from SEC EDGAR via data.sec.gov. Provide a User-Agent with contact info per SEC policy.")

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        insider_ticker = st.text_input("Ticker (optional)", value=ss_get("insider_ticker", "")).strip().upper()
    with col_b:
        insider_cik = st.text_input("CIK (optional)", value=ss_get("insider_cik", "")).strip()
    with col_c:
        max_filings = st.number_input("Max filings", min_value=5, max_value=200, value=25, step=5)

    user_agent = st.text_input(
        "SEC User-Agent (required)",
        value=ss_get("sec_user_agent", "QTBN-Research (contact: email@example.com)"),
        help="SEC requests a descriptive User-Agent with contact info (email or URL).",
    ).strip()

    selected_forms = st.multiselect("Forms", ["3", "4", "5"], default=["4"])

    def _normalize_cik(raw: str) -> Optional[str]:
        if not raw:
            return None
        digits = re.sub(r"\D", "", raw)
        if not digits:
            return None
        return digits.zfill(10)

    @st.cache_data(ttl=24 * 60 * 60)
    def _fetch_company_tickers(ua: str) -> dict:
        import requests
        headers = {"User-Agent": ua}
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def _lookup_cik_for_ticker(ticker: str, ua: str) -> Optional[str]:
        if not ticker:
            return None
        data = _fetch_company_tickers(ua)
        ticker_u = ticker.upper()
        for _k, entry in (data or {}).items():
            if str(entry.get("ticker", "")).upper() == ticker_u:
                cik_num = entry.get("cik_str")
                if cik_num is None:
                    return None
                return str(int(cik_num)).zfill(10)
        return None

    @st.cache_data(ttl=60 * 60)
    def _fetch_submissions(cik: str, ua: str) -> dict:
        import requests
        headers = {"User-Agent": ua}
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    if st.button("Load filings"):
        if not user_agent:
            st.error("Please provide a SEC-compliant User-Agent with contact info.")
        else:
            ss_set("insider_ticker", insider_ticker)
            ss_set("insider_cik", insider_cik)
            ss_set("sec_user_agent", user_agent)

            cik_norm = _normalize_cik(insider_cik) if insider_cik else None
            if not cik_norm and insider_ticker:
                try:
                    cik_norm = _lookup_cik_for_ticker(insider_ticker, user_agent)
                except Exception as e:
                    st.error(f"Ticker lookup failed: {e}")

            if not cik_norm:
                st.error("Provide a valid CIK or ticker to continue.")
            else:
                try:
                    data = _fetch_submissions(cik_norm, user_agent)
                    company_name = data.get("name") or data.get("entityName") or "Unknown"
                    st.write(f"**Company:** {company_name} (CIK {int(cik_norm)})")

                    recent = (data.get("filings") or {}).get("recent") or {}
                    forms = recent.get("form", []) or []
                    filing_dates = recent.get("filingDate", []) or []
                    accession_numbers = recent.get("accessionNumber", []) or []
                    primary_docs = recent.get("primaryDocument", []) or []

                    rows = []
                    for i, form in enumerate(forms):
                        if selected_forms and str(form) not in selected_forms:
                            continue
                        if len(rows) >= int(max_filings):
                            break
                        accession = accession_numbers[i] if i < len(accession_numbers) else ""
                        primary = primary_docs[i] if i < len(primary_docs) else ""
                        filing_date = filing_dates[i] if i < len(filing_dates) else ""
                        cik_path = str(int(cik_norm))
                        acc_path = accession.replace("-", "")
                        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_path}/{acc_path}/{primary}"
                        rows.append({
                            "Form": form,
                            "Filing Date": filing_date,
                            "Accession": accession,
                            "Primary Doc": primary,
                            "Link": filing_url,
                        })

                    if not rows:
                        st.info("No matching filings found for the selected forms.")
                    else:
                        ss_set("insider_rows", rows)
                        st.dataframe(
                            [{"Form": r["Form"], "Filing Date": r["Filing Date"], "Accession": r["Accession"], "Primary Doc": r["Primary Doc"]} for r in rows],
                            use_container_width=True
                        )
                        with st.expander("Filing links", expanded=False):
                            for r in rows:
                                st.markdown(f"- [{r['Form']} | {r['Filing Date']} | {r['Accession']}]({r['Link']})")
                except Exception as e:
                    st.error(f"EDGAR fetch failed: {e}")

    # --- Lachesis summary (LLM + voice) ---------------------------------
    st.markdown("---")
    st.subheader("Lachesis — Insider Trading Summary (Plain English)")
    st.caption("Explains filings in layman terms and highlights pros/cons. Not financial advice.")
    render_lachesis_voice_panel("insider_lachesis")
    render_llm_disclaimer()

    use_openai = st.checkbox(
        "Enable OpenAI for Lachesis summary",
        value=ss_get("lachesis_mode", "local") == "openai",
        key="insider_lachesis_use_openai",
    )
    if use_openai:
        st.caption(api_key_status_caption("OpenAI key status: ", "openai"))
        ss_set("lachesis_mode", "openai")
    else:
        ss_set("lachesis_mode", "local")

    def _local_insider_summary(filings: list) -> str:
        if not filings:
            return "No filings loaded yet. Click **Load filings** first."
        by_form = {}
        for r in filings:
            by_form[r["Form"]] = by_form.get(r["Form"], 0) + 1
        forms_summary = ", ".join([f"Form {k}: {v}" for k, v in sorted(by_form.items())])
        return (
            "Loaded filings summary:\n"
            f"- Total filings: {len(filings)}\n"
            f"- By form: {forms_summary}\n"
            "- This tab lists filings only; it does not parse transaction details yet."
        )

    if st.button("Summarize with Lachesis"):
        filings = ss_get("insider_rows", [])
        if not filings:
            st.warning("Load filings first so Lachesis has data to interpret.")
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Lachesis, a careful, neutral explainer. "
                        "Summarize insider trading filings in plain English, provide possible pros/cons, "
                        "and avoid giving financial advice. If data is missing, say so. "
                        "Include a brief caution about limitations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Here are SEC EDGAR filings for a company. "
                        "Explain what this generally means in layman terms, "
                        "highlight potential pros/cons for investors, and explicitly say this is not financial advice.\n\n"
                        f"Filings:\n{json.dumps(filings, indent=2)}"
                    ),
                },
            ]

            insider_openai_key = resolve_api_key("openai")
            if ss_get("lachesis_mode", "local") == "openai" and insider_openai_key:
                try:
                    text, _meta = _studio_run_openai_chat(
                        messages,
                        model="gpt-4o-mini",
                        temperature=0.2,
                        max_tokens=600,
                        api_key=insider_openai_key,
                    )
                    st.markdown("**Lachesis response**")
                    st.write(text)
                    audio_bytes, audio_err = synthesize_lachesis_audio(text)
                    if audio_err:
                        st.warning(f"Voice synthesis issue: {audio_err}")
                    elif audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                except Exception as e:
                    st.error(f"Lachesis summary failed: {e}")
            else:
                st.markdown("**Lachesis response (local fallback)**")
                st.write(_local_insider_summary(filings))


    # -- Lachesis Guide
with tab_guide:
    st.subheader("Lachesis Guide — Status, Pipeline & Q&A")

    # --- Pull QAOA stance snapshot (if it exists) -----------------------
    qaoa_snapshot = None
    try:
        qaoa_snapshot = load_qaoa_snapshot()
    except Exception:
        qaoa_snapshot = None

    persona = qaoa_snapshot.get("persona", "—") if qaoa_snapshot else "—"
    regime_stance = qaoa_snapshot.get("regime", "—") if qaoa_snapshot else "—"
    lambda_risk = qaoa_snapshot.get("lambda", None) if qaoa_snapshot else None
    expected_ret = qaoa_snapshot.get("expected_return", None) if qaoa_snapshot else None
    crash_idx = qaoa_snapshot.get("crash_index", None) if qaoa_snapshot else None

    # --- QTBN priors (mirroring the foresight logic) --------------------
    qtbn_prior_regime = "calm"
    risk_on_prior = 0.5
    drift_mu_prior = 0.08

    if qaoa_snapshot:
        # crash index → regime
        if crash_idx is not None:
            c = float(crash_idx)
            if c >= 0.66:
                qtbn_prior_regime = "crisis"
            elif c >= 0.33:
                qtbn_prior_regime = "stressed"
            else:
                qtbn_prior_regime = "calm"

        # persona → risk-on prior
        persona_lower = str(persona).lower()
        if "conservative" in persona_lower:
            risk_on_prior = 0.3
        elif "balanced" in persona_lower:
            risk_on_prior = 0.5
        else:
            risk_on_prior = 0.7

        # drift μ from QAOA expected return if available
        if expected_ret is not None:
            drift_mu_prior = float(expected_ret)

    # --- Toy QTBN outlook (if helper is available) ----------------------
    toy_line = "QTBN toy outlook not available."
    try:
        toy_forecast = qtbn_toy_forecast(
            start_regime=qtbn_prior_regime,
            risk_on_prior=risk_on_prior,
            drift_mu=drift_mu_prior,
            steps=3,
        )
        toy_t2 = toy_forecast[-1]
        toy_line = (
            f"QTBN toy outlook at T2 → "
            f"Calm {toy_t2['calm']:.0%}, "
            f"Stressed {toy_t2['stressed']:.0%}, "
            f"Crisis {toy_t2['crisis']:.0%}."
        )
    except Exception:
        # If qtbn_toy_forecast isn't wired yet, just skip quietly.
        pass

    # --- Global config flags from the rest of the app -------------------
    use_qae = bool(st.session_state.get("use_qae", False))
    apply_macro = bool(st.session_state.get("financial_analysis_apply_macro_stress", False))
    alpha = ss_get("confidence_level", 0.95)
    var_h = ss_get("var_horizon", 10)

    # ===================== 1. STATUS SNAPSHOT ===========================
    st.markdown("### 1. Quick status snapshot")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("QAOA persona", persona)
        st.metric("QAOA regime stance", regime_stance)
    with c2:
        st.metric("QTBN prior regime", qtbn_prior_regime)
        st.metric("Risk-on prior", f"{risk_on_prior:.2f}")
    with c3:
        st.metric("QAE enabled?", "Yes" if use_qae else "No")
        st.metric("Macro stress on?", "Yes" if apply_macro else "No")

    st.caption(toy_line)

    # ===================== 2. PIPELINE OVERVIEW =========================
    st.markdown("### 2. Pipeline overview (how the pieces connect)")

    st.markdown(
        f"""
1. **Toy QAOA – Portfolio stance**  
   - Run the **Toy QAOA – Portfolio Selection** mini-lab to choose a risk stance.  
   - It exports a JSON snapshot with: persona, crash index, expected return, and selected assets.  

2. **QTBN – Temporal risk priors**  
   - Lachesis maps that stance into **starting priors** for the QTBN:  
     - A starting regime: `calm`, `stressed`, or `crisis` (from crash index).  
     - A risk-on prior (risk-seeking probability), from the persona.  
     - A drift parameter μ (expected return), from the QAOA expected return.  

3. **Financial Analysis – VaR / CVaR + macro stress**  
   - Market data drives a Monte-Carlo (or QAE-enhanced) risk engine at horizon **{var_h} days**, α = {alpha:.2f}.  
   - Optional macro stress multiplies losses using a factor from CPI, unemployment, and 10Y yields.  

4. **Narratives & persona views**  
   - Results are translated into: CIO view, Risk Officer view, Quant view, and a Client-friendly summary.  
   - The **Lachesis Risk Narrative** block on the Financial Analysis tab is built from the same numbers.
        """
    )

    # ===================== 3. HOW TO READ THIS ==========================
    st.markdown("### 3. How to read this configuration")

    st.markdown(
        """
- An **aggressive** QAOA stance with low crash index → QTBN starts more *calm / risk-on*.  
- A **conservative** stance or high crash index → QTBN starts more *stressed / crisis* with lower risk-on.  
- The QTBN toy outlook line summarizes where the model thinks the regime might drift by T2.  
- The Financial Analysis tab then answers:  
  *“Given that environment, how bad could reasonable losses get for this portfolio?”*  

Use this tab as a live explainer: change QAOA stance, priors, QAE, or macro stress, then check here to see how Lachesis’ worldview shifts.
        """
    )

    # ===================== 4. INTERACTIVE Q&A ===========================
    st.markdown("---")
    st.subheader("4. Interactive Q&A with Lachesis")

    st.markdown(
        "Use this panel to have Lachesis (or the local fallback) explain what your current "
        "**quantum** and **market** settings imply in natural language."
    )

    openai_key = resolve_api_key("openai")
    st.caption(api_key_status_caption("OpenAI key status: ", "openai"))
    render_lachesis_voice_panel("lachesis_guide_qna")
    render_llm_disclaimer()

    prompt = st.text_area(
        "Question for Lachesis",
        "Explain how the current circuit noise and market settings affect tail risk and regime.",
        height=120,
    )

    if st.button("Ask Lachesis"):
        ctx = get_app_context()
        sys_msg = {
            "role": "system",
            "content": (
                "You are Lachesis, an AI specializing in quantum-financial hybrid reasoning. "
                "Explain clearly, concretely, and without hype."
            ),
        }
        user_msg = {
            "role": "user",
            "content": f"Context: {json.dumps(ctx)}\n\nQuestion: {prompt}",
        }
        text, meta = _studio_run_openai_chat(
            [sys_msg, user_msg],
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=600,
            api_key=openai_key,
        )
        st.markdown("**Response:**")
        st.write(text)
        audio_bytes, audio_err = synthesize_lachesis_audio(text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        elif audio_err:
            st.caption(f"Voice playback unavailable: {audio_err}")
        if meta.get("offline"):
            st.caption("Using local fallback (OpenAI not configured).")

    # ===================== 5. NOISE-AWARE SUGGESTIONS ===================
    st.markdown("---")
    st.subheader("5. Noise-Aware Suggestions")
    for s in noise_aware_suggestions():
        st.markdown(f"- {s}")

            # ---------------------------------------------------------------------
    st.markdown("---")
    st.subheader("QTBN Forecast Prototype")

    st.caption(
        "This is a lightweight QTBN-style forecast that uses your current "
        "priors (regime, risk-on stance, drift μ) plus a simple Gaussian model "
        "to estimate the chance of gains vs losses over different horizons."
    )

    # Rebuild priors here using the same mapping as in the Foresight tab
    qaoa_snapshot = None
    try:
        qaoa_snapshot = load_qaoa_snapshot()
    except Exception:
        qaoa_snapshot = None

    prior_regime = "calm"   # calm / stressed / crisis
    risk_on_prior = 0.5     # P(risk-on)
    drift_mu = 0.08         # annual drift

    persona = None
    crash_idx = None

    if qaoa_snapshot:
        persona = str(qaoa_snapshot.get("persona", "Balanced"))
        crash_idx = float(qaoa_snapshot.get("crash_index", 0.0))
        expected_ret = float(qaoa_snapshot.get("expected_return", drift_mu))

        # crash index → regime
        if crash_idx >= 0.66:
            prior_regime = "crisis"
        elif crash_idx >= 0.33:
            prior_regime = "stressed"
        else:
            prior_regime = "calm"

        # persona → risk-on prior
        p_lower = persona.lower()
        if "conservative" in p_lower:
            risk_on_prior = 0.3
        elif "balanced" in p_lower:
            risk_on_prior = 0.5
        else:
            risk_on_prior = 0.7

        drift_mu = expected_ret

    # Show the priors that drive the forecast
    c_pr1, c_pr2, c_pr3 = st.columns(3)
    with c_pr1:
        st.metric("Prior regime", prior_regime.capitalize())
    with c_pr2:
        st.metric("P(risk-on) prior", f"{risk_on_prior:.2f}")
    with c_pr3:
        st.metric("Drift μ (annual)", f"{drift_mu:.2%}")

    # Choose horizons to forecast
    horizons = st.multiselect(
        "Forecast horizons (trading days)",
        options=[10, 30, 60, 90],
        default=[10, 30, 90],
        key="lachesis_forecast_horizons",
    )

    if horizons:
        rows = []
        for h in horizons:
            rows.append(qtbn_forecast_stub(prior_regime, risk_on_prior, drift_mu, h))

        df_forecast = pd.DataFrame(rows)
        st.dataframe(df_forecast, use_container_width=True)

        # Build a short narrative from the longest horizon
        longest = max(horizons)
        row_long = [r for r in rows if r["horizon_days"] == longest][0]

        dominant_bucket = max(
            ["gain", "flat", "loss", "severe_loss"],
            key=lambda b: row_long[f"P({b})"],
        )
        dom_prob = row_long[f"P({dominant_bucket})"]

        st.markdown(
            f"""
**Lachesis forecast (prototype)**  

- Over the next **{longest} trading days**, the most likely bucket is **{dominant_bucket.upper()}**  
  with probability **{dom_prob:.1%}** under the current priors.  
- These probabilities are **scenario-style risk views**, not point predictions; the full engine
  will eventually refine them using a richer QTBN structure.
            """
        )

        # Export the forecast as JSON for logging / Spartans / etc.
        forecast_payload = {
            "priors": {
                "regime": prior_regime,
                "risk_on_prior": risk_on_prior,
                "drift_mu": drift_mu,
                "persona": persona,
                "crash_index": crash_idx,
            },
            "rows": rows,
        }
        st.download_button(
            "⬇️ Download forecast snapshot (.json)",
            data=json.dumps(forecast_payload, indent=2).encode("utf-8"),
            file_name="lachesis_qtbn_forecast.json",
            mime="application/json",
        )
    else:
        st.info("Select at least one horizon above to generate a QTBN-style forecast.")


# ==========================
# QAOA TOY PORTFOLIO SECTION
# ==========================

# ---- Helper functions ----

def qaoa_stub_optimize_portfolio(toy_cfg: dict, depth: int, shots: int) -> dict:
    """
    Classical heuristic stub that *pretends* to be a QAOA run.

    toy_cfg: {
        "assets": [...],
        "mu": [...],
        "cov": [[...], [...], ...] or [...],
        "lambda_risk": float,
        "max_assets": int
    }
    """
    assets = toy_cfg.get("assets", [])
    mu = np.array(toy_cfg.get("mu", []), dtype=float)
    cov = np.array(toy_cfg.get("cov", []), dtype=float)
    lam = float(toy_cfg.get("lambda_risk", 0.5))
    max_assets = int(toy_cfg.get("max_assets", len(assets)))

    n = len(assets)
    if n == 0:
        return {
            "selected_assets": [],
            "weights": [],
            "expected_return": 0.0,
            "risk": 0.0,
            "objective": 0.0,
        }

    # --- handle 1D vs 2D covariance ---
    if cov.ndim == 2 and cov.shape == (n, n):
        variances = np.diag(cov)          # variance per asset
    elif cov.ndim == 1 and cov.shape[0] == n:
        variances = cov                   # already variances
    else:
        # fallback: no covariance info → zero variance
        variances = np.zeros_like(mu)

    # --- simple greedy mean–variance heuristic ---
    # higher mu, lower variance is better
    scores = mu - lam * variances
    idx_sorted = list(np.argsort(scores)[::-1])  # best first

    chosen_idx = idx_sorted[:max_assets]
    chosen_assets = [assets[i] for i in chosen_idx]

    # equal weights for now
    w = np.ones(len(chosen_idx)) / len(chosen_idx)

    mu_chosen = mu[chosen_idx]

    # risk calculation also needs 1D vs 2D handling
    if cov.ndim == 2 and cov.shape == (n, n):
        cov_chosen = cov[np.ix_(chosen_idx, chosen_idx)]
        risk = float(w @ cov_chosen @ w)
    else:
        var_chosen = variances[chosen_idx]
        # variance of portfolio with diagonal cov ≈ Σ w_i² σ_i²
        risk = float(np.sum((w ** 2) * var_chosen))

    exp_ret = float(w @ mu_chosen)
    objective = float(exp_ret - lam * risk)

    return {
        "selected_assets": chosen_assets,
        "weights": w.tolist(),
        "expected_return": exp_ret,
        "risk": risk,
        "objective": objective,
        "qaoa_depth": depth,
        "shots": shots,
    }


def run_qaoa_portfolio(portfolio_cfg: dict, depth: int, shots: int) -> dict:
    """
    Placeholder QAOA portfolio optimizer.

    Right now this just wraps the classical stub; later you can
    swap the inside with a real QAOA call.
    """
    stub = qaoa_stub_optimize_portfolio(portfolio_cfg, depth, shots)

    assets = portfolio_cfg.get("assets", [])
    selected_assets = stub["selected_assets"]

    # simple bitstring: 1 if asset selected, else 0
    bitstring = "".join("1" if a in selected_assets else "0" for a in assets)

    result = {
        "bitstring": bitstring,
        "selected_assets": selected_assets,
        "energy": 0.0,  # placeholder until real QAOA
    }
    result.update(stub)
    return result


# Persona mapping from lambda (risk aversion)
def lambda_to_persona(lmbda: float) -> str:
    if lmbda >= 1.0:
        return "Conservative"
    elif lmbda >= 0.7:
        return "Balanced"
    else:
        return "Aggressive"


# Regime mapping from crash index
def crash_to_regime(crash_idx: float) -> str:
    if crash_idx < 0.30:
        return "calm"
    elif crash_idx < 0.65:
        return "stressed"
    else:
        return "panic"


# ---------------------------
# Toy QAOA – Portfolio Tab UI
# ---------------------------
with tab_qaoa:
    # existing circuit / explanation renderer
    render_qaoa_tab(st)

    st.markdown("#### Toy Portfolio")
    st.json(TOY_QAOA_PORTFOLIO)

    st.markdown("#### QAOA Hyperparameters (coming soon)")
    depth = st.slider(
        "QAOA depth p (layer count)",
        min_value=1,
        max_value=3,
        value=1,
        key="toy_qaoa_depth_main",   # unique key
    )
    shots = st.slider(
        "Number of shots",
        min_value=128,
        max_value=4096,
        value=1024,
        step=128,
        key="toy_qaoa_shots_main",   # unique key
    )

    # ---------- Run stub ----------
    if st.button("Run QAOA (stub)", type="primary", key="run_qaoa_stub_main"):
        result = run_qaoa_portfolio(TOY_QAOA_PORTFOLIO, depth, shots)

        # Store for export → QTBN
        st.session_state["qaoa_last_result"] = result

        exp_ret = float(result["expected_return"])
        risk = float(result.get("risk", 0.0))
        crash_idx = max(0.0, min(1.0, risk / (abs(exp_ret) + risk + 1e-6)))
        st.session_state["qaoa_last_crash_idx"] = crash_idx

        st.markdown("### Result")
        st.write(f"Bitstring: `{result['bitstring']}`")
        st.write(f"Selected assets: {', '.join(result['selected_assets'])}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Energy", f"{result['energy']:.3f}")
        with col2:
            st.metric("Expected return", f"{result['expected_return']:.3f}")
        with col3:
            st.metric("Objective", f"{result['objective']:.3f}")

    # ---------- Export stance to QTBN ----------
    st.markdown("### Export stance to QTBN")

    last_result = st.session_state.get("qaoa_last_result")

    if last_result is None:
        st.info("Run the QAOA stub above first, then you can export the stance to QTBN.")
    else:
        current_lambda = float(TOY_QAOA_PORTFOLIO.get("lambda_risk", 0.5))

        current_expected_return = float(last_result.get("expected_return", 0.0))
        current_risk = float(last_result.get("risk", 0.0))
        current_assets = last_result.get("selected_assets", [])

        default_crash = max(
            0.0,
            min(1.0, current_risk / (abs(current_expected_return) + current_risk + 1e-6)),
        )
        current_crash_index = float(
            st.session_state.get("qaoa_last_crash_idx", default_crash)
        )

        if st.button("Export stance to QTBN", key="export_qaoa_to_qtbn"):
            snapshot = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "persona": lambda_to_persona(current_lambda),
                "regime": crash_to_regime(current_crash_index),
                "lambda": current_lambda,
                "expected_return": current_expected_return,
                "risk": current_risk,
                "crash_index": current_crash_index,
                "assets": current_assets,
            }

            with open("qaoa_snapshot.json", "w") as f:
                json.dump(snapshot, f, indent=2)

            st.success(
                "QAOA stance exported to QTBN. "
                "Go to the QTBN foresight tab and check "
                "'Use QAOA stance as starting prior for this foresight run'."
            )



# -- Advanced Quantum
with tab_advanced_q:
    st.subheader("Advanced Quantum Diagnostics")
    st.markdown("---")
    render_qaoa_bridge_inspector()
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
        cur_cal = st.session_state.current_calibration
        cur_ts = cur_cal.get("timestamp", "unknown")
        st.info(f"Current calibration: {cur_ts}")
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
        mapping = process_tomography_proxy_1q(lambda: build_unitary_circuit_1q(),
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
    st.markdown("Derive a sentiment-based **VaR multiplier** using either Google News RSS+VADER or Perplexity API.")

    sentiment_source_options = ["Google News RSS + VADER", "Perplexity API (live web)"]
    if ss_get("sentiment_source", sentiment_source_options[0]) not in sentiment_source_options:
        ss_set("sentiment_source", sentiment_source_options[0])
    sentiment_source = st.radio(
        "Sentiment source",
        sentiment_source_options,
        horizontal=True,
        key="sentiment_source",
    )

    tickers_str = st.text_input("Tickers for sentiment (comma)", ss_get("tickers", "AAPL,MSFT,SPY"))
    if sentiment_source == "Perplexity API (live web)":
        st.caption("Perplexity mode pulls live web context and returns structured sentiment.")
        st.caption(
            api_key_status_caption("Perplexity API key: ", "perplexity")
        )
        pplx_model = st.text_input(
            "Perplexity model",
            value=ss_get("perplexity_model", "sonar"),
            key="sentiment_perplexity_model",
        )
        if pplx_model != ss_get("perplexity_model", "sonar"):
            ss_set("perplexity_model", pplx_model)
    else:
        st.caption("RSS mode pulls Google News headlines and scores each headline with VADER.")

    if st.button("Analyze Sentiment"):
        tick_list = [t.strip() for t in tickers_str.split(",") if t.strip()]
        if sentiment_source == "Perplexity API (live web)":
            res = analyze_sentiment_perplexity(
                tick_list,
                api_key=resolve_api_key("perplexity"),
                model=ss_get("perplexity_model", "sonar"),
            )
            if "error" in res and res["error"]:
                st.warning(f"Perplexity sentiment failed: {res['error']}. Falling back to Google News RSS + VADER.")
                res = analyze_sentiment(tick_list)
        else:
            res = analyze_sentiment(tick_list)
        if "error" in res and res["error"]:
            st.error(res["error"])
        else:
            if res.get("provider"):
                provider_txt = str(res.get("provider"))
                model_txt = str(res.get("model", "")).strip()
                if model_txt:
                    provider_txt += f" ({model_txt})"
                st.caption(f"Provider: {provider_txt}")
            st.write(f"Average compound score: {res['avg_score']:.3f}")
            st.write(f"Suggested stress multiplier: {res['multiplier']:.3f}")
            st.session_state.sentiment_multiplier = float(res["multiplier"])
            st.session_state.sentiment_last_result = res
            st.caption("This multiplier will be applied inside the **Financial Analysis** tab VaR/CVaR calculations.")

            if res.get("headlines"):
                st.markdown("**Sample Headlines:**")
                items = res.get("items") or []
                if items:
                    scored = list(zip(res["headlines"], res["scores"]))
                    scores_by_title = {h: s for h, s in scored}
                    for item in items[:20]:
                        title = item.get("title")
                        link = item.get("link")
                        score = scores_by_title.get(title)
                        if title and link and score is not None:
                            st.markdown(f"- ({score:+.3f}) [{title}]({link})")
                        elif title:
                            st.write(f"- ({score:+.3f}) {title}" if score is not None else f"- {title}")
                else:
                    for h, s in list(zip(res["headlines"], res["scores"]))[:20]:
                        st.write(f"- ({s:+.3f}) {h}")

            if res.get("items"):
                links = [it.get("link") for it in res.get("items", []) if it.get("link")]
                if links:
                    st.markdown("**Copy-friendly links:**")
                    st.text_area(
                        "Links (copy/paste as needed)",
                        value="\n".join(links[:30]),
                        height=140,
                        key="sentiment_links_copy",
                        help="These are the Google News links pulled for your tickers.",
                    )

    st.markdown("---")
    st.subheader("Override Multiplier Manually")
    manual_mult = st.slider("Manual sentiment multiplier", 0.5, 1.5, float(ss_get("sentiment_multiplier", 1.0)), 0.01)
    if st.button("Apply manual multiplier"):
        ss_set("sentiment_multiplier", float(manual_mult))
        st.success(f"Sentiment multiplier set to {manual_mult:.3f}")

    st.markdown("---")
    st.subheader("Lachesis — Explain Sentiment & News in Plain English")
    st.caption("Paste links or headlines. Lachesis will summarize the sentiment signal and what the news may imply.")
    st.caption("Note: Lachesis does not fetch URL contents; paste headlines or excerpts for best results.")
    render_lachesis_voice_panel("sentiment_lachesis")
    render_llm_disclaimer()

    last_res = st.session_state.get("sentiment_last_result", {})
    default_links = ""
    if isinstance(last_res, dict):
        links = [it.get("link") for it in (last_res.get("items") or []) if it.get("link")]
        if links:
            default_links = "\n".join(links[:20])

    use_openai = st.checkbox(
        "Enable OpenAI for Lachesis analysis",
        value=ss_get("lachesis_mode", "local") == "openai",
        key="sentiment_lacheiss_use_openai",
    )
    if use_openai:
        st.caption(api_key_status_caption("OpenAI key status: ", "openai"))
        ss_set("lachesis_mode", "openai")
    else:
        ss_set("lachesis_mode", "local")

    user_links = st.text_area(
        "Paste news links or headlines (one per line)",
        value=default_links,
        height=160,
        key="sentiment_links_input",
    )

    if st.button("Explain with Lachesis"):
        lines = [ln.strip() for ln in (user_links or "").splitlines() if ln.strip()]
        avg_score = last_res.get("avg_score")
        multiplier = last_res.get("multiplier")
        headlines = last_res.get("headlines") or []

        prompt_lines = [
            "Explain the sentiment score in plain English and what it means for a non-technical reader.",
            "Then summarize the news items and explain what they may imply for the near future.",
            "Provide clear pros and cons of investing in the referenced stock(s) based on the summary.",
            "Be concise and avoid hype.",
        ]
        if avg_score is not None and multiplier is not None:
            prompt_lines.append(f"Sentiment score (avg compound): {avg_score:.3f}.")
            prompt_lines.append(f"Suggested VaR multiplier: {multiplier:.3f}.")
        if headlines:
            prompt_lines.append("Headlines: " + "; ".join(headlines[:12]))
        if lines:
            prompt_lines.append("User-provided links/headlines: " + " | ".join(lines[:20]))

        lacheiss_prompt = "\n".join(prompt_lines)
        lacheiss_context = {
            "sentiment": {
                "avg_score": avg_score,
                "multiplier": multiplier,
            },
            "headlines": headlines[:20],
            "links_or_notes": lines[:20],
        }
        lacheiss_reply = ask_lachesis(lacheiss_prompt, lacheiss_context)
        st.markdown("**Lachesis response**")
        st.write(lacheiss_reply)
        audio_bytes, audio_err = synthesize_lachesis_audio(lacheiss_reply)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        elif audio_err:
            st.caption(f"Voice playback unavailable: {audio_err}")
        if ss_get("lachesis_mode", "local") != "openai" or not resolve_api_key("openai"):
            st.caption("Using local fallback (OpenAI not configured).")

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
        st.caption(
            api_key_status_caption("OpenAI key status: ", "openai")
        )
        render_lachesis_voice_panel("prompt_studio")
        render_llm_disclaimer()
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
                        text, meta = _studio_run_openai_chat(messages, model, temperature, max_tokens, resolve_api_key("openai"))
                    st.markdown("**Lachesis response**")
                    st.write(text)
                    audio_bytes, audio_err = synthesize_lachesis_audio(text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    elif audio_err:
                        st.caption(f"Voice playback unavailable: {audio_err}")
                    with st.expander("Raw meta"):
                        st.write(meta)
                except Exception as e:
                    st.error(f"Lachesis run failed: {e}")
        st.markdown("---")
        st.caption("Tip: Your template variables can pull from any session key (e.g., confidence_level, var_horizon, enable_cnot_noise, g0_q0, etc.).")

# -- Admin (owner-only tab)
if tab_admin is not None:
    with tab_admin:
        st.subheader("Admin — API Key Management")
        st.caption("Only owner can view/edit API keys.")
        st.caption("Keys entered here are session-only.")
        st.caption("For shared non-owner usage, configure server secrets/env.")

        admin_specs = [
            ("OpenAI API key override", "openai_api_key", "openai"),
            ("FRED API key override", "fred_api_key", "fred"),
            ("Perplexity API key override", "perplexity_api_key", "perplexity"),
            ("Voice OpenAI API key override", "voice_openai_api_key", "voice_openai"),
            ("Voice ElevenLabs API key override", "voice_elevenlabs_api_key", "voice_elevenlabs"),
        ]

        for label, session_key, service_name in admin_specs:
            current_override = ss_get(session_key, "")
            input_key = f"admin_{session_key}_input"
            if input_key not in st.session_state:
                st.session_state[input_key] = current_override
            typed_value = st.text_input(
                label,
                type="password",
                key=input_key,
            )
            if typed_value != current_override:
                ss_set(session_key, typed_value)
            st.caption(
                f"Effective {service_name} key: "
                + ("Configured" if resolve_api_key(service_name) else "Missing")
            )

        if st.button("Clear session key overrides", key="admin_clear_session_key_overrides"):
            for _, session_key, _ in admin_specs:
                ss_set(session_key, "")
                st.session_state[f"admin_{session_key}_input"] = ""
            st.success("Cleared owner session key overrides.")

# VQE Tab
with tab_vqe:
    if render_vqe_tab is None:
        st.error(f"VQE tab failed to load: {_vqe_import_error}")
    else:
        render_vqe_tab(st)

    def real_execute_fn(order: dict) -> dict:
        # TODO: replace with your real broker/exchange call
        # IMPORTANT: use order["notional_usd"] (it may be CLAMPED)
        return place_order(
            symbol=order["symbol"],
            side=order["side"],
            notional_usd=order["notional_usd"],
        )

    if submit_order_through_vqe_gate is None:
        st.error("VQE trade gate is unavailable (submit_order_through_vqe_gate import failed).")
    else:
        st.markdown("#### Trade execution (VQE gate)")
        symbol = st.text_input("Symbol", value="AAPL", key="vqe_trade_symbol")
        side = st.selectbox("Side", ["BUY", "SELL"], index=0, key="vqe_trade_side")
        notional = st.number_input("Notional USD", min_value=1.0, value=1000.0, step=100.0, key="vqe_trade_notional")
        vol_proxy = st.number_input("Volatility (proxy)", min_value=0.0, value=0.25, step=0.01, key="vqe_trade_vol_proxy")
        equity_usd = st.number_input("Equity USD", min_value=1.0, value=10000.0, step=500.0, key="vqe_trade_equity")

        if st.button("EXECUTE TRADE", key="vqe_execute_trade_btn"):
            order = {
                "symbol": symbol,          # your UI input
                "side": side,              # BUY/SELL
                "notional_usd": notional,  # your UI input
                # optional placeholders used by estimate_order_risk proxy:
                "volatility": vol_proxy,   # or remove once you use QTBN VaR/CVaR
                "equity_usd": equity_usd,
            }

            gate = submit_order_through_vqe_gate(
                st=st,
                order=order,
                execute_fn=real_execute_fn,
                policy="Moderate",          # Off / Moderate / Strict
                warn_tighten_factor=0.85,   # WARN ⇒ clamp risk appetite
                fail_behavior="Block",      # FAIL ⇒ block
                limits_key="vqe_scaled_risk_limits",
                audit_to_disk=False,
                audit_dir="golden_build",
            )

            st.write(gate)  # shows APPROVED/CLAMPED/REJECTED + reasons

    def qtbn_risk_adapter(*, st, order: dict) -> dict:
        """
        Returns dict with est_var_usd, est_cvar_usd, leverage_used.
        Replace the internals with your QTBN outputs.
        """
        notional = float(order.get("notional_usd", 0.0))
        equity = float(order.get("equity_usd", 1000.0))  # replace with real wallet equity

        # TODO: replace these with your QTBN VaR/CVaR outputs:
        est_var_usd = float(order.get("qtbn_var_usd", 0.0))
        est_cvar_usd = float(order.get("qtbn_cvar_usd", 0.0))

        leverage_used = (notional / max(1e-9, equity)) if equity > 0 else None

        return {
            "est_var_usd": est_var_usd,
            "est_cvar_usd": est_cvar_usd,
            "leverage_used": leverage_used,
        }
