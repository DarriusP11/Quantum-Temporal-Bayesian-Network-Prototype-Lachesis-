# vqe_tab.py
# ------------------------------------------------------------
# VQE Tab (Mini-Lab, isolated) — reproducible runs + diagnostics
#
# IMPLEMENTED (per roadmap):
# 1) MaxCut Graph Generator:
#    - seeded generator
#    - weighted/unweighted edges
#    - button "Generate → fill Edges"
#    - optional "Set qubits = n nodes"
# 2) Graph visualization:
#    - Plotly if available; otherwise edge table fallback
# 3) Hamiltonian preview:
#    - constant shift, H=-cost toggle, #terms, first ~10 Pauli terms
# 4) Result interpretation panel:
#    - VQE implied MaxCut score (cost)
#    - exact best cut (when n <= exact_max_n) + best bitstring
#    - gap %
# 5) VQE bitstring decoding (MaxCut/Ising):
#    - use optimal VQE parameters to sample / compute probabilities
#    - report best sampled bitstring, expected cost/energy, top-K outcomes
#    - supports Statevector (exact probs) when available; Aer shots when available; safe fallbacks
#
# NEW (previous update):
# 8.8) RISK GATE CHOKE-POINT INTEGRATION (bridge-ready):
#    - builds scaled risk limits from VQE multiplier
#    - stores st.session_state["vqe_scaled_risk_limits"]
#    - provides enforce_risk_gate_and_execute(...) for downstream trade execution
#    - includes a "simulate order" panel so you can validate gate behavior immediately
#
# NEW (this update):
# 8.9) EXECUTION PATH WIRING TEMPLATE (smoke policy + risk estimation + audit trail):
#    - estimate_order_risk(...) placeholder you can upgrade to QTBN/VaR/CVaR
#    - enforce_smoke_policy_and_execute(...) blocks/clamps based on vqe_smoke PASS/WARN/FAIL
#    - submit_order_through_vqe_gate(...) single "order submit" wrapper (the integration target)
#    - trade audit log stored in st.session_state["trade_audit_log"] (+ optional disk write)
#
# Notes:
# - Tab is isolated: only uses vqe_* keys + exports artifacts.
# - Does NOT mutate non-vqe keys except shared bridge keys explicitly documented:
#     - st.session_state["vqe_scaled_risk_limits"]
#     - st.session_state["risk_gate_last"]
#     - st.session_state["trade_audit_log"]   (NEW)
# - Safe fallbacks for missing deps (plotly/networkx/pandas/qiskit).
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import glob
import math
import datetime as dt
from typing import Any, Dict, Optional, Tuple, List, Callable
from collections import OrderedDict
import numpy as np

# -----------------------------
# ON/OFF SWITCH (requested)
# -----------------------------
VQE_TAB_ENABLED = True  # flip to False to hard-disable this tab safely

# Optional Plotly (safe fallback if unavailable)
try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None

# Optional pandas (for nicer tables)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

# Optional networkx (graph generation + layout)
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None


GOLDEN_DIR_DEFAULT = "golden_build"


# ============================================================
# STEP 8.8 — RISK GATE CHOKE-POINT (BRIDGE UTILITIES)
# ============================================================
def _finite_or_none(x: Any) -> Optional[float]:
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else None
    except Exception:
        return None


def build_scaled_risk_limits(
    *,
    base_limits: Dict[str, float],
    risk_budget_multiplier: float,
) -> Dict[str, float]:
    """
    Scale base risk limits using the VQE-derived multiplier in [0.5, 1.5].

    Interpretation (default):
      higher multiplier => allow more risk budget
      lower multiplier  => tighten risk budget

    By default, we scale "maximum allowed" limits upward/downward:
      - max_notional_usd
      - max_position_usd
      - max_daily_loss_usd
      - max_var_usd
      - max_cvar_usd
      - max_leverage

    You can change which fields are scaled by editing `scale_keys`.
    """
    mult = float(max(0.1, min(5.0, risk_budget_multiplier)))
    scale_keys = {
        "max_notional_usd",
        "max_position_usd",
        "max_daily_loss_usd",
        "max_var_usd",
        "max_cvar_usd",
        "max_leverage",
    }

    out: Dict[str, float] = {}
    for k, v in (base_limits or {}).items():
        try:
            vv = float(v)
        except Exception:
            continue
        out[k] = float(vv * mult) if k in scale_keys else float(vv)

    out["risk_budget_multiplier"] = float(mult)
    return out


def apply_risk_gates(
    *,
    requested_notional_usd: float,
    est_var_usd: Optional[float],
    est_cvar_usd: Optional[float],
    leverage_used: Optional[float],
    limits: Dict[str, float],
) -> Dict[str, Any]:
    """
    Core gate decision:
      - REJECTED: violates hard limits (VaR/CVaR/Leverage) or missing required data (if configured).
      - CLAMPED: notional exceeds max_notional_usd (or max_position_usd).
      - APPROVED: safe.

    Returns a normalized gate dict with final_notional_usd and reasons.
    """
    reasons: List[str] = []
    status = "APPROVED"

    req = float(max(0.0, requested_notional_usd))
    max_notional = float(limits.get("max_notional_usd", req))
    max_position = float(limits.get("max_position_usd", max_notional))

    max_var = _finite_or_none(limits.get("max_var_usd", None))
    max_cvar = _finite_or_none(limits.get("max_cvar_usd", None))
    max_lev = _finite_or_none(limits.get("max_leverage", None))

    varv = _finite_or_none(est_var_usd)
    cvarv = _finite_or_none(est_cvar_usd)
    levv = _finite_or_none(leverage_used)

    # Hard rejects
    if max_var is not None and varv is not None and varv > max_var:
        status = "REJECTED"
        reasons.append(f"VaR breach: est_var_usd={varv:.4f} > max_var_usd={max_var:.4f}")

    if status != "REJECTED" and max_cvar is not None and cvarv is not None and cvarv > max_cvar:
        status = "REJECTED"
        reasons.append(f"CVaR breach: est_cvar_usd={cvarv:.4f} > max_cvar_usd={max_cvar:.4f}")

    if status != "REJECTED" and max_lev is not None and levv is not None and levv > max_lev:
        status = "REJECTED"
        reasons.append(f"Leverage breach: leverage_used={levv:.4f} > max_leverage={max_lev:.4f}")

    # Clamp notional
    final_notional = req
    clamp_cap = float(min(max_notional, max_position))
    if status != "REJECTED" and final_notional > clamp_cap:
        status = "CLAMPED"
        reasons.append(f"Notional clamped: requested={req:.2f} -> cap={clamp_cap:.2f}")
        final_notional = clamp_cap

    return {
        "status": status,
        "requested_notional_usd": float(req),
        "final_notional_usd": float(final_notional),
        "est_var_usd": varv,
        "est_cvar_usd": cvarv,
        "leverage_used": levv,
        "limits": dict(limits or {}),
        "reasons": reasons,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }


def enforce_risk_gate_and_execute(
    *,
    st,
    requested_notional_usd: float,
    est_var_usd: Optional[float],
    est_cvar_usd: Optional[float],
    leverage_used: Optional[float],
    execute_fn: Callable[[float], Any],
    limits_key: str = "vqe_scaled_risk_limits",
) -> Dict[str, Any]:
    """
    STEP 8.8 CHOKE-POINT FUNCTION (downstream trade execution should call this)

    - Pulls scaled limits from st.session_state[limits_key]
    - Applies risk gates
    - If REJECTED: does NOT execute
    - If CLAMPED: executes with reduced notional
    - If APPROVED: executes with requested notional

    Returns the gate dict, plus optional "execution" output if executed.
    """
    limits = st.session_state.get(limits_key, None)
    if not isinstance(limits, dict) or not limits:
        gate = {
            "status": "REJECTED",
            "reasons": [f"Missing risk limits in session_state['{limits_key}']"],
            "requested_notional_usd": float(requested_notional_usd),
            "final_notional_usd": 0.0,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state["risk_gate_last"] = gate
        return gate

    gate = apply_risk_gates(
        requested_notional_usd=float(requested_notional_usd),
        est_var_usd=est_var_usd,
        est_cvar_usd=est_cvar_usd,
        leverage_used=leverage_used,
        limits=limits,
    )

    exec_out = None
    if gate["status"] in ("APPROVED", "CLAMPED"):
        try:
            exec_out = execute_fn(float(gate["final_notional_usd"]))
        except Exception as e:
            gate["status"] = "REJECTED"
            gate["reasons"].append(f"Execution error: {e}")
            exec_out = None

    if exec_out is not None:
        gate["execution"] = exec_out

    st.session_state["risk_gate_last"] = gate
    return gate


# ============================================================
# STEP 8.9 — EXECUTION PATH WIRING TEMPLATE (SMOKE + ESTIMATE + AUDIT)
# ============================================================
def estimate_order_risk(
    *,
    order: Dict[str, Any],
    confidence: float = 0.95,
    z_095: float = 1.65,
    cvar_multiplier: float = 1.30,
) -> Dict[str, Optional[float]]:
    """
    Placeholder risk estimator (upgrade target).

    Intended final version:
      - call your QTBN / VaR / CVaR estimator using live returns + horizon + regime
      - return est_var_usd, est_cvar_usd, leverage_used

    This placeholder uses a simple proxy:
      - var ≈ notional * volatility * z
      - cvar ≈ var * cvar_multiplier
      - leverage ≈ notional / equity (if equity provided)

    Order dict fields (optional):
      - "notional_usd" (required for proxy)
      - "volatility" (e.g. daily stdev; default 0.02)
      - "equity_usd"  (for leverage proxy)
    """
    notional = _finite_or_none(order.get("notional_usd", None))
    if notional is None:
        return {"est_var_usd": None, "est_cvar_usd": None, "leverage_used": None}

    vol = _finite_or_none(order.get("volatility", None))
    if vol is None:
        vol = 0.02  # safe proxy default

    z = float(z_095) if float(confidence) >= 0.90 else 1.28
    est_var = float(abs(notional) * abs(vol) * z)
    est_cvar = float(est_var * float(max(1.0, cvar_multiplier)))

    equity = _finite_or_none(order.get("equity_usd", None))
    lev = None
    if equity is not None and equity > 0:
        lev = float(abs(notional) / equity)

    return {"est_var_usd": est_var, "est_cvar_usd": est_cvar, "leverage_used": lev}


def _get_smoke_status(st) -> str:
    s = st.session_state.get("vqe_smoke", None)
    if isinstance(s, dict):
        return str(s.get("status", "WARN")).upper()
    return "WARN"


def enforce_smoke_policy_and_execute(
    *,
    st,
    requested_notional_usd: float,
    est_var_usd: Optional[float],
    est_cvar_usd: Optional[float],
    leverage_used: Optional[float],
    execute_fn: Callable[[float], Any],
    policy: str = "Moderate",
    warn_tighten_factor: float = 0.85,
    fail_behavior: str = "Block",
    limits_key: str = "vqe_scaled_risk_limits",
) -> Dict[str, Any]:
    """
    STEP 8.9 wrapper:
      - reads vqe_smoke status (PASS/WARN/FAIL)
      - applies a policy BEFORE calling the Step 8.8 choke-point

    Policy modes:
      - "Off"       : ignore smoke entirely
      - "Moderate"  : WARN => tighten notional/limits by factor; FAIL => block or clamp
      - "Strict"    : WARN => block unless user overrides (here: treated as FAIL behavior)

    fail_behavior:
      - "Block" : hard reject (no execution)
      - "Clamp" : allow but force notional to 0 (effectively block) OR you can change to small cap

    Returns a gate dict (same shape as Step 8.8), with added "smoke" metadata.
    """
    smoke_status = _get_smoke_status(st)
    pol = str(policy).strip().title()
    fb = str(fail_behavior).strip().title()

    # Off => route straight to risk gate
    if pol == "Off":
        gate = enforce_risk_gate_and_execute(
            st=st,
            requested_notional_usd=float(requested_notional_usd),
            est_var_usd=est_var_usd,
            est_cvar_usd=est_cvar_usd,
            leverage_used=leverage_used,
            execute_fn=execute_fn,
            limits_key=limits_key,
        )
        gate["smoke"] = {"status": smoke_status, "policy": pol}
        return gate

    # Strict treats WARN like FAIL
    if pol == "Strict" and smoke_status in ("WARN", "FAIL"):
        smoke_status = "FAIL"

    # FAIL behavior
    if smoke_status == "FAIL":
        if fb == "Clamp":
            # You can choose to allow a tiny amount; by default we block by setting to 0.
            gate = {
                "status": "REJECTED",
                "requested_notional_usd": float(requested_notional_usd),
                "final_notional_usd": 0.0,
                "est_var_usd": _finite_or_none(est_var_usd),
                "est_cvar_usd": _finite_or_none(est_cvar_usd),
                "leverage_used": _finite_or_none(leverage_used),
                "reasons": [f"Smoke FAIL under policy={pol}: execution prevented."],
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            }
        else:
            gate = {
                "status": "REJECTED",
                "requested_notional_usd": float(requested_notional_usd),
                "final_notional_usd": 0.0,
                "est_var_usd": _finite_or_none(est_var_usd),
                "est_cvar_usd": _finite_or_none(est_cvar_usd),
                "leverage_used": _finite_or_none(leverage_used),
                "reasons": [f"Smoke FAIL under policy={pol}: execution blocked."],
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            }
        st.session_state["risk_gate_last"] = gate
        gate["smoke"] = {"status": "FAIL", "policy": pol, "fail_behavior": fb}
        return gate

    # WARN => tighten by scaling down requested notional (and effectively making gate stricter)
    # If you want to tighten limits instead, you can also pre-scale st.session_state[limits_key].
    req = float(requested_notional_usd)
    if smoke_status == "WARN":
        f = float(max(0.05, min(1.0, warn_tighten_factor)))
        req = float(req * f)

    gate = enforce_risk_gate_and_execute(
        st=st,
        requested_notional_usd=float(req),
        est_var_usd=est_var_usd,
        est_cvar_usd=est_cvar_usd,
        leverage_used=leverage_used,
        execute_fn=execute_fn,
        limits_key=limits_key,
    )
    gate["smoke"] = {"status": smoke_status, "policy": pol, "warn_tighten_factor": float(warn_tighten_factor)}
    if smoke_status == "WARN":
        gate.setdefault("reasons", [])
        gate["reasons"].append(f"Smoke WARN: tightened requested_notional by factor={float(warn_tighten_factor):.2f}")
    return gate


def _append_trade_audit(st, record: Dict[str, Any], *, max_len: int = 200) -> None:
    """
    Append an audit record to st.session_state['trade_audit_log'] (bounded).
    """
    if "trade_audit_log" not in st.session_state or not isinstance(st.session_state.get("trade_audit_log"), list):
        st.session_state["trade_audit_log"] = []
    log: List[Dict[str, Any]] = st.session_state["trade_audit_log"]  # type: ignore
    log.append(record)
    if len(log) > int(max_len):
        st.session_state["trade_audit_log"] = log[-int(max_len):]


def _write_audit_json(gb_dir: str, record: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort audit persistence.
    """
    try:
        os.makedirs(gb_dir, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(gb_dir, f"trade_audit_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        return path
    except Exception:
        return None


def submit_order_through_vqe_gate(
    *,
    st,
    order: Dict[str, Any],
    execute_fn: Callable[[Dict[str, Any]], Any],
    policy: str = "Moderate",
    warn_tighten_factor: float = 0.85,
    fail_behavior: str = "Block",
    limits_key: str = "vqe_scaled_risk_limits",
    audit_to_disk: bool = False,
    audit_dir: str = GOLDEN_DIR_DEFAULT,
) -> Dict[str, Any]:
    """
    STEP 8.9: This is the function you wire into your REAL order execution button.

    Flow:
      1) Estimate risk for this order (placeholder or QTBN call).
      2) Apply smoke policy (PASS/WARN/FAIL) + Step 8.8 risk gate.
      3) If approved/clamped => execute_fn(order_with_final_notional)
      4) Audit everything into session_state['trade_audit_log'] (and optional disk)

    Expected order keys:
      - "symbol", "side", "notional_usd" (at minimum for useful logs)
      - optional "volatility", "equity_usd" for placeholder risk estimate
    """
    # 1) risk estimate
    risk = estimate_order_risk(order=order)
    est_var = risk.get("est_var_usd", None)
    est_cvar = risk.get("est_cvar_usd", None)
    lev = risk.get("leverage_used", None)

    req_notional = _finite_or_none(order.get("notional_usd", None))
    if req_notional is None:
        gate = {
            "status": "REJECTED",
            "requested_notional_usd": 0.0,
            "final_notional_usd": 0.0,
            "reasons": ["Order missing notional_usd."],
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state["risk_gate_last"] = gate
        audit = {
            "timestamp": gate["timestamp"],
            "order": dict(order),
            "risk_estimate": dict(risk),
            "gate": dict(gate),
            "vqe_snapshot_meta": st.session_state.get("vqe_snapshot", {}).get("timestamp") if isinstance(st.session_state.get("vqe_snapshot"), dict) else None,
        }
        _append_trade_audit(st, audit)
        if audit_to_disk:
            _write_audit_json(audit_dir, audit)
        return gate

    # 2) smoke policy + gate
    def _execute_with_final_notional(final_notional: float) -> Any:
        order2 = dict(order)
        order2["notional_usd"] = float(final_notional)
        return execute_fn(order2)

    gate = enforce_smoke_policy_and_execute(
        st=st,
        requested_notional_usd=float(req_notional),
        est_var_usd=est_var,
        est_cvar_usd=est_cvar,
        leverage_used=lev,
        execute_fn=lambda final_notional: _execute_with_final_notional(final_notional),
        policy=policy,
        warn_tighten_factor=float(warn_tighten_factor),
        fail_behavior=fail_behavior,
        limits_key=limits_key,
    )

    # 3) audit
    audit = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "order": dict(order),
        "risk_estimate": dict(risk),
        "gate": dict(gate),
        "vqe_smoke": st.session_state.get("vqe_smoke") if isinstance(st.session_state.get("vqe_smoke"), dict) else None,
        "vqe_snapshot_timestamp": st.session_state.get("vqe_snapshot", {}).get("timestamp") if isinstance(st.session_state.get("vqe_snapshot"), dict) else None,
    }
    _append_trade_audit(st, audit)
    if audit_to_disk:
        _write_audit_json(audit_dir, audit)

    return gate


# -----------------------------
# Filesystem helpers (safe)
# -----------------------------
def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _write_json(path: str, obj: Any) -> bool:
    try:
        parent = os.path.dirname(path)
        if parent:
            _ensure_dir(parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return True
    except Exception:
        return False


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _latest_snapshot_path(gb_dir: str) -> Optional[str]:
    try:
        paths = sorted(glob.glob(os.path.join(gb_dir, "vqe_snapshot_*.json")))
        return paths[-1] if paths else None
    except Exception:
        return None


def _latest_smoke_path(gb_dir: str) -> Optional[str]:
    try:
        paths = sorted(glob.glob(os.path.join(gb_dir, "vqe_smoke_*.json")))
        return paths[-1] if paths else None
    except Exception:
        return None


# -----------------------------
# Parsing helpers
# -----------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _parse_pauli_list(text: str) -> List[Tuple[str, float]]:
    """
    Accepts formats like:
      "ZZ:1, XI:0.4, IX:0.4"
      "ZZ=1;XI=0.4;IX=0.4"
      "ZZ 1, XI 0.4, IX 0.4"
    Returns list of (pauli_string, coeff).
    """
    if not text or not str(text).strip():
        return [("ZZ", 1.0), ("XI", 0.4), ("IX", 0.4)]

    raw = str(text).replace(";", ",").replace("\n", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[Tuple[str, float]] = []
    for p in parts:
        if ":" in p:
            a, b = p.split(":", 1)
        elif "=" in p:
            a, b = p.split("=", 1)
        else:
            toks = p.split()
            if len(toks) >= 2:
                a, b = toks[0], toks[1]
            else:
                continue
        pauli = a.strip().upper().replace(" ", "")
        coeff = _safe_float(b.strip(), 0.0)
        if pauli:
            out.append((pauli, coeff))

    return out if out else [("ZZ", 1.0), ("XI", 0.4), ("IX", 0.4)]


def _parse_edges(text: str) -> List[Tuple[int, int, float]]:
    """
    Parse MaxCut edges.
    Supports any of:
      0-1
      0 1
      0,1
      0 1 0.7   (weighted)
      0-1:0.7   (weighted)
    Separate by commas or newlines.
    Returns list of (i, j, w).
    """
    if not text or not str(text).strip():
        return [(0, 1, 1.0)]

    raw = str(text).replace(";", "\n").replace(",", "\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    edges: List[Tuple[int, int, float]] = []
    for ln in lines:
        w = 1.0
        core = ln

        if ":" in ln:
            core, wtxt = ln.split(":", 1)
            w = _safe_float(wtxt.strip(), 1.0)

        toks = core.replace("-", " ").replace(",", " ").split()
        if len(toks) >= 3:
            i = _safe_int(toks[0], 0)
            j = _safe_int(toks[1], 0)
            w = _safe_float(toks[2], w)
        elif len(toks) >= 2:
            i = _safe_int(toks[0], 0)
            j = _safe_int(toks[1], 0)
        else:
            continue

        if i == j:
            continue
        if i > j:
            i, j = j, i
        edges.append((i, j, float(w)))

    # de-dup (keep last weight)
    dedup: Dict[Tuple[int, int], float] = {}
    for i, j, w in edges:
        dedup[(i, j)] = float(w)

    out = [(i, j, w) for (i, j), w in dedup.items()]
    out.sort(key=lambda t: (t[0], t[1]))
    return out if out else [(0, 1, 1.0)]


def _infer_n_from_edges(edges: List[Tuple[int, int, float]]) -> int:
    if not edges:
        return 0
    mx = 0
    for i, j, _ in edges:
        mx = max(mx, int(i), int(j))
    return mx + 1


def _parse_h_vector(text: str, n: int) -> List[float]:
    """
    Parse Ising local fields h_i.
    Accepts comma/space separated floats. If fewer than n, pad with zeros.
    """
    if not text or not str(text).strip():
        return [0.0] * n
    raw = str(text).replace(",", " ").replace(";", " ")
    toks = [t for t in raw.split() if t.strip()]
    hs = [_safe_float(t, 0.0) for t in toks[:n]]
    if len(hs) < n:
        hs += [0.0] * (n - len(hs))
    return hs


def _parse_J_couplings(text: str) -> List[Tuple[int, int, float]]:
    """
    Parse Ising couplings J_ij.
    Each line like:
      i j J
      i-j:J
      i-j
    Returns list of (i,j,J).
    """
    if not text or not str(text).strip():
        return []

    raw = str(text).replace(";", "\n").replace(",", "\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    out: List[Tuple[int, int, float]] = []
    for ln in lines:
        J = 1.0
        core = ln
        if ":" in ln:
            core, jtxt = ln.split(":", 1)
            J = _safe_float(jtxt.strip(), 1.0)

        toks = core.replace("-", " ").replace(",", " ").split()
        if len(toks) >= 3:
            i = _safe_int(toks[0], 0)
            j = _safe_int(toks[1], 0)
            J = _safe_float(toks[2], J)
        elif len(toks) >= 2:
            i = _safe_int(toks[0], 0)
            j = _safe_int(toks[1], 0)
        else:
            continue

        if i == j:
            continue
        if i > j:
            i, j = j, i
        out.append((i, j, float(J)))

    # de-dup keep last
    dedup: Dict[Tuple[int, int], float] = {}
    for i, j, J in out:
        dedup[(i, j)] = float(J)
    final = [(i, j, J) for (i, j), J in dedup.items()]
    final.sort(key=lambda t: (t[0], t[1]))
    return final


# -----------------------------
# Pauli term builders
# -----------------------------
def _pauli_Z(n: int, i: int) -> str:
    s = ["I"] * n
    s[i] = "Z"
    return "".join(s)


def _pauli_ZZ(n: int, i: int, j: int) -> str:
    s = ["I"] * n
    s[i] = "Z"
    s[j] = "Z"
    return "".join(s)


# -----------------------------
# Math helpers
# -----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _energy_to_risk_multiplier(energy: float) -> float:
    """
    Interpretable mapping:
      lower energy => "more stable" => allow more risk budget (up to 1.5)
      higher energy => "less stable" => tighten risk budget (down to 0.5)
    """
    if not np.isfinite(energy):
        return 1.0
    energy_norm = float(np.tanh(energy))  # [-1, 1]
    tilt = -energy_norm                  # invert: lower energy => positive tilt
    mult = 1.0 + 0.5 * tilt              # [-1,1] => [0.5,1.5]
    return _clamp(mult, 0.5, 1.5)


def _smoke_eval(
    *,
    energy: float,
    reference_energy: Optional[float],
    pass_delta: float,
    warn_delta: float,
) -> Dict[str, Any]:
    """
    Smoke status:
      - PASS if |ΔE| <= pass_delta
      - WARN if |ΔE| <= warn_delta
      - FAIL otherwise
    If no reference: WARN.
    """
    if not np.isfinite(energy):
        return {
            "status": "FAIL",
            "reason": "Energy is not finite",
            "best_energy": None,
            "reference_energy": reference_energy,
            "abs_delta": None,
            "thresholds": {"pass_delta": float(pass_delta), "warn_delta": float(warn_delta)},
        }

    if reference_energy is None or (not np.isfinite(reference_energy)):
        return {
            "status": "WARN",
            "reason": "No reference energy available (set one or load previous snapshot).",
            "best_energy": float(energy),
            "reference_energy": None,
            "abs_delta": None,
            "thresholds": {"pass_delta": float(pass_delta), "warn_delta": float(warn_delta)},
        }

    d = float(abs(float(energy) - float(reference_energy)))
    if d <= float(pass_delta):
        s = "PASS"
    elif d <= float(warn_delta):
        s = "WARN"
    else:
        s = "FAIL"

    return {
        "status": s,
        "best_energy": float(energy),
        "reference_energy": float(reference_energy),
        "abs_delta": float(d),
        "thresholds": {"pass_delta": float(pass_delta), "warn_delta": float(warn_delta)},
    }


# -----------------------------
# Exact reference helpers (optional, small n)
# -----------------------------
def _ising_energy_bitstring(bits: int, n: int, h: List[float], J: List[Tuple[int, int, float]]) -> float:
    """
    Classical Ising energy for assignment bits in {0,1} mapped to spins z in {+1,-1} via:
      bit 0 -> +1, bit 1 -> -1  (z = 1 - 2*bit)
    Energy: sum_i h_i z_i + sum_{i<j} J_ij z_i z_j
    """
    z = [1.0 if ((bits >> i) & 1) == 0 else -1.0 for i in range(n)]
    e = 0.0
    for i in range(n):
        e += float(h[i]) * z[i]
    for i, j, Jij in J:
        e += float(Jij) * z[i] * z[j]
    return float(e)


def _maxcut_cost_bitstring(bits: int, n: int, edges: List[Tuple[int, int, float]]) -> float:
    """
    MaxCut cost for assignment bits in {0,1}:
      cost = sum_{(i,j)} w_ij * [bit_i != bit_j]
    """
    c = 0.0
    for i, j, w in edges:
        bi = (bits >> i) & 1
        bj = (bits >> j) & 1
        if bi != bj:
            c += float(w)
    return float(c)


def _exact_maxcut_best_cost_and_bits(
    *, n: int, edges: List[Tuple[int, int, float]], max_n: int = 12
) -> Optional[Dict[str, Any]]:
    """
    Returns exact best MaxCut cost and argmax bitstring (for n <= max_n).
    """
    if n <= 0 or n > max_n:
        return None
    best_cost = None
    best_bits = None
    for bits in range(1 << n):
        c = _maxcut_cost_bitstring(bits, n, edges)
        if best_cost is None or c > best_cost:
            best_cost = c
            best_bits = bits
    if best_cost is None or best_bits is None:
        return None
    return {"best_cost": float(best_cost), "best_bits": int(best_bits)}


def _exact_reference_energy(
    *,
    problem: str,
    n: int,
    edges: Optional[List[Tuple[int, int, float]]] = None,
    h: Optional[List[float]] = None,
    J: Optional[List[Tuple[int, int, float]]] = None,
    max_n: int = 12,
    maxcut_use_negative_cost_hamiltonian: bool = True,
) -> Optional[float]:
    """
    Returns an exact reference energy for small n.
    - Ising: returns min energy over all bitstrings.
    - MaxCut: if using Hamiltonian = -cost, returns min energy = -max(cost).
    """
    if n <= 0 or n > max_n:
        return None

    try:
        best = None
        for bits in range(1 << n):
            if problem.startswith("Ising"):
                if h is None or J is None:
                    return None
                e = _ising_energy_bitstring(bits, n, h, J)
                if best is None or e < best:
                    best = e
            elif problem.startswith("MaxCut"):
                if edges is None:
                    return None
                c = _maxcut_cost_bitstring(bits, n, edges)
                e = -c if maxcut_use_negative_cost_hamiltonian else c
                if best is None or e < best:
                    best = e
            else:
                return None
        return float(best) if best is not None else None
    except Exception:
        return None


# -----------------------------
# Qiskit builders (best-effort)
# -----------------------------
def _build_hamiltonian(pauli_list: List[Tuple[str, float]]) -> Tuple[Any, str]:
    """
    Prefer SparsePauliOp, fallback to PauliSumOp if needed.
    Returns: (hamiltonian_obj, description)
    """
    ham_desc = ", ".join([f"{p}:{c}" for p, c in pauli_list])

    try:
        from qiskit.quantum_info import SparsePauliOp  # type: ignore
        ham = SparsePauliOp.from_list([(p, float(c)) for p, c in pauli_list])
        return ham, f"SparsePauliOp({ham_desc})"
    except Exception:
        pass

    try:
        from qiskit.opflow import PauliSumOp  # type: ignore
        ham = PauliSumOp.from_list([(p, float(c)) for p, c in pauli_list])
        return ham, f"PauliSumOp({ham_desc})"
    except Exception:
        pass

    raise RuntimeError("Unable to construct Hamiltonian (SparsePauliOp/PauliSumOp not available).")


def _build_ansatz(ansatz_name: str, num_qubits: int, reps: int) -> Tuple[Any, str, int]:
    ansatz_name = str(ansatz_name)

    if ansatz_name.startswith("TwoLocal"):
        from qiskit.circuit.library import TwoLocal  # type: ignore
        ansatz = TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cx",
            entanglement="linear" if num_qubits > 1 else "full",
            reps=int(reps),
            insert_barriers=False,
        )
        return ansatz, f"TwoLocal({num_qubits}q,reps={reps})", int(getattr(ansatz, "num_parameters", 0))

    if ansatz_name.startswith("EfficientSU2"):
        from qiskit.circuit.library import EfficientSU2  # type: ignore
        ansatz = EfficientSU2(num_qubits=num_qubits, reps=int(reps))
        return ansatz, f"EfficientSU2({num_qubits}q,reps={reps})", int(getattr(ansatz, "num_parameters", 0))

    from qiskit.circuit.library import RealAmplitudes  # type: ignore
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=int(reps))
    return ansatz, f"RealAmplitudes({num_qubits}q,reps={reps})", int(getattr(ansatz, "num_parameters", 0))


def _build_optimizer(opt_name: str, maxiter: int, seed: Optional[int] = None) -> Tuple[Any, str]:
    opt_name = str(opt_name)

    try:
        from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP  # type: ignore
        if opt_name == "SPSA":
            try:
                opt = SPSA(maxiter=int(maxiter), seed=int(seed) if seed is not None else None)
            except Exception:
                opt = SPSA(maxiter=int(maxiter))
            return opt, f"SPSA(maxiter={maxiter})"
        if opt_name == "SLSQP":
            opt = SLSQP(maxiter=int(maxiter))
            return opt, f"SLSQP(maxiter={maxiter})"
        opt = COBYLA(maxiter=int(maxiter))
        return opt, f"COBYLA(maxiter={maxiter})"
    except Exception:
        pass

    try:
        from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP  # type: ignore
        if opt_name == "SPSA":
            opt = SPSA(maxiter=int(maxiter))
            return opt, f"SPSA(maxiter={maxiter})"
        if opt_name == "SLSQP":
            opt = SLSQP(maxiter=int(maxiter))
            return opt, f"SLSQP(maxiter={maxiter})"
        opt = COBYLA(maxiter=int(maxiter))
        return opt, f"COBYLA(maxiter={maxiter})"
    except Exception:
        pass

    raise RuntimeError("Optimizer class not available (COBYLA/SPSA/SLSQP).")


def _build_estimator(backend_choice: str) -> Tuple[Optional[Any], str]:
    backend_choice = str(backend_choice)

    if backend_choice.startswith("AerEstimator"):
        try:
            from qiskit_aer.primitives import Estimator as AerEstimator  # type: ignore
            est = AerEstimator()
            return est, "qiskit_aer.primitives.Estimator"
        except Exception as e:
            return None, f"AerEstimator unavailable: {e}"

    try:
        from qiskit.primitives import Estimator  # type: ignore
        est = Estimator()
        return est, "qiskit.primitives.Estimator"
    except Exception as e:
        return None, f"Estimator unavailable: {e}"


def _set_algorithm_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        from qiskit_algorithms.utils import algorithm_globals  # type: ignore
        algorithm_globals.random_seed = int(seed)
        return
    except Exception:
        pass
    try:
        from qiskit.utils import algorithm_globals  # type: ignore
        algorithm_globals.random_seed = int(seed)
        return
    except Exception:
        pass


# -----------------------------
# Step 5 — bitstring decode helpers
# -----------------------------
def _bits_int_from_lsb_string(lsb_str: str) -> int:
    """
    lsb_str: index 0 corresponds to bit0 (node0/qubit0).
    Example: "101" => bits 0 and 2 set => 0b101 => 5.
    """
    b = 0
    s = str(lsb_str).strip()
    for i, ch in enumerate(s):
        if ch == "1":
            b |= (1 << i)
    return int(b)


def _qiskit_key_to_lsb_string(key: str) -> str:
    """
    Qiskit probability/count keys are usually MSB...LSB order (|q_{n-1}...q_0>).
    We reverse so returned string index 0 corresponds to qubit0/node0.
    """
    return str(key)[::-1]


def _safe_assign_parameters(circuit: Any, values: List[float]) -> Any:
    """
    Bind parameters in a version-tolerant way.
    values are in the order of circuit.parameters.
    """
    try:
        params = list(getattr(circuit, "parameters", []))
        if not params:
            return circuit
        if len(values) != len(params):
            return circuit
        bind_map = {p: float(v) for p, v in zip(params, values)}
        try:
            return circuit.assign_parameters(bind_map, inplace=False)
        except Exception:
            return circuit.bind_parameters(bind_map)  # type: ignore
    except Exception:
        return circuit


def _try_statevector_probabilities(ansatz_circuit: Any) -> Optional[Dict[str, float]]:
    """
    Returns probabilities dict keyed by Qiskit bitstrings (MSB...LSB), or None if unavailable.
    """
    try:
        from qiskit.quantum_info import Statevector  # type: ignore
        sv = Statevector.from_instruction(ansatz_circuit)
        probs = sv.probabilities_dict()
        out = {str(k): float(v) for k, v in probs.items()}
        return out
    except Exception:
        return None


def _try_aer_counts(meas_circuit: Any, shots: int, seed: Optional[int]) -> Optional[Dict[str, int]]:
    """
    Try running shots via AerSimulator; returns counts (keys MSB...LSB).
    """
    try:
        from qiskit_aer import AerSimulator  # type: ignore
        from qiskit import transpile  # type: ignore

        backend = AerSimulator()
        tqc = transpile(meas_circuit, backend)
        run_kwargs = {"shots": int(shots)}
        if seed is not None:
            run_kwargs["seed_simulator"] = int(seed)
        job = backend.run(tqc, **run_kwargs)
        res = job.result()
        counts = res.get_counts()
        if isinstance(counts, list) and counts:
            counts = counts[0]
        if isinstance(counts, dict):
            return {str(k): int(v) for k, v in counts.items()}
        return None
    except Exception:
        return None


def _add_measure_all(qc: Any) -> Any:
    """
    Add measurement to all qubits, version tolerant.
    """
    try:
        from qiskit import QuantumCircuit  # type: ignore
        nq = int(getattr(qc, "num_qubits", 0))
        if nq <= 0:
            return qc
        try:
            out = QuantumCircuit(nq, nq)
            out.compose(qc, inplace=True)
        except Exception:
            out = qc.copy()
            if int(getattr(out, "num_clbits", 0)) < nq:
                out.add_register(type(out.cregs[0])(nq))  # type: ignore
        try:
            out.measure_all()
        except Exception:
            for i in range(nq):
                out.measure(i, i)
        return out
    except Exception:
        return qc


def _decode_vqe_bitstrings_for_problem(
    *,
    problem: str,
    n: int,
    ansatz_name: str,
    reps: int,
    best_param_values: Optional[List[float]],
    edges: Optional[List[Tuple[int, int, float]]] = None,
    ising_h: Optional[List[float]] = None,
    ising_J: Optional[List[Tuple[int, int, float]]] = None,
    maxcut_negative_cost_h: bool = True,
    method: str = "Auto",
    shots: int = 2048,
    topk: int = 12,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"enabled": True, "method": str(method), "shots": int(shots), "topk": int(topk)}
    if best_param_values is None or not isinstance(best_param_values, list) or len(best_param_values) == 0:
        out["enabled"] = False
        out["reason"] = "No optimal parameters available to decode bitstrings."
        return out

    try:
        ansatz, ans_desc, n_params = _build_ansatz(str(ansatz_name), int(n), int(reps))
        out["ansatz"] = ans_desc
        out["num_params"] = int(n_params)
        if int(n_params) != len(best_param_values):
            out["enabled"] = False
            out["reason"] = f"Param length mismatch: got {len(best_param_values)} values, expected {n_params}."
            return out
        bound = _safe_assign_parameters(ansatz, best_param_values)
    except Exception as e:
        out["enabled"] = False
        out["reason"] = f"Could not build/bind ansatz: {e}"
        return out

    probs = None
    counts = None
    used = None

    want = str(method)
    if want == "Statevector (exact probs)":
        probs = _try_statevector_probabilities(bound)
        used = "Statevector"
    elif want == "Aer (shots)":
        meas = _add_measure_all(bound)
        counts = _try_aer_counts(meas, int(shots), seed)
        used = "Aer"
    elif want == "Sample from probs":
        probs = _try_statevector_probabilities(bound)
        used = "SampleFromProbs"
    else:
        probs = _try_statevector_probabilities(bound)
        if probs is not None:
            used = "Statevector"
        else:
            meas = _add_measure_all(bound)
            counts = _try_aer_counts(meas, int(shots), seed)
            if counts is not None:
                used = "Aer"
            else:
                used = "None"

    out["used_backend"] = used

    if probs is None and counts is None:
        out["enabled"] = False
        out["reason"] = "No Statevector or Aer sampling available in this environment."
        return out

    dist: Dict[str, float] = {}
    if probs is not None:
        for k, v in probs.items():
            lsb = _qiskit_key_to_lsb_string(k)
            dist[lsb] = float(v)
    else:
        total = float(sum(counts.values())) if counts else 0.0
        if total <= 0:
            out["enabled"] = False
            out["reason"] = "Aer returned empty counts."
            return out
        for k, c in counts.items():
            lsb = _qiskit_key_to_lsb_string(k)
            dist[lsb] = float(c) / total

    if used == "SampleFromProbs" and probs is not None:
        try:
            keys = list(dist.keys())
            pvals = np.array([dist[k] for k in keys], dtype=float)
            pvals = pvals / max(1e-12, float(pvals.sum()))
            rng = np.random.default_rng(int(seed) if seed is not None else 12345)
            draws = rng.choice(len(keys), size=int(shots), replace=True, p=pvals)
            freq: Dict[str, int] = {}
            for idx in draws:
                s = keys[int(idx)]
                freq[s] = int(freq.get(s, 0)) + 1
            dist = {k: float(v) / float(shots) for k, v in freq.items()}
            out["used_backend"] = "SampleFromProbs(shots)"
        except Exception:
            pass

    scored: List[Dict[str, Any]] = []
    expected_obj = 0.0

    if str(problem).startswith("MaxCut"):
        if edges is None:
            out["enabled"] = False
            out["reason"] = "No edges provided for MaxCut decode."
            return out
        for lsb_str, prob in dist.items():
            bits = _bits_int_from_lsb_string(lsb_str)
            cost = _maxcut_cost_bitstring(bits, int(n), edges)
            expected_obj += float(prob) * float(cost)
            scored.append({"bitstring_lsb": lsb_str, "prob": float(prob), "maxcut_cost": float(cost)})
        scored.sort(key=lambda r: (float(r["maxcut_cost"]), float(r["prob"])), reverse=True)
        best = scored[0] if scored else None
        out["objective"] = "MaxCut cost"
        out["expected_cost"] = float(expected_obj)
        if best is not None:
            out["best_bitstring_lsb"] = best["bitstring_lsb"]
            out["best_cost"] = float(best["maxcut_cost"])
        out["topk"] = scored[: int(max(1, topk))]

    elif str(problem).startswith("Ising"):
        if ising_h is None or ising_J is None:
            out["enabled"] = False
            out["reason"] = "No h/J provided for Ising decode."
            return out
        for lsb_str, prob in dist.items():
            bits = _bits_int_from_lsb_string(lsb_str)
            e = _ising_energy_bitstring(bits, int(n), ising_h, ising_J)
            expected_obj += float(prob) * float(e)
            scored.append({"bitstring_lsb": lsb_str, "prob": float(prob), "ising_energy": float(e)})
        scored.sort(key=lambda r: (float(r["ising_energy"]), -float(r["prob"])))
        best = scored[0] if scored else None
        out["objective"] = "Ising energy"
        out["expected_energy"] = float(expected_obj)
        if best is not None:
            out["best_bitstring_lsb"] = best["bitstring_lsb"]
            out["best_energy"] = float(best["ising_energy"])
        out["topk"] = scored[: int(max(1, topk))]

    else:
        out["enabled"] = False
        out["reason"] = "Decode is only implemented for MaxCut and Ising."
        return out

    return out


def _try_run_real_vqe(
    *,
    num_qubits: int,
    pauli_list: List[Tuple[str, float]],
    ansatz_name: str,
    reps: int,
    optimizer_name: str,
    maxiter: int,
    backend_choice: str,
    seed: Optional[int],
) -> Tuple[bool, float, Dict[str, Any], str, List[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []

    try:
        _set_algorithm_seed(seed)

        ham, ham_desc = _build_hamiltonian(pauli_list)
        estimator, estimator_name = _build_estimator(backend_choice)
        if estimator is None:
            return False, float("nan"), {}, estimator_name, history

        ansatz, ansatz_desc, n_params = _build_ansatz(ansatz_name, int(num_qubits), int(reps))
        optimizer, opt_desc = _build_optimizer(optimizer_name, int(maxiter), seed=seed)

        try:
            from qiskit_algorithms import VQE  # type: ignore
        except Exception:
            from qiskit.algorithms.minimum_eigensolvers import VQE  # type: ignore

        def callback(eval_count=None, params=None, mean=None, std=None, *args, **kwargs):
            rec: Dict[str, Any] = {"t": len(history)}
            if eval_count is not None:
                rec["eval_count"] = eval_count
            if mean is not None:
                try:
                    rec["energy"] = float(np.real(mean))
                except Exception:
                    rec["energy"] = mean
            if std is not None:
                rec["std"] = std
            if params is not None:
                try:
                    arr = np.array(params, dtype=float).reshape(-1)
                    rec["params_head"] = [float(x) for x in arr[: min(6, len(arr))]]
                except Exception:
                    rec["params_head"] = None
            history.append(rec)

        vqe_notes = ""
        try:
            vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, callback=callback)
            vqe_notes = "VQE(callback=...)"
        except Exception:
            vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
            try:
                setattr(vqe, "callback", callback)
                vqe_notes = "VQE + vqe.callback"
            except Exception:
                vqe_notes = "VQE (no callback available)"

        energy = float("nan")
        best_params: Any = None
        nfev = None

        try:
            res = vqe.compute_minimum_eigenvalue(ham)
            ev = getattr(res, "eigenvalue", None)
            if ev is None:
                ev = getattr(res, "optimal_value", None)
            energy = float(np.real(ev))
            best_params = getattr(res, "optimal_parameters", None)
            nfev = getattr(res, "optimizer_evals", None)
        except Exception:
            res = vqe.solve(ham)  # type: ignore
            ev = getattr(res, "eigenvalue", getattr(res, "optimal_value", None))
            energy = float(np.real(ev))
            best_params = getattr(res, "optimal_parameters", None)
            nfev = getattr(res, "optimizer_evals", None)

        # Store optimal param vector aligned to ansatz.parameters
        best_param_values: Optional[List[float]] = None
        best_param_names: Optional[List[str]] = None
        try:
            params_order = list(getattr(ansatz, "parameters", []))
            best_param_names = [str(p) for p in params_order]

            if best_params is None:
                best_param_values = None
            elif isinstance(best_params, dict):
                vals: List[float] = []
                ok_all = True
                for p in params_order:
                    if p in best_params:
                        vals.append(float(best_params[p]))
                    elif str(p) in best_params:
                        vals.append(float(best_params[str(p)]))  # type: ignore
                    else:
                        ok_all = False
                        break
                best_param_values = vals if ok_all and len(vals) == len(params_order) else None
            else:
                arr = np.array(best_params, dtype=float).reshape(-1)
                if len(arr) == len(params_order):
                    best_param_values = [float(x) for x in arr]
                else:
                    best_param_values = None
        except Exception:
            best_param_values = None
            best_param_names = None

        meta = {
            "backend": estimator_name,
            "backend_choice": backend_choice,
            "hamiltonian": ham_desc,
            "ansatz": ansatz_desc,
            "optimizer": opt_desc,
            "maxiter": int(maxiter),
            "seed": seed,
            "callback": vqe_notes,
            "num_qubits": int(num_qubits),
            "num_params": int(n_params),
            "optimizer_evals": nfev,
        }

        if best_param_values is not None and best_param_names is not None:
            meta["best_param_values"] = best_param_values
            meta["best_param_names"] = best_param_names

        if best_params is not None:
            try:
                meta["best_params"] = {str(k): float(v) for k, v in best_params.items()}
            except Exception:
                meta["best_params"] = str(best_params)

        return True, energy, meta, "Real VQE run succeeded.", history

    except Exception as e:
        return False, float("nan"), {}, f"Real VQE unavailable: {e}", history


# -----------------------------
# Toy fallback
# -----------------------------
def _run_toy_energy(seed: Optional[int] = None) -> Tuple[float, Dict[str, Any], str, List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(seed) if seed is not None else 12345)
    base = float(rng.normal(loc=0.0, scale=0.7))
    hist = []
    cur = base + 0.5
    for t in range(30):
        cur = float(cur - 0.03 + rng.normal(0, 0.01))
        hist.append({"t": t, "energy": cur})
    energy = float(hist[-1]["energy"]) if hist else base
    meta = {"seed": seed, "generator": "toy(normal + fake descent)"}
    return energy, meta, "Toy fallback energy (Qiskit VQE deps missing or failed).", hist


# -----------------------------
# Reference-energy helpers
# -----------------------------
def _extract_energy_from_snapshot(obj: Any) -> Optional[float]:
    if not isinstance(obj, dict):
        return None
    e = obj.get("energy", None)
    try:
        if e is None:
            return None
        ef = float(e)
        return ef if np.isfinite(ef) else None
    except Exception:
        return None


def _load_reference_from_latest_snapshot(gb_dir: str) -> Optional[float]:
    lp = _latest_snapshot_path(gb_dir)
    if not lp:
        return None
    obj = _read_json(lp)
    return _extract_energy_from_snapshot(obj)


# -----------------------------
# Problem builders
# -----------------------------
def _build_problem_paulis(
    *,
    problem: str,
    n: int,
    pauli_text: str = "",
    maxcut_edges_text: str = "",
    maxcut_negative_cost_h: bool = True,
    ising_h_text: str = "",
    ising_J_text: str = "",
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """
    Returns (pauli_list, problem_meta)
    """
    problem = str(problem)

    if problem.startswith("Custom Pauli"):
        paulis = _parse_pauli_list(pauli_text)
        return paulis, {"problem": "CustomPauli", "pauli_list": paulis}

    if problem.startswith("Toy Hamiltonian"):
        pauli_list: List[Tuple[str, float]] = []
        if n <= 1:
            pauli_list = [("Z", 1.0), ("X", 0.3)]
        else:
            for i in range(n - 1):
                pauli_list.append((_pauli_ZZ(n, i, i + 1), 1.0))
            for i in range(n):
                s = ["I"] * n
                s[i] = "X"
                pauli_list.append(("".join(s), 0.2))
        return pauli_list, {"problem": "Toy", "pauli_list": pauli_list}

    if problem.startswith("MaxCut"):
        edges = _parse_edges(maxcut_edges_text)
        inferred_n = _infer_n_from_edges(edges)

        # MaxCut cost: C = Σ w(1 - Z_i Z_j)/2
        # If we want VQE-min to correspond to MAX cost, set H = -C.
        paulis: List[Tuple[str, float]] = []
        const = 0.0
        for i, j, w in edges:
            const += 0.5 * w
            if i < n and j < n:
                paulis.append((_pauli_ZZ(n, i, j), -0.5 * w))

        I = "I" * n if n > 1 else "I"
        paulis.append((I, const))

        if maxcut_negative_cost_h:
            paulis = [(p, -c) for (p, c) in paulis]  # H = -C

        return paulis, {
            "problem": "MaxCut",
            "edges": edges,
            "inferred_n_from_edges": inferred_n,
            "hamiltonian_is_negative_cost": bool(maxcut_negative_cost_h),
            "constant_shift": float((-const) if maxcut_negative_cost_h else const),
            "pauli_list": paulis,
        }

    if problem.startswith("Ising"):
        h = _parse_h_vector(ising_h_text, n)
        J = _parse_J_couplings(ising_J_text)
        paulis: List[Tuple[str, float]] = []
        for i in range(n):
            if abs(float(h[i])) > 0.0:
                paulis.append((_pauli_Z(n, i), float(h[i])))
        for i, j, Jij in J:
            if i < n and j < n:
                paulis.append((_pauli_ZZ(n, i, j), float(Jij)))

        if not paulis:
            paulis.append((_pauli_Z(n, 0), 0.0))

        return paulis, {
            "problem": "Ising",
            "h": h,
            "J": J,
            "pauli_list": paulis,
        }

    paulis = _parse_pauli_list(pauli_text)
    return paulis, {"problem": "CustomPauli", "pauli_list": paulis}


# -----------------------------
# MaxCut graph generator
# -----------------------------
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _gen_edges_manual(
    *, generator: str, n: int, p: float, d: int, seed: int
) -> List[Tuple[int, int, float]]:
    """
    Manual fallback when networkx is unavailable.
    Supported:
      - Complete K_n
      - Erdos-Renyi G(n,p)
      - Cycle C_n
      - Path P_n
    """
    n = max(2, int(n))
    r = _rng(seed)
    edges: List[Tuple[int, int, float]] = []

    gen = str(generator)
    if gen == "Complete K_n":
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, 1.0))
        return edges

    if gen == "Cycle C_n":
        for i in range(n):
            j = (i + 1) % n
            a, b = (i, j) if i < j else (j, i)
            edges.append((a, b, 1.0))
        return _dedup_edges(edges)

    if gen == "Path P_n":
        for i in range(n - 1):
            edges.append((i, i + 1, 1.0))
        return edges

    p = float(max(0.0, min(1.0, p)))
    for i in range(n):
        for j in range(i + 1, n):
            if r.random() < p:
                edges.append((i, j, 1.0))
    if not edges:
        edges = [(0, 1, 1.0)]
    return edges


def _dedup_edges(edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    dedup: Dict[Tuple[int, int], float] = {}
    for i, j, w in edges:
        a, b = (i, j) if i < j else (j, i)
        if a == b:
            continue
        dedup[(a, b)] = float(w)
    out = [(i, j, w) for (i, j), w in dedup.items()]
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _apply_weights(
    *, edges: List[Tuple[int, int, float]], weighted: bool, wmin: float, wmax: float, seed: int
) -> List[Tuple[int, int, float]]:
    if not edges:
        return edges
    if not weighted:
        return [(i, j, 1.0) for i, j, _ in edges]
    r = _rng(seed + 999)
    lo = float(min(wmin, wmax))
    hi = float(max(wmin, wmax))
    return [(i, j, float(r.uniform(lo, hi))) for i, j, _ in edges]


def _edges_to_text(edges: List[Tuple[int, int, float]], weighted: bool) -> str:
    lines = []
    for i, j, w in edges:
        if weighted:
            lines.append(f"{i}-{j}:{w:.3f}")
        else:
            lines.append(f"{i}-{j}")
    return "\n".join(lines).strip()


def _build_graph_from_edges(edges: List[Tuple[int, int, float]], n: Optional[int] = None) -> Optional[Any]:
    if nx is None:
        return None
    G = nx.Graph()
    if n is None:
        n = _infer_n_from_edges(edges)
    for k in range(int(n)):
        G.add_node(k)
    for i, j, w in edges:
        G.add_edge(int(i), int(j), weight=float(w))
    return G


def _plot_graph_plotly(edges: List[Tuple[int, int, float]], n: int) -> Optional[Any]:
    if go is None:
        return None

    if nx is not None:
        G = _build_graph_from_edges(edges, n=n)
        if G is None:
            return None
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = {}
        for i in range(n):
            ang = 2 * math.pi * (i / max(1, n))
            pos[i] = (math.cos(ang), math.sin(ang))

    edge_x = []
    edge_y = []
    for i, j, _w in edges:
        if i not in pos or j not in pos:
            continue
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(width=2),
        name="edges",
    )

    node_x = []
    node_y = []
    node_text = []
    for i in range(n):
        x, y = pos.get(i, (0.0, 0.0))
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"node {i}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[str(i) for i in range(n)],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=14),
        name="nodes",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="MaxCut graph",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )

    weighted = any(abs(float(w) - 1.0) > 1e-9 for _, _, w in edges)
    if weighted:
        ann = []
        for i, j, w in edges:
            if i not in pos or j not in pos:
                continue
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ann.append(dict(x=xm, y=ym, text=f"{float(w):.2f}", showarrow=False, font=dict(size=11)))
        fig.update_layout(annotations=ann)

    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
def render_vqe_tab(st, *, default_golden_dir: str = GOLDEN_DIR_DEFAULT) -> None:
    if not VQE_TAB_ENABLED:
        st.warning("VQE tab is disabled (VQE_TAB_ENABLED=False).")
        return

    st.subheader("VQE Mini-Lab — isolated runs + diagnostics")

    # Session slots (isolated)
    if "vqe_risk_signal" not in st.session_state:
        st.session_state["vqe_risk_signal"] = None
    if "vqe_snapshot" not in st.session_state:
        st.session_state["vqe_snapshot"] = None
    if "vqe_smoke" not in st.session_state:
        st.session_state["vqe_smoke"] = None
    if "trade_audit_log" not in st.session_state:
        st.session_state["trade_audit_log"] = []

    # ---------- Controls ----------
    gb_dir = st.text_input("Golden build directory", default_golden_dir, key="vqe_gb_dir")
    _ensure_dir(gb_dir)

    st.markdown("### Run controls")

    cA, cB, cC, cD = st.columns(4)
    with cA:
        use_real = st.checkbox("Try real VQE (Qiskit)", value=True, key="vqe_try_real")
    with cB:
        backend_choice = st.selectbox(
            "Backend",
            ["Estimator (default)", "AerEstimator (qiskit_aer)"],
            index=0,
            key="vqe_backend_choice",
        )
    with cC:
        seed_txt = st.text_input("Seed (optional)", "42", key="vqe_seed_txt")
        seed = int(seed_txt) if seed_txt.strip().isdigit() else None
    with cD:
        st.caption("Bridge output: **risk budget multiplier** ∈ [0.5, 1.5]")

    # NEW: smoke/reference controls
    st.markdown("### Golden Build smoke check")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        ref_mode = st.selectbox(
            "Reference energy source",
            ["Latest snapshot in golden_build", "Manual value", "Exact (bruteforce, small n)", "None (no reference)"],
            index=0,
            key="vqe_ref_mode",
        )
    with s2:
        pass_delta = st.number_input("PASS |ΔE| ≤", min_value=0.0, max_value=10.0, value=0.05, step=0.01, key="vqe_pass_delta")
    with s3:
        warn_delta = st.number_input("WARN |ΔE| ≤", min_value=0.0, max_value=10.0, value=0.15, step=0.01, key="vqe_warn_delta")
    with s4:
        exact_max_n = st.number_input("Exact max n", min_value=4, max_value=20, value=12, step=1, key="vqe_exact_max_n")

    manual_ref = None
    if ref_mode == "Manual value":
        manual_ref_txt = st.text_input("Manual reference energy", "-1.0", key="vqe_manual_ref_txt")
        try:
            manual_ref = float(manual_ref_txt)
        except Exception:
            manual_ref = None

    c1, c2, c3 = st.columns(3)
    with c1:
        problem = st.selectbox(
            "Problem",
            ["Toy Hamiltonian (default)", "Custom Pauli Hamiltonian", "MaxCut (graph)", "Ising (h/J)"],
            index=0,
            key="vqe_problem",
        )
    with c2:
        ansatz_name = st.selectbox(
            "Ansatz",
            ["TwoLocal", "EfficientSU2", "RealAmplitudes"],
            index=0,
            key="vqe_ansatz",
        )
    with c3:
        optimizer_name = st.selectbox(
            "Optimizer",
            ["COBYLA", "SPSA", "SLSQP"],
            index=0,
            key="vqe_optimizer",
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        num_qubits = st.number_input("Qubits", min_value=1, max_value=20, value=2, step=1, key="vqe_num_qubits")
    with c5:
        reps = st.number_input("Ansatz reps", min_value=1, max_value=8, value=2, step=1, key="vqe_reps")
    with c6:
        maxiter = st.number_input("Max iterations", min_value=10, max_value=2000, value=80, step=10, key="vqe_maxiter")

    # Step 5 (decode) controls
    st.markdown("### Step 5 — Decode solution bitstrings (NEW)")
    d1, d2, d3, d4 = st.columns([1.2, 1.0, 1.0, 1.0])
    with d1:
        decode_enabled = st.checkbox("Enable decode (MaxCut/Ising)", value=True, key="vqe_decode_enabled")
    with d2:
        decode_method = st.selectbox(
            "Decode method",
            ["Auto", "Statevector (exact probs)", "Aer (shots)", "Sample from probs"],
            index=0,
            key="vqe_decode_method",
        )
    with d3:
        decode_shots = st.number_input("Decode shots", min_value=128, max_value=200_000, value=4096, step=256, key="vqe_decode_shots")
    with d4:
        decode_topk = st.number_input("Top-K outcomes", min_value=5, max_value=50, value=12, step=1, key="vqe_decode_topk")

    # Problem-specific inputs
    pauli_text = ""
    maxcut_edges_text = ""
    maxcut_neg_cost = True
    ising_h_text = ""
    ising_J_text = ""

    if problem == "Custom Pauli Hamiltonian":
        pauli_text = st.text_area(
            "Pauli terms (e.g., `ZZ:1, XI:0.4, IX:0.4`)",
            "ZZ:1, XI:0.4, IX:0.4",
            height=70,
            key="vqe_pauli_text",
        )

    # -----------------------------
    # MaxCut Graph Builder + Visualization + Hamiltonian preview
    # -----------------------------
    if problem == "MaxCut (graph)":
        st.markdown("### MaxCut Graph Builder (NEW)")
        st.caption("Generate an edge list automatically, then it will populate the **Edges** box below.")

        gcol1, gcol2, gcol3, gcol4 = st.columns([1.2, 1.0, 1.0, 1.2])
        with gcol1:
            gen_choice = st.selectbox(
                "Generator",
                ["Complete K_n", "Erdos-Renyi G(n,p)", "d-regular", "Cycle C_n", "Path P_n", "Barabasi-Albert"],
                index=0,
                key="vqe_maxcut_gen_choice",
            )
        with gcol2:
            gen_n = st.number_input("Nodes (n)", min_value=2, max_value=30, value=int(max(2, num_qubits)), step=1, key="vqe_maxcut_gen_n")
        with gcol3:
            gen_seed = st.number_input("Gen seed", min_value=0, max_value=10_000_000, value=int(seed if seed is not None else 42), step=1, key="vqe_maxcut_gen_seed")
        with gcol4:
            if st.button("Set qubits = n nodes", key="vqe_set_qubits_equals_nodes"):
                st.session_state["vqe_num_qubits"] = int(gen_n)
                st.rerun()

        g2a, g2b, g2c, g2d = st.columns([1.2, 1.0, 1.0, 1.0])
        with g2a:
            p_val = st.slider("p (for G(n,p))", min_value=0.0, max_value=1.0, value=0.35, step=0.01, key="vqe_maxcut_p")
        with g2b:
            d_val = st.number_input("d (for d-regular)", min_value=1, max_value=29, value=3, step=1, key="vqe_maxcut_d")
        with g2c:
            weighted_edges = st.checkbox("Weighted edges", value=True, key="vqe_maxcut_weighted")
        with g2d:
            wmin = st.number_input("w min", min_value=0.01, max_value=100.0, value=0.50, step=0.05, key="vqe_maxcut_wmin")
            wmax = st.number_input("w max", min_value=0.01, max_value=100.0, value=1.50, step=0.05, key="vqe_maxcut_wmax")

        preview_edges: List[Tuple[int, int, float]] = []
        try:
            n_nodes = int(gen_n)
            if nx is not None:
                if gen_choice == "Complete K_n":
                    G = nx.complete_graph(n_nodes)
                elif gen_choice == "Cycle C_n":
                    G = nx.cycle_graph(n_nodes)
                elif gen_choice == "Path P_n":
                    G = nx.path_graph(n_nodes)
                elif gen_choice == "d-regular":
                    dd = int(max(1, min(n_nodes - 1, int(d_val))))
                    if (n_nodes * dd) % 2 == 1:
                        dd = max(1, dd - 1)
                    G = nx.random_regular_graph(dd, n_nodes, seed=int(gen_seed))
                elif gen_choice == "Barabasi-Albert":
                    m = int(max(1, min(n_nodes - 1, max(1, int(d_val) // 2))))
                    G = nx.barabasi_albert_graph(n_nodes, m, seed=int(gen_seed))
                else:
                    G = nx.erdos_renyi_graph(n_nodes, float(p_val), seed=int(gen_seed))
                preview_edges = [(int(u), int(v), 1.0) for (u, v) in G.edges()]
                preview_edges = _dedup_edges(preview_edges)
            else:
                preview_edges = _gen_edges_manual(
                    generator=gen_choice,
                    n=int(gen_n),
                    p=float(p_val),
                    d=int(d_val),
                    seed=int(gen_seed),
                )

            preview_edges = _apply_weights(
                edges=preview_edges,
                weighted=bool(weighted_edges),
                wmin=float(wmin),
                wmax=float(wmax),
                seed=int(gen_seed),
            )
            preview_edges = _dedup_edges(preview_edges)
        except Exception:
            preview_edges = [(0, 1, 1.0)]

        st.caption(f"Preview: {len(preview_edges)} edges")

        with st.expander("Preview edges", expanded=False):
            st.json(preview_edges[: min(200, len(preview_edges))])

        b1, b2, b3 = st.columns([1.2, 1.2, 2.6])
        with b1:
            if st.button("Generate → fill Edges", key="vqe_maxcut_fill_edges"):
                st.session_state["vqe_maxcut_edges"] = _edges_to_text(preview_edges, bool(weighted_edges))
                st.rerun()
        with b2:
            st.caption("Copy preview into clipboard (manual)")
        with b3:
            st.caption("The generator writes into the **Edges** box below via `st.session_state['vqe_maxcut_edges']`.")

        st.markdown("---")

        st.caption("Edge list formats: `0-1`, `0 1`, `0-1:0.7`, one per line or comma-separated.")
        maxcut_edges_text = st.text_area(
            "Edges",
            st.session_state.get("vqe_maxcut_edges", "0-1\n1-2\n2-3\n3-0"),
            height=110,
            key="vqe_maxcut_edges",
        )
        maxcut_neg_cost = st.checkbox(
            "Use Hamiltonian = -cost (so VQE min-energy corresponds to MAX cut)",
            value=True,
            key="vqe_maxcut_neg_cost",
        )

        edges_now = _parse_edges(maxcut_edges_text)
        inferred_n = _infer_n_from_edges(edges_now)
        if inferred_n != int(num_qubits):
            st.warning(
                f"Edges imply **n={inferred_n}** nodes, but Qubits is set to **{int(num_qubits)}**. "
                f"For correct Hamiltonian sizing, set Qubits = n."
            )
            if st.button("Sync Qubits to inferred n", key="vqe_sync_qubits_inferred"):
                st.session_state["vqe_num_qubits"] = int(inferred_n)
                st.rerun()

        st.markdown("### Graph visualization")
        fig = _plot_graph_plotly(edges_now, n=int(max(int(num_qubits), inferred_n)))
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            rows = [{"i": i, "j": j, "w": float(w)} for i, j, w in edges_now]
            if pd is not None:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.write(rows)

        st.markdown("### Hamiltonian preview")
        try:
            n_for_preview = int(num_qubits)
            pauli_list_preview, pm = _build_problem_paulis(
                problem="MaxCut",
                n=n_for_preview,
                maxcut_edges_text=maxcut_edges_text,
                maxcut_negative_cost_h=bool(maxcut_neg_cost),
            )
            const_shift = pm.get("constant_shift", None)
            negflag = bool(pm.get("hamiltonian_is_negative_cost", True))
            st.write(
                {
                    "using_H_equals": "-cost" if negflag else "+cost",
                    "constant_shift (I term coeff)": float(const_shift) if const_shift is not None else None,
                    "num_pauli_terms": int(len(pauli_list_preview)),
                }
            )
            preview_terms = pauli_list_preview[:10]
            st.caption("First ~10 Pauli terms (pauli_string, coeff):")
            st.code("\n".join([f"{p:>10s}  {c:+.6f}" for p, c in preview_terms]))
        except Exception as e:
            st.warning(f"Hamiltonian preview unavailable: {e}")

    if problem == "Ising (h/J)":
        st.caption("Ising energy: Σ h_i Z_i + Σ J_ij Z_i Z_j. h length n; J lines like `0-1:1.0` or `0 1 -0.7`.")
        ising_h_text = st.text_input(
            "h vector (comma/space separated)",
            "0,0,0,0",
            key="vqe_ising_h",
        )
        ising_J_text = st.text_area(
            "J couplings (one per line)",
            "0-1:1.0\n1-2:1.0\n2-3:1.0\n0-3:1.0",
            height=90,
            key="vqe_ising_J",
        )

    st.caption(
        "This is a **prototype mini-lab** for reproducible VQE runs, convergence diagnostics, "
        "and artifact export. Not financial advice."
    )

    # ---------- Run ----------
    run_col1, run_col2, run_col3 = st.columns([1, 1, 2])
    with run_col1:
        run_btn = st.button("Run VQE", key="vqe_run_btn")
    with run_col2:
        clear_btn = st.button("Clear VQE session result", key="vqe_clear_btn")
    with run_col3:
        st.caption("Tip: if Real VQE fails, the tab auto-falls back to a toy run + still produces artifacts.")

    if clear_btn:
        st.session_state["vqe_risk_signal"] = None
        st.session_state["vqe_snapshot"] = None
        st.session_state["vqe_smoke"] = None
        st.success("Cleared VQE results from session_state.")

    if run_btn:
        ts_fs = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        started = dt.datetime.now()

        progress = st.progress(0.0)
        status = st.empty()

        ok = False
        energy = float("nan")
        meta: Dict[str, Any] = {}
        notes = ""
        history: List[Dict[str, Any]] = []

        try:
            n = int(num_qubits)

            if problem == "Custom Pauli Hamiltonian":
                pauli_text = st.session_state.get("vqe_pauli_text", "")
            if problem == "MaxCut (graph)":
                maxcut_edges_text = st.session_state.get("vqe_maxcut_edges", "")
                maxcut_neg_cost = bool(st.session_state.get("vqe_maxcut_neg_cost", True))
            if problem == "Ising (h/J)":
                ising_h_text = st.session_state.get("vqe_ising_h", "")
                ising_J_text = st.session_state.get("vqe_ising_J", "")

            pauli_list, prob_meta = _build_problem_paulis(
                problem=str(problem),
                n=n,
                pauli_text=pauli_text,
                maxcut_edges_text=maxcut_edges_text,
                maxcut_negative_cost_h=bool(maxcut_neg_cost),
                ising_h_text=ising_h_text,
                ising_J_text=ising_J_text,
            )

            status.text("Running VQE…")
            progress.progress(0.05)

            if use_real:
                ok, energy, meta, notes, history = _try_run_real_vqe(
                    num_qubits=n,
                    pauli_list=pauli_list,
                    ansatz_name=str(ansatz_name),
                    reps=int(reps),
                    optimizer_name=str(optimizer_name),
                    maxiter=int(maxiter),
                    backend_choice=str(backend_choice),
                    seed=seed,
                )

            progress.progress(0.75)

            if not ok:
                status.text("Real VQE failed/unavailable — running toy fallback…")
                energy, meta2, notes2, history2 = _run_toy_energy(seed=seed)
                meta = {
                    **meta2,
                    "real_vqe_attempted": bool(use_real),
                    "real_vqe_notes": notes,
                    "requested": {
                        "num_qubits": n,
                        "ansatz": str(ansatz_name),
                        "reps": int(reps),
                        "optimizer": str(optimizer_name),
                        "maxiter": int(maxiter),
                        "backend_choice": str(backend_choice),
                        "problem": str(problem),
                        "problem_meta": prob_meta,
                    },
                }
                notes = notes2
                history = history2
            else:
                meta = {**meta, "problem": str(problem), "problem_meta": prob_meta}

            progress.progress(0.9)

            mult = _energy_to_risk_multiplier(energy)

            reference_energy: Optional[float] = None
            if ref_mode == "Latest snapshot in golden_build":
                reference_energy = _load_reference_from_latest_snapshot(gb_dir)
            elif ref_mode == "Manual value":
                reference_energy = manual_ref
            elif ref_mode == "Exact (bruteforce, small n)":
                if str(problem).startswith("MaxCut"):
                    edges = prob_meta.get("edges", None)
                    reference_energy = _exact_reference_energy(
                        problem="MaxCut",
                        n=n,
                        edges=edges if isinstance(edges, list) else None,
                        max_n=int(exact_max_n),
                        maxcut_use_negative_cost_hamiltonian=bool(prob_meta.get("hamiltonian_is_negative_cost", True)),
                    )
                elif str(problem).startswith("Ising"):
                    h = prob_meta.get("h", None)
                    J = prob_meta.get("J", None)
                    reference_energy = _exact_reference_energy(
                        problem="Ising",
                        n=n,
                        h=h if isinstance(h, list) else None,
                        J=J if isinstance(J, list) else None,
                        max_n=int(exact_max_n),
                    )
                else:
                    reference_energy = None
            else:
                reference_energy = None

            smoke = _smoke_eval(
                energy=float(energy),
                reference_energy=reference_energy,
                pass_delta=float(pass_delta),
                warn_delta=float(warn_delta),
            )
            smoke["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
            smoke["ref_mode"] = ref_mode

            interpretation: Dict[str, Any] = {}
            if str(problem).startswith("MaxCut"):
                negflag = bool(prob_meta.get("hamiltonian_is_negative_cost", True))
                implied_cost = (-float(energy)) if negflag and np.isfinite(energy) else (float(energy) if np.isfinite(energy) else None)
                interpretation["maxcut_implied_cost"] = implied_cost
                interpretation["maxcut_mapping"] = "cost = -energy" if negflag else "cost = +energy"

                edges = prob_meta.get("edges", [])
                if isinstance(edges, list):
                    exact = _exact_maxcut_best_cost_and_bits(n=n, edges=edges, max_n=int(exact_max_n))
                    if exact is not None:
                        best_cost = float(exact["best_cost"])
                        best_bits = int(exact["best_bits"])
                        interpretation["maxcut_exact_best_cost"] = best_cost
                        interpretation["maxcut_exact_best_bits"] = best_bits
                        interpretation["maxcut_exact_best_bitstring"] = format(best_bits, f"0{n}b")[::-1]
                        if implied_cost is not None and best_cost > 0:
                            gap = 100.0 * max(0.0, (best_cost - float(implied_cost))) / best_cost
                            interpretation["maxcut_gap_percent"] = float(gap)

            decode_result: Optional[Dict[str, Any]] = None
            if bool(decode_enabled) and ok:
                best_param_values = None
                try:
                    bpv = meta.get("best_param_values", None)
                    if isinstance(bpv, list) and all(isinstance(x, (int, float)) for x in bpv):
                        best_param_values = [float(x) for x in bpv]
                except Exception:
                    best_param_values = None

                if str(problem).startswith("MaxCut"):
                    edges = prob_meta.get("edges", None)
                    decode_result = _decode_vqe_bitstrings_for_problem(
                        problem="MaxCut",
                        n=int(n),
                        ansatz_name=str(ansatz_name),
                        reps=int(reps),
                        best_param_values=best_param_values,
                        edges=edges if isinstance(edges, list) else None,
                        maxcut_negative_cost_h=bool(prob_meta.get("hamiltonian_is_negative_cost", True)),
                        method=str(decode_method),
                        shots=int(decode_shots),
                        topk=int(decode_topk),
                        seed=seed,
                    )
                    interpretation["decode_vqe_solution"] = decode_result

                elif str(problem).startswith("Ising"):
                    h = prob_meta.get("h", None)
                    J = prob_meta.get("J", None)
                    decode_result = _decode_vqe_bitstrings_for_problem(
                        problem="Ising",
                        n=int(n),
                        ansatz_name=str(ansatz_name),
                        reps=int(reps),
                        best_param_values=best_param_values,
                        ising_h=h if isinstance(h, list) else None,
                        ising_J=J if isinstance(J, list) else None,
                        method=str(decode_method),
                        shots=int(decode_shots),
                        topk=int(decode_topk),
                        seed=seed,
                    )
                    interpretation["decode_vqe_solution"] = decode_result

            risk_signal = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "model": "VQE" if ok else "TOY",
                "energy": float(energy) if np.isfinite(energy) else None,
                "risk_budget_multiplier": float(mult),
                "notes": str(notes),
                "meta": meta,
                "interpretation": interpretation,
            }

            snapshot = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "started_at": started.isoformat(timespec="seconds"),
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                "model": "VQE" if ok else "TOY",
                "energy": float(energy) if np.isfinite(energy) else None,
                "risk_budget_multiplier": float(mult),
                "notes": str(notes),
                "meta": meta,
                "history": history,
                "smoke": smoke,
                "interpretation": interpretation,
            }

            st.session_state["vqe_risk_signal"] = risk_signal
            st.session_state["vqe_snapshot"] = snapshot
            st.session_state["vqe_smoke"] = smoke

            out_snap = os.path.join(gb_dir, f"vqe_snapshot_{ts_fs}.json")
            out_smoke = os.path.join(gb_dir, f"vqe_smoke_{ts_fs}.json")
            wrote_snap = _write_json(out_snap, snapshot)
            wrote_smoke = _write_json(out_smoke, smoke)

            progress.progress(1.0)
            status.empty()
            progress.empty()

            st.success("VQE run complete.")
            st.metric("Energy", f"{energy:.6f}" if np.isfinite(energy) else "—")
            st.metric("Risk budget multiplier", f"{mult:.2f}")

            if str(problem).startswith("MaxCut"):
                st.markdown("### Result interpretation (MaxCut)")
                implied = interpretation.get("maxcut_implied_cost", None)
                st.write({"mapping": interpretation.get("maxcut_mapping", "—")})
                st.metric("VQE implied cut cost", f"{float(implied):.6f}" if implied is not None else "—")

                if "maxcut_exact_best_cost" in interpretation:
                    best_cost = float(interpretation["maxcut_exact_best_cost"])
                    st.metric("Exact best cut cost", f"{best_cost:.6f}")
                    st.caption(f"Exact best bitstring (LSB→node0): {interpretation.get('maxcut_exact_best_bitstring')}")
                    if "maxcut_gap_percent" in interpretation:
                        st.metric("Gap (%)", f"{float(interpretation['maxcut_gap_percent']):.2f}%")
                else:
                    st.caption(f"Exact best cut not computed (n={n} exceeds exact_max_n={int(exact_max_n)}).")

            if isinstance(interpretation.get("decode_vqe_solution"), dict):
                dec = interpretation["decode_vqe_solution"]
                st.markdown("### Step 5 — Decoded solution (from VQE params)")
                if not bool(dec.get("enabled", False)):
                    st.warning(f"Decode unavailable: {dec.get('reason', 'unknown reason')}")
                else:
                    st.write({"used_backend": dec.get("used_backend"), "ansatz": dec.get("ansatz")})
                    if dec.get("objective") == "MaxCut cost":
                        st.metric("Expected cut cost (from distribution)", f"{float(dec.get('expected_cost', 0.0)):.6f}")
                        if "best_cost" in dec:
                            st.metric("Best decoded cost", f"{float(dec.get('best_cost')):.6f}")
                            st.caption(f"Best decoded bitstring (LSB→node0): {dec.get('best_bitstring_lsb')}")
                    elif dec.get("objective") == "Ising energy":
                        st.metric("Expected energy (from distribution)", f"{float(dec.get('expected_energy', 0.0)):.6f}")
                        if "best_energy" in dec:
                            st.metric("Best decoded energy", f"{float(dec.get('best_energy')):.6f}")
                            st.caption(f"Best decoded bitstring (LSB→q0): {dec.get('best_bitstring_lsb')}")
                    with st.expander("Top-K decoded outcomes", expanded=False):
                        topk_rows = dec.get("topk", [])
                        if isinstance(topk_rows, list) and topk_rows:
                            if pd is not None:
                                st.dataframe(pd.DataFrame(topk_rows), use_container_width=True)
                            else:
                                st.write(topk_rows)
                        else:
                            st.info("No top-K rows available.")

            sstat = str(smoke.get("status", "WARN")).upper()
            if sstat == "PASS":
                st.success(f"✅ VQE Smoke: {sstat}")
            elif sstat == "WARN":
                st.warning(f"⚠️ VQE Smoke: {sstat}")
            else:
                st.error(f"❌ VQE Smoke: {sstat}")

            if smoke.get("abs_delta") is not None and smoke.get("reference_energy") is not None:
                st.caption(
                    f"|ΔE|={float(smoke['abs_delta']):.6f} "
                    f"(ref={float(smoke['reference_energy']):.6f}, pass≤{float(pass_delta):.3f}, warn≤{float(warn_delta):.3f})"
                )
            else:
                st.caption(str(smoke.get("reason", "No delta computed.")))

            if wrote_snap:
                st.caption(f"Saved snapshot: {out_snap}")
            else:
                st.warning("Could not write snapshot to disk (filesystem may be restricted).")

            if wrote_smoke:
                st.caption(f"Saved smoke: {out_smoke}")
            else:
                st.warning("Could not write smoke to disk (filesystem may be restricted).")

        except Exception as e:
            try:
                progress.empty()
                status.empty()
            except Exception:
                pass
            st.error(f"VQE run failed: {e}")

    # ============================================================
    # STEP 8.8 — BRIDGE PANEL (Scaled limits + simulate gate)
    # ============================================================
    st.markdown("---")
    st.markdown("## Step 8.8 — Risk Gate bridge")
    st.caption(
        "This section converts the VQE risk multiplier into **scaled risk limits** and provides a "
        "single choke-point (`enforce_risk_gate_and_execute`) your trading engine can call."
    )

    # Base limits inputs (safe defaults)
    base_defaults = {
        "max_notional_usd": 500.0,
        "max_position_usd": 500.0,
        "max_daily_loss_usd": 100.0,
        "max_var_usd": 60.0,
        "max_cvar_usd": 90.0,
        "max_leverage": 2.0,
    }

    cur_sig = st.session_state.get("vqe_risk_signal", {})
    cur_mult = 1.0
    try:
        if isinstance(cur_sig, dict):
            cur_mult = float(cur_sig.get("risk_budget_multiplier", 1.0))
    except Exception:
        cur_mult = 1.0

    r1, r2, r3 = st.columns(3)
    with r1:
        base_max_notional = st.number_input("Base max_notional_usd", min_value=0.0, max_value=1e9, value=float(base_defaults["max_notional_usd"]), step=50.0, key="vqe_base_max_notional")
        base_max_position = st.number_input("Base max_position_usd", min_value=0.0, max_value=1e9, value=float(base_defaults["max_position_usd"]), step=50.0, key="vqe_base_max_position")
    with r2:
        base_max_var = st.number_input("Base max_var_usd", min_value=0.0, max_value=1e9, value=float(base_defaults["max_var_usd"]), step=5.0, key="vqe_base_max_var")
        base_max_cvar = st.number_input("Base max_cvar_usd", min_value=0.0, max_value=1e9, value=float(base_defaults["max_cvar_usd"]), step=5.0, key="vqe_base_max_cvar")
    with r3:
        base_max_lev = st.number_input("Base max_leverage", min_value=0.0, max_value=1000.0, value=float(base_defaults["max_leverage"]), step=0.25, key="vqe_base_max_lev")
        base_max_daily_loss = st.number_input("Base max_daily_loss_usd", min_value=0.0, max_value=1e9, value=float(base_defaults["max_daily_loss_usd"]), step=10.0, key="vqe_base_max_daily_loss")

    base_limits = {
        "max_notional_usd": float(base_max_notional),
        "max_position_usd": float(base_max_position),
        "max_daily_loss_usd": float(base_max_daily_loss),
        "max_var_usd": float(base_max_var),
        "max_cvar_usd": float(base_max_cvar),
        "max_leverage": float(base_max_lev),
    }

    scaled = build_scaled_risk_limits(base_limits=base_limits, risk_budget_multiplier=float(cur_mult))
    st.session_state["vqe_scaled_risk_limits"] = scaled

    st.write(
        {
            "current_vqe_multiplier": float(cur_mult),
            "scaled_limits_saved_to": "st.session_state['vqe_scaled_risk_limits']",
        }
    )
    with st.expander("Scaled risk limits (JSON)", expanded=False):
        st.json(scaled)

    st.markdown("### Simulate an order through the gate (validation)")
    sA, sB, sC, sD = st.columns(4)
    with sA:
        sim_notional = st.number_input("Requested notional (USD)", min_value=0.0, max_value=1e9, value=600.0, step=50.0, key="vqe_sim_notional")
    with sB:
        sim_var = st.number_input("Estimated VaR (USD)", min_value=0.0, max_value=1e9, value=50.0, step=5.0, key="vqe_sim_var")
    with sC:
        sim_cvar = st.number_input("Estimated CVaR (USD)", min_value=0.0, max_value=1e9, value=80.0, step=5.0, key="vqe_sim_cvar")
    with sD:
        sim_lev = st.number_input("Leverage used", min_value=0.0, max_value=1000.0, value=1.0, step=0.25, key="vqe_sim_lev")

    if st.button("Run gate simulation", key="vqe_run_gate_sim"):
        gate = apply_risk_gates(
            requested_notional_usd=float(sim_notional),
            est_var_usd=float(sim_var),
            est_cvar_usd=float(sim_cvar),
            leverage_used=float(sim_lev),
            limits=scaled,
        )
        st.session_state["risk_gate_last"] = gate

    last_gate = st.session_state.get("risk_gate_last", None)
    if isinstance(last_gate, dict):
        stat = str(last_gate.get("status", "—")).upper()
        if stat == "APPROVED":
            st.success(f"✅ Gate: {stat}")
        elif stat == "CLAMPED":
            st.warning(f"⚠️ Gate: {stat}")
        else:
            st.error(f"❌ Gate: {stat}")

        st.write(
            {
                "requested_notional_usd": last_gate.get("requested_notional_usd"),
                "final_notional_usd": last_gate.get("final_notional_usd"),
                "reasons": last_gate.get("reasons", []),
            }
        )
        with st.expander("risk_gate_last (full JSON)", expanded=False):
            st.json(last_gate)
    else:
        st.info("No risk gate decision yet. Click **Run gate simulation**.")

    # ============================================================
    # STEP 8.9 — INTEGRATED EXECUTION SIM (smoke policy + estimator + audit)
    # ============================================================
    st.markdown("---")
    st.markdown("## Step 8.9 — Wire the gate into execution")
    st.caption(
        "This panel demonstrates the exact integration pattern you should use in your real trade button: "
        "**estimate risk → smoke policy → risk gate → execute → audit**."
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        smoke_policy = st.selectbox("Smoke policy", ["Moderate", "Strict", "Off"], index=0, key="vqe_smoke_policy")
    with p2:
        warn_tighten = st.number_input("WARN tighten factor", min_value=0.05, max_value=1.0, value=0.85, step=0.05, key="vqe_warn_tighten")
    with p3:
        fail_behavior = st.selectbox("FAIL behavior", ["Block", "Clamp"], index=0, key="vqe_fail_behavior")
    with p4:
        audit_to_disk = st.checkbox("Write audit JSON to golden_build/", value=False, key="vqe_audit_to_disk")

    st.markdown("### Simulate a full order submit (end-to-end)")
    o1, o2, o3, o4, o5 = st.columns([1.2, 1.0, 1.0, 1.0, 1.2])
    with o1:
        sim_symbol = st.text_input("Symbol", "BTC-USD", key="vqe_order_symbol")
    with o2:
        sim_side = st.selectbox("Side", ["BUY", "SELL"], index=0, key="vqe_order_side")
    with o3:
        sim_notional2 = st.number_input("Notional (USD)", min_value=0.0, max_value=1e9, value=600.0, step=50.0, key="vqe_order_notional")
    with o4:
        sim_vol = st.number_input("Volatility (proxy)", min_value=0.0001, max_value=5.0, value=0.02, step=0.005, key="vqe_order_vol")
    with o5:
        sim_equity = st.number_input("Equity (USD)", min_value=0.0, max_value=1e9, value=1000.0, step=100.0, key="vqe_order_equity")

    def _mock_execute(order_obj: Dict[str, Any]) -> Dict[str, Any]:
        # Replace this with your real broker/exchange execution call.
        return {
            "ok": True,
            "mock_order_id": f"MOCK-{dt.datetime.now().strftime('%H%M%S')}",
            "executed_notional_usd": float(order_obj.get("notional_usd", 0.0)),
            "symbol": order_obj.get("symbol"),
            "side": order_obj.get("side"),
        }

    if st.button("Submit order through VQE gate", key="vqe_submit_order_btn"):
        order = {
            "symbol": str(sim_symbol).strip(),
            "side": str(sim_side),
            "notional_usd": float(sim_notional2),
            "volatility": float(sim_vol),
            "equity_usd": float(sim_equity) if float(sim_equity) > 0 else None,
        }
        gate = submit_order_through_vqe_gate(
            st=st,
            order=order,
            execute_fn=_mock_execute,
            policy=str(smoke_policy),
            warn_tighten_factor=float(warn_tighten),
            fail_behavior=str(fail_behavior),
            limits_key="vqe_scaled_risk_limits",
            audit_to_disk=bool(audit_to_disk),
            audit_dir=str(gb_dir),
        )
        st.session_state["risk_gate_last"] = gate

    # Show last audit
    if isinstance(st.session_state.get("trade_audit_log"), list) and st.session_state["trade_audit_log"]:
        with st.expander("Latest trade audit record", expanded=False):
            st.json(st.session_state["trade_audit_log"][-1])

    st.markdown("---")
    st.markdown("### Results & diagnostics")

    cur_smoke = st.session_state.get("vqe_smoke")
    if isinstance(cur_smoke, dict):
        sstat = str(cur_smoke.get("status", "WARN")).upper()
        if sstat == "PASS":
            st.success(f"✅ Current vqe_smoke: {sstat}")
        elif sstat == "WARN":
            st.warning(f"⚠️ Current vqe_smoke: {sstat}")
        else:
            st.error(f"❌ Current vqe_smoke: {sstat}")
        with st.expander("vqe_smoke (bridge object)", expanded=False):
            st.json(cur_smoke)
    else:
        st.info("No vqe_smoke yet. Run VQE to create it.")

    snap = st.session_state.get("vqe_snapshot")
    if not isinstance(snap, dict):
        st.info("No VQE snapshot yet. Run VQE to generate results.")
    else:
        cL, cM, cR = st.columns(3)
        with cL:
            st.metric("Model", str(snap.get("model", "—")))
        with cM:
            e = snap.get("energy", None)
            st.metric("Energy", f"{float(e):.6f}" if e is not None else "—")
        with cR:
            rbm = snap.get("risk_budget_multiplier", 1.0)
            st.metric("Risk budget", f"{float(rbm):.2f}")

        interp = snap.get("interpretation", {})
        if isinstance(interp, dict) and len(interp) > 0:
            with st.expander("Interpretation (if available)", expanded=False):
                st.json(interp)

        hist = snap.get("history", [])
        energies, iters = [], []
        if isinstance(hist, list):
            for rec in hist:
                if isinstance(rec, dict) and ("energy" in rec):
                    try:
                        energies.append(float(rec["energy"]))
                        iters.append(int(rec.get("t", len(iters))))
                    except Exception:
                        pass

        if energies:
            st.caption("Convergence (energy vs iteration)")
            try:
                st.line_chart({"energy": energies})
            except Exception:
                st.write({"iter": iters, "energy": energies})
        else:
            st.info("No convergence history captured (this Qiskit VQE stack may not expose callback).")

        with st.expander("Show ansatz circuit (best-effort)", expanded=False):
            meta2 = snap.get("meta", {}) if isinstance(snap.get("meta"), dict) else {}
            try:
                nq = int(meta2.get("num_qubits", st.session_state.get("vqe_num_qubits", 2)))
                reps2 = int(st.session_state.get("vqe_reps", 2))
                ans_name = str(st.session_state.get("vqe_ansatz", "TwoLocal"))

                ansatz, ans_desc, n_params = _build_ansatz(ans_name, nq, reps2)
                st.write({"ansatz": ans_desc, "num_params": n_params})

                try:
                    drawing = ansatz.draw(output="text")
                    st.code(str(drawing))
                except Exception:
                    st.info("Circuit drawing unavailable in this environment.")
            except Exception as e:
                st.warning(f"Could not build/draw ansatz: {e}")

        with st.expander("Snapshot JSON (full)", expanded=False):
            st.json(snap)

        st.markdown("#### Export & persistence")
        snap_json = json.dumps(snap, indent=2)

        cE1, cE2, cE3 = st.columns([1, 1, 2])
        with cE1:
            st.download_button(
                "⬇️ Download snapshot (.json)",
                data=snap_json.encode("utf-8"),
                file_name=f"vqe_snapshot_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="vqe_dl_snapshot",
            )
        with cE2:
            if st.button("💾 Write snapshot to golden_build/", key="vqe_write_snapshot_btn"):
                ts_fs = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(gb_dir, f"vqe_snapshot_{ts_fs}.json")
                ok2 = _write_json(out_path, snap)
                if ok2:
                    st.success(f"Wrote: {out_path}")
                else:
                    st.warning("Could not write snapshot to disk (filesystem may be restricted).")
        with cE3:
            if st.button("💾 Write smoke to golden_build/", key="vqe_write_smoke_btn"):
                if isinstance(st.session_state.get("vqe_smoke"), dict):
                    ts_fs = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(gb_dir, f"vqe_smoke_{ts_fs}.json")
                    ok3 = _write_json(out_path, st.session_state["vqe_smoke"])
                    if ok3:
                        st.success(f"Wrote: {out_path}")
                    else:
                        st.warning("Could not write smoke to disk (filesystem may be restricted).")
                else:
                    st.info("No vqe_smoke to write yet.")

    st.markdown("---")
    st.markdown("### Load latest snapshot/smoke from golden_build")

    lp = _latest_snapshot_path(gb_dir)
    ls = _latest_smoke_path(gb_dir)

    cols = st.columns(2)
    with cols[0]:
        if lp:
            st.caption(f"Latest snapshot: {lp}")
            if st.button("Load latest snapshot → session", key="vqe_load_latest"):
                obj = _read_json(lp)
                if obj is None:
                    st.error("Failed to read snapshot.")
                else:
                    st.session_state["vqe_snapshot"] = obj
                    if isinstance(obj, dict):
                        st.session_state["vqe_risk_signal"] = {
                            "timestamp": obj.get("timestamp"),
                            "model": obj.get("model"),
                            "energy": obj.get("energy"),
                            "risk_budget_multiplier": obj.get("risk_budget_multiplier"),
                            "notes": obj.get("notes", ""),
                            "meta": obj.get("meta", {}),
                            "interpretation": obj.get("interpretation", {}),
                        }
                        embedded_smoke = obj.get("smoke")
                        if isinstance(embedded_smoke, dict):
                            st.session_state["vqe_smoke"] = embedded_smoke
                    st.success("Loaded snapshot (and refreshed vqe_risk_signal; smoke if present).")
        else:
            st.info("No vqe_snapshot_*.json found yet in this golden build directory.")

    with cols[1]:
        if ls:
            st.caption(f"Latest smoke: {ls}")
            if st.button("Load latest smoke → session", key="vqe_load_latest_smoke"):
                obj = _read_json(ls)
                if obj is None:
                    st.error("Failed to read smoke.")
                else:
                    st.session_state["vqe_smoke"] = obj
                    st.success("Loaded smoke into session_state['vqe_smoke'].")
        else:
            st.info("No vqe_smoke_*.json found yet in this golden build directory.")

    st.markdown("---")
    st.markdown("### Current bridge output (vqe_risk_signal)")
    sig = st.session_state.get("vqe_risk_signal")
    if isinstance(sig, dict):
        try:
            st.metric("Risk budget multiplier", f"{float(sig.get('risk_budget_multiplier', 1.0)):.2f}")
        except Exception:
            st.metric("Risk budget multiplier", "—")
        st.json(sig)
    else:
        st.info("No vqe_risk_signal yet. Run VQE to create one.")

    st.markdown("---")
    st.markdown("### Trade audit log (Step 8.9)")
    if st.button("Clear trade audit log", key="vqe_clear_trade_audit"):
        st.session_state["trade_audit_log"] = []
        st.success("Cleared trade audit log.")
    log = st.session_state.get("trade_audit_log", [])
    if isinstance(log, list) and log:
        st.caption(f"{len(log)} audit records stored (bounded).")
        with st.expander("Show audit log (JSON)", expanded=False):
            st.json(log)
    else:
        st.info("No audit records yet. Submit an order through the gate to create one.")
