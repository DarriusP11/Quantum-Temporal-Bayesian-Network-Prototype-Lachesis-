"""
credit_behavior_sim.py — Credit Behavior Simulator (educational, illustrative)

NOT a real credit-bureau algorithm. This is a classical, consumer-behavior
teaching tool that walks a hypothetical monthly "what if" scenario (on-time vs.
missed payments, utilization changes, opening new accounts) through a
simplified, publicly-documented FICO factor-weighting scheme and produces an
illustrative month-by-month score trajectory. It is unrelated to — and does
not reuse — credit_risk.py's quantum Gaussian-Conditional-Independence / IQAE
institutional-portfolio model. Only the FICO-band lookup (`fico_to_pd`) is
reused from credit_risk.py, purely for illustrative narrative context.

Purely classical — no quantum dependency, never gated by QUANTUM_FEATURES_ENABLED.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

try:
    from credit_risk import fico_to_pd  # illustrative narrative context only
except Exception:
    def fico_to_pd(fico_score: int):  # fallback if credit_risk.py is unavailable
        return None, "Unknown"

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Standard, publicly-documented FICO factor weighting — used only as an
# illustrative basis for this educational simulation.
FACTOR_WEIGHTS: Dict[str, float] = {
    "payment_history": 0.35,
    "utilization": 0.30,
    "history_length": 0.15,
    "credit_mix": 0.10,
    "new_credit": 0.10,
}

DISCLAIMER = "Educational simulation only — not a real FICO/VantageScore calculation."

_SCORE_MIN, _SCORE_MAX = 300, 850

_TIP_TEXT: Dict[str, str] = {
    "payment_history": (
        "Set up autopay or reminders — payment history is the single biggest "
        "factor (35%). Even one missed payment causes a sharp drop that takes "
        "months of on-time payments to recover from."
    ),
    "utilization": (
        "Keep credit utilization below 30% of your limit, ideally under 10%, "
        "for the strongest score. Utilization responds quickly in both directions."
    ),
    "history_length": (
        "Avoid closing your oldest credit accounts — a longer average account "
        "age helps your score, and it only builds slowly over time."
    ),
    "credit_mix": (
        "A healthy mix of credit types (credit cards, installment loans) can "
        "modestly help your score, though it's the smallest factor."
    ),
    "new_credit": (
        "Space out new credit applications. Each new account causes a temporary "
        "dip that recovers over the following months — avoid opening several at once."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _fico_to_baseline_subscore(starting_fico: int) -> float:
    """Map a 300-850 starting FICO to an initial 0-100 sub-score baseline."""
    return _clip(((starting_fico - _SCORE_MIN) / (_SCORE_MAX - _SCORE_MIN)) * 100.0)


def _blend_to_fico(sub_scores: Dict[str, float]) -> int:
    blend = sum(sub_scores[f] * w for f, w in FACTOR_WEIGHTS.items())
    return round(_SCORE_MIN + (blend / 100.0) * (_SCORE_MAX - _SCORE_MIN))


def _step_payment_history(score: float, behavior: str) -> float:
    if behavior == "missed":
        return _clip(score - 60.0)   # sharp drop
    if behavior == "minimum_only":
        return _clip(score - 5.0)    # mild drop
    return _clip(score + 2.0)        # on_time: slow recovery/creep


def _step_utilization(utilization_pct: float) -> float:
    u = max(0.0, float(utilization_pct))
    if u <= 30.0:
        score = 100.0 - u            # roughly linear penalty below 30%
    else:
        score = 70.0 - (u - 30.0) * 1.5   # mildly convex penalty above 30%
    return _clip(score)


def _step_history_length(score: float, new_account_opened: bool) -> float:
    if new_account_opened:
        return _clip(score - 10.0)   # opening an account lowers avg account age
    return _clip(score + 0.5)        # creeps up slowly every month


def _step_new_credit(score: float, new_account_opened: bool) -> float:
    if new_account_opened:
        return _clip(score - 30.0)   # temporary dip
    return _clip(score + 3.0)        # recovers over time


def _generate_tips(final_breakdown: Dict[str, float]) -> List[str]:
    weakest_first = sorted(final_breakdown.items(), key=lambda kv: kv[1])
    tips: List[str] = []
    for factor, score in weakest_first:
        if len(tips) >= 5:
            break
        if len(tips) < 3 or score < 60.0:
            tips.append(_TIP_TEXT[factor])
    return tips


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def simulate_trajectory(
    starting_fico: int,
    monthly_income: float,
    monthly_debt: float,
    months: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pure function — no I/O. Walks `months` behavior day-by-day (month-by-month)
    and produces an illustrative 300-850 score trajectory."""
    baseline = _fico_to_baseline_subscore(starting_fico)
    sub_scores: Dict[str, float] = {
        "payment_history": baseline,
        "utilization": baseline,
        "history_length": baseline,
        "credit_mix": baseline,     # no behavior input drives this in Phase 1
        "new_credit": baseline,
    }

    trajectory: List[Dict[str, Any]] = []
    for i, m in enumerate(months or []):
        behavior = (m.get("payment_behavior") or "on_time").lower()
        utilization_pct = float(m.get("utilization_pct") or 0.0)
        new_account_opened = bool(m.get("new_account_opened") or False)

        sub_scores["payment_history"] = _step_payment_history(sub_scores["payment_history"], behavior)
        sub_scores["utilization"] = _step_utilization(utilization_pct)
        sub_scores["history_length"] = _step_history_length(sub_scores["history_length"], new_account_opened)
        sub_scores["new_credit"] = _step_new_credit(sub_scores["new_credit"], new_account_opened)
        # credit_mix stays constant — no per-month input drives it in Phase 1

        score = _blend_to_fico(sub_scores)
        trajectory.append({
            "month": i + 1,
            "score": score,
            "factor_breakdown": {k: round(v, 1) for k, v in sub_scores.items()},
        })

    ending_fico = trajectory[-1]["score"] if trajectory else int(starting_fico)
    dti = (monthly_debt / monthly_income) if monthly_income else None
    final_breakdown = trajectory[-1]["factor_breakdown"] if trajectory else {
        k: round(v, 1) for k, v in sub_scores.items()
    }

    return {
        "trajectory": trajectory,
        "starting_fico": int(starting_fico),
        "ending_fico": ending_fico,
        "dti": dti,
        "disclaimer": DISCLAIMER,
        "tips": _generate_tips(final_breakdown),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE PERSISTENCE (self-contained, see budgeting.py for the rationale)
# ══════════════════════════════════════════════════════════════════════════════

def _select(table: str, user_id: str, select: str = "*") -> list:
    supabase_url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url or not service_key:
        return []
    headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}"}
    r = requests.get(
        f"{supabase_url}/rest/v1/{table}?user_id=eq.{user_id}&select={select}",
        headers=headers, timeout=10,
    )
    return r.json() if r.ok else []


def _upsert(table: str, row: dict) -> dict:
    supabase_url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url or not service_key:
        raise RuntimeError("Persistence is not configured on this server")
    headers = {
        "apikey": service_key, "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=representation",
    }
    r = requests.post(f"{supabase_url}/rest/v1/{table}?on_conflict=user_id", json=row, headers=headers, timeout=10)
    if not r.ok:
        raise RuntimeError(f"Failed to save to {table}: {r.text[:200]}")
    rows = r.json()
    return rows[0] if rows else row


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """No persistence — just runs the trajectory for the given payload."""
    return simulate_trajectory(
        starting_fico=payload["starting_fico"],
        monthly_income=payload["monthly_income"],
        monthly_debt=payload["monthly_debt"],
        months=payload.get("months") or [],
    )


def get_or_create_profile(user_id: str) -> Dict[str, Any]:
    rows = _select("user_credit_sim_profiles", user_id)
    if rows:
        return rows[0]
    return {
        "user_id": user_id,
        "starting_fico": 700,
        "monthly_income": 0.0,
        "monthly_debt": 0.0,
        "behavior_assumptions": {},
        "last_trajectory": None,
    }


def save_profile(
    user_id: str,
    starting_fico: int,
    monthly_income: float,
    monthly_debt: float,
    behavior_assumptions: dict,
    last_trajectory: Optional[dict] = None,
) -> Dict[str, Any]:
    row = {
        "user_id": user_id,
        "starting_fico": starting_fico,
        "monthly_income": monthly_income,
        "monthly_debt": monthly_debt,
        "behavior_assumptions": behavior_assumptions,
        "last_trajectory": last_trajectory,
    }
    return _upsert("user_credit_sim_profiles", row)
