"""
debt_management.py — Debt payoff simulator (Minimum Only vs. Snowball vs. Avalanche)

Simulates paying off a list of debts (student loans, credit cards, personal
loans, business loans) month by month under three strategies, so a user can
see how much interest and time each strategy saves:

  - Minimum Only — baseline: pay every debt's minimum, nothing extra.
  - Snowball     — extra budget targets the SMALLEST-BALANCE debt first;
                    once a debt is paid off, its freed-up minimum payment
                    rolls into the pool available for the next target.
  - Avalanche    — same rollover mechanic, but targets the HIGHEST-APR debt
                    first (mathematically minimizes total interest paid).

Purely classical — no quantum dependency, never gated by QUANTUM_FEATURES_ENABLED.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Default amortization term used only for the *suggested* minimum-payment hint —
# the simulation itself always uses the user's entered minimum_payment.
_DEFAULT_TERM_MONTHS = {
    "student_loan": 120,
    "personal_loan": 60,
    "business_loan": 84,
}

# Safety cap so a debt whose minimum payment doesn't cover accruing interest
# can't loop forever.
_MAX_MONTHS = 600


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _amortized_payment(principal: float, annual_rate_pct: float, term_months: int) -> float:
    if term_months <= 0 or principal <= 0:
        return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0:
        return principal / term_months
    factor = (1 + r) ** term_months
    return principal * r * factor / (factor - 1)


def compute_minimum_payment_hint(debt_type: str, balance: float, apr_pct: float) -> float:
    """A *suggested* minimum, shown as a UI hint only — never used by the
    simulation itself, which always uses the user's entered minimum_payment."""
    balance = float(balance or 0.0)
    apr_pct = float(apr_pct or 0.0)
    if debt_type == "credit_card":
        return round(max(25.0, 0.02 * balance), 2)
    term = _DEFAULT_TERM_MONTHS.get(debt_type, 60)
    return round(_amortized_payment(balance, apr_pct, term), 2)


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def simulate_payoff(debts: List[Dict[str, Any]], extra_monthly_payment: float, strategy: str) -> Dict[str, Any]:
    """Pure function — no I/O. Simulates one strategy month by month."""
    debts = list(debts or [])
    ids = [d["id"] for d in debts]
    balances = {d["id"]: max(0.0, float(d.get("balance") or 0.0)) for d in debts}
    min_payments = {d["id"]: max(0.0, float(d.get("minimum_payment") or 0.0)) for d in debts}
    apr = {d["id"]: float(d.get("apr_pct") or 0.0) for d in debts}

    total_min = sum(min_payments.values())
    # "Minimum only" never adds the extra budget — that's what makes it the baseline.
    total_budget = total_min + (float(extra_monthly_payment or 0.0) if strategy != "minimum_only" else 0.0)

    total_interest = 0.0
    payoff_month: Dict[str, int] = {}
    timeline: List[Dict[str, Any]] = []
    month = 0

    while any(balances[d] > 0.005 for d in ids) and month < _MAX_MONTHS:
        month += 1
        active = [d for d in ids if balances[d] > 0.005]

        for d in active:
            interest = balances[d] * (apr[d] / 100.0 / 12.0)
            balances[d] += interest
            total_interest += interest

        spent = 0.0
        for d in active:
            pay = min(min_payments[d], balances[d])
            balances[d] -= pay
            spent += pay
            if balances[d] <= 0.005:
                balances[d] = 0.0
                payoff_month.setdefault(d, month)

        # Roll whatever's left of the fixed monthly budget — including any
        # freed-up minimums from debts already paid off — onto the strategy's target.
        leftover = max(0.0, total_budget - spent)
        if leftover > 0:
            still_active = [d for d in ids if balances[d] > 0.005]
            if strategy == "snowball":
                target_order = sorted(still_active, key=lambda d: balances[d])
            elif strategy == "avalanche":
                target_order = sorted(still_active, key=lambda d: -apr[d])
            else:
                target_order = []
            for d in target_order:
                if leftover <= 0:
                    break
                pay = min(leftover, balances[d])
                balances[d] -= pay
                leftover -= pay
                if balances[d] <= 0.005:
                    balances[d] = 0.0
                    payoff_month.setdefault(d, month)

        timeline.append({"month": month, "total_remaining_balance": round(sum(balances.values()), 2)})

    debt_free = all(balances[d] <= 0.005 for d in ids)

    return {
        "strategy": strategy,
        "months_to_debt_free": month if debt_free else None,
        "total_interest_paid": round(total_interest, 2),
        "payoff_order": [{"id": d, "payoff_month": payoff_month.get(d)} for d in ids],
        "timeline": timeline,
        "hit_cap": not debt_free,
    }


def compare_strategies(debts: List[Dict[str, Any]], extra_monthly_payment: float) -> Dict[str, Any]:
    results = {
        "minimum_only": simulate_payoff(debts, extra_monthly_payment, "minimum_only"),
        "snowball": simulate_payoff(debts, extra_monthly_payment, "snowball"),
        "avalanche": simulate_payoff(debts, extra_monthly_payment, "avalanche"),
    }

    def _savings(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        interest_saved = base["total_interest_paid"] - other["total_interest_paid"]
        months_saved = None
        if base["months_to_debt_free"] is not None and other["months_to_debt_free"] is not None:
            months_saved = base["months_to_debt_free"] - other["months_to_debt_free"]
        return {"interest_saved": round(interest_saved, 2), "months_saved": months_saved}

    summary = {
        "snowball_vs_minimum": _savings(results["minimum_only"], results["snowball"]),
        "avalanche_vs_minimum": _savings(results["minimum_only"], results["avalanche"]),
        "avalanche_vs_snowball": _savings(results["snowball"], results["avalanche"]),
    }

    return {"strategies": results, "summary": summary}


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

def get_or_create_plan(user_id: str) -> Dict[str, Any]:
    rows = _select("user_debt_plans", user_id)
    if rows:
        return rows[0]
    return {"user_id": user_id, "debts": [], "extra_monthly_payment": 0.0}


def save_plan(user_id: str, debts: List[Dict[str, Any]], extra_monthly_payment: float) -> Dict[str, Any]:
    row = {"user_id": user_id, "debts": debts, "extra_monthly_payment": extra_monthly_payment}
    return _upsert("user_debt_plans", row)
