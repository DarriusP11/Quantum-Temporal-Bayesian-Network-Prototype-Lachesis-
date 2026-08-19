"""
home_planning.py — Housing cost simulator (buy vs. rent, editable utilities)

Lets a user evaluate the monthly and upfront cost of four housing types —
Home (buy), Apartment (rent), Dorm, Mobile Home — and, when one "buy" type
(home/mobile_home) and one "rent" type (apartment/dorm) are both provided,
produces a rent-vs-buy cumulative-cost timeline with a breakeven year.

Scope (Phase 2, as approved): a cash-flow comparison plus a simple breakeven
estimate factoring in closing costs and home appreciation — NOT a full
opportunity-cost/NPV model (no modeling of investing the down payment
elsewhere, no mortgage-interest tax deduction).

Purely classical — no quantum dependency, never gated by QUANTUM_FEATURES_ENABLED.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Rough national-average estimates — editable placeholders, not authoritative.
DEFAULT_UTILITIES: Dict[str, Dict[str, Any]] = {
    "electricity": {"label": "Electricity", "default_monthly": 135.0},
    "water_sewer": {"label": "Water & Sewer", "default_monthly": 45.0},
    "gas_heating": {"label": "Gas / Heating", "default_monthly": 70.0},
    "internet": {"label": "Internet", "default_monthly": 65.0},
    "trash": {"label": "Trash & Recycling", "default_monthly": 25.0},
}

DEFAULT_UTILITY_VALUES: Dict[str, float] = {k: v["default_monthly"] for k, v in DEFAULT_UTILITIES.items()}

# Housing types where a separate monthly utility bill is realistic. Dorm is
# excluded by default since utilities are typically bundled into the fee.
_UTILITY_ELIGIBLE = {"home", "apartment", "mobile_home"}

_BUY_TYPES = {"home", "mobile_home"}
_RENT_TYPES = {"apartment", "dorm"}

_DEFAULT_COMPARISON_SETTINGS = {
    "horizon_years": 10,
    "appreciation_rate_pct": 3.0,
    "selling_cost_pct": 6.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_mortgage_payment(principal: float, annual_rate_pct: float, term_years: int) -> float:
    """Standard amortization formula: M = P*r(1+r)^n / ((1+r)^n - 1)."""
    n = int(term_years) * 12
    if n <= 0 or principal <= 0:
        return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0:
        return principal / n
    factor = (1 + r) ** n
    return principal * r * factor / (factor - 1)


def _remaining_balance(principal: float, annual_rate_pct: float, term_years: int, months_paid: int) -> float:
    """Outstanding loan balance after `months_paid` payments."""
    n = int(term_years) * 12
    if n <= 0 or principal <= 0:
        return 0.0
    months_paid = max(0, min(months_paid, n))
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0:
        return max(0.0, principal * (1 - months_paid / n))
    factor_n = (1 + r) ** n
    factor_k = (1 + r) ** months_paid
    return max(0.0, principal * (factor_n - factor_k) / (factor_n - 1))


def sum_utilities(utilities: Optional[Dict[str, Any]]) -> float:
    return sum(float(v or 0.0) for v in (utilities or {}).values())


def with_utilities(evaluation: Dict[str, Any], utilities: Optional[Dict[str, Any]], include: bool) -> Dict[str, Any]:
    utilities_total = sum_utilities(utilities) if include else 0.0
    out = dict(evaluation)
    out["utilities_total"] = round(utilities_total, 2)
    out["grand_total_monthly"] = round(out.get("monthly_total", 0.0) + utilities_total, 2)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC — per-housing-type evaluators (all pure functions, no I/O)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_home(inputs: Dict[str, Any]) -> Dict[str, Any]:
    price = float(inputs.get("purchase_price") or 0.0)
    down_pct = float(inputs.get("down_payment_pct") if inputs.get("down_payment_pct") is not None else 20.0)
    rate = float(inputs.get("mortgage_rate_pct") if inputs.get("mortgage_rate_pct") is not None else 7.0)
    term = int(inputs.get("term_years") or 30)
    tax_rate = float(inputs.get("property_tax_rate_pct") if inputs.get("property_tax_rate_pct") is not None else 1.1)
    annual_insurance = float(inputs.get("annual_insurance") if inputs.get("annual_insurance") is not None else 1500.0)
    hoa = float(inputs.get("hoa_monthly") or 0.0)
    closing_pct = float(inputs.get("closing_costs_pct") if inputs.get("closing_costs_pct") is not None else 3.0)

    down_payment = price * down_pct / 100.0
    loan_amount = max(0.0, price - down_payment)
    monthly_pi = compute_mortgage_payment(loan_amount, rate, term)
    monthly_tax = price * tax_rate / 100.0 / 12.0
    monthly_insurance = annual_insurance / 12.0
    # Simplified PMI: ~0.5%/yr of the loan balance, only while down payment < 20%.
    monthly_pmi = (loan_amount * 0.005 / 12.0) if down_pct < 20.0 else 0.0
    monthly_total = monthly_pi + monthly_tax + monthly_insurance + hoa + monthly_pmi
    closing_costs = price * closing_pct / 100.0
    upfront_cost = down_payment + closing_costs
    total_interest_paid = max(0.0, monthly_pi * term * 12 - loan_amount)

    return {
        "type": "home",
        "purchase_price": round(price, 2),
        "down_payment": round(down_payment, 2),
        "loan_amount": round(loan_amount, 2),
        "mortgage_rate_pct": rate,
        "term_years": term,
        "monthly_pi": round(monthly_pi, 2),
        "monthly_tax": round(monthly_tax, 2),
        "monthly_insurance": round(monthly_insurance, 2),
        "monthly_hoa": round(hoa, 2),
        "monthly_pmi": round(monthly_pmi, 2),
        "monthly_total": round(monthly_total, 2),
        "closing_costs": round(closing_costs, 2),
        "upfront_cost": round(upfront_cost, 2),
        "total_interest_paid": round(total_interest_paid, 2),
    }


def evaluate_apartment(inputs: Dict[str, Any]) -> Dict[str, Any]:
    rent = float(inputs.get("monthly_rent") or 0.0)
    deposit_multiplier = float(inputs.get("deposit_multiplier") if inputs.get("deposit_multiplier") is not None else 1.0)
    renters_insurance = float(inputs.get("renters_insurance_monthly") if inputs.get("renters_insurance_monthly") is not None else 15.0)

    monthly_total = rent + renters_insurance
    upfront_cost = rent * deposit_multiplier + rent  # security deposit + first month's rent

    return {
        "type": "apartment",
        "monthly_rent": round(rent, 2),
        "monthly_insurance": round(renters_insurance, 2),
        "monthly_total": round(monthly_total, 2),
        "upfront_cost": round(upfront_cost, 2),
    }


def evaluate_dorm(inputs: Dict[str, Any]) -> Dict[str, Any]:
    cost_per_semester = float(inputs.get("cost_per_semester") or 0.0)
    semesters_per_year = float(inputs.get("semesters_per_year") if inputs.get("semesters_per_year") is not None else 2.0)
    meal_plan_included = bool(inputs.get("meal_plan_included", True))

    monthly_total = (cost_per_semester * semesters_per_year) / 12.0

    return {
        "type": "dorm",
        "cost_per_semester": round(cost_per_semester, 2),
        "semesters_per_year": semesters_per_year,
        "meal_plan_included": meal_plan_included,
        "monthly_total": round(monthly_total, 2),
        "upfront_cost": round(cost_per_semester, 2),  # typically billed per semester upfront
        "note": "Utilities (and often meals) are typically bundled into dorm costs.",
    }


def evaluate_mobile_home(inputs: Dict[str, Any]) -> Dict[str, Any]:
    price = float(inputs.get("purchase_price") or 0.0)
    down_pct = float(inputs.get("down_payment_pct") if inputs.get("down_payment_pct") is not None else 10.0)
    rate = float(inputs.get("loan_rate_pct") if inputs.get("loan_rate_pct") is not None else 9.0)
    term = int(inputs.get("term_years") or 20)
    lot_rent = float(inputs.get("lot_rent_monthly") or 0.0)
    annual_insurance = float(inputs.get("annual_insurance") if inputs.get("annual_insurance") is not None else 800.0)

    down_payment = price * down_pct / 100.0
    loan_amount = max(0.0, price - down_payment)
    monthly_pi = compute_mortgage_payment(loan_amount, rate, term)
    monthly_insurance = annual_insurance / 12.0
    monthly_total = monthly_pi + lot_rent + monthly_insurance

    return {
        "type": "mobile_home",
        "purchase_price": round(price, 2),
        "down_payment": round(down_payment, 2),
        "loan_amount": round(loan_amount, 2),
        "mortgage_rate_pct": rate,
        "term_years": term,
        "monthly_pi": round(monthly_pi, 2),
        "monthly_lot_rent": round(lot_rent, 2),
        "monthly_insurance": round(monthly_insurance, 2),
        "monthly_total": round(monthly_total, 2),
        "upfront_cost": round(down_payment, 2),
    }


_EVALUATORS = {
    "home": evaluate_home,
    "apartment": evaluate_apartment,
    "dorm": evaluate_dorm,
    "mobile_home": evaluate_mobile_home,
}


def compare_rent_vs_buy(
    buy_eval: Dict[str, Any],
    rent_eval: Dict[str, Any],
    horizon_years: int,
    appreciation_rate_pct: float,
    selling_cost_pct: float,
) -> Dict[str, Any]:
    """Cumulative cash-flow comparison, net of a hypothetical sale of the home
    at each year (home value minus selling costs minus remaining loan balance).
    Not a full NPV/opportunity-cost model — see module docstring."""
    buy_monthly = buy_eval.get("grand_total_monthly", buy_eval.get("monthly_total", 0.0))
    rent_monthly = rent_eval.get("grand_total_monthly", rent_eval.get("monthly_total", 0.0))
    buy_upfront = buy_eval.get("upfront_cost", 0.0)
    rent_upfront = rent_eval.get("upfront_cost", 0.0)
    purchase_price = float(buy_eval.get("purchase_price") or 0.0)
    loan_amount = float(buy_eval.get("loan_amount") or 0.0)
    rate_pct = float(buy_eval.get("mortgage_rate_pct") or 0.0)
    term_years = int(buy_eval.get("term_years") or 30)

    appreciation_rate = appreciation_rate_pct / 100.0
    timeline: List[Dict[str, Any]] = []
    cumulative_buy_spend = buy_upfront
    cumulative_rent_spend = rent_upfront
    breakeven_year: Optional[int] = None

    for year in range(1, int(horizon_years) + 1):
        cumulative_buy_spend += buy_monthly * 12
        cumulative_rent_spend += rent_monthly * 12
        home_value = purchase_price * ((1 + appreciation_rate) ** year)
        remaining_balance = _remaining_balance(loan_amount, rate_pct, term_years, year * 12)
        net_sale_proceeds = home_value * (1 - selling_cost_pct / 100.0) - remaining_balance
        net_buy_cost = cumulative_buy_spend - net_sale_proceeds
        net_rent_cost = cumulative_rent_spend

        if breakeven_year is None and net_buy_cost <= net_rent_cost:
            breakeven_year = year

        timeline.append({
            "year": year,
            "cumulative_buy_spend": round(cumulative_buy_spend, 2),
            "cumulative_rent_spend": round(cumulative_rent_spend, 2),
            "home_value": round(home_value, 2),
            "net_buy_cost": round(net_buy_cost, 2),
            "net_rent_cost": round(net_rent_cost, 2),
        })

    return {
        "timeline": timeline,
        "breakeven_year": breakeven_year,
        "upfront_comparison": {"buy": round(buy_upfront, 2), "rent": round(rent_upfront, 2)},
        "monthly_comparison": {"buy": round(buy_monthly, 2), "rent": round(rent_monthly, 2)},
    }


def run_simulation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates whichever of option_a/option_b are provided, and — only when
    one side is a buy type (home/mobile_home) and the other is a rent type
    (apartment/dorm) — adds the rent-vs-buy comparison timeline."""
    utilities = payload.get("utilities") or DEFAULT_UTILITY_VALUES
    result: Dict[str, Any] = {}
    option_evals: Dict[str, Dict[str, Any]] = {}

    for key in ("option_a", "option_b"):
        opt = payload.get(key)
        if not opt:
            continue
        housing_type = opt.get("type")
        inputs = opt.get("inputs") or {}
        evaluator = _EVALUATORS.get(housing_type)
        if evaluator is None:
            continue
        ev = evaluator(inputs)
        ev = with_utilities(ev, utilities, include=housing_type in _UTILITY_ELIGIBLE)
        option_evals[key] = ev
        result[key] = ev

    if "option_a" in option_evals and "option_b" in option_evals:
        type_a = option_evals["option_a"]["type"]
        type_b = option_evals["option_b"]["type"]
        horizon = int(payload.get("horizon_years") or _DEFAULT_COMPARISON_SETTINGS["horizon_years"])
        appreciation = float(payload.get("appreciation_rate_pct") if payload.get("appreciation_rate_pct") is not None else _DEFAULT_COMPARISON_SETTINGS["appreciation_rate_pct"])
        selling_cost = float(payload.get("selling_cost_pct") if payload.get("selling_cost_pct") is not None else _DEFAULT_COMPARISON_SETTINGS["selling_cost_pct"])

        if type_a in _BUY_TYPES and type_b in _RENT_TYPES:
            result["comparison"] = compare_rent_vs_buy(option_evals["option_a"], option_evals["option_b"], horizon, appreciation, selling_cost)
        elif type_b in _BUY_TYPES and type_a in _RENT_TYPES:
            result["comparison"] = compare_rent_vs_buy(option_evals["option_b"], option_evals["option_a"], horizon, appreciation, selling_cost)
        # Two buy types or two rent types: no equity-based comparison is computed —
        # the per-option monthly/upfront totals above are still directly comparable.

    result["utilities_used"] = utilities
    return result


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
    rows = _select("user_home_plans", user_id)
    if rows:
        return rows[0]
    return {
        "user_id": user_id,
        "option_a": None,
        "option_b": None,
        "utilities": DEFAULT_UTILITY_VALUES,
        "comparison_settings": _DEFAULT_COMPARISON_SETTINGS,
    }


def save_plan(
    user_id: str,
    option_a: Optional[dict],
    option_b: Optional[dict],
    utilities: dict,
    comparison_settings: dict,
) -> Dict[str, Any]:
    row = {
        "user_id": user_id,
        "option_a": option_a,
        "option_b": option_b,
        "utilities": utilities,
        "comparison_settings": comparison_settings,
    }
    return _upsert("user_home_plans", row)
