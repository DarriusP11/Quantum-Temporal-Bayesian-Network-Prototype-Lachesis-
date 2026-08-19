"""
retirement.py — Classical compound-growth retirement projector (Phase 1 scope)

Deliberately simple: a single fixed monthly-return compounding loop, no tax
brackets, no decumulation phase, no Monte Carlo. Ports RetirementDashboard.tsx's
`computeGrowth()` exactly so the API and the UI never disagree on the numbers.
A deeper engine (tax brackets, decumulation, Monte Carlo) is a later phase.

Purely classical — no quantum dependency, never gated by QUANTUM_FEATURES_ENABLED.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PLAN = {
    "plan_name": "My Plan",
    "current_age": 22,
    "retirement_age": 65,
    "current_savings": 0.0,
    "monthly_contribution": 200.0,
    "expected_return_rate": 7.0,
    "inflation_rate": 0.0,
    "retirement_goal": "",
    # Phase 4 fields — decumulation/withdrawal analysis
    "roth_pct": 0.0,
    "life_expectancy": 90,
    "withdrawal_rate_pct": 4.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_series(
    start_age: int,
    retirement_age: int,
    starting_balance: float,
    monthly_contribution: float,
    annual_return_pct: float,
) -> List[Dict[str, float]]:
    """Month-by-month compounding, one point per year — mirrors computeGrowth()."""
    r = annual_return_pct / 100.0 / 12.0
    years = max(0, retirement_age - start_age)
    balance = float(starting_balance)
    series: List[Dict[str, float]] = []
    for y in range(years + 1):
        series.append({"age": start_age + y, "balance": round(balance, 2)})
        for _m in range(12):
            balance = balance * (1 + r) + monthly_contribution
    return series


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def compute_growth(
    current_age: int,
    retirement_age: int,
    current_savings: float,
    monthly_contribution: float,
    annual_return_pct: float,
) -> Dict[str, Any]:
    """Pure function — no I/O. Returns the 'start now' series plus a 'start at
    30' comparison series exactly like RetirementDashboard.tsx's two-line chart."""
    years = max(0, retirement_age - current_age)

    series = _simulate_series(
        current_age, retirement_age, current_savings, monthly_contribution, annual_return_pct
    )

    late_start_age = max(current_age + 1, 30)
    late_start_series = _simulate_series(
        late_start_age, retirement_age, 0.0, monthly_contribution, annual_return_pct
    )

    projected_balance = series[-1]["balance"] if series else float(current_savings)
    late_start_balance = late_start_series[-1]["balance"] if late_start_series else 0.0

    total_contributed = float(current_savings) + monthly_contribution * 12 * years
    total_growth = projected_balance - total_contributed
    growth_multiple = (projected_balance / total_contributed) if total_contributed > 0 else None

    return {
        "series": series,
        "late_start_series": late_start_series,
        "late_start_age": late_start_age,
        "years": years,
        "projected_balance": projected_balance,
        "late_start_balance": late_start_balance,
        "total_contributed": total_contributed,
        "total_growth": total_growth,
        "growth_multiple": growth_multiple,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — TAX BRACKETS, WITHDRAWAL/DECUMULATION, SEQUENCE-OF-RETURNS MONTE CARLO
#
# Deliberately simplified and disclaimed: 2024 federal single-filer brackets
# only (no state tax, no other filing statuses), brackets held constant in
# nominal terms across the simulation horizon. Educational illustration only —
# not tax advice.
# ══════════════════════════════════════════════════════════════════════════════

FEDERAL_TAX_BRACKETS_2024_SINGLE: List[Tuple[float, float]] = [
    (0, 0.10),
    (11600, 0.12),
    (47150, 0.22),
    (100525, 0.24),
    (191950, 0.32),
    (243725, 0.35),
    (609350, 0.37),
]
STANDARD_DEDUCTION_2024_SINGLE = 14600.0


def compute_federal_tax(gross_taxable_income: float) -> float:
    """Progressive federal tax on ordinary income after the standard deduction."""
    taxable = max(0.0, float(gross_taxable_income or 0.0) - STANDARD_DEDUCTION_2024_SINGLE)
    if taxable <= 0:
        return 0.0
    tax = 0.0
    for i, (lower, rate) in enumerate(FEDERAL_TAX_BRACKETS_2024_SINGLE):
        upper = FEDERAL_TAX_BRACKETS_2024_SINGLE[i + 1][0] if i + 1 < len(FEDERAL_TAX_BRACKETS_2024_SINGLE) else float("inf")
        if taxable <= lower:
            break
        tax += (min(taxable, upper) - lower) * rate
    return round(tax, 2)


# Widely-cited, non-fabricated long-run US large-cap statistics — used ONLY as
# a fallback if the live historical data fetch below fails.
_FALLBACK_MEAN_RETURN = 0.10
_FALLBACK_STD_RETURN = 0.16

_spy_returns_cache: Dict[str, Any] = {"returns": None, "fetched_at": None}
_CACHE_TTL_SECONDS = 24 * 3600


def _fallback_annual_returns(n: int = 200) -> List[float]:
    rng = np.random.default_rng()
    return [float(x) for x in rng.normal(_FALLBACK_MEAN_RETURN, _FALLBACK_STD_RETURN, size=n)]


def _fetch_spy_annual_returns() -> Tuple[List[float], str]:
    """Live SPY (dividend-adjusted) annual returns since 1993, cached in-process
    for _CACHE_TTL_SECONDS to avoid refetching on every request. Falls back to a
    parametric normal distribution (see above) if the live fetch fails, matching
    this app's established graceful-degradation convention (see /api/financial/analyze).
    Never silently swaps in the fallback without disclosing it via the returned label."""
    now = time.time()
    cached = _spy_returns_cache.get("returns")
    fetched_at = _spy_returns_cache.get("fetched_at")
    if cached and fetched_at and (now - fetched_at) < _CACHE_TTL_SECONDS:
        return cached, "SPY (cached, dividend-adjusted, since 1993)"

    try:
        import yfinance as yf
        data = yf.download("SPY", start="1993-01-01", auto_adjust=True, progress=False)
        closes = data["Close"]
        if hasattr(closes, "columns"):  # yfinance may return a MultiIndex column frame
            closes = closes.iloc[:, 0]
        annual = closes.resample("YE").last().dropna()
        pct_returns = annual.pct_change().dropna()
        values = [float(r) for r in pct_returns.tolist() if math.isfinite(r)]
        if len(values) < 5:
            raise ValueError("Not enough historical data points returned")
        _spy_returns_cache["returns"] = values
        _spy_returns_cache["fetched_at"] = now
        return values, "SPY (live, dividend-adjusted, since 1993)"
    except Exception:
        return (
            _fallback_annual_returns(),
            "parametric fallback (~10% mean / ~16% stdev) — live market data unavailable",
        )


def _bootstrap_return_sequence(historical_returns: List[float], horizon_years: int, block_size: int, rng) -> List[float]:
    """Circular block bootstrap: builds a horizon_years-long return sequence by
    concatenating randomly-chosen overlapping blocks (with wraparound), sampled
    with replacement from the real historical series — preserves some real
    sequential correlation rather than pure IID annual resampling."""
    n = len(historical_returns)
    seq: List[float] = []
    while len(seq) < horizon_years:
        start = int(rng.integers(0, n))
        for i in range(block_size):
            seq.append(historical_returns[(start + i) % n])
            if len(seq) >= horizon_years:
                break
    return seq[:horizon_years]


def _simulate_decumulation_path(
    starting_balance: float,
    roth_pct: float,
    withdrawal_rate_pct: float,
    inflation_rate_pct: float,
    horizon_years: int,
    return_sequence: List[float],
) -> Dict[str, Any]:
    """One simulated path: grows Traditional/Roth balances by the sampled
    sequence, withdraws the inflation-adjusted 4%-rule amount proportionally
    from both pots each year, taxes only the Traditional portion, tracks
    whether/when the portfolio depletes."""
    roth_balance = starting_balance * (roth_pct / 100.0)
    traditional_balance = starting_balance - roth_balance

    withdrawal_this_year = starting_balance * (withdrawal_rate_pct / 100.0)
    depleted_year: Optional[int] = None
    yearly: List[Dict[str, Any]] = []

    for year in range(1, horizon_years + 1):
        r = return_sequence[year - 1] if year - 1 < len(return_sequence) else 0.0
        traditional_balance *= (1 + r)
        roth_balance *= (1 + r)
        total_balance = traditional_balance + roth_balance

        if total_balance <= 0:
            if depleted_year is None:
                depleted_year = year
            traditional_balance = 0.0
            roth_balance = 0.0
            yearly.append({"year": year, "balance": 0.0, "withdrawal": 0.0, "tax_paid": 0.0})
            withdrawal_this_year *= (1 + inflation_rate_pct / 100.0)
            continue

        trad_ratio = traditional_balance / total_balance
        withdrawal = min(withdrawal_this_year, total_balance)
        trad_withdrawal = withdrawal * trad_ratio
        roth_withdrawal = withdrawal - trad_withdrawal
        tax = compute_federal_tax(trad_withdrawal)

        traditional_balance = max(0.0, traditional_balance - trad_withdrawal)
        roth_balance = max(0.0, roth_balance - roth_withdrawal)
        total_balance = traditional_balance + roth_balance

        yearly.append({
            "year": year, "balance": round(total_balance, 2),
            "withdrawal": round(withdrawal, 2), "tax_paid": round(tax, 2),
        })

        if total_balance <= 0 and depleted_year is None:
            depleted_year = year

        withdrawal_this_year *= (1 + inflation_rate_pct / 100.0)

    ending_balance = yearly[-1]["balance"] if yearly else starting_balance
    return {"yearly": yearly, "depleted_year": depleted_year, "ending_balance": ending_balance}


_N_SIMULATIONS_DEFAULT = 500
_BLOCK_SIZE_YEARS = 5


def run_retirement_risk_analysis(
    starting_balance: float,
    roth_pct: float,
    withdrawal_rate_pct: float,
    inflation_rate_pct: float,
    horizon_years: int,
    n_simulations: int = _N_SIMULATIONS_DEFAULT,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs the block-bootstrap Monte Carlo across n_simulations paths and
    aggregates success rate, percentile balance timeline, and lifetime taxes."""
    historical_returns, data_source = _fetch_spy_annual_returns()
    rng = np.random.default_rng(seed)
    horizon_years = max(1, int(horizon_years))

    paths = []
    for _ in range(max(1, int(n_simulations))):
        seq = _bootstrap_return_sequence(historical_returns, horizon_years, _BLOCK_SIZE_YEARS, rng)
        paths.append(_simulate_decumulation_path(starting_balance, roth_pct, withdrawal_rate_pct, inflation_rate_pct, horizon_years, seq))

    success_count = sum(1 for p in paths if p["depleted_year"] is None)
    success_rate_pct = round(100.0 * success_count / len(paths), 1) if paths else 0.0

    percentile_timeline = []
    for y in range(horizon_years):
        balances = np.array([p["yearly"][y]["balance"] for p in paths if y < len(p["yearly"])])
        if balances.size == 0:
            continue
        percentile_timeline.append({
            "year": y + 1,
            "p10": round(float(np.percentile(balances, 10)), 2),
            "median": round(float(np.percentile(balances, 50)), 2),
            "p90": round(float(np.percentile(balances, 90)), 2),
        })

    ending_balances = np.array([p["ending_balance"] for p in paths]) if paths else np.array([])
    lifetime_taxes = np.array([sum(y["tax_paid"] for y in p["yearly"]) for p in paths]) if paths else np.array([])

    initial_withdrawal = starting_balance * (withdrawal_rate_pct / 100.0)
    initial_trad_withdrawal = initial_withdrawal * (1 - roth_pct / 100.0)
    initial_roth_withdrawal = initial_withdrawal - initial_trad_withdrawal

    return {
        "success_rate_pct": success_rate_pct,
        "median_ending_balance": round(float(np.median(ending_balances)), 2) if ending_balances.size else 0.0,
        "worst_case_ending_balance": round(float(np.percentile(ending_balances, 5)), 2) if ending_balances.size else 0.0,
        "best_case_ending_balance": round(float(np.percentile(ending_balances, 95)), 2) if ending_balances.size else 0.0,
        "median_lifetime_taxes_paid": round(float(np.median(lifetime_taxes)), 2) if lifetime_taxes.size else 0.0,
        "percentile_timeline": percentile_timeline,
        "n_simulations": len(paths),
        "data_source": data_source,
        "initial_annual_withdrawal": round(initial_withdrawal, 2),
        "initial_taxable_withdrawal": round(initial_trad_withdrawal, 2),
        "initial_tax_free_withdrawal": round(initial_roth_withdrawal, 2),
        "initial_year_tax": compute_federal_tax(initial_trad_withdrawal),
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

def get_or_create_plan(user_id: str) -> Dict[str, Any]:
    rows = _select("user_retirement_plans", user_id)
    row = rows[0] if rows else {"user_id": user_id, **_DEFAULT_PLAN}
    # Backfill any fields missing from an older row (e.g. saved before the
    # Phase 4 migration added roth_pct/life_expectancy/withdrawal_rate_pct).
    merged = {**_DEFAULT_PLAN, **row}

    growth = compute_growth(
        current_age=int(merged["current_age"]),
        retirement_age=int(merged["retirement_age"]),
        current_savings=float(merged["current_savings"]),
        monthly_contribution=float(merged["monthly_contribution"]),
        annual_return_pct=float(merged["expected_return_rate"]),
    )
    return {**merged, **growth}


def save_plan(
    user_id: str,
    plan_name: str,
    current_age: int,
    retirement_age: int,
    current_savings: float,
    monthly_contribution: float,
    expected_return_rate: float,
    inflation_rate: float = 0.0,
    retirement_goal: str = "",
    roth_pct: float = 0.0,
    life_expectancy: int = 90,
    withdrawal_rate_pct: float = 4.0,
) -> Dict[str, Any]:
    row = {
        "user_id": user_id,
        "plan_name": plan_name,
        "current_age": current_age,
        "retirement_age": retirement_age,
        "current_savings": current_savings,
        "monthly_contribution": monthly_contribution,
        "expected_return_rate": expected_return_rate,
        "inflation_rate": inflation_rate,
        "retirement_goal": retirement_goal,
        "roth_pct": roth_pct,
        "life_expectancy": life_expectancy,
        "withdrawal_rate_pct": withdrawal_rate_pct,
    }
    saved = _upsert("user_retirement_plans", row)
    growth = compute_growth(
        current_age=current_age,
        retirement_age=retirement_age,
        current_savings=current_savings,
        monthly_contribution=monthly_contribution,
        annual_return_pct=expected_return_rate,
    )
    return {**{**_DEFAULT_PLAN, **saved}, **growth}
