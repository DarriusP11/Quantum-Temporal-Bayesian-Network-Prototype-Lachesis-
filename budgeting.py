"""
budgeting.py — Classical monthly budget planner (50/30/20 rule)

Mirrors the frontend's BudgetingDashboard.tsx logic exactly so the API and the
UI never disagree on the numbers:
  needs_pct   = (housing + food + transportation + healthcare) / income
  savings_pct = savings / income
  wants_pct   = total_spend_pct - needs_pct   (floored at 0)

Purely classical — no quantum dependency, never gated by QUANTUM_FEATURES_ENABLED.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CATEGORIES: List[Dict[str, Any]] = [
    {"id": "housing",        "emoji": "🏠", "label": "Housing",        "items": []},
    {"id": "food",           "emoji": "🍕", "label": "Food",           "items": []},
    {"id": "transportation", "emoji": "🚗", "label": "Transportation", "items": []},
    {"id": "entertainment",  "emoji": "🏀", "label": "Entertainment",  "items": []},
    {"id": "education",      "emoji": "📚", "label": "Education",      "items": []},
    {"id": "healthcare",     "emoji": "💊", "label": "Healthcare",     "items": []},
    {"id": "clothing",       "emoji": "👕", "label": "Clothing",       "items": []},
    {"id": "subscriptions",  "emoji": "📱", "label": "Subscriptions",  "items": []},
    {"id": "savings",        "emoji": "🐷", "label": "Savings",        "items": []},
]

# 50/30/20 rule targets (percent of income)
_TARGETS = {"needs": 50, "wants": 30, "savings": 20}

# Category ids that count toward "needs" in the 50/30/20 breakdown
_NEEDS_CATEGORY_IDS = {"housing", "food", "transportation", "healthcare"}
_SAVINGS_CATEGORY_ID = "savings"


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _category_total(category: Dict[str, Any]) -> float:
    return sum(float(item.get("amount") or 0.0) for item in category.get("items") or [])


def _verdict(actual: float, target: float) -> str:
    if actual > target:
        return "over"
    if actual < target:
        return "under"
    return "on_track"


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def compute_budget_summary(income: float, categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure function — no I/O. Same 50/30/20 formula as BudgetingDashboard.tsx."""
    income = float(income or 0.0)
    categories = categories or []

    category_totals: Dict[str, float] = {c.get("id", ""): _category_total(c) for c in categories}
    total_spend = sum(category_totals.values())
    surplus = income - total_spend

    if income > 0:
        needs_total = sum(v for cid, v in category_totals.items() if cid in _NEEDS_CATEGORY_IDS)
        savings_total = category_totals.get(_SAVINGS_CATEGORY_ID, 0.0)
        needs_pct = round((needs_total / income) * 100)
        savings_pct = round((savings_total / income) * 100)
        total_spend_pct = round((total_spend / income) * 100)
        wants_pct = max(0, total_spend_pct - needs_pct)
    else:
        needs_pct = 0
        savings_pct = 0
        wants_pct = 0

    verdicts = {
        "needs": _verdict(needs_pct, _TARGETS["needs"]),
        "wants": _verdict(wants_pct, _TARGETS["wants"]),
        "savings": _verdict(savings_pct, _TARGETS["savings"]),
    }

    return {
        "category_totals": category_totals,
        "total_spend": total_spend,
        "surplus": surplus,
        "needs_pct": needs_pct,
        "wants_pct": wants_pct,
        "savings_pct": savings_pct,
        "targets": _TARGETS,
        "verdicts": verdicts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE PERSISTENCE (self-contained — mirrors api_server.py's generic REST
# helpers; duplicated locally so this module has no import-time dependency on
# api_server.py, consistent with every other helper module in this codebase)
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

def get_or_create_budget(user_id: str) -> Dict[str, Any]:
    rows = _select("user_budgets", user_id)
    if rows:
        row = rows[0]
        income = row.get("income", 0.0)
        categories = row.get("categories") or DEFAULT_CATEGORIES
    else:
        row = {"user_id": user_id, "income": 0.0, "categories": DEFAULT_CATEGORIES}
        income = 0.0
        categories = DEFAULT_CATEGORIES

    return {**row, **compute_budget_summary(income, categories)}


def save_budget(user_id: str, income: float, categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    row = {"user_id": user_id, "income": income, "categories": categories}
    saved = _upsert("user_budgets", row)
    return {**saved, **compute_budget_summary(income, categories)}
