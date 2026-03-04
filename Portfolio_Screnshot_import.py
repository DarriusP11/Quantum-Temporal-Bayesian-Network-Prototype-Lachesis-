# portfolio_screenshot_import.py
from __future__ import annotations

import base64, json, os, re
from typing import Any, Dict, Optional

PORTFOLIO_SCREENSHOT_IMPORT_ENABLED = True  # on/off switch

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        # remove $ and commas
        s = s.replace("$", "").replace(",", "")
        return float(s)
    except Exception:
        return None

def _infer_total_portfolio_value(market_value: Optional[float], diversity_pct: Optional[float]) -> Optional[float]:
    if market_value is None or diversity_pct is None:
        return None
    if diversity_pct <= 0 or diversity_pct > 100:
        return None
    return market_value / (diversity_pct / 100.0)

def extract_portfolio_from_screenshot_via_openai(
    image_bytes: bytes,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Uses a vision-capable model to extract structured position data from a brokerage screenshot.
    Requires OPENAI_API_KEY set in env.
    """
    api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return {"ok": False, "error": "Missing OPENAI_API_KEY environment variable."}

    # Lazy import so your app still runs without the package
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        return {"ok": False, "error": f"openai package not available: {e}"}

    client = OpenAI(api_key=api_key)

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    system = (
        "You extract brokerage/portfolio screenshot data into strict JSON. "
        "Return ONLY valid JSON. No prose."
    )

    schema = {
        "broker": "Robinhood|Other|Unknown",
        "currency": "USD|Other|Unknown",
        "positions": [
            {
                "ticker": "string|nullable",
                "shares": "number|nullable",
                "market_value": "number|nullable",
                "average_cost": "number|nullable",
                "today_return": "number|nullable",
                "total_return": "number|nullable",
                "portfolio_diversity_pct": "number|nullable",
                "price": "number|nullable"
            }
        ]
    }

    user = (
        "Extract any visible position fields. If a field isn't present, set it to null.\n"
        "JSON schema example:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Important:\n"
        "- ticker should be uppercase like NVDA\n"
        "- portfolio_diversity_pct should be a number like 37.67 (not 0.3767)\n"
        "- Return ONLY JSON."
    )

    # NOTE: model name must support image input in your OpenAI account.
    # If you already have a Lachesis/OpenAI integration, reuse your existing model config.
    try:
        resp = client.responses.create(
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_base64": b64},
                ]},
            ],
        )
        text = resp.output_text
        data = json.loads(text)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def normalize_extracted_position(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and add inferred_total_portfolio_value if possible.
    """
    positions = data.get("positions", []) if isinstance(data, dict) else []
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
        "raw": data,
    }
