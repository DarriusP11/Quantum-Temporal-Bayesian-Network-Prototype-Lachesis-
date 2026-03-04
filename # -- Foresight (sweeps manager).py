# =========================
# FORESIGHT TAB (merged + fixed)
# =========================


import io
import json
import math
import csv
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


def _csv_download(df: pd.DataFrame, filename: str, label: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="text/csv")


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
    """
    _init_foresight_state()

    keys = ["0", "1"] if int(num_qubits) == 1 else ["00", "01", "10", "11"]

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

    st.checkbox("Enforce gate: disable sweeps when Gate Status = FAIL", key="fx_enforce_gate")

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
                pr = float(meta["p_risk_on"])
                scenario_p = {"0": 1.0 - pr, "1": pr}
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

    # Enforce gate
    sweeps_disabled = False
    if st.session_state.fx_enforce_gate and (gate["status"] == "FAIL" or (not blessed_ok)):
        sweeps_disabled = True
        st.warning("Gate Status = FAIL (or Blessing FAIL). Sweeps are disabled by enforcement setting.")

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
            "Consider tightening exposure if gate trends toward WARN/FAIL.",
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
        st.json({"baseline": baseline, "gate": gate, "blessing": blessed_detail})


# =========================
# END FORESIGHT TAB
# =========================

with tab_fx:
    try:
        render_foresight_tab(SAMPLE_SCENARIOS, int(st.session_state.get("num_qubits", 1) or 1))
    except Exception as e:
        st.error(f"Foresight tab failed: {e}")
        st.code(traceback.format_exc())
