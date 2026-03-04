from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Lachesis UI Prototype",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="collapsed",
)


TAB_NAMES = [
    "Statevector",
    "Reduced States",
    "Measurement",
    "Fidelity & Export",
    "Presets",
    "Present Scenarios",
    "Foresight",
    "Financial Analysis",
    "Insider Trading",
    "Lachesis Guide",
    "Advanced Quantum",
    "Toy QAOA",
    "Sentiment Analysis",
    "Prompt Studio",
    "VQE",
]


def inject_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg-0: #060915;
  --bg-1: #0b1022;
  --bg-2: #12182f;
  --card: rgba(17, 23, 45, 0.78);
  --card-strong: rgba(20, 27, 53, 0.9);
  --edge: rgba(133, 90, 255, 0.35);
  --edge-soft: rgba(133, 90, 255, 0.18);
  --txt: #d9e6ff;
  --txt-dim: #9eb1da;
  --brand: #7b5cff;
  --brand-2: #5d7dff;
  --brand-3: #64e3ff;
  --ok: #58e3a8;
  --warn: #ffcf6a;
}

html, body, [class*="css"] {
  font-family: "Space Grotesk", system-ui, sans-serif;
  color: var(--txt);
}

.stApp {
  background:
    radial-gradient(1200px 480px at 14% -6%, rgba(112, 78, 255, 0.22), transparent 62%),
    radial-gradient(1000px 540px at 94% 4%, rgba(90, 142, 255, 0.18), transparent 60%),
    linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 56%, var(--bg-2) 100%);
}

[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 1480px;
  padding-top: 1.05rem;
  padding-bottom: 2rem;
}

h1, h2, h3, h4, h5 {
  font-family: "Sora", "Space Grotesk", sans-serif;
  letter-spacing: 0.2px;
}

.proto-shell {
  border: 1px solid var(--edge-soft);
  background: linear-gradient(
      180deg,
      rgba(28, 36, 70, 0.82) 0%,
      rgba(13, 18, 37, 0.76) 100%
    );
  border-radius: 18px;
  padding: 1rem 1.1rem 0.9rem;
  box-shadow: 0 18px 40px rgba(1, 3, 10, 0.45);
  backdrop-filter: blur(10px);
}

.proto-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}

.brand-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.brand-mark {
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--brand), var(--brand-2));
  color: white;
  font-weight: 700;
  font-size: 0.95rem;
}

.brand-title {
  font-family: "Sora", sans-serif;
  font-size: 1.55rem;
  color: #7d97ff;
  font-weight: 700;
  line-height: 1.1;
}

.brand-sub {
  color: var(--txt-dim);
  font-size: 0.88rem;
  margin-top: 0.15rem;
}

.top-right {
  display: flex;
  gap: 0.45rem;
  flex-wrap: wrap;
  justify-content: flex-end;
  margin-top: 0.15rem;
}

.pill {
  border-radius: 999px;
  border: 1px solid rgba(120, 160, 255, 0.35);
  background: rgba(17, 23, 45, 0.75);
  color: #c8d8ff;
  padding: 0.24rem 0.62rem;
  font-size: 0.72rem;
  font-weight: 600;
}

.pill.user {
  border-color: rgba(255, 207, 106, 0.35);
  color: #f3dd9b;
}

.proto-note {
  margin-top: 0.7rem;
  color: #bdd0ff;
  background: rgba(86, 116, 255, 0.11);
  border: 1px solid rgba(86, 116, 255, 0.28);
  border-radius: 10px;
  padding: 0.5rem 0.65rem;
  font-size: 0.8rem;
}

.section-title {
  margin-bottom: 0.2rem;
}

.section-title h3 {
  margin: 0;
  font-size: 1.06rem;
}

.section-title p {
  margin: 0.08rem 0 0;
  color: var(--txt-dim);
  font-size: 0.85rem;
}

.chat-card {
  border: 1px solid var(--edge-soft);
  background: linear-gradient(
      160deg,
      rgba(22, 29, 53, 0.86) 0%,
      rgba(12, 18, 35, 0.86) 100%
    );
  border-radius: 14px;
  padding: 0.8rem 0.95rem;
}

.chat-head {
  display: flex;
  align-items: center;
  gap: 0.7rem;
}

.chat-avatar {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  background: linear-gradient(135deg, #7b5cff, #5d7dff 55%, #64e3ff);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  color: white;
}

.chat-bubble {
  margin-top: 0.65rem;
  border-left: 3px solid #7b5cff;
  border-radius: 9px;
  background: rgba(47, 60, 118, 0.18);
  padding: 0.55rem 0.65rem;
  color: #cfddff;
  font-size: 0.9rem;
}

[data-baseweb="tab-list"] {
  gap: 0.25rem;
  background: rgba(13, 20, 42, 0.7);
  border: 1px solid var(--edge-soft);
  border-radius: 14px;
  padding: 0.3rem;
}

[data-baseweb="tab"] {
  border-radius: 10px;
  padding: 0.42rem 0.7rem;
  color: #9eb1da;
  text-transform: none;
  font-size: 0.77rem;
  font-weight: 600;
}

[aria-selected="true"] {
  background: linear-gradient(
      135deg,
      rgba(123, 92, 255, 0.3),
      rgba(93, 125, 255, 0.26)
    ) !important;
  border: 1px solid rgba(123, 92, 255, 0.45) !important;
  color: #dbe6ff !important;
}

[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--edge-soft);
  border-radius: 12px;
  padding: 0.45rem 0.65rem;
}

.stButton > button {
  border-radius: 10px;
  border: 1px solid rgba(100, 120, 255, 0.4);
  background: linear-gradient(135deg, #5239ff, #5f76ff);
  color: #f8faff;
  font-weight: 600;
}

.stButton > button:hover {
  border-color: rgba(100, 170, 255, 0.7);
  background: linear-gradient(135deg, #5e47ff, #6d86ff);
}

div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
  border: 1px solid var(--edge-soft);
  border-radius: 10px;
  padding: 0.2rem;
  background: rgba(10, 15, 30, 0.45);
}

@media (max-width: 980px) {
  .proto-top {
    flex-direction: column;
    align-items: flex-start;
  }
  .top-right {
    justify-content: flex-start;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def app_header() -> None:
    st.markdown(
        """
<div class="proto-shell">
  <div class="proto-top">
    <div>
      <div class="brand-row">
        <span class="brand-mark">L</span>
        <span class="brand-title">Lachesis</span>
      </div>
      <div class="brand-sub">Quantum-Enhanced Financial Analytics &amp; Foresight Platform</div>
    </div>
    <div class="top-right">
      <span class="pill">AI-Powered</span>
      <span class="pill">Quantum-Ready</span>
      <span class="pill user">Dimp11</span>
      <span class="pill user">Sign Out</span>
    </div>
  </div>
  <div class="proto-note">
    Prototype only: this UI mirrors the structure and flows of <code>qtbn_simulator_clean.py</code>,
    but is intentionally disconnected from backend analysis logic.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
<div class="section-title">
  <h3>{title}</h3>
  {sub}
</div>
        """,
        unsafe_allow_html=True,
    )


def mock_distribution(num_qubits: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + (17 * num_qubits))
    states = [format(i, f"0{num_qubits}b") for i in range(2 ** num_qubits)]
    p = rng.random(len(states))
    p = p / p.sum()
    return pd.DataFrame({"State": states, "Probability": p})


def mock_price_frame(days: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.015, size=days)
    px = 100 * np.exp(np.cumsum(rets))
    dt_idx = pd.date_range(end=dt.date.today(), periods=days, freq="B")
    return pd.DataFrame({"Price": px}, index=dt_idx)


def render_quantum_left_rail(prefix: str) -> dict:
    with st.container(border=True):
        section_title("Quantum Configuration")
        nq = st.selectbox("Number of qubits", [1, 2, 3, 4], index=0, key=f"{prefix}_nq")
        shots = st.number_input("Shots", min_value=128, max_value=8192, value=2048, step=128, key=f"{prefix}_shots")
        seed = st.number_input("Random seed", min_value=1, max_value=99_999, value=17, step=1, key=f"{prefix}_seed")
        st.caption("Backend simulator support here is mocked for UI prototyping.")

    with st.container(border=True):
        section_title("Noise Models")
        c1, c2 = st.columns(2)
        with c1:
            dep = st.checkbox("Depolarizing", key=f"{prefix}_dep")
            phs = st.checkbox("Phase Damping", key=f"{prefix}_phs")
        with c2:
            amp = st.checkbox("Amplitude Damping", key=f"{prefix}_amp")
            cnot = st.checkbox("CNOT Noise", key=f"{prefix}_cnot")

    with st.container(border=True):
        section_title("Gate Configuration - Step 0")
        st.selectbox("Qubit 0 gate", ["H", "X", "Y", "Z", "RX", "RY", "RZ"], key=f"{prefix}_gate_q0")
        if nq > 1:
            st.selectbox("Qubit 1 gate", ["I", "H", "X", "RX", "RY"], key=f"{prefix}_gate_q1")

    return {
        "num_qubits": int(nq),
        "shots": int(shots),
        "seed": int(seed),
        "dep": dep,
        "amp": amp,
        "phs": phs,
        "cnot": cnot,
    }


def render_statevector_tab() -> None:
    left, right = st.columns([0.33, 0.67], gap="large")
    with left:
        cfg = render_quantum_left_rail("sv")
    with right:
        with st.container(border=True):
            section_title("Statevector", "Pre-measurement amplitudes and basis weights (mocked output).")
            probs = mock_distribution(cfg["num_qubits"], cfg["seed"])
            st.bar_chart(probs.set_index("State"))
            st.code(
                "psi = [0.7071+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.7071+0.0000j]",
                language="text",
            )


def render_reduced_tab() -> None:
    left, right = st.columns([0.33, 0.67], gap="large")
    with left:
        cfg = render_quantum_left_rail("red")
    with right:
        with st.container(border=True):
            section_title("Reduced States", "Subsystem density summaries and entropy proxies.")
            probs = mock_distribution(max(1, cfg["num_qubits"]), cfg["seed"] + 3)
            top = probs.sort_values("Probability", ascending=False).head(6).reset_index(drop=True)
            st.dataframe(top, use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Purity", "0.87")
            c2.metric("Entropy", "0.41")
            c3.metric("Mutual Info", "0.18")


def render_measurement_tab() -> None:
    left, right = st.columns([0.33, 0.67], gap="large")
    with left:
        cfg = render_quantum_left_rail("meas")
    with right:
        with st.container(border=True):
            section_title("Measurement", "Counts histogram and quick diagnostics.")
            probs = mock_distribution(cfg["num_qubits"], cfg["seed"] + 11)
            counts = (probs["Probability"] * cfg["shots"]).round().astype(int)
            df = pd.DataFrame({"State": probs["State"], "Counts": counts})
            st.bar_chart(df.set_index("State"))
            st.dataframe(df.sort_values("Counts", ascending=False), use_container_width=True, hide_index=True)


def render_fidelity_tab() -> None:
    left, right = st.columns([0.33, 0.67], gap="large")
    with left:
        render_quantum_left_rail("fid")
    with right:
        with st.container(border=True):
            section_title("Fidelity & Export", "Global fidelity and artifact export controls.")
            st.metric("Global fidelity", "0.962", "ideal vs noisy")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.button("Compute Fidelity", use_container_width=True, key="proto_compute_fid")
            with c2:
                st.download_button(
                    "Download Snapshot (.json)",
                    data='{"prototype": true, "fidelity": 0.962}',
                    file_name="lachesis_fidelity_prototype.json",
                    mime="application/json",
                    use_container_width=True,
                )


def render_presets_tab() -> None:
    left, right = st.columns([0.33, 0.67], gap="large")
    with left:
        render_quantum_left_rail("presets")
    with right:
        with st.container(border=True):
            section_title("Presets", "Fast starter templates for common configurations.")
            c1, c2, c3 = st.columns(3)
            c1.button("Bell prep (H + CX)", key="preset_bell", use_container_width=True)
            c2.button("Dephasing stress", key="preset_dephase", use_container_width=True)
            c3.button("Amplitude relaxation", key="preset_amp", use_container_width=True)
            st.selectbox("Scenario -> Circuit", ["Bell", "GHZ-lite", "Noise-Robust"], key="preset_circuit")
            st.button("Apply Scenario", key="apply_scenario")


def render_present_scenarios_tab() -> None:
    left, right = st.columns([0.33, 0.67], gap="large")
    with left:
        render_quantum_left_rail("present")
    with right:
        with st.container(border=True):
            section_title("Present Scenarios", "Quick-check and compare noisy distributions with TV robustness.")
            st.button("Analyze current scenario", key="analyze_current_scenario")
            st.caption("Prototype output")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Scenario": "Base", "TV Distance": 0.031, "Status": "PASS"},
                        {"Scenario": "Noisy", "TV Distance": 0.084, "Status": "WARN"},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_foresight_tab() -> None:
    with st.container(border=True):
        section_title("Q-TBN Forecast Engine (Backend)", "Lachesis QTBN forecast and toy temporal regime projection.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.selectbox("Prior regime", ["Calm", "Stressed", "Crisis"], key="fx_prior_regime")
        with c2:
            st.slider("Risk-on prior (0-1)", 0.0, 1.0, 0.60, 0.01, key="fx_risk_on")
        with c3:
            st.number_input("Drift Mu", value=0.05, step=0.01, key="fx_drift_mu")
        with c4:
            st.number_input("Horizon (days)", min_value=5, max_value=365, value=30, key="fx_horizon")
        c_run, c_reset = st.columns([0.9, 0.1])
        c_run.button("Execute Q-TBN Analysis", key="fx_execute", use_container_width=True)
        c_reset.button("Reset", key="fx_reset", use_container_width=True)


def render_financial_tab() -> None:
    with st.container(border=True):
        section_title("Financial Analytics Configuration (Backend)")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Tickers (comma-separated)", "SPY,QQQ,AAPL", key="fin_tickers")
        with c2:
            st.number_input("Lookback days", min_value=30, max_value=1500, value=365, key="fin_lookback")
        c3, c4, c5 = st.columns([1, 1, 1.3])
        with c3:
            st.number_input("Horizon days", min_value=1, max_value=60, value=10, key="fin_horizon")
        with c4:
            st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="fin_alpha")
        with c5:
            st.number_input("Monte Carlo simulations", min_value=1000, max_value=300000, value=50000, step=1000, key="fin_sims")
        st.checkbox("Demo mode (force synthetic market data)", key="fin_demo")
        st.button("Fetch & Analyze Market Data", key="fin_fetch", use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Regime", "High Volatility")
    m2.metric("MC VaR (10d)", "-6.31%")
    m3.metric("MC CVaR (10d)", "-7.95%")
    m4.metric("Dollar CVaR", "USD 79,532")

    with st.container(border=True):
        px = mock_price_frame()
        section_title("Market Preview", "Synthetic line preview for UI prototyping.")
        st.line_chart(px, use_container_width=True)

    with st.container(border=True):
        section_title("Lachesis - Interactive Risk Explainer")
        st.text_area(
            "Question for Lachesis",
            "Explain the difference between VaR and CVaR in plain language.",
            key="fin_lachesis_q",
        )
        st.button("Ask Lachesis", key="fin_lachesis_btn")
        st.caption("This prototype response area is intentionally not wired to backend calls.")


def render_insider_tab() -> None:
    with st.container(border=True):
        section_title("Insider Trading", "SEC filing ingestion and plain-English explanation workflow.")
        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        c1.text_input("Ticker", "AAPL", key="ins_ticker")
        c2.number_input("Max filings", min_value=5, max_value=200, value=30, key="ins_max")
        c3.button("Load filings", key="ins_load", use_container_width=True)

    filings = pd.DataFrame(
        [
            {"Form": "4", "Filing Date": "2026-02-10", "Accession": "0001-01", "Primary Doc": "xslF345.xml"},
            {"Form": "4", "Filing Date": "2026-02-08", "Accession": "0001-00", "Primary Doc": "xslF345.xml"},
            {"Form": "13D", "Filing Date": "2026-02-03", "Accession": "0000-91", "Primary Doc": "d13d.htm"},
        ]
    )
    st.dataframe(filings, use_container_width=True, hide_index=True)
    st.button("Summarize with Lachesis", key="ins_summarize")


def render_lachesis_guide_tab() -> None:
    st.text_input(
        "Enter your OpenAI API key...",
        value="",
        key="guide_openai_key",
        placeholder="Enter your OpenAI API key...",
    )
    st.text_input(
        "Enter your ElevenLabs API key for voice...",
        value="",
        key="guide_voice_key",
        placeholder="Enter your ElevenLabs API key for voice...",
    )

    st.markdown(
        """
<div class="chat-card">
  <div class="chat-head">
    <div class="chat-avatar">L</div>
    <div>
      <div style="font-weight:700; font-size:1.05rem; line-height:1.1;">Lachesis</div>
      <div style="font-size:0.82rem; color:#9eb1da;">Your AI Integrated Quantum Assistant</div>
    </div>
  </div>
  <div class="chat-bubble">
    Hello! I'm Lachesis, your AI Integrated Quantum Assistant. I specialize in quantum computing,
    financial analytics, and risk assessment. How can I help you navigate the quantum-enhanced financial
    landscape today?
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.text_input(
        "Ask Lachesis about quantum computing, financial analytics, or risk assessment...",
        key="guide_prompt",
    )
    q1, q2, q3, q4 = st.columns(4)
    q1.button("Quantum Circuits", key="guide_quick_1", use_container_width=True)
    q2.button("Risk Analysis", key="guide_quick_2", use_container_width=True)
    q3.button("Sentiment Integration", key="guide_quick_3", use_container_width=True)
    q4.button("QTBN Explained", key="guide_quick_4", use_container_width=True)


def render_advanced_quantum_tab() -> None:
    with st.container(border=True):
        section_title("Advanced Quantum", "Tomography, benchmarking, calibration snapshots.")
        c1, c2, c3 = st.columns(3)
        c1.metric("RB Decay", "0.991")
        c2.metric("Process Proxy", "0.947")
        c3.metric("Volume Proxy", "16")
        st.button("Run tomography sweep", key="adv_tomo")
        st.button("Capture calibration snapshot", key="adv_snap")


def render_qaoa_tab() -> None:
    with st.container(border=True):
        section_title("Toy QAOA", "Persona-driven allocation and crash index visualization.")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Persona", ["Conservative", "Balanced", "Aggressive"], key="qaoa_persona")
        c2.slider("Lambda (risk preference)", 0.0, 1.0, 0.55, 0.01, key="qaoa_lambda")
        c3.number_input("Budget", min_value=10_000, max_value=10_000_000, value=250_000, step=10_000, key="qaoa_budget")
        st.button("Run Toy QAOA", key="qaoa_run", use_container_width=True)

    df = pd.DataFrame(
        {
            "Lambda": np.linspace(0.05, 0.95, 10),
            "Objective": [0.38, 0.41, 0.46, 0.48, 0.52, 0.57, 0.59, 0.61, 0.62, 0.63],
        }
    )
    st.line_chart(df.set_index("Lambda"), use_container_width=True)


def render_sentiment_tab() -> None:
    with st.container(border=True):
        section_title("Sentiment Analysis", "News sentiment extraction and VaR multiplier workflow.")
        c1, c2 = st.columns([1, 1])
        c1.selectbox("Source", ["Google News RSS + VADER", "Perplexity API"], key="sent_source")
        c2.text_input("Tickers", "AAPL,MSFT,NVDA", key="sent_tickers")
        st.text_area("Paste links or headlines", height=110, key="sent_links")
        st.button("Analyze sentiment", key="sent_analyze")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg score", "0.18")
    c2.metric("VaR multiplier", "1.07")
    c3.metric("Headlines scanned", "24")
    st.button("Explain with Lachesis", key="sent_explain")


def render_prompt_studio_tab() -> None:
    c1, c2 = st.columns([0.52, 0.48], gap="large")
    with c1:
        with st.container(border=True):
            section_title("Prompt Templates")
            st.text_input("Template name", "Tail-risk explainer", key="studio_name")
            st.text_area(
                "System prompt",
                "You are Lachesis, a precise, neutral explainer for quantum-financial workflows.",
                height=100,
                key="studio_sys",
            )
            st.text_area(
                "User template",
                "Explain how current settings influence VaR/CVaR over {h} days at alpha={alpha}.",
                height=100,
                key="studio_user",
            )
            st.button("Save template", key="studio_save")
    with c2:
        with st.container(border=True):
            section_title("Run Panel")
            st.selectbox("Template", ["Tail-risk explainer", "QAOA digest", "Sentiment recap"], key="studio_pick")
            st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "o4-mini"], key="studio_model")
            st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, key="studio_temp")
            st.button("Run with Lachesis", key="studio_run", use_container_width=True)
            st.text_area("Output", "Prototype output area (no backend connection).", height=180, key="studio_output")


def render_vqe_tab() -> None:
    with st.container(border=True):
        section_title("VQE", "Variational optimization, smoke tests, and artifact snapshots.")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Ansatz depth", min_value=1, max_value=12, value=3, key="vqe_depth")
        c2.number_input("Iterations", min_value=10, max_value=500, value=120, key="vqe_iters")
        c3.selectbox("Optimizer", ["COBYLA", "SPSA", "L-BFGS-B"], key="vqe_opt")
        st.button("Run VQE smoke", key="vqe_smoke", use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Energy", "-1.0312")
    c2.metric("Convergence", "Stable")
    c3.metric("Artifact", "vqe_snapshot.json")
    st.dataframe(
        pd.DataFrame(
            [
                {"Artifact": "vqe_snapshot_20260213_1800.json", "Type": "snapshot"},
                {"Artifact": "vqe_smoke_20260213_1800.json", "Type": "smoke"},
                {"Artifact": "vqe_sweep_20260213_1800.csv", "Type": "sweep"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


inject_theme()
app_header()
st.markdown("")

(
    tab_sv,
    tab_red,
    tab_meas,
    tab_fid,
    tab_presets,
    tab_present,
    tab_fx,
    tab_fin,
    tab_insider,
    tab_guide,
    tab_advanced_q,
    tab_qaoa,
    tab_sentiment,
    tab_prompt,
    tab_vqe,
) = st.tabs(TAB_NAMES)

with tab_sv:
    render_statevector_tab()

with tab_red:
    render_reduced_tab()

with tab_meas:
    render_measurement_tab()

with tab_fid:
    render_fidelity_tab()

with tab_presets:
    render_presets_tab()

with tab_present:
    render_present_scenarios_tab()

with tab_fx:
    render_foresight_tab()

with tab_fin:
    render_financial_tab()

with tab_insider:
    render_insider_tab()

with tab_guide:
    render_lachesis_guide_tab()

with tab_advanced_q:
    render_advanced_quantum_tab()

with tab_qaoa:
    render_qaoa_tab()

with tab_sentiment:
    render_sentiment_tab()

with tab_prompt:
    render_prompt_studio_tab()

with tab_vqe:
    render_vqe_tab()

