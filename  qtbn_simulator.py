# qtbn_simulator.py
# Qiskit 2.x • Aer-safe • Streamlit UI
# Repo-backed scenarios.json • Custom scenarios • Reload/Save
# Statevector • Reduced states • Counts • Fidelity • Presets
# Present analysis • Foresight: sweep + suggestion + Top-3
# Save/Load sweeps • A/B compare • Export compare CSV • Clone endpoint→scenario
# NEW: Analytical Foresight (Phase 1)
#  - Scenario scoring: alignment×impact×trend×(1–uncertainty)
#  - Trend models + "now" slider
#  - Uncertainty via multi-seed variability
#  - Scoreboard table + bar chart

import io, os, json, math, hashlib, time, csv
import datetime as dt
from pathlib import Path
from typing import Optional  # <-- 3.9-safe

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- Qiskit 2.x
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, Pauli, state_fidelity

st.set_page_config(page_title="Q‑TBN Simulator (Qiskit 2.x)", layout="wide")
st.caption("Repo‑backed scenarios, statevector, noisy counts, fidelity, present analysis, foresight (sweep + suggestions + scoring), Save/Load sweeps, A/B compare, export, and cloning endpoints to scenarios.")

# ========== Built-in SAMPLE scenario seeds (always available) ==========
SAMPLE_SCENARIOS = {
    "Balanced-1q":        {"keys": ["0","1"],             "p": {"0":0.50,"1":0.50},                                 "note":"Fair superposition", "impact": 1.0},
    "Ground-skew-1q":     {"keys": ["0","1"],             "p": {"0":0.85,"1":0.15},                                 "note":"Relaxation-like",    "impact": 1.1},
    "Bell-like-2q":       {"keys": ["00","01","10","11"], "p": {"00":0.48,"01":0.02,"10":0.02,"11":0.48},           "note":"Near-Bell",          "impact": 1.2},
    "Bitflip-2q":         {"keys": ["00","01","10","11"], "p": {"00":0.05,"01":0.45,"10":0.45,"11":0.05},           "note":"Bitflip dominated",  "impact": 0.9},
    "Uniform-2q":         {"keys": ["00","01","10","11"], "p": {"00":0.25,"01":0.25,"10":0.25,"11":0.25},           "note":"Max entropy",        "impact": 0.8},
}

# ========== Disk scenarios (repo-backed) ==========
SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios.json"

def load_disk_scenarios():
    """Return dict[name] = {keys, p, note, impact?} from scenarios.json if present."""
    out = {}
    try:
        if SCENARIOS_PATH.exists():
            blob = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
            for item in blob.get("scenarios", []):
                name = item.get("name")
                keys = item.get("keys")
                p    = item.get("p")
                note = item.get("note", "")
                impact = float(item.get("impact", 1.0))
                if name and isinstance(keys, list) and isinstance(p, dict):
                    out[name] = {"keys": keys, "p": p, "note": note, "impact": impact}
    except Exception as e:
        st.sidebar.warning(f"Failed to read scenarios.json: {e}")
    return out

def save_disk_scenarios(all_entries: dict):
    """Persist dict[name] -> meta into scenarios.json."""
    try:
        payload = {"scenarios": []}
        for name, meta in all_entries.items():
            payload["scenarios"].append({
                "name": name,
                "keys": meta["keys"],
                "p":    meta["p"],
                "note": meta.get("note",""),
                "impact": float(meta.get("impact", 1.0))
            })
        SCENARIOS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True, f"Saved {len(payload['scenarios'])} scenario(s) to {SCENARIOS_PATH.name}"
    except Exception as e:
        return False, f"Save failed: {e}"

# ========== Caching ==========
@st.cache_resource
def get_simulator(method: Optional[str] = None, seed: Optional[int] = None):
    kw = {}
    if method: kw["method"] = method
    if seed is not None: kw["seed_simulator"] = int(seed)
    return AerSimulator(**kw)

def _qc_key(qc: QuantumCircuit)->str:
    s = str(qc.draw(output="text", fold=-1))
    return hashlib.sha1(s.encode()).hexdigest()

def _to_qpy(qc: QuantumCircuit)->bytes:
    from qiskit import qpy as qpy_io
    buf = io.BytesIO(); qpy_io.dump([qc], buf); buf.seek(0)
    return buf.read()

@st.cache_data(show_spinner=False)
def cached_transpile(sim_desc: str, circ_key: str, qpy_bytes: bytes, optimization_level: int=1)->bytes:
    from qiskit import qpy as qpy_io
    qc_loaded = list(qpy_io.load(io.BytesIO(qpy_bytes)))[0]
    sim = get_simulator(method=None)
    tqc = transpile(qc_loaded, sim, optimization_level=optimization_level)
    out = io.BytesIO(); qpy_io.dump([tqc], out); out.seek(0)
    return out.read()

@st.cache_data(show_spinner=False)
def run_counts_cached(sim_method: str, tqc_qpy: bytes, shots: int, seed: Optional[int]):
    from qiskit import qpy as qpy_io
    tqc = list(qpy_io.load(io.BytesIO(tqc_qpy)))[0]
    sim = get_simulator(method=None if sim_method=="default" else sim_method, seed=seed)
    return sim.run(tqc, shots=shots).result().get_counts()

# ========== Distance helpers ==========
def kldiv(p, q, eps=1e-12):
    ks = set(p) | set(q)
    s = 0.0
    for k in ks:
        a = max(p.get(k, 0.0), eps)
        b = max(q.get(k, 0.0), eps)
        s += a * math.log(a / b)
    return float(s)

def tvdist(p, q):
    ks = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in ks)

# ========== Session defaults ==========
def ss_get(k, default):
    if k not in st.session_state: st.session_state[k]=default
    return st.session_state[k]
def ss_set(k,v): st.session_state[k]=v

def ensure_defaults():
    ss_get("num_qubits",1)
    ss_get("shots",1024)
    ss_get("use_seed",True)
    ss_get("seed_val",42)
    ss_get("enable_dep",True)
    ss_get("enable_amp",True)
    ss_get("enable_phs",True)
    ss_get("enable_cnot_noise",False)
    ss_get("snap_labels",True)
    ss_get("enable_export",True)
    base = {
        "g0_q0":"H","a0_q0":0.5, "g0_q1":"None","a0_q1":0.0, "cnot0":False,
        "g1_q0":"None","a1_q0":0.0, "g1_q1":"None","a1_q1":0.0, "cnot1":False,
        "g2_q0":"RX","a2_q0":2.0, "g2_q1":"None","a2_q1":0.0, "cnot2":False,
    }
    for k,v in base.items(): ss_get(k,v)
    for s in (0,1,2):
        ss_get(f"pdep{s}",0.0); ss_get(f"pamp{s}",0.0); ss_get(f"pph{s}",0.0)
        ss_get(f"pcnot{s}",0.0)
    ss_get("foresight_sweeps", {})   # saved sweeps (name -> payload)
    ss_get("custom_scenarios", {})   # session-defined scenarios
    ss_get("disk_scenarios", load_disk_scenarios())  # repo-backed scenarios
ensure_defaults()

# ========== Sidebar ==========
st.sidebar.subheader("Qubit configuration")
num_qubits = st.sidebar.radio("Number of qubits",[1,2],
                              index=0 if ss_get("num_qubits",1)==1 else 1, key="num_qubits")

st.sidebar.markdown("---")
shots = st.sidebar.number_input("Shots per counts run",256,16384,ss_get("shots",1024),256,key="shots")
use_seed = st.sidebar.checkbox("Use reproducible seed",value=ss_get("use_seed",True),key="use_seed")
seed_val = st.sidebar.number_input("Seed (int)",0,10_000_000,ss_get("seed_val",42),1,key="seed_val")

st.sidebar.markdown("---")
speed_mode = st.sidebar.radio("Speed mode",["Fast","Accurate"],index=0, key="speed_mode")
if speed_mode=="Fast":
    if st.session_state.shots>2048: st.session_state.shots=512
else:
    if st.session_state.shots<1024: st.session_state.shots=2048

with st.sidebar.expander("ℹ️ Noise legend"):
    st.write("- **Depolarizing (p)** randomizes; **Amplitude (γ)** relaxes |1>→|0>; **Phase (λ)** kills phase; **CNOT depol (p2)** after CX.")

GATES=["None","H","X","Z","RX","RY"]
def gate_ui(step,qname,gkey,akey,gdef="None",adef=0.0):
    st.sidebar.text(f"{step} — {qname}")
    st.sidebar.selectbox(f"{step} {qname} Gate",GATES,index=GATES.index(ss_get(gkey,gdef)),key=gkey)
    st.sidebar.slider(f"{step} {qname} Angle (RX/RY)",0.0,math.pi,ss_get(akey,adef),0.01,key=akey)

st.sidebar.markdown("---"); st.sidebar.text("Temporal Gates (T0 → T1 → T2)")
gate_ui("T0","q0","g0_q0","a0_q0","H",0.5)
if num_qubits==2:
    gate_ui("T0","q1","g0_q1","a0_q1"); st.sidebar.checkbox("T0: CNOT q0→q1",value=ss_get("cnot0",False),key="cnot0")
else:
    ss_set("g0_q1","None"); ss_set("a0_q1",0.0); ss_set("cnot0",False)
gate_ui("T1","q0","g1_q0","a1_q0")
if num_qubits==2:
    gate_ui("T1","q1","g1_q1","a1_q1"); st.sidebar.checkbox("T1: CNOT q0→q1",value=ss_get("cnot1",False),key="cnot1")
else:
    ss_set("g1_q1","None"); ss_set("a1_q1",0.0); ss_set("cnot1",False)
gate_ui("T2","q0","g2_q0","a2_q0","RX",2.0)
if num_qubits==2:
    gate_ui("T2","q1","g2_q1","a2_q1"); st.sidebar.checkbox("T2: CNOT q0→q1",value=ss_get("cnot2",False),key="cnot2")
else:
    ss_set("g2_q1","None"); ss_set("a2_q1",0.0); ss_set("cnot2",False)

st.sidebar.markdown("---"); st.sidebar.text("Enable / disable channels")
enable_dep = st.sidebar.checkbox("Enable depolarizing", value=ss_get("enable_dep",True), key="enable_dep")
enable_amp = st.sidebar.checkbox("Enable amplitude damping", value=ss_get("enable_amp",True), key="enable_amp")
enable_phs = st.sidebar.checkbox("Enable phase damping", value=ss_get("enable_phs",True), key="enable_phs")
enable_cnot_noise = st.sidebar.checkbox("Enable CNOT depolarizing (2‑qubit only)", value=ss_get("enable_cnot_noise", num_qubits==2), key="enable_cnot_noise")

st.sidebar.markdown("---"); st.sidebar.text("Per‑step single‑qubit noise")
for s,lbl in enumerate(["T0","T1","T2"]):
    c = st.sidebar.container()
    c.slider(f"{lbl} depolarizing p",0.0,0.3,ss_get(f"pdep{s}",0.0),0.01,key=f"pdep{s}")
    c.slider(f"{lbl} amplitude γ",  0.0,0.3,ss_get(f"pamp{s}",0.0),0.01,key=f"pamp{s}")
    c.slider(f"{lbl} phase λ",      0.0,0.3,ss_get(f"pph{s}",0.0), 0.01,key=f"pph{s}")

if num_qubits==2:
    st.sidebar.markdown("---"); st.sidebar.text("Per‑step CNOT noise")
    st.sidebar.slider("T0 CNOT depol p2",0.0,0.3,ss_get("pcnot0",0.0),0.01,key="pcnot0")
    st.sidebar.slider("T1 CNOT depol p2",0.0,0.3,ss_get("pcnot1",0.0),0.01,key="pcnot1")
    st.sidebar.slider("T2 CNOT depol p2",0.0,0.3,ss_get("pcnot2",0.0),0.01,key="pcnot2")
else:
    ss_set("pcnot0",0.0); ss_set("pcnot1",0.0); ss_set("pcnot2",0.0)

st.sidebar.markdown("---")
st.sidebar.checkbox("Snap labels (1‑qubit phase wheel)",value=ss_get("snap_labels",True),key="snap_labels")
st.sidebar.checkbox("Enable JSON export (Fidelity tab)",value=ss_get("enable_export",True),key="enable_export")

# ---- Save/Load scenario (export current UI state)
st.sidebar.markdown("---"); st.sidebar.subheader("Scenario I/O (current UI state)")
def _snapshot_state()->dict:
    keys=["num_qubits","shots","use_seed","seed_val","enable_dep","enable_amp","enable_phs","enable_cnot_noise","snap_labels","enable_export",
          "g0_q0","a0_q0","g0_q1","a0_q1","cnot0",
          "g1_q0","a1_q0","g1_q1","a1_q1","cnot1",
          "g2_q0","a2_q0","g2_q1","a2_q1","cnot2",
          "pdep0","pdep1","pdep2","pamp0","pamp1","pamp2","pph0","pph1","pph2","pcnot0","pcnot1","pcnot2"]
    return {k: st.session_state.get(k) for k in keys}
def _restore_state(d:dict):
    for k,v in d.items():
        if k in st.session_state: st.session_state[k]=v
c1,c2 = st.sidebar.columns(2)
with c1:
    if st.sidebar.button("💾 Save"):
        buf=io.BytesIO(); buf.write(json.dumps(_snapshot_state(),indent=2).encode()); buf.seek(0)
        st.sidebar.download_button("Download JSON",buf,file_name="qtbn_scenario.json",mime="application/json")
with c2:
    up = st.sidebar.file_uploader("Load",type=["json"],label_visibility="collapsed")
    if up is not None:
        try:
            _restore_state(json.load(up)); st.sidebar.success("Scenario loaded."); st.rerun()
        except Exception as e:
            st.sidebar.error(f"Load failed: {e}")

# ========== Builders ==========
def apply_gate(qc,q,g,ang):
    if g=="H": qc.h(q)
    elif g=="X": qc.x(q)
    elif g=="Z": qc.z(q)
    elif g=="RX": qc.rx(ang,q)
    elif g=="RY": qc.ry(ang,q)

def one_qubit_noise_instrs(p_dep,p_amp,p_ph):
    out=[]
    if st.session_state.enable_dep and p_dep>0: out.append(depolarizing_error(p_dep,1).to_instruction())
    if st.session_state.enable_amp and p_amp>0: out.append(amplitude_damping_error(p_amp).to_instruction())
    if st.session_state.enable_phs and p_ph>0: out.append(phase_damping_error(p_ph).to_instruction())
    return out

def two_qubit_depol_instr(p2):
    if not st.session_state.enable_cnot_noise or p2<=0: return None
    return depolarizing_error(p2,2).to_instruction()

def build_unitary_circuit():
    nq=st.session_state.num_qubits; qc=QuantumCircuit(nq)
    # T0
    apply_gate(qc,0,st.session_state.g0_q0,st.session_state.a0_q0)
    if nq==2:
        apply_gate(qc,1,st.session_state.g0_q1,st.session_state.a0_q1)
        if st.session_state.cnot0: qc.cx(0,1)
    qc.barrier()
    # T1
    apply_gate(qc,0,st.session_state.g1_q0,st.session_state.a1_q0)
    if nq==2:
        apply_gate(qc,1,st.session_state.g1_q1,st.session_state.a1_q1)
        if st.session_state.cnot1: qc.cx(0,1)
    qc.barrier()
    # T2
    apply_gate(qc,0,st.session_state.g2_q0,st.session_state.a2_q0)
    if nq==2:
        apply_gate(qc,1,st.session_state.g2_q1,st.session_state.a2_q1)
        if st.session_state.cnot2: qc.cx(0,1)
    qc.barrier(); return qc

def build_measure_circuit_with_noise():
    nq=st.session_state.num_qubits; qc=QuantumCircuit(nq,nq)
    # T0
    apply_gate(qc,0,st.session_state.g0_q0,st.session_state.a0_q0)
    if nq==2:
        apply_gate(qc,1,st.session_state.g0_q1,st.session_state.a0_q1)
        if st.session_state.cnot0:
            qc.cx(0,1); inst2=two_qubit_depol_instr(st.session_state.pcnot0)
            if inst2: qc.append(inst2,[0,1])
    for q in range(nq):
        for inst in one_qubit_noise_instrs(st.session_state.pdep0,st.session_state.pamp0,st.session_state.pph0): qc.append(inst,[q])
    qc.barrier()
    # T1
    apply_gate(qc,0,st.session_state.g1_q0,st.session_state.a1_q0)
    if nq==2:
        apply_gate(qc,1,st.session_state.g1_q1,st.session_state.a1_q1)
        if st.session_state.cnot1:
            qc.cx(0,1); inst2=two_qubit_depol_instr(st.session_state.pcnot1)
            if inst2: qc.append(inst2,[0,1])
    for q in range(nq):
        for inst in one_qubit_noise_instrs(st.session_state.pdep1,st.session_state.pamp1,st.session_state.pph1): qc.append(inst,[q])
    qc.barrier()
    # T2
    apply_gate(qc,0,st.session_state.g2_q0,st.session_state.a2_q0)
    if nq==2:
        apply_gate(qc,1,st.session_state.g2_q1,st.session_state.a2_q1)
        if st.session_state.cnot2:
            qc.cx(0,1); inst2=two_qubit_depol_instr(st.session_state.pcnot2)
            if inst2: qc.append(inst2,[0,1])
    for q in range(nq):
        for inst in one_qubit_noise_instrs(st.session_state.pdep2,st.session_state.pamp2,st.session_state.pph2): qc.append(inst,[q])
    qc.barrier(); qc.measure(range(nq),range(nq)); return qc

def build_noisy_unitary_for_density():
    nq=st.session_state.num_qubits; qc=QuantumCircuit(nq)
    # T0
    apply_gate(qc,0,st.session_state.g0_q0,st.session_state.a0_q0)
    if nq==2:
        apply_gate(qc,1,st.session_state.g0_q1,st.session_state.a0_q1)
        if st.session_state.cnot0:
            qc.cx(0,1); inst2=two_qubit_depol_instr(st.session_state.pcnot0)
            if inst2: qc.append(inst2,[0,1])
    for q in range(nq):
        for inst in one_qubit_noise_instrs(st.session_state.pdep0,st.session_state.pamp0,st.session_state.pph0): qc.append(inst,[q])
    qc.barrier()
    # T1
    apply_gate(qc,0,st.session_state.g1_q0,st.session_state.a1_q0)
    if nq==2:
        apply_gate(qc,1,st.session_state.g1_q1,st.session_state.a1_q1)
        if st.session_state.cnot1:
            qc.cx(0,1); inst2=two_qubit_depol_instr(st.session_state.pcnot1)
            if inst2: qc.append(inst2,[0,1])
    for q in range(nq):
        for inst in one_qubit_noise_instrs(st.session_state.pdep1,st.session_state.pamp1,st.session_state.pph1): qc.append(inst,[q])
    qc.barrier()
    # T2
    apply_gate(qc,0,st.session_state.g2_q0,st.session_state.a2_q0)
    if nq==2:
        apply_gate(qc,1,st.session_state.g2_q1,st.session_state.a2_q1)
        if st.session_state.cnot2:
            qc.cx(0,1); inst2=two_qubit_depol_instr(st.session_state.pcnot2)
            if inst2: qc.append(inst2,[0,1])
    for q in range(nq):
        for inst in one_qubit_noise_instrs(st.session_state.pdep2,st.session_state.pamp2,st.session_state.pph2): qc.append(inst,[q])
    return qc

# ========== Tiny viz helpers ==========
def show_amp_line(label, amp: complex):
    prob = amp.real**2 + amp.imag**2
    phase = math.atan2(amp.imag, amp.real)
    st.text(f"{label:<4} amp={amp.real:+.6f} {amp.imag:+.6f}j   |amp|^2={prob:.4f}   phase={phase:.3f} rad")

def quadrant_anchor(angle_rad):
    a=(angle_rad+math.pi)%(2*math.pi)-math.pi
    centers=[math.pi/4,3*math.pi/4,-3*math.pi/4,-math.pi/4]
    return min(centers,key=lambda c:abs(a-c))

def phase_wheel_1q(alpha: complex, beta: complex, snap: bool):
    a_prob=alpha.real**2+alpha.imag**2; b_prob=beta.real**2+beta.imag**2
    a_phase=math.atan2(alpha.imag,alpha.real); b_phase=math.atan2(beta.imag,beta.real)
    fig,ax=plt.subplots(); th=np.linspace(0,2*math.pi,201); R=1.02
    ax.plot(R*np.cos(th),R*np.sin(th),linewidth=0.8,alpha=0.5,zorder=1)
    ax.axhline(0,linewidth=0.5,alpha=0.6); ax.axvline(0,linewidth=0.5,alpha=0.6)
    def draw_vec(cplx,label,color):
        r=(cplx.real**2+cplx.imag**2)**0.5; ang=math.atan2(cplx.imag,cplx.real)
        tip=min(r,1.0)*0.96
        ax.arrow(0,0,tip*math.cos(ang),tip*math.sin(ang),head_width=0.03,length_includes_head=True,zorder=3,color=color)
        base=tip+0.08
        if snap: t=quadrant_anchor(ang); ax.text(base*math.cos(t),base*math.sin(t),label,ha="center",va="center",color=color)
        else: ax.text(base*math.cos(ang+0.06),base*math.sin(ang+0.06),label,ha="center",va="center",color=color)
    ax.legend(handles=[
        Line2D([0],[0],color="C0",lw=3,label=f"α: |α|²={a_prob:.3f}, ϕ={a_phase:.3f}"),
        Line2D([0],[0],color="C1",lw=3,label=f"β: |β|²={b_prob:.3f}, ϕ={b_phase:.3f}")
    ],loc="upper right",framealpha=0.85)
    draw_vec(alpha,"α (|0>)","C0"); draw_vec(beta,"β (|1>)","C1")
    ax.set_aspect("equal"); ax.set_xlim(-1.15,1.15); ax.set_ylim(-1.15,1.15); ax.set_title("Phase wheel (radius≈|amp|, angle=phase)")
    st.pyplot(fig)

def bloch_expectations(rho_1q: DensityMatrix):
    X,Y,Z = Pauli("X"),Pauli("Y"),Pauli("Z")
    ex=float(np.real(np.trace(rho_1q.data @ X.to_matrix())))
    ey=float(np.real(np.trace(rho_1q.data @ Y.to_matrix())))
    ez=float(np.real(np.trace(rho_1q.data @ Z.to_matrix())))
    purity=float(np.real(np.trace(rho_1q.data @ rho_1q.data)))
    return ex,ey,ez,purity

def tiny_bloch_plots(ex,ey,ez,title):
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(); th=np.linspace(0,2*math.pi,181)
        ax.plot(np.cos(th),np.sin(th),alpha=0.5,linewidth=0.8); ax.scatter([ex],[ey])
        ax.set_aspect("equal"); ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_title(f"{title}: XY"); st.pyplot(fig)
    with c2:
        fig2,ax2=plt.subplots(); ax2.bar(["Z"],[ez]); ax2.set_ylim(-1.05,1.05); ax2.set_title(f"{title}: Z"); st.pyplot(fig2)

# ========== ASCII circuit ==========
unitary_qc = build_unitary_circuit()
st.text("Circuit (ASCII, no measurement)")
st.code(str(unitary_qc.draw(output="text")), language="text")
st.caption("Legend: q[i] quantum wires; c[i] classical; barriers enforce step order; measures map q→c.")

# ========== Tabs ==========
tab_sv, tab_red, tab_meas, tab_fid, tab_presets, tab_present, tab_fx = st.tabs(
    ["Statevector","Reduced States (2q)","Measurement","Fidelity & Export","Presets","Present Scenarios","Foresight (mock)"]
)

# --- Statevector
with tab_sv:
    st.text("Statevector (before measurement)")
    if st.button("Run Statevector"):
        try:
            sv = Statevector.from_instruction(unitary_qc).data
            if st.session_state.num_qubits==1:
                alpha, beta = complex(sv[0]), complex(sv[1])
                p0 = alpha.real**2+alpha.imag**2; p1 = beta.real**2+beta.imag**2
                c1,c2=st.columns(2)
                with c1:
                    show_amp_line("|0>",alpha); show_amp_line("|1>",beta)
                    phase_wheel_1q(alpha,beta,st.session_state.snap_labels)
                with c2:
                    st.text("Ideal probabilities"); st.bar_chart({"P":[p0,p1]})
            else:
                labels=["|00>","|01>","|10>","|11>"]  # <-- fixed "|10|"
                probs=[]
                st.text("Amplitudes & ideal probabilities (2‑qubit):")
                for i,lab in enumerate(labels):
                    amp=complex(sv[i]); show_amp_line(lab,amp); probs.append(amp.real**2+amp.imag**2)
                st.bar_chart({"P":probs})
        except Exception as e:
            st.error(f"Statevector failed: {e}")

# --- Reduced States (2q)
with tab_red:
    st.text("Reduced density matrices (2‑qubit only)")
    if st.session_state.num_qubits==1:
        st.info("Switch to 2 qubits to view reduced states.")
    else:
        if st.button("Compute Reduced States"):
            try:
                noisy_qc=build_noisy_unitary_for_density()
                sim_dm=get_simulator(method="density_matrix",
                                     seed=st.session_state.seed_val if st.session_state.use_seed else None)
                noisy_qc.save_density_matrix()
                tqc=transpile(noisy_qc,sim_dm)
                dm=DensityMatrix(sim_dm.run(tqc).result().get_density_matrix(0))
                rho_q0=partial_trace(dm,[1]); rho_q1=partial_trace(dm,[0])
                ex0,ey0,ez0,pur0=bloch_expectations(DensityMatrix(rho_q0))
                ex1,ey1,ez1,pur1=bloch_expectations(DensityMatrix(rho_q1))
                st.text(f"q0 purity={pur0:.4f}, Bloch=(X={ex0:.3f}, Y={ey0:.3f}, Z={ez0:.3f})"); tiny_bloch_plots(ex0,ey0,ez0,"q0")
                st.text(f"q1 purity={pur1:.4f}, Bloch=(X={ex1:.3f}, Y={ey1:.3f}, Z={ez1:.3f})"); tiny_bloch_plots(ex1,ey1,ez1,"q1")
            except Exception as e:
                st.error(f"Reduced states failed: {e}")

# --- Measurement (cached)
last_counts_ideal, last_counts_noisy = {}, {}
with tab_meas:
    st.text("Measurement (counts) — ideal vs noisy")
    if st.button("Run Counts"):
        try:
            nq=st.session_state.num_qubits
            qc_i=QuantumCircuit(nq,nq); qc_i.compose(build_unitary_circuit(),inplace=True); qc_i.measure(range(nq),range(nq))
            qc_n=build_measure_circuit_with_noise()

            key_i=_qc_key(qc_i); key_n=_qc_key(qc_n)
            tqc_qpy_i=cached_transpile("default",key_i,_to_qpy(qc_i),1)
            tqc_qpy_n=cached_transpile("default",key_n,_to_qpy(qc_n),1)
            seed=st.session_state.seed_val if st.session_state.use_seed else None
            counts_i=run_counts_cached("default",tqc_qpy_i,st.session_state.shots,seed)
            counts_n=run_counts_cached("default",tqc_qpy_n,st.session_state.shots,seed)

            last_counts_ideal.update(counts_i); last_counts_noisy.update(counts_n)
            st.text("Counts (ideal)"); st.write(counts_i)
            st.text("Counts (noisy)"); st.write(counts_n)

            keys=["0","1"] if nq==1 else ["00","01","10","11"]
            c1,c2=st.columns(2)
            with c1: st.text("Ideal"); st.bar_chart({"counts":[counts_i.get(k,0) for k in keys]})
            with c2: st.text("Noisy"); st.bar_chart({"counts":[counts_n.get(k,0) for k in keys]})
        except Exception as e:
            st.error(f"Counts run failed: {e}")

# --- Fidelity & Export
with tab_fid:
    st.text("Global fidelity (ideal vs noisy) + per‑qubit (2q)")
    if st.button("Compute Fidelity"):
        try:
            nq=st.session_state.num_qubits
            ideal_sv=Statevector.from_instruction(build_unitary_circuit()); ideal_dm=DensityMatrix(ideal_sv)
            noisy_qc=build_noisy_unitary_for_density()
            sim_dm=get_simulator(method="density_matrix",
                                 seed=st.session_state.seed_val if st.session_state.use_seed else None)
            noisy_qc.save_density_matrix(); tqc=transpile(noisy_qc,sim_dm)
            noisy_dm=DensityMatrix(sim_dm.run(tqc).result().get_density_matrix(0))
            F_global=float(state_fidelity(ideal_dm,noisy_dm)); st.text(f"Global fidelity: {F_global:.6f}")

            perq={}
            if nq==2:
                ideal_q0=partial_trace(ideal_dm,[1]); ideal_q1=partial_trace(ideal_dm,[0])
                noisy_q0=partial_trace(noisy_dm,[1]); noisy_q1=partial_trace(noisy_dm,[0])
                F_q0=float(state_fidelity(DensityMatrix(ideal_q0),DensityMatrix(noisy_q0)))
                F_q1=float(state_fidelity(DensityMatrix(ideal_q1),DensityMatrix(noisy_q1)))
                st.text(f"q0 fidelity: {F_q0:.6f}"); st.text(f"q1 fidelity: {F_q1:.6f}"); perq={"q0":F_q0,"q1":F_q1}

            if st.session_state.enable_export:
                data={"num_qubits":nq,"shots":st.session_state.shots,
                      "seed_used":int(st.session_state.seed_val) if st.session_state.use_seed else None,
                      "gates":{"T0":{"q0":[st.session_state.g0_q0,st.session_state.a0_q0],"q1":[st.session_state.g0_q1,st.session_state.a0_q1],"cnot":bool(st.session_state.cnot0)},
                               "T1":{"q0":[st.session_state.g1_q0,st.session_state.a1_q0],"q1":[st.session_state.g1_q1,st.session_state.a1_q1],"cnot":bool(st.session_state.cnot1)},
                               "T2":{"q0":[st.session_state.g2_q0,st.session_state.a2_q0],"q1":[st.session_state.g2_q1,st.session_state.a2_q1],"cnot":bool(st.session_state.cnot2)}},
                      "noise_enabled":{"depolarizing":st.session_state.enable_dep,"amplitude":st.session_state.enable_amp,"phase":st.session_state.enable_phs,"cnot_depol":st.session_state.enable_cnot_noise},
                      "noise_single":{"T0":{"p_dep":st.session_state.pdep0,"p_amp":st.session_state.pamp0,"p_ph":st.session_state.pph0},
                                      "T1":{"p_dep":st.session_state.pdep1,"p_amp":st.session_state.pamp1,"p_ph":st.session_state.pph1},
                                      "T2":{"p_dep":st.session_state.pdep2,"p_amp":st.session_state.pamp2,"p_ph":st.session_state.pph2}},
                      "noise_cnot":{"T0":st.session_state.pcnot0,"T1":st.session_state.pcnot1,"T2":st.session_state.pcnot2},
                      "fidelity":{"global":F_global,**perq}}
                buf=io.BytesIO(); buf.write(json.dumps(data,indent=2).encode()); buf.seek(0)
                st.download_button("⬇️ Download run JSON",buf,file_name="qtbn_run.json",mime="application/json")
        except Exception as e:
            st.error(f"Fidelity failed: {e}")

# --- Presets
def apply_preset_bell():
    ss_set("num_qubits",2)
    ss_set("g0_q0","H"); ss_set("a0_q0",0.5); ss_set("g0_q1","None"); ss_set("a0_q1",0.0); ss_set("cnot0",True)
    ss_set("g1_q0","None"); ss_set("a1_q0",0.0); ss_set("g1_q1","None"); ss_set("a1_q1",0.0); ss_set("cnot1",False)
    ss_set("g2_q0","None"); ss_set("a2_q0",0.0); ss_set("g2_q1","None"); ss_set("a2_q1",0.0); ss_set("cnot2",False)
    ss_set("enable_dep",True); ss_set("enable_amp",False); ss_set("enable_phs",False); ss_set("enable_cnot_noise",True)
    ss_set("pdep0",0.01); ss_set("pdep1",0.02); ss_set("pdep2",0.02)
    ss_set("pamp0",0.00); ss_set("pamp1",0.00); ss_set("pamp2",0.00)
    ss_set("pph0",0.00); ss_set("pph1",0.00); ss_set("pph2",0.00)
    ss_set("pcnot0",0.02); ss_set("pcnot1",0.00); ss_set("pcnot2",0.00); st.rerun()

def apply_preset_dephasing():
    ss_set("num_qubits",1)
    ss_set("g0_q0","H"); ss_set("a0_q0",0.5); ss_set("g1_q0","None"); ss_set("a1_q0",0.0); ss_set("g2_q0","None"); ss_set("a2_q0",0.0)
    ss_set("cnot0",False); ss_set("cnot1",False); ss_set("cnot2",False)
    ss_set("enable_dep",False); ss_set("enable_amp",False); ss_set("enable_phs",True); ss_set("enable_cnot_noise",False)
    ss_set("pdep0",0.00); ss_set("pdep1",0.00); ss_set("pdep2",0.00)
    ss_set("pamp0",0.00); ss_set("pamp1",0.00); ss_set("pamp2",0.00)
    ss_set("pph0",0.00); ss_set("pph1",0.15); ss_set("pph2",0.15); st.rerun()

def apply_preset_amplitude():
    ss_set("num_qubits",1)
    ss_set("g0_q0","X"); ss_set("a0_q0",0.0); ss_set("g1_q0","None"); ss_set("a1_q0",0.0); ss_set("g2_q0","None"); ss_set("a2_q0",0.0)
    ss_set("cnot0",False); ss_set("cnot1",False); ss_set("cnot2",False)
    ss_set("enable_dep",False); ss_set("enable_amp",True); ss_set("enable_phs",False); ss_set("enable_cnot_noise",False)
    ss_set("pdep0",0.00); ss_set("pdep1",0.00); ss_set("pdep2",0.00)
    ss_set("pamp0",0.00); ss_set("pamp1",0.20); ss_set("pamp2",0.20)
    ss_set("pph0",0.00); ss_set("pph1",0.00); ss_set("pph2",0.00); st.rerun()

with tab_presets:
    st.subheader("Preset Scenarios")
    c1,c2,c3=st.columns(3)
    c1.button("Bell prep (H→CX, light depol)",on_click=apply_preset_bell)
    c2.button("Dephasing stress (H + λ)",on_click=apply_preset_dephasing)
    c3.button("Amplitude relaxation (X + γ)",on_click=apply_preset_amplitude)
    st.caption("Pick a preset, then run Statevector / Measurement / Fidelity with reproducible seed.")

# --- Present Scenarios
with tab_present:
    st.subheader("Present Scenarios — current settings snapshot")
    st.caption("Quick check: ideal vs noisy counts, Δ probabilities, robustness (1 − TV distance).")
    if st.button("Analyze current scenario"):
        try:
            nq=st.session_state.num_qubits
            qc_i=QuantumCircuit(nq,nq); qc_i.compose(build_unitary_circuit(),inplace=True); qc_i.measure(range(nq),range(nq))
            qc_n=build_measure_circuit_with_noise()

            key_i=_qc_key(qc_i); key_n=_qc_key(qc_n)
            tqc_qpy_i=cached_transpile("default",key_i,_to_qpy(qc_i),1)
            tqc_qpy_n=cached_transpile("default",key_n,_to_qpy(qc_n),1)
            seed=st.session_state.seed_val if st.session_state.use_seed else None
            ri=run_counts_cached("default",tqc_qpy_i,st.session_state.shots,seed)
            rn=run_counts_cached("default",tqc_qpy_n,st.session_state.shots,seed)

            keys=["0","1"] if nq==1 else ["00","01","10","11"]
            st.text("Counts (ideal)"); st.write(ri)
            st.text("Counts (noisy)"); st.write(rn)

            def normed(d):
                N=sum(d.values()) or 1
                return {k: d.get(k,0)/N for k in keys}
            pi,pn=normed(ri),normed(rn)
            delta={k: pn.get(k,0)-pi.get(k,0) for k in keys}
            st.text("Δ probability (noisy - ideal)"); st.write(delta)
            tv=0.5*sum(abs(pi[k]-pn[k]) for k in keys); robustness=max(0.0,1.0-tv)
            st.text(f"Robustness (1 − TV): {robustness:.4f}")

            c1,c2=st.columns(2)
            with c1: st.text("Ideal"); st.bar_chart({"counts":[ri.get(k,0) for k in keys]})
            with c2: st.text("Noisy"); st.bar_chart({"counts":[rn.get(k,0) for k in keys]})
        except Exception as e:
            st.error(f"Present scenarios failed: {e}")

# --- Foresight (mock) + Suggestions + Save/Load + A/B + Export compare CSV + Clone → scenario + Repo scenarios panel + Scoring
with tab_fx:
    st.subheader("Analytical Foresight (mock Lachesis)")
    st.caption("Sweep a noise knob; compare to scenario; scoring = alignment×impact×trend×(1–uncertainty); save/load sweeps, A/B compare, export, clone, and repo scenarios.")

    # Keys by qubit count
    keys = ["0","1"] if st.session_state.num_qubits == 1 else ["00","01","10","11"]

    # Combine sample + repo disk + custom
    def combined_scenarios_for_keys(keys_):
        combined = {k:{**v} for k,v in SAMPLE_SCENARIOS.items() if v["keys"] == keys_}
        for name, meta in st.session_state.disk_scenarios.items():
            if meta["keys"] == keys_: combined[name]={**meta}
        for name, meta in st.session_state.custom_scenarios.items():
            if meta["keys"] == keys_: combined[name]={**meta}
        # default impact if missing
        for name, meta in combined.items():
            if "impact" not in meta: meta["impact"]=1.0
        return combined

    ACTIVE_SCENARIOS = combined_scenarios_for_keys(keys)

    # -------- Trend model (mock) ----------
    st.markdown("### Trend model")
    tcol1, tcol2, tcol3 = st.columns([1,1,2])
    with tcol1:
        trend_model = st.selectbox("Trend", ["Flat","Upward","Downward","Cyclical"], index=0)
    with tcol2:
        trend_now = st.slider("Now (0→1)", 0.0, 1.0, 0.6, 0.01)
    with tcol3:
        global_impact_scale = st.slider("Global impact scale", 0.5, 1.5, 1.0, 0.01, help="Multiplies each scenario's impact")

    def trend_factor(model:str, t:float)->float:
        t = max(0.0, min(1.0, t))
        if model == "Flat":
            return 1.0
        if model == "Upward":
            return 0.8 + 0.4*t          # 0.8→1.2
        if model == "Downward":
            return 1.2 - 0.4*t          # 1.2→0.8
        if model == "Cyclical":
            return 1.0 + 0.2*math.sin(2*math.pi*t)  # 0.8→1.2 sinusoid
        return 1.0

    T_factor = trend_factor(trend_model, trend_now)

    # -------- Baseline current noisy distribution + uncertainty via seeds ----------
    st.markdown("### Current baseline (multi‑seed for uncertainty)")
    seeds_str_base = st.text_input("Seeds for baseline uncertainty (comma)", "11,17,29")
    base_seeds = [int(s.strip()) for s in seeds_str_base.split(",") if s.strip().isdigit()]
    shots_hint = 256 if st.session_state.get("speed_mode","Fast") == "Fast" else 1024

    def run_noisy_p(seed):
        qc_n = build_measure_circuit_with_noise()
        tqc_qpy = cached_transpile("default", _qc_key(qc_n), _to_qpy(qc_n), 1)
        counts = run_counts_cached("default", tqc_qpy, shots_hint, seed)
        N = sum(counts.values()) or 1
        return {k: counts.get(k, 0)/N for k in keys}

    try:
        samples = []
        if base_seeds:
            for sd in base_seeds: samples.append(run_noisy_p(sd))
        else:
            samples.append(run_noisy_p(None))
        # mean distribution
        current_p = {k: float(np.mean([s.get(k,0.0) for s in samples])) for k in keys}
        # std across seeds → uncertainty [0..1] (clipped & scaled)
        stds = {k: float(np.std([s.get(k,0.0) for s in samples])) for k in keys}
        avg_std = float(np.mean(list(stds.values()))) if stds else 0.0
        # heuristic: 0 std → 0 penalty, >=0.15 → ~max penalty
        uncertainty = max(0.0, min(1.0, avg_std/0.15))
        st.write({"mean_p": current_p, "avg_std": round(avg_std,6), "uncertainty_penalty": round(uncertainty,3)})
    except Exception as e:
        current_p = {k: 1.0/len(keys) for k in keys}
        uncertainty = 0.5
        st.info(f"Baseline failed; using uniform fallback. ({e})")

    # -------- Suggestions + Top‑3 ----------
    st.markdown("### Suggestions")
    ranked = []
    for name, meta in ACTIVE_SCENARIOS.items():
        tgt = meta["p"]
        ranked.append((name, kldiv(current_p, tgt), tvdist(current_p, tgt), meta.get("note","")))
    ranked.sort(key=lambda x: (x[1], x[2]))
    valid_names = [n for n,_,_,_ in ranked]
    if ranked:
        s_name, s_kl, s_tv, s_note = ranked[0]
        st.markdown(f"**Suggested scenario:** `{s_name}` — KL≈`{s_kl:.4f}`, TV≈`{s_tv:.4f}`  ({s_note})")
        cols = st.columns([1,1,2])
        with cols[0]:
            if st.button("Use suggested scenario"):
                st.session_state["sc_choice"] = s_name
                st.rerun()
        with cols[1]:
            if st.button("Refresh suggestion"):
                st.rerun()
        st.markdown("**Closest scenarios (Top‑3):**")
        topN = ranked[:3]
        st.dataframe(
            {"Scenario":[n for n,_,_,_ in topN],
             "KL":[round(kl,6) for _,kl,_,_ in topN],
             "TV":[round(tv,6) for _,_,tv,_ in topN],
             "Note":[note for _,_,_,note in topN]}, use_container_width=True
        )

    # -------- Scenario scoring (Phase 1) ----------
    st.markdown("### Scenario scoring (alignment × impact × trend × (1–uncertainty))")

    def alignment_score(cur, tgt):
        return max(0.0, 1.0 - tvdist(cur, tgt))

    def foresight_score(cur, meta):
        align = alignment_score(cur, meta["p"])
        impact = float(meta.get("impact", 1.0)) * float(global_impact_scale)
        trend = float(T_factor)
        penalty = float(uncertainty)
        raw = align * impact * trend * (1.0 - penalty)
        return min(100.0, 100.0 * raw), align, impact, trend, penalty

    rows = []
    for name, meta in ACTIVE_SCENARIOS.items():
        score, align, imp, tr, pen = foresight_score(current_p, meta)
        rows.append({
            "Scenario": name,
            "Score": round(score,3),
            "Alignment(1-TV)": round(align,4),
            "Impact": round(imp,3),
            "Trend": round(tr,3),
            "Uncertainty": round(pen,3),
            "Note": meta.get("note","")
        })
    df_scores = pd.DataFrame(rows).sort_values(by="Score", ascending=False).reset_index(drop=True)
    st.dataframe(df_scores, use_container_width=True)
    st.bar_chart(df_scores.set_index("Scenario")["Score"])

    default_index = 0
    if "sc_choice" in st.session_state and st.session_state["sc_choice"] in valid_names:
        default_index = valid_names.index(st.session_state["sc_choice"])
    sc_name = st.selectbox("Scenario to compare against", valid_names, index=default_index, key="sc_choice")
    scenario_p = ACTIVE_SCENARIOS[sc_name]["p"] if sc_name else None
    scenario_note = ACTIVE_SCENARIOS[sc_name].get("note","") if sc_name else ""

    # ---- Sweep controls ----
    st.markdown("---")
    st.subheader("Sweep (what‑if)")

    chan = st.selectbox(
        "Channel to sweep",
        ["None (no sweep)", "Depolarizing (p)", "Amplitude damping (γ)", "Phase damping (λ)"] +
        (["CNOT depolarizing (p2)"] if st.session_state.num_qubits==2 else [])
    )
    step_label = st.selectbox("Temporal step", ["T0","T1","T2"], index=1)
    step_idx = {"T0":0,"T1":1,"T2":2}[step_label]

    c1,c2,c3=st.columns(3)
    with c1: v_start=st.number_input("Start",0.0,0.5,0.0,0.01)
    with c2: v_end  =st.number_input("End",  0.0,0.5,0.2,0.01)
    with c3: n_pts  =st.number_input("Points",3,61,7 if st.session_state["speed_mode"]=="Fast" else 21,1)

    seeds_str=st.text_input("Seeds (comma)","7,13,23")
    seeds_fx=[int(s.strip()) for s in seeds_str.split(",") if s.strip().isdigit()]
    shots_fx=st.number_input("Shots (per point)",128,8192,512 if st.session_state["speed_mode"]=="Fast" else 2048,128)

    def run_counts_noisy_once(seed_val):
        qc_n=build_measure_circuit_with_noise()
        tqc_qpy=cached_transpile("default",_qc_key(qc_n),_to_qpy(qc_n),1)
        return run_counts_cached("default",tqc_qpy,int(shots_fx),seed_val)

    def agg_counts(list_of_counts):
        tot={k:0 for k in keys}
        for c in list_of_counts:
            for k in keys: tot[k]+=c.get(k,0)
        N=sum(tot.values()) or 1
        return {k:tot[k]/N for k in keys}

    if st.button("Run sweep"):
        if chan=="None (no sweep)":
            st.info("Choose a channel to sweep.")
        else:
            try:
                X=np.linspace(v_start,v_end,int(n_pts)).tolist()
                series={k:[] for k in keys}
                originals={"pdep":[st.session_state.pdep0,st.session_state.pdep1,st.session_state.pdep2],
                           "pamp":[st.session_state.pamp0,st.session_state.pamp1,st.session_state.pamp2],
                           "pph":[st.session_state.pph0,st.session_state.pph1,st.session_state.pph2],
                           "pcnot":[st.session_state.pcnot0,st.session_state.pcnot1,st.session_state.pcnot2]}
                def set_step_param(kind, idx, val):
                    if   kind=="pdep":  st.session_state[f"pdep{idx}"]=val
                    elif kind=="pamp":  st.session_state[f"pamp{idx}"]=val
                    elif kind=="pph":   st.session_state[f"pph{idx}"]=val
                    elif kind=="pcnot": st.session_state[f"pcnot{idx}"]=val
                if chan=="Depolarizing (p)": param="pdep";   st.session_state.enable_dep=True
                elif chan=="Amplitude damping (γ)": param="pamp"; st.session_state.enable_amp=True
                elif chan=="Phase damping (λ)": param="pph";  st.session_state.enable_phs=True
                elif chan=="CNOT depolarizing (p2)": param="pcnot"; st.session_state.enable_cnot_noise=True

                for v in X:
                    set_step_param(param,step_idx,float(v))
                    bag=[]
                    for sd in (seeds_fx or [None]): bag.append(run_counts_noisy_once(sd))
                    agg=agg_counts(bag)
                    for k in keys: series[k].append(agg.get(k,0.0))

                # restore sliders
                st.session_state.pdep0,st.session_state.pdep1,st.session_state.pdep2 = originals["pdep"]
                st.session_state.pamp0,st.session_state.pamp1,st.session_state.pamp2 = originals["pamp"]
                st.session_state.pph0, st.session_state.pph1, st.session_state.pph2  = originals["pph"]
                st.session_state.pcnot0,st.session_state.pcnot1,st.session_state.pcnot2 = originals["pcnot"]

                # plot
                st.subheader(f"Sweep results: {chan} @ {step_label}")
                data={"x":X};  [data.setdefault(k,series[k]) for k in keys]
                st.line_chart(data, x="x")

                # Compare LAST sweep point to chosen scenario + score it
                current_end={k:series[k][-1] for k in keys}
                if scenario_p:
                    D=kldiv(current_end,scenario_p); TV=tvdist(current_end,scenario_p)
                    st.text(f"Scenario: {sc_name} — {scenario_note}")
                    st.text(f"KL(current || scenario) at end of sweep: {D:.6f} | TV: {TV:.6f}")
                    c1,c2=st.columns(2)
                    with c1: st.text("Current (end of sweep)"); st.bar_chart({"p":[current_end.get(k,0.0) for k in keys]})
                    with c2: st.text("Scenario target");         st.bar_chart({"p":[scenario_p.get(k,0.0) for k in keys]})

                    meta = ACTIVE_SCENARIOS[sc_name]
                    score_end, align_end, imp_end, tr_end, pen_end = foresight_score(current_end, meta)
                    st.success(f"Foresight Score at endpoint = {score_end:.2f} (align={align_end:.3f}, impact={imp_end:.2f}, trend={tr_end:.2f}, 1−uncert={1-pen_end:.2f})")

                # Save sweep CSV + memory
                ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                sweep_name_default = f"{chan}@{step_label}_{ts}"
                sweep_name = st.text_input("Name this sweep (for save & compare)", sweep_name_default, key=f"sweep_name_{ts}")

                df = pd.DataFrame({"x": X})
                for k in keys: df[k] = series[k]
                meta = {
                    "qubits": 1 if keys==["0","1"] else 2,
                    "keys": keys,
                    "channel": chan,
                    "step": step_label,
                    "start": float(v_start),
                    "end": float(v_end),
                    "points": int(n_pts),
                    "shots_per_point": int(shots_fx),
                    "seeds": seeds_fx,
                    "scenario": sc_name,
                    "trend_model": trend_model,
                    "trend_now": trend_now,
                    "global_impact_scale": global_impact_scale
                }

                csv_buf = io.StringIO()
                csv_buf.write("# meta=" + json.dumps(meta) + "\n")
                df.to_csv(csv_buf, index=False)
                st.download_button("⬇️ Download sweep CSV", csv_buf.getvalue().encode(), file_name=f"{sweep_name}.csv", mime="text/csv")

                st.session_state.foresight_sweeps[sweep_name] = {
                    "meta": meta, "x": X, "series": series, "df": df.to_dict(orient="list"), "saved_at": ts
                }
                st.success(f"Sweep saved in memory as: {sweep_name}")

            except Exception as e:
                st.error(f"Sweep failed: {e}")

    st.markdown("---")
    st.subheader("Manage sweeps (Load CSV & Compare)")

    # Load CSV → memory
    up_csv = st.file_uploader("Load a sweep CSV", type=["csv"])
    if up_csv is not None:
        try:
            first = up_csv.readline().decode("utf-8")
            if first.startswith("# meta="):
                meta_json = first[len("# meta="):].strip()
                meta = json.loads(meta_json)
            else:
                meta = {}
            df_loaded = pd.read_csv(up_csv)
            if "x" not in df_loaded.columns:
                st.error("CSV missing 'x' column.")
            else:
                loaded_keys = [c for c in df_loaded.columns if c != "x"]
                sweep_name = meta.get("name") or f"loaded_{int(time.time())}"
                st.session_state.foresight_sweeps[sweep_name] = {
                    "meta": {**meta, "keys": loaded_keys},
                    "x": df_loaded["x"].tolist(),
                    "series": {k: df_loaded[k].tolist() for k in loaded_keys},
                    "df": df_loaded.to_dict(orient="list"),
                    "saved_at": dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                st.success(f"Loaded sweep '{sweep_name}' into memory.")
        except Exception as e:
            st.error(f"Load failed: {e}")

    # Compare two sweeps + export + clone endpoints
    sweep_names = sorted(list(st.session_state.foresight_sweeps.keys()))
    if len(sweep_names) >= 2:
        cols = st.columns(2)
        with cols[0]:
            a_name = st.selectbox("Sweep A", sweep_names, key="cmp_a")
        with cols[1]:
            b_name = st.selectbox("Sweep B", sweep_names, index=min(1, len(sweep_names)-1), key="cmp_b")

        if a_name != b_name:
            A = st.session_state.foresight_sweeps[a_name]
            B = st.session_state.foresight_sweeps[b_name]
            keysA = list(A["series"].keys()); keysB = list(B["series"].keys())

            if set(keysA) != set(keysB):
                st.warning("Sweeps have different outcome keys; cannot compare directly.")
            else:
                XA, XB = A["x"], B["x"]
                keys_cmp = keysA
                grids_match = (len(XA)==len(XB)) and all(abs(a-b) < 1e-9 for a,b in zip(XA,XB))

                def dist_at_index(i):
                    p = {k: A["series"][k][i] for k in keys_cmp}
                    q = {k: B["series"][k][i] for k in keys_cmp}
                    return kldiv(p,q), tvdist(p,q)

                st.markdown(f"**A:** {a_name}  |  **B:** {b_name}")
                KLs, TVs = [], []
                if grids_match:
                    for i in range(len(XA)):
                        kl,tv = dist_at_index(i)
                        KLs.append(kl); TVs.append(tv)
                    st.text("Distance (pointwise over sweep X)")
                    st.line_chart({"x": XA, "KL(A||B)": KLs, "TV(A,B)": TVs}, x="x")
                else:
                    st.info("X-grids differ; computing only endpoint distances.")

                # Endpoint distances
                pA_end = {k: A["series"][k][-1] for k in keys_cmp}
                pB_end = {k: B["series"][k][-1] for k in keys_cmp}
                KL_end = kldiv(pA_end, pB_end); TV_end = tvdist(pA_end, pB_end)
                st.text(f"Endpoint distances — KL(A||B): {KL_end:.6f} | TV(A,B): {TV_end:.6f}")

                c1,c2 = st.columns(2)
                with c1:
                    st.text("Sweep A (end)")
                    st.bar_chart({"p":[pA_end.get(k,0.0) for k in keys_cmp]})
                with c2:
                    st.text("Sweep B (end)")
                    st.bar_chart({"p":[pB_end.get(k,0.0) for k in keys_cmp]})

                st.markdown("**Per-key trajectories (A vs B):**")
                for k in keys_cmp:
                    if grids_match:
                        st.line_chart({"x": XA, f"{k} (A)": A["series"][k], f"{k} (B)": B["series"][k]}, x="x")
                    else:
                        st.line_chart({"x": XA, f"{k} (A)": A["series"][k]}, x="x")
                        st.line_chart({"x": XB, f"{k} (B)": B["series"][k]}, x="x")

                # Export compare CSV
                st.markdown("---")
                st.subheader("Export A/B compare distances (CSV)")
                cmp_buf = io.StringIO()
                w = csv.writer(cmp_buf)
                meta_cmp = {"A": a_name, "B": b_name, "keys": keys_cmp, "grids_match": grids_match}
                w.writerow(["# meta=" + json.dumps(meta_cmp)])
                if grids_match:
                    w.writerow(["x","KL_A||B","TV_A_B"])
                    for x, kl, tv in zip(XA, KLs, TVs): w.writerow([x, kl, tv])
                else:
                    w.writerow(["note","KL_end","TV_end"])
                    w.writerow(["grids differ", KL_end, TV_end])
                st.download_button("⬇️ Download compare CSV", data=cmp_buf.getvalue().encode(), file_name=f"compare_{a_name}_VS_{b_name}.csv", mime="text/csv")

                # Clone endpoints → custom scenarios (session)
                st.markdown("---")
                st.subheader("Clone endpoint to a new scenario")
                cc1, cc2 = st.columns(2)
                with cc1:
                    newA = st.text_input("Name for scenario from A endpoint", f"{a_name}_END_SCN")
                    if st.button("Clone A endpoint → scenario"):
                        st.session_state.custom_scenarios[newA] = {
                            "keys": keys_cmp,
                            "p": {k: float(pA_end[k]) for k in keys_cmp},
                            "note": f"Cloned from sweep '{a_name}' endpoint",
                            "impact": 1.0
                        }
                        st.success(f"Added custom scenario: {newA}")
                with cc2:
                    newB = st.text_input("Name for scenario from B endpoint", f"{b_name}_END_SCN")
                    if st.button("Clone B endpoint → scenario"):
                        st.session_state.custom_scenarios[newB] = {
                            "keys": keys_cmp,
                            "p": {k: float(pB_end[k]) for k in keys_cmp},
                            "note": f"Cloned from sweep '{b_name}' endpoint",
                            "impact": 1.0
                        }
                        st.success(f"Added custom scenario: {newB}")
                st.caption("Custom scenarios appear in the scenario dropdown automatically (session‑local).")
    else:
        st.info("Run a sweep (and/or load a CSV) to enable A/B comparison.")

    # ---- Repo scenarios panel ----
    st.markdown("---")
    st.subheader("Repo scenarios (scenarios.json)")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Reload scenarios.json"):
            st.session_state.disk_scenarios = load_disk_scenarios()
            st.success("Reloaded disk scenarios.")
            st.rerun()
    with c2:
        if st.button("💾 Save custom scenarios → scenarios.json"):
            merged = dict(st.session_state.disk_scenarios)
            merged.update(st.session_state.custom_scenarios)  # customs overwrite on name collision
            ok, msg = save_disk_scenarios(merged)
            if ok:
                st.success(msg)
                st.session_state.disk_scenarios = load_disk_scenarios()
            else:
                st.error(msg)

    if st.checkbox("Show current disk scenarios"):
        st.json(st.session_state.disk_scenarios)
