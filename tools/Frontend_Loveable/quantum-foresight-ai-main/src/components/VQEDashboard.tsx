import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import {
  apiVQERiskGate, apiVQEAudit, apiVQESolve,
  VQERiskGateResponse, VQESolveResponse,
} from "@/lib/api";
import {
  Shield, AlertTriangle, CheckCircle, XCircle,
  History, AlertCircle, RefreshCw, FlaskConical, TrendingDown,
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";

// ── Risk Gate ─────────────────────────────────────────────────────────────────
const STATUS_CONFIG = {
  APPROVED: { color: "text-green-400", bg: "bg-green-500/10 border-green-500/30", icon: CheckCircle },
  PARTIAL:  { color: "text-yellow-400", bg: "bg-yellow-500/10 border-yellow-500/30", icon: AlertTriangle },
  BLOCKED:  { color: "text-red-400",   bg: "bg-red-500/10 border-red-500/30",   icon: XCircle },
} as const;

const POLICIES = ["Conservative", "Moderate", "Aggressive"];
const POLICY_DESCRIPTIONS: Record<string, string> = {
  Conservative: "Max notional $50K · VaR $2K · CVaR $3.5K · 1.5× leverage",
  Moderate:     "Max notional $250K · VaR $10K · CVaR $18K · 3× leverage",
  Aggressive:   "Max notional $1M · VaR $50K · CVaR $90K · 6× leverage",
};

// ── Advanced Metrics config ───────────────────────────────────────────────────
const PROBLEMS = [
  { value: "Toy Hamiltonian",       label: "Toy Problem (default)",         desc: "A simple built-in problem — good for testing. No setup needed." },
  { value: "MaxCut (graph)",        label: "Graph Partition (MaxCut)",       desc: "Divide graph nodes into two groups to maximise cut edges. Enter edges below." },
  { value: "Ising (h/J)",           label: "Spin System (Ising model)",      desc: "Simulate interacting particles with local fields and couplings." },
  { value: "Custom Pauli Hamiltonian", label: "Custom Energy Function",      desc: "Enter Pauli operator terms directly (advanced)." },
];

const ANSATZE = [
  { value: "RealAmplitudes",  label: "Simple Rotations (RealAmplitudes)",  desc: "Lightweight circuit using Y-rotations. Fast and efficient." },
  { value: "EfficientSU2",   label: "Balanced Circuit (EfficientSU2)",     desc: "More expressive than Simple Rotations. Good all-around choice." },
  { value: "TwoLocal",       label: "Full Rotation Circuit (TwoLocal)",    desc: "Both Y and Z rotations. Most expressive, but slowest." },
];

const OPTIMIZERS = [
  { value: "COBYLA",  label: "Derivative-Free (COBYLA)",   desc: "Doesn't need gradients. Robust and reliable for small problems." },
  { value: "SPSA",    label: "Stochastic Gradient (SPSA)", desc: "Estimates gradients by random perturbation. Good for noisy problems." },
  { value: "SLSQP",   label: "Gradient-Based (SLSQP)",     desc: "Uses exact gradients. Fast when they are available." },
];

// ── Risk multiplier colour ────────────────────────────────────────────────────
function multiplierColor(m: number) {
  if (m >= 1.2) return "text-green-400";
  if (m >= 0.9) return "text-yellow-400";
  return "text-red-400";
}

// ════════════════════════════════════════════════════════════════════════════════
export const VQEDashboard = () => {
  // ── Risk Gate state ──────────────────────────────────────────────────────────
  const [notional, setNotional]   = useState(100000);
  const [price, setPrice]         = useState(150);
  const [volPct, setVolPct]       = useState(1.5);
  const [leverage, setLeverage]   = useState(1.0);
  const [policy, setPolicy]       = useState("Moderate");
  const [gateLoading, setGateLoading] = useState(false);
  const [gateResult, setGateResult]   = useState<VQERiskGateResponse | null>(null);
  const [audit, setAudit]         = useState<VQERiskGateResponse[]>([]);
  const [gateError, setGateError] = useState<string | null>(null);
  const [showAudit, setShowAudit] = useState(false);

  // ── Advanced Metrics state ───────────────────────────────────────────────────
  const [problem, setProblem]         = useState("Toy Hamiltonian");
  const [ansatzName, setAnsatzName]   = useState("RealAmplitudes");
  const [optimizerName, setOptimizer] = useState("COBYLA");
  const [numQubits, setNumQubits]     = useState(2);
  const [reps, setReps]               = useState(2);
  const [maxiter, setMaxiter]         = useState(80);
  const [pauliText, setPauliText]     = useState("ZZ:1, XI:0.4, IX:0.4");
  const [edgesText, setEdgesText]     = useState("0-1:1.0\n1-2:0.8\n0-2:0.6");
  const [isingH, setIsingH]           = useState("0.5, -0.5");
  const [isingJ, setIsingJ]           = useState("0 1 1.0");
  const [solveLoading, setSolveLoading] = useState(false);
  const [solveResult, setSolveResult]   = useState<VQESolveResponse | null>(null);
  const [solveError, setSolveError]     = useState<string | null>(null);

  useEffect(() => {
    apiVQEAudit(20).then(r => setAudit(r.records)).catch(() => {});
  }, []);

  // ── Risk Gate handler ────────────────────────────────────────────────────────
  const checkGate = async () => {
    setGateLoading(true);
    setGateError(null);
    try {
      const res = await apiVQERiskGate({ requested_notional_usd: notional, price_usd: price, vol_daily_pct: volPct, leverage, policy });
      setGateResult(res);
      setAudit(prev => [res, ...prev].slice(0, 20));
    } catch (e: unknown) {
      setGateError(e instanceof Error ? e.message : String(e));
    } finally {
      setGateLoading(false);
    }
  };

  // ── VQE Solve handler ────────────────────────────────────────────────────────
  const runVQE = async () => {
    setSolveLoading(true);
    setSolveError(null);
    try {
      const res = await apiVQESolve({
        problem, ansatz_name: ansatzName, optimizer_name: optimizerName,
        num_qubits: numQubits, reps, maxiter, seed: 42,
        pauli_text: pauliText, maxcut_edges_text: edgesText,
        ising_h_text: isingH, ising_J_text: isingJ,
      });
      setSolveResult(res);
    } catch (e: unknown) {
      setSolveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSolveLoading(false);
    }
  };

  const StatusIcon = gateResult ? STATUS_CONFIG[gateResult.status].icon : Shield;

  // Helper: description for currently selected options
  const problemDesc  = PROBLEMS.find(p => p.value === problem)?.desc ?? "";
  const ansatzDesc   = ANSATZE.find(a => a.value === ansatzName)?.desc ?? "";
  const optimizerDesc = OPTIMIZERS.find(o => o.value === optimizerName)?.desc ?? "";

  // Convergence chart data — only entries that have an energy value
  const chartData = solveResult?.history.filter(h => h.energy != null).map(h => ({
    t: h.t,
    energy: typeof h.energy === "number" ? parseFloat(h.energy.toFixed(5)) : null,
  })) ?? [];

  return (
    <div className="space-y-6">

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/* RISK GATE                                                              */}
      {/* ══════════════════════════════════════════════════════════════════════ */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary" />
            VQE Risk Gate
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              Python Backend
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-base font-medium">Stress-tests your portfolio against different market scenarios using a quantum algorithm.</p>
          {/* Policy */}
          <div>
            <Label>Risk Policy</Label>
            <div className="flex gap-3 mt-2">
              {POLICIES.map(p => (
                <Button key={p} variant={policy === p ? "default" : "outline"} size="sm" onClick={() => setPolicy(p)} className="flex-1">
                  {p}
                </Button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-1">{POLICY_DESCRIPTIONS[policy]}</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <Label htmlFor="notional">Notional (USD)</Label>
              <Input id="notional" type="number" value={notional} onChange={e => setNotional(parseFloat(e.target.value) || 0)} step={10000} />
            </div>
            <div>
              <Label htmlFor="price">Price per Unit (USD)</Label>
              <Input id="price" type="number" value={price} onChange={e => setPrice(parseFloat(e.target.value) || 1)} step={10} />
            </div>
            <div>
              <Label htmlFor="vol">Daily Vol (%)</Label>
              <Input id="vol" type="number" value={volPct} onChange={e => setVolPct(parseFloat(e.target.value) || 1)} step={0.1} min={0.1} />
            </div>
            <div>
              <Label htmlFor="lev">Leverage (×)</Label>
              <Input id="lev" type="number" value={leverage} onChange={e => setLeverage(parseFloat(e.target.value) || 1)} step={0.5} min={1} />
            </div>
          </div>

          <Button onClick={checkGate} disabled={gateLoading} className="w-full h-12">
            {gateLoading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Checking Risk Gate...</>
              : <><Shield className="w-4 h-4 mr-2" />Check Risk Gate</>}
          </Button>
        </CardContent>
      </Card>

      {gateError && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{gateError}</span>
          </CardContent>
        </Card>
      )}

      {/* Gate result */}
      {gateResult && (
        <Card className={`border ${STATUS_CONFIG[gateResult.status].bg}`}>
          <CardHeader>
            <CardTitle className={`flex items-center gap-3 ${STATUS_CONFIG[gateResult.status].color}`}>
              <StatusIcon className="w-6 h-6" />
              Trade {gateResult.status}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {[
                { label: "Policy",             val: gateResult.policy },
                { label: "Requested Notional", val: `$${gateResult.requested_notional_usd.toLocaleString()}` },
                { label: "Final Notional",     val: `$${gateResult.final_notional_usd.toLocaleString()}` },
                { label: "Est. VaR (95%)",     val: `$${gateResult.est_var_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
                { label: "Est. CVaR",          val: `$${gateResult.est_cvar_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
                { label: "Leverage",           val: `${gateResult.leverage_used.toFixed(1)}×` },
              ].map(({ label, val }) => (
                <div key={label} className="p-3 border border-accent/20 rounded">
                  <p className="text-xs text-muted-foreground">{label}</p>
                  <p className="font-semibold mt-0.5">{val}</p>
                </div>
              ))}
            </div>
            {gateResult.reasons.length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Gate Reasons</p>
                <ul className="space-y-1">
                  {gateResult.reasons.map((r, i) => (
                    <li key={i} className={`text-sm flex items-center gap-2 ${STATUS_CONFIG[gateResult.status].color}`}>
                      <AlertTriangle className="w-3 h-3 flex-shrink-0" />{r}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <div>
              <p className="text-sm font-medium mb-2">Policy Limits ({gateResult.policy})</p>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(gateResult.limits).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-xs p-2 bg-muted/20 rounded">
                    <span className="text-muted-foreground">{k.replace(/_/g, " ")}</span>
                    <span>{typeof v === "number" && v > 10 ? `$${v.toLocaleString()}` : `${v}×`}</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Audit log */}
      {audit.length > 0 && (
        <Card className="border-accent/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 cursor-pointer" onClick={() => setShowAudit(!showAudit)}>
              <History className="w-4 h-4" />
              Audit Log ({audit.length})
              <span className="text-xs text-muted-foreground ml-auto">{showAudit ? "▲ collapse" : "▼ expand"}</span>
            </CardTitle>
          </CardHeader>
          {showAudit && (
            <CardContent>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {audit.map((r, i) => (
                  <div key={i} className="flex items-center gap-3 text-sm p-2 border border-accent/10 rounded">
                    <Badge variant="outline" className={STATUS_CONFIG[r.status].color}>{r.status}</Badge>
                    <span className="flex-1">${r.requested_notional_usd.toLocaleString()}</span>
                    <span className="text-muted-foreground text-xs">{r.policy}</span>
                    <span className="text-muted-foreground text-xs">{new Date(r.timestamp).toLocaleTimeString()}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          )}
        </Card>
      )}

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/* ADVANCED METRICS divider                                               */}
      {/* ══════════════════════════════════════════════════════════════════════ */}
      <div className="flex items-center gap-4 pt-2">
        <Separator className="flex-1" />
        <div className="flex items-center gap-2 px-3 py-1 rounded-full border border-primary/30 bg-primary/5">
          <FlaskConical className="w-4 h-4 text-primary" />
          <span className="text-sm font-semibold text-primary tracking-wide">Advanced Metrics</span>
        </div>
        <Separator className="flex-1" />
      </div>
      <p className="text-xs text-muted-foreground -mt-3 text-center">
        Variational Quantum Eigensolver — find the lowest-energy solution to an optimization problem using a quantum circuit.
      </p>

      {/* ── Problem / Ansatz / Optimizer config ─────────────────────────────── */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FlaskConical className="w-5 h-5 text-primary" />
            VQE Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">

          {/* Row 1 — Problem / Ansatz / Optimizer */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-1">
              <Label className="text-xs">Problem type</Label>
              <Select value={problem} onValueChange={setProblem}>
                <SelectTrigger className="h-9 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PROBLEMS.map(p => (
                    <SelectItem key={p.value} value={p.value} className="text-xs">{p.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground leading-snug">{problemDesc}</p>
            </div>

            <div className="space-y-1">
              <Label className="text-xs">Circuit template (Ansatz)</Label>
              <Select value={ansatzName} onValueChange={setAnsatzName}>
                <SelectTrigger className="h-9 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ANSATZE.map(a => (
                    <SelectItem key={a.value} value={a.value} className="text-xs">{a.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground leading-snug">{ansatzDesc}</p>
            </div>

            <div className="space-y-1">
              <Label className="text-xs">Search strategy (Optimizer)</Label>
              <Select value={optimizerName} onValueChange={setOptimizer}>
                <SelectTrigger className="h-9 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {OPTIMIZERS.map(o => (
                    <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground leading-snug">{optimizerDesc}</p>
            </div>
          </div>

          {/* Row 2 — Numerical controls */}
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-1">
              <Label className="text-xs">Qubits</Label>
              <Input type="number" min={1} max={10} value={numQubits}
                onChange={e => setNumQubits(Math.max(1, parseInt(e.target.value) || 1))}
                className="h-8 text-xs" />
              <p className="text-xs text-muted-foreground">Number of quantum bits</p>
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Circuit depth (reps)</Label>
              <Input type="number" min={1} max={8} value={reps}
                onChange={e => setReps(Math.max(1, parseInt(e.target.value) || 1))}
                className="h-8 text-xs" />
              <p className="text-xs text-muted-foreground">More reps = more expressive, slower</p>
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Max iterations</Label>
              <Input type="number" min={10} max={2000} step={10} value={maxiter}
                onChange={e => setMaxiter(Math.max(10, parseInt(e.target.value) || 80))}
                className="h-8 text-xs" />
              <p className="text-xs text-muted-foreground">How long the optimizer runs</p>
            </div>
          </div>

          {/* Problem-specific inputs */}
          {problem === "Custom Pauli Hamiltonian" && (
            <div className="space-y-1">
              <Label className="text-xs">Pauli terms (e.g. ZZ:1, XI:0.4, IX:0.4)</Label>
              <Textarea value={pauliText} onChange={e => setPauliText(e.target.value)}
                className="h-20 text-xs font-mono" placeholder="ZZ:1, XI:0.4, IX:0.4" />
            </div>
          )}

          {problem === "MaxCut (graph)" && (
            <div className="space-y-1">
              <Label className="text-xs">Graph edges (one per line: node-node:weight)</Label>
              <Textarea value={edgesText} onChange={e => setEdgesText(e.target.value)}
                className="h-24 text-xs font-mono" placeholder={"0-1:1.0\n1-2:0.8\n0-2:0.6"} />
              <p className="text-xs text-muted-foreground">Each line is an edge. Weights are optional (default 1.0).</p>
            </div>
          )}

          {problem === "Ising (h/J)" && (
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <Label className="text-xs">Local fields h (comma-separated)</Label>
                <Input value={isingH} onChange={e => setIsingH(e.target.value)} className="h-8 text-xs font-mono" placeholder="0.5, -0.5" />
                <p className="text-xs text-muted-foreground">One value per qubit</p>
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Couplings J (i j value, one per line)</Label>
                <Textarea value={isingJ} onChange={e => setIsingJ(e.target.value)}
                  className="h-16 text-xs font-mono" placeholder={"0 1 1.0\n1 2 -0.5"} />
              </div>
            </div>
          )}

          <Button onClick={runVQE} disabled={solveLoading} className="w-full h-12">
            {solveLoading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Running VQE optimizer...</>
              : <><FlaskConical className="w-4 h-4 mr-2" />Run VQE</>}
          </Button>
        </CardContent>
      </Card>

      {solveError && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{solveError}</span>
          </CardContent>
        </Card>
      )}

      {/* ── VQE Results ──────────────────────────────────────────────────────── */}
      {solveResult && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="border-accent/20">
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Minimum energy found</p>
                <p className="text-2xl font-mono font-bold text-primary mt-1">
                  {solveResult.energy != null ? solveResult.energy.toFixed(4) : "—"}
                </p>
                <p className="text-xs text-muted-foreground mt-1">Lower = better solution</p>
              </CardContent>
            </Card>
            <Card className="border-accent/20">
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Risk budget multiplier</p>
                <p className={`text-2xl font-mono font-bold mt-1 ${multiplierColor(solveResult.risk_multiplier)}`}>
                  {solveResult.risk_multiplier.toFixed(3)}×
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {solveResult.risk_multiplier >= 1.2 ? "Expand risk budget" :
                   solveResult.risk_multiplier >= 0.9 ? "Neutral" : "Tighten risk budget"}
                </p>
              </CardContent>
            </Card>
            <Card className="border-accent/20">
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Convergence</p>
                <p className="text-lg font-semibold mt-1">
                  {solveResult.converged
                    ? <span className="text-green-400">Converged</span>
                    : solveResult.used_fallback
                      ? <span className="text-yellow-400">Fallback used</span>
                      : <span className="text-orange-400">Partial</span>}
                </p>
                <p className="text-xs text-muted-foreground mt-1">{solveResult.estimator}</p>
              </CardContent>
            </Card>
            <Card className="border-accent/20">
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Problem info</p>
                <p className="text-sm font-semibold mt-1">{solveResult.problem_type}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {solveResult.num_pauli_terms} Pauli terms · {solveResult.num_qubits}q
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Technical detail strip */}
          <Card className="border-accent/20">
            <CardContent className="pt-4">
              <div className="flex flex-wrap gap-2 text-xs">
                <Badge variant="outline" className="font-mono border-primary/30 text-primary/80">{solveResult.ansatz_desc}</Badge>
                <Badge variant="outline" className="font-mono border-accent/50">{solveResult.optimizer_desc}</Badge>
                {solveResult.used_fallback && (
                  <Badge variant="outline" className="border-yellow-500/40 text-yellow-400">
                    Qiskit not available — toy fallback energy shown
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Convergence chart */}
          {chartData.length > 1 && (
            <Card className="border-accent/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <TrendingDown className="w-4 h-4 text-primary" />
                  Energy Convergence
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 11 }} label={{ value: "Iteration", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} width={60} tickFormatter={v => v.toFixed(2)} />
                    <Tooltip
                      contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6 }}
                      labelStyle={{ color: "#94a3b8", fontSize: 11 }}
                      formatter={(v: number) => [v.toFixed(5), "Energy"]}
                    />
                    <Line type="monotone" dataKey="energy" stroke="#7c3aed" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
                <p className="text-xs text-muted-foreground mt-2 text-center">
                  The optimizer drives the energy downward — a steep descent means fast convergence.
                </p>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
};
