/**
 * CircuitInspectorDashboard.tsx — Circuit Inspector with four sub-tabs:
 *   1. Statevector  2. Measurement  3. QASM Editor  4. IBM Hardware
 */
import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell,
} from "recharts";
import {
  AlertCircle, RefreshCw, Atom, Activity, Code2, Cpu, CheckCircle,
} from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import {
  apiQuantumSimulate, QuantumSimulateResponse, post,
  apiQASMValidate, QASMValidateResponse,
  apiIBMListBackends, apiIBMRunCircuit,
  IBMBackend, IBMRunCircuitResponse,
} from "@/lib/api";

// ── Helpers ───────────────────────────────────────────────────────────────────

function complexAngle(re: number, im: number) {
  return Math.atan2(im, re) * (180 / Math.PI);
}
function phaseColor(deg: number) {
  const h = ((deg % 360) + 360) % 360;
  return `hsl(${h}, 80%, 55%)`;
}
function tvColor(tv: number) {
  if (tv < 0.05) return "text-green-400 border-green-500/30 bg-green-500/10";
  if (tv < 0.15) return "text-yellow-400 border-yellow-500/30 bg-yellow-500/10";
  return "text-red-400 border-red-500/30 bg-red-500/10";
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface MeasurementResponse {
  ideal_counts:  Record<string, number>;
  noisy_counts:  Record<string, number>;
  ideal_probs:   Record<string, number>;
  noisy_probs:   Record<string, number>;
  tv_distance:   number;
  all_states:    string[];
  num_qubits:    number;
}

// ── Statevector sub-tab ───────────────────────────────────────────────────────

function StatevectorPanel() {
  const { state, buildQuantumRequest } = useAppContext();
  const [result, setResult]   = useState<QuantumSimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const run = async () => {
    setLoading(true); setError(null);
    try {
      const req = buildQuantumRequest() as Parameters<typeof apiQuantumSimulate>[0];
      setResult(await apiQuantumSimulate(req));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const nq = state.num_qubits;
  const dim = result ? result.statevector_real.length : Math.pow(2, nq);

  const chartData = result ? result.statevector_real.map((re, i) => {
    const im    = result.statevector_imag[i];
    const amp   = Math.sqrt(re * re + im * im);
    const phase = complexAngle(re, im);
    return {
      state:       `|${i.toString(2).padStart(nq, "0")}⟩`,
      amplitude:   parseFloat(amp.toFixed(4)),
      probability: parseFloat((result.probabilities[i] * 100).toFixed(2)),
      phase:       parseFloat(phase.toFixed(1)),
      phaseColor:  phaseColor(phase),
    };
  }) : [];

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Atom className="w-5 h-5 text-primary" />
            Statevector Analysis
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              {dim}-dim Hilbert space
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            Displays ideal (noise-free) statevector amplitudes and phases from the sidebar circuit.
          </p>
          <Button onClick={run} disabled={loading} className="w-full h-10">
            {loading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Simulating…</>
              : <><Atom className="w-4 h-4 mr-2" />Run Statevector Simulation</>}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span className="text-sm">{error}</span>
          </CardContent>
        </Card>
      )}

      {result && (
        <>
          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-base">Probability Amplitudes |ψ|²</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, "Probability"]} />
                  <Bar dataKey="probability" name="Probability (%)">
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={`hsl(${210 + i * 30}, 70%, 55%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-base">Phase (degrees) per Basis State</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                  <YAxis domain={[-180, 180]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(1)}°`, "Phase"]} />
                  <Bar dataKey="phase" name="Phase (°)">
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={d.phaseColor} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-base">Amplitude Table</CardTitle></CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-accent/20 text-muted-foreground text-xs">
                      <th className="text-left py-2 pr-4">State</th>
                      <th className="text-right py-2 pr-4">Re(α)</th>
                      <th className="text-right py-2 pr-4">Im(α)</th>
                      <th className="text-right py-2 pr-4">|α|</th>
                      <th className="text-right py-2 pr-4">|α|²</th>
                      <th className="text-right py-2">Phase (°)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {chartData.map((d, i) => (
                      <tr key={i} className="border-b border-accent/10 hover:bg-accent/5">
                        <td className="py-1.5 pr-4 font-mono">{d.state}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{result.statevector_real[i].toFixed(4)}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{result.statevector_imag[i].toFixed(4)}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{d.amplitude.toFixed(4)}</td>
                        <td className="py-1.5 pr-4 text-right font-mono text-primary">{(d.probability / 100).toFixed(4)}</td>
                        <td className="py-1.5 text-right font-mono" style={{ color: d.phaseColor }}>{d.phase.toFixed(1)}°</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {result.circuit_lines.length > 0 && (
            <Card className="border-accent/20">
              <CardHeader><CardTitle className="text-base">Circuit Diagram</CardTitle></CardHeader>
              <CardContent>
                <pre className="text-xs font-mono bg-black/20 p-3 rounded overflow-x-auto text-green-400">
                  {result.circuit_lines.join("\n")}
                </pre>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

// ── Measurement sub-tab ───────────────────────────────────────────────────────

function MeasurementPanel() {
  const { buildQuantumRequest, state } = useAppContext();
  const [result, setResult]   = useState<MeasurementResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const noiseEnabled = state.noise.enable_depolarizing || state.noise.enable_amplitude_damping
    || state.noise.enable_phase_damping || state.noise.enable_cnot_noise;

  const run = async () => {
    setLoading(true); setError(null);
    try {
      setResult(await post<MeasurementResponse>("/api/quantum/measurement", buildQuantumRequest()));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const chartData = result ? result.all_states.map(st => ({
    state: `|${st}⟩`,
    ideal: parseFloat(((result.ideal_probs[st] ?? 0) * 100).toFixed(2)),
    noisy: parseFloat(((result.noisy_probs[st] ?? 0) * 100).toFixed(2)),
  })) : [];

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary" />
            Measurement Comparison
            <Badge variant="outline" className={`ml-auto text-xs ${noiseEnabled ? "border-yellow-500/30 bg-yellow-500/10 text-yellow-400" : "border-accent/30"}`}>
              {noiseEnabled ? "Noise ON" : "Noise OFF"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            Compares ideal vs noisy measurement distributions. TV distance measures how much noise corrupts the output — enable noise channels in the sidebar.
          </p>
          <Button onClick={run} disabled={loading} className="w-full h-10">
            {loading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Running measurements…</>
              : <><Activity className="w-4 h-4 mr-2" />Run Measurement</>}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span className="text-sm">{error}</span>
          </CardContent>
        </Card>
      )}

      {result && (
        <>
          <Card className={`border ${tvColor(result.tv_distance)}`}>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-muted-foreground">Total Variation Distance (TV)</p>
                  <p className="text-3xl font-bold font-mono mt-1">{result.tv_distance.toFixed(4)}</p>
                </div>
                <Badge className={`text-sm px-3 py-1 ${tvColor(result.tv_distance)}`}>
                  {result.tv_distance < 0.05 ? "Robust" : result.tv_distance < 0.15 ? "Moderate noise" : "High noise"}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                TV ∈ [0,1] — 0 means perfect match, 1 means completely different distributions.
              </p>
            </CardContent>
          </Card>

          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-base">Ideal vs Noisy Probabilities (%)</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`]} />
                  <Legend />
                  <Bar dataKey="ideal" name="Ideal (%)"  fill="hsl(210,80%,60%)" />
                  <Bar dataKey="noisy" name="Noisy (%)" fill="hsl(30,80%,55%)" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-base">Raw Counts</CardTitle></CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-accent/20 text-muted-foreground text-xs">
                      <th className="text-left py-2">State</th>
                      <th className="text-right py-2">Ideal count</th>
                      <th className="text-right py-2">Noisy count</th>
                      <th className="text-right py-2">Ideal %</th>
                      <th className="text-right py-2">Noisy %</th>
                      <th className="text-right py-2">|Δ| %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.all_states.map(st => {
                      const ip   = result.ideal_probs[st] ?? 0;
                      const np_  = result.noisy_probs[st] ?? 0;
                      const diff = Math.abs(ip - np_) * 100;
                      return (
                        <tr key={st} className="border-b border-accent/10 hover:bg-accent/5">
                          <td className="py-1.5 font-mono">{`|${st}⟩`}</td>
                          <td className="py-1.5 text-right font-mono">{result.ideal_counts[st] ?? 0}</td>
                          <td className="py-1.5 text-right font-mono">{result.noisy_counts[st] ?? 0}</td>
                          <td className="py-1.5 text-right font-mono">{(ip * 100).toFixed(2)}%</td>
                          <td className="py-1.5 text-right font-mono">{(np_ * 100).toFixed(2)}%</td>
                          <td className={`py-1.5 text-right font-mono ${diff > 5 ? "text-red-400" : "text-green-400"}`}>
                            {diff.toFixed(2)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

// ── QASM Editor sub-tab ───────────────────────────────────────────────────────

const DEFAULT_QASM = `OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];`;

function QASMEditorPanel() {
  const [qasmText, setQasmText]         = useState(DEFAULT_QASM);
  const [validation, setValidation]     = useState<QASMValidateResponse | null>(null);
  const [simResult, setSimResult]       = useState<QuantumSimulateResponse | null>(null);
  const [validating, setValidating]     = useState(false);
  const [simulating, setSimulating]     = useState(false);
  const [error, setError]               = useState<string | null>(null);

  const validate = async () => {
    setValidating(true); setError(null);
    try {
      setValidation(await apiQASMValidate(qasmText));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setValidating(false);
    }
  };

  const simulate = async () => {
    setSimulating(true); setError(null);
    try {
      setSimResult(await apiQuantumSimulate({ qasm_str: qasmText }));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSimulating(false);
    }
  };

  const svData = simResult ? simResult.statevector_real.map((re, i) => {
    const im  = simResult.statevector_imag[i];
    const nq  = simResult.num_qubits;
    const amp = Math.sqrt(re * re + im * im);
    return {
      state: `|${i.toString(2).padStart(nq, "0")}⟩`,
      probability: parseFloat((simResult.probabilities[i] * 100).toFixed(2)),
      amplitude: parseFloat(amp.toFixed(4)),
    };
  }) : [];

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code2 className="w-5 h-5 text-primary" />
            OpenQASM 2.0 Editor
            <Badge variant="outline" className="ml-auto border-accent/30 text-xs">qiskit.QuantumCircuit.from_qasm_str</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Write or paste an OpenQASM 2.0 circuit. Validate its syntax, then simulate it to get the full statevector.
          </p>
          <Textarea
            value={qasmText}
            onChange={e => setQasmText(e.target.value)}
            className="font-mono text-xs bg-black/30 border-accent/30 min-h-[160px] resize-y"
            spellCheck={false}
          />
          <div className="flex gap-2">
            <Button variant="outline" onClick={validate} disabled={validating} className="flex-1">
              {validating
                ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Validating…</>
                : <><CheckCircle className="w-4 h-4 mr-2" />Validate QASM</>}
            </Button>
            <Button onClick={simulate} disabled={simulating} className="flex-1">
              {simulating
                ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Simulating…</>
                : <><Atom className="w-4 h-4 mr-2" />Simulate</>}
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span className="text-sm font-mono">{error}</span>
          </CardContent>
        </Card>
      )}

      {validation && (
        <Card className={validation.valid ? "border-green-500/30 bg-green-500/5" : "border-red-500/30 bg-red-500/5"}>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              {validation.valid
                ? <CheckCircle className="w-4 h-4 text-green-400" />
                : <AlertCircle className="w-4 h-4 text-red-400" />}
              {validation.valid ? "Valid circuit" : "Parse error"}
            </CardTitle>
          </CardHeader>
          {validation.valid ? (
            <CardContent className="space-y-3">
              <div className="grid grid-cols-4 gap-3 text-center">
                {[
                  { label: "Qubits",  value: validation.num_qubits },
                  { label: "Cbits",   value: validation.num_clbits },
                  { label: "Depth",   value: validation.depth },
                  { label: "Gates",   value: validation.num_gates },
                ].map(({ label, value }) => (
                  <div key={label} className="rounded-md border border-accent/20 bg-background/40 py-2">
                    <div className="text-xl font-bold font-mono text-primary">{value}</div>
                    <div className="text-xs text-muted-foreground">{label}</div>
                  </div>
                ))}
              </div>
              {validation.circuit_lines.length > 0 && (
                <pre className="text-xs font-mono bg-black/20 p-3 rounded overflow-x-auto text-green-400">
                  {validation.circuit_lines.join("\n")}
                </pre>
              )}
            </CardContent>
          ) : (
            <CardContent>
              <p className="text-sm font-mono text-red-400">{validation.error}</p>
            </CardContent>
          )}
        </Card>
      )}

      {simResult && svData.length > 0 && (
        <Card className="border-accent/20">
          <CardHeader><CardTitle className="text-base">Statevector — Probability Distribution</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={svData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, "Probability"]} />
                <Bar dataKey="probability" name="Probability (%)">
                  {svData.map((_, i) => <Cell key={i} fill={`hsl(${210 + i * 30}, 70%, 55%)`} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            {simResult.circuit_lines.length > 0 && (
              <pre className="mt-3 text-xs font-mono bg-black/20 p-3 rounded overflow-x-auto text-green-400">
                {simResult.circuit_lines.join("\n")}
              </pre>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ── IBM Hardware sub-tab ──────────────────────────────────────────────────────

function IBMHardwarePanel() {
  const [token, setToken]               = useState("");
  const [showToken, setShowToken]       = useState(false);
  const [backends, setBackends]         = useState<IBMBackend[]>([]);
  const [listLoading, setListLoading]   = useState(false);
  const [listError, setListError]       = useState<string | null>(null);

  const [selectedBackend, setSelectedBackend] = useState("");
  const [qasmText, setQasmText]         = useState(DEFAULT_QASM);
  const [shots, setShots]               = useState(1024);
  const [runLoading, setRunLoading]     = useState(false);
  const [runResult, setRunResult]       = useState<IBMRunCircuitResponse | null>(null);
  const [runError, setRunError]         = useState<string | null>(null);

  const listBackends = async () => {
    if (!token.trim()) return;
    setListLoading(true); setListError(null);
    try {
      const res = await apiIBMListBackends(token.trim());
      setBackends(res.backends);
      if (res.backends.length > 0) setSelectedBackend(res.backends[0].name);
    } catch (e: unknown) {
      setListError(e instanceof Error ? e.message : String(e));
    } finally {
      setListLoading(false);
    }
  };

  const runCircuit = async () => {
    if (!token.trim() || !selectedBackend) return;
    setRunLoading(true); setRunError(null); setRunResult(null);
    try {
      setRunResult(await apiIBMRunCircuit(token.trim(), selectedBackend, qasmText, shots));
    } catch (e: unknown) {
      setRunError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunLoading(false);
    }
  };

  const countData = runResult
    ? Object.entries(runResult.probabilities).map(([state, prob]) => ({
        state: `|${state}⟩`,
        probability: parseFloat((prob * 100).toFixed(2)),
      }))
    : [];

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-primary" />
            IBM Quantum Hardware
            <Badge variant="outline" className="ml-auto border-amber-500/30 bg-amber-500/10 text-amber-400 text-xs">
              Real QPU
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Connect your IBM Quantum account to run circuits on real hardware. Token is used only for this request and never stored.
          </p>

          {/* Token input */}
          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground font-medium">IBM Quantum API Token</label>
            <div className="flex gap-2">
              <Input
                type={showToken ? "text" : "password"}
                value={token}
                onChange={e => setToken(e.target.value)}
                placeholder="Paste your IBM Quantum API token…"
                className="font-mono text-xs bg-background/60 border-accent/30 flex-1"
              />
              <Button variant="outline" size="sm" onClick={() => setShowToken(s => !s)} className="shrink-0">
                {showToken ? "Hide" : "Show"}
              </Button>
            </div>
          </div>

          <Button onClick={listBackends} disabled={listLoading || !token.trim()} className="w-full">
            {listLoading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Fetching backends…</>
              : <><Cpu className="w-4 h-4 mr-2" />Load Available Backends</>}
          </Button>

          {listError && (
            <div className="flex items-center gap-2 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span className="font-mono">{listError}</span>
            </div>
          )}
        </CardContent>
      </Card>

      {backends.length > 0 && (
        <>
          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-sm">Available Backends ({backends.length})</CardTitle></CardHeader>
            <CardContent>
              <div className="grid gap-2">
                {backends.map(b => (
                  <button
                    key={b.name}
                    onClick={() => setSelectedBackend(b.name)}
                    className={`flex items-center justify-between p-3 rounded-md border text-left transition-colors ${
                      selectedBackend === b.name
                        ? "border-primary/60 bg-primary/10"
                        : "border-accent/20 hover:border-accent/40 bg-background/40"
                    }`}
                  >
                    <div>
                      <div className="text-sm font-mono font-medium">{b.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {b.num_qubits ?? "?"} qubits · {b.pending_jobs} pending jobs
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {b.simulator && <Badge variant="outline" className="text-xs border-blue-500/30 text-blue-400">Simulator</Badge>}
                      <Badge
                        variant="outline"
                        className={`text-xs ${b.operational ? "border-green-500/30 text-green-400" : "border-red-500/30 text-red-400"}`}
                      >
                        {b.operational ? "Online" : "Offline"}
                      </Badge>
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-sm">Run Circuit on {selectedBackend || "—"}</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={qasmText}
                onChange={e => setQasmText(e.target.value)}
                className="font-mono text-xs bg-black/30 border-accent/30 min-h-[120px] resize-y"
                spellCheck={false}
              />
              <div className="flex items-center gap-3">
                <label className="text-xs text-muted-foreground whitespace-nowrap">Shots:</label>
                <Input
                  type="number"
                  min={64}
                  max={20000}
                  value={shots}
                  onChange={e => setShots(Number(e.target.value))}
                  className="w-28 text-xs bg-background/60 border-accent/30"
                />
              </div>
              <Button
                onClick={runCircuit}
                disabled={runLoading || !selectedBackend}
                className="w-full"
              >
                {runLoading
                  ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Submitting to IBM…</>
                  : <><Cpu className="w-4 h-4 mr-2" />Run on {selectedBackend || "selected backend"}</>}
              </Button>
              {runError && (
                <div className="flex items-center gap-2 text-red-400 text-sm">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span className="font-mono text-xs">{runError}</span>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {runResult && (
        <Card className="border-accent/20">
          <CardHeader>
            <CardTitle className="text-base">
              Results — {runResult.backend} ({runResult.shots.toLocaleString()} shots)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={countData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, "Probability"]} />
                <Bar dataKey="probability" name="Probability (%)" fill="hsl(30,80%,55%)" />
              </BarChart>
            </ResponsiveContainer>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-accent/20 text-muted-foreground text-xs">
                    <th className="text-left py-2">State</th>
                    <th className="text-right py-2">Counts</th>
                    <th className="text-right py-2">Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(runResult.counts).map(([state, count]) => (
                    <tr key={state} className="border-b border-accent/10 hover:bg-accent/5">
                      <td className="py-1.5 font-mono">{`|${state}⟩`}</td>
                      <td className="py-1.5 text-right font-mono">{count}</td>
                      <td className="py-1.5 text-right font-mono text-primary">
                        {((runResult.probabilities[state] ?? 0) * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ── Combined tab ──────────────────────────────────────────────────────────────

export function CircuitInspectorDashboard() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">Circuit Inspector</h2>
        <p className="text-sm text-muted-foreground">
          Examine your circuit from four angles: statevector, measurements, custom QASM input, and real IBM hardware.
        </p>
      </div>

      <Tabs defaultValue="statevector">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="statevector" className="flex items-center gap-1.5 text-xs">
            <Atom className="w-3.5 h-3.5" />Statevector
          </TabsTrigger>
          <TabsTrigger value="measurement" className="flex items-center gap-1.5 text-xs">
            <Activity className="w-3.5 h-3.5" />Measurement
          </TabsTrigger>
          <TabsTrigger value="qasm" className="flex items-center gap-1.5 text-xs">
            <Code2 className="w-3.5 h-3.5" />QASM Editor
          </TabsTrigger>
          <TabsTrigger value="ibm" className="flex items-center gap-1.5 text-xs">
            <Cpu className="w-3.5 h-3.5" />IBM Hardware
          </TabsTrigger>
        </TabsList>

        <TabsContent value="statevector" className="mt-4">
          <StatevectorPanel />
        </TabsContent>
        <TabsContent value="measurement" className="mt-4">
          <MeasurementPanel />
        </TabsContent>
        <TabsContent value="qasm" className="mt-4">
          <QASMEditorPanel />
        </TabsContent>
        <TabsContent value="ibm" className="mt-4">
          <IBMHardwarePanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
