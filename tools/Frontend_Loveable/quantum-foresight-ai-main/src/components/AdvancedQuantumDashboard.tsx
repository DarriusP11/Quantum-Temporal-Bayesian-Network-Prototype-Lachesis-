import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  apiQuantumTomography, apiQuantumBenchmarking, apiQuantumCalibrate, apiQuantumFidelity,
  TomographyResponse, BenchmarkingResponse, CalibrateResponse, FidelityResponse,
} from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts';
import { Atom, Activity, BarChart3, Gauge, RefreshCw, AlertCircle } from "lucide-react";

const GATES_1Q = ["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"];

export const AdvancedQuantumDashboard = () => {
  // Shared state
  const [selectedGate, setSelectedGate] = useState("H");
  const [angle, setAngle]               = useState(0.5);
  const [shots, setShots]               = useState(4096);
  const [seed, setSeed]                 = useState(17);

  // Per-tab results
  const [tomoResult, setTomoResult]   = useState<TomographyResponse | null>(null);
  const [rbResult, setRBResult]       = useState<BenchmarkingResponse | null>(null);
  const [calResult, setCalResult]     = useState<CalibrateResponse | null>(null);
  const [fidResult, setFidResult]     = useState<FidelityResponse | null>(null);

  // Loading / error per tab
  const [loadingTab, setLoadingTab]   = useState<string | null>(null);
  const [error, setError]             = useState<string | null>(null);

  const run = async (tab: string, fn: () => Promise<void>) => {
    setLoadingTab(tab);
    setError(null);
    try {
      await fn();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingTab(null);
    }
  };

  // Tomography
  const runTomography = () =>
    run("tomo", async () => {
      const res = await apiQuantumTomography(selectedGate, angle, shots, seed);
      setTomoResult(res);
    });

  // Benchmarking
  const [rbLengths, setRBLengths] = useState("2,4,8,16,32,48,64");
  const [rbSeeds, setRBSeeds]     = useState(8);

  const runBenchmarking = () =>
    run("rb", async () => {
      const lengths = rbLengths.split(",").map(v => parseInt(v.trim())).filter(v => !isNaN(v));
      const res = await apiQuantumBenchmarking(lengths, rbSeeds, shots, seed);
      setRBResult(res);
    });

  // Calibration
  const runCalibration = () =>
    run("cal", async () => {
      const res = await apiQuantumCalibrate(shots, seed);
      setCalResult(res);
    });

  // Fidelity
  const runFidelity = () =>
    run("fid", async () => {
      const res = await apiQuantumFidelity(selectedGate, angle * Math.PI, shots, seed);
      setFidResult(res);
    });

  // Chart helpers
  const blochData = tomoResult
    ? [
        { axis: "X", value: +(tomoResult.bloch_x * 100).toFixed(1) },
        { axis: "Y", value: +(tomoResult.bloch_y * 100).toFixed(1) },
        { axis: "Z", value: +(tomoResult.bloch_z * 100).toFixed(1) },
      ]
    : [];

  const rbChartData = rbResult
    ? rbResult.lengths.map((l, i) => ({ m: l, survival: +(rbResult.survival[i] * 100).toFixed(2) }))
    : [];

  const calTableRows = calResult
    ? Object.entries(calResult.posteriors)
    : [];

  return (
    <div className="space-y-6">
      {/* Shared controls */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Atom className="w-5 h-5 text-primary" />
            Advanced Quantum Analysis
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              Python · Qiskit
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <Label>Gate</Label>
              <Select value={selectedGate} onValueChange={setSelectedGate}>
                <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {GATES_1Q.map(g => <SelectItem key={g} value={g}>{g}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Angle (× π): {angle.toFixed(2)}</Label>
              <Slider value={[angle]} onValueChange={([v]) => setAngle(v)} min={0} max={2} step={0.05} className="mt-2" />
            </div>
            <div>
              <Label htmlFor="aq-shots">Shots</Label>
              <Input id="aq-shots" type="number" value={shots} onChange={e => setShots(parseInt(e.target.value) || 4096)} min={512} max={32768} step={512} />
            </div>
            <div>
              <Label htmlFor="aq-seed">Seed</Label>
              <Input id="aq-seed" type="number" value={seed} onChange={e => setSeed(parseInt(e.target.value) || 17)} />
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{error}</span>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="tomo" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="tomo">Tomography</TabsTrigger>
          <TabsTrigger value="rb">Benchmarking</TabsTrigger>
          <TabsTrigger value="cal">Noise Calibration</TabsTrigger>
          <TabsTrigger value="fid">Process Fidelity</TabsTrigger>
        </TabsList>

        {/* ── State Tomography ─────────────────────────────────────────────── */}
        <TabsContent value="tomo" className="space-y-4">
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" />
                1-Qubit State Tomography
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-base font-medium mb-1">Deep diagnostics for your qubits — measures quantum properties and benchmarks circuit performance.</p>
              <p className="text-sm text-muted-foreground">
                Measures the Bloch sphere coordinates ⟨X⟩, ⟨Y⟩, ⟨Z⟩ of the state produced by the selected gate.
              </p>
              <Button onClick={runTomography} disabled={loadingTab === "tomo"} className="w-full">
                {loadingTab === "tomo"
                  ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Running Tomography...</>
                  : <><Activity className="w-4 h-4 mr-2" />Run Tomography</>}
              </Button>

              {tomoResult && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
                  <div className="space-y-3">
                    {[
                      { axis: "X", val: tomoResult.bloch_x },
                      { axis: "Y", val: tomoResult.bloch_y },
                      { axis: "Z", val: tomoResult.bloch_z },
                    ].map(({ axis, val }) => (
                      <div key={axis}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-mono">⟨{axis}⟩</span>
                          <span className={val > 0 ? "text-green-400" : val < 0 ? "text-red-400" : "text-muted-foreground"}>
                            {val >= 0 ? "+" : ""}{val.toFixed(4)}
                          </span>
                        </div>
                        <Progress value={(val + 1) / 2 * 100} className="h-2" />
                      </div>
                    ))}
                    <div className="flex justify-between text-sm pt-2 border-t border-accent/20">
                      <span>Purity |r|²</span>
                      <Badge variant="outline">{tomoResult.purity.toFixed(4)}</Badge>
                    </div>
                  </div>

                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={blochData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="axis" />
                      <YAxis domain={[-100, 100]} />
                      <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {blochData.map((entry, i) => (
                          <Cell key={i} fill={entry.value >= 0 ? "#22c55e" : "#ef4444"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Randomized Benchmarking ──────────────────────────────────────── */}
        <TabsContent value="rb" className="space-y-4">
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-primary" />
                Randomized Benchmarking
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Measures gate error per Clifford (EPG) by fitting a survival probability curve across sequence lengths.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="rb-lengths">Sequence lengths</Label>
                  <Input
                    id="rb-lengths"
                    value={rbLengths}
                    onChange={e => setRBLengths(e.target.value)}
                    placeholder="2,4,8,16,32,48,64"
                  />
                </div>
                <div>
                  <Label htmlFor="rb-seeds">Seeds per length</Label>
                  <Input
                    id="rb-seeds"
                    type="number"
                    value={rbSeeds}
                    onChange={e => setRBSeeds(parseInt(e.target.value) || 8)}
                    min={1} max={32}
                  />
                </div>
              </div>
              <Button onClick={runBenchmarking} disabled={loadingTab === "rb"} className="w-full">
                {loadingTab === "rb"
                  ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Benchmarking...</>
                  : <><BarChart3 className="w-4 h-4 mr-2" />Run RB</>}
              </Button>

              {rbResult && (
                <div className="space-y-4 mt-4">
                  <div className="grid grid-cols-3 gap-3">
                    <Card className="border-accent/20">
                      <CardContent className="pt-4 text-center">
                        <p className="text-xs text-muted-foreground">Decay rate p</p>
                        <p className="text-2xl font-bold text-primary">{rbResult.fit.p.toFixed(4)}</p>
                      </CardContent>
                    </Card>
                    <Card className="border-accent/20">
                      <CardContent className="pt-4 text-center">
                        <p className="text-xs text-muted-foreground">EPG (Error/Gate)</p>
                        <p className="text-2xl font-bold text-orange-400">{rbResult.EPG.toFixed(4)}</p>
                      </CardContent>
                    </Card>
                    <Card className="border-accent/20">
                      <CardContent className="pt-4 text-center">
                        <p className="text-xs text-muted-foreground">Fit: A·pᵐ+B</p>
                        <p className="text-lg font-bold">
                          {rbResult.fit.A.toFixed(2)}·p^m+{rbResult.fit.B.toFixed(2)}
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={rbChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="m" label={{ value: "Sequence length m", position: "insideBottomRight", offset: -5 }} />
                      <YAxis domain={[0, 100]} label={{ value: "Survival (%)", angle: -90, position: "insideLeft" }} />
                      <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
                      <Line
                        type="monotone"
                        dataKey="survival"
                        stroke="hsl(263 70% 50%)"
                        strokeWidth={2}
                        dot={{ r: 5, fill: "hsl(263 70% 50%)" }}
                        name="Survival"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Noise Calibration ────────────────────────────────────────────── */}
        <TabsContent value="cal" className="space-y-4">
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Gauge className="w-5 h-5 text-primary" />
                Bayesian Noise Calibration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Estimates posterior distributions for depolarizing, amplitude damping, and phase damping noise
                parameters using Bayesian beta posteriors with 95% credible intervals.
              </p>
              <Button onClick={runCalibration} disabled={loadingTab === "cal"} className="w-full">
                {loadingTab === "cal"
                  ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Calibrating...</>
                  : <><Gauge className="w-4 h-4 mr-2" />Run Calibration</>}
              </Button>

              {calResult && (
                <div className="space-y-3 mt-4">
                  {calTableRows.map(([param, post]) => (
                    <div key={param} className="p-3 border border-accent/20 rounded space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-sm">{param.replace(/_/g, " ")}</span>
                        <Badge variant="outline" className="text-xs">{post.gate_label}</Badge>
                      </div>
                      <div className="grid grid-cols-4 gap-2 text-xs text-center">
                        {[
                          { label: "Mean",    val: post.mean.toFixed(4) },
                          { label: "α",       val: post.alpha.toFixed(1) },
                          { label: "β",       val: post.beta.toFixed(1) },
                          { label: "95% CI",  val: `[${post.ci_low.toFixed(3)}, ${post.ci_high.toFixed(3)}]` },
                        ].map(({ label, val }) => (
                          <div key={label} className="bg-muted/20 rounded p-1">
                            <div className="text-muted-foreground">{label}</div>
                            <div className="font-mono font-semibold">{val}</div>
                          </div>
                        ))}
                      </div>
                      <Progress value={post.mean * 100} className="h-1.5" />
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Process Fidelity ─────────────────────────────────────────────── */}
        <TabsContent value="fid" className="space-y-4">
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Gauge className="w-5 h-5 text-primary" />
                Process Fidelity
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Compares ideal vs noisy gate execution across three basis states (|0⟩, |+⟩, |i+⟩)
                and reports average process fidelity.
              </p>
              <Button onClick={runFidelity} disabled={loadingTab === "fid"} className="w-full">
                {loadingTab === "fid"
                  ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Computing Fidelity...</>
                  : <><Gauge className="w-4 h-4 mr-2" />Compute Process Fidelity</>}
              </Button>

              {fidResult && (
                <div className="space-y-4 mt-4">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground mb-2">Average Process Fidelity</p>
                    <p className={`text-5xl font-bold ${fidResult.fidelity > 0.95 ? "text-green-400" : fidResult.fidelity > 0.85 ? "text-yellow-400" : "text-red-400"}`}>
                      {(fidResult.fidelity * 100).toFixed(2)}%
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">Gate: {fidResult.gate}</p>
                  </div>
                  <Progress value={fidResult.fidelity * 100} className="h-3" />

                  <div>
                    <p className="text-sm font-medium mb-2">Per-Basis Fidelities</p>
                    {["∣0⟩ basis", "∣+⟩ basis", "∣i+⟩ basis"].map((label, i) => (
                      <div key={label} className="flex justify-between items-center text-sm py-2 border-b border-accent/10">
                        <span className="font-mono text-muted-foreground">{label}</span>
                        <span className={fidResult.per_basis[i] > 0.95 ? "text-green-400 font-semibold" : "text-yellow-400 font-semibold"}>
                          {((fidResult.per_basis[i] ?? 0) * 100).toFixed(2)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
