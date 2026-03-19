import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  apiQAOAPortfolios, apiQAOAOptimize, apiQAOASweep, apiQAOAGetScenarios,
  apiQAOASaveScenario, apiQAOAGetLog,
  QAOAOptimizeResponse, QAOASweepPoint,
} from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts';
import { Zap, TrendingUp, Save, History, AlertCircle, RefreshCw } from "lucide-react";

const BACKENDS = [
  "Classical brute-force",
  "Qiskit QAOA",
  "QAOA (Aer Sampler)",
];

const PERSONAS: { label: string; lam: number }[] = [
  { label: "Conservative", lam: 0.6 },
  { label: "Balanced",     lam: 1.0 },
  { label: "Aggressive",   lam: 1.6 },
];

const REGIMES = ["None", "Bull regime", "Bear regime", "Shock regime"];

export const QAOADashboard = () => {
  const [portfolios, setPortfolios]     = useState<string[]>([]);
  const [portfolio, setPortfolio]       = useState("Toy 3-asset tech portfolio");
  const [backend, setBackend]           = useState("Classical brute-force");
  const [depth, setDepth]               = useState(1);
  const [shots, setShots]               = useState(1024);
  const [lam, setLam]                   = useState(1.0);
  const [regime, setRegime]             = useState("None");
  const [isLoading, setIsLoading]       = useState(false);
  const [isSweeping, setIsSweeping]     = useState(false);
  const [result, setResult]             = useState<QAOAOptimizeResponse | null>(null);
  const [sweepData, setSweepData]       = useState<QAOASweepPoint[]>([]);
  const [scenarios, setScenarios]       = useState<{ name: string; timestamp?: string }[]>([]);
  const [logRows, setLogRows]           = useState<Record<string, string>[]>([]);
  const [scenarioName, setScenarioName] = useState("");
  const [error, setError]               = useState<string | null>(null);

  useEffect(() => {
    apiQAOAPortfolios().then(r => setPortfolios(r.portfolios)).catch(() => {});
    apiQAOAGetScenarios().then(r => setScenarios(r.scenarios)).catch(() => {});
    apiQAOAGetLog().then(r => setLogRows(r.rows)).catch(() => {});
  }, []);

  const runOptimize = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await apiQAOAOptimize({
        portfolio,
        depth,
        shots,
        lam,
        backend,
        regime: regime === "None" ? null : regime,
      });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsLoading(false);
    }
  };

  const runSweep = async () => {
    setIsSweeping(true);
    setError(null);
    try {
      const res = await apiQAOASweep(portfolio, 0.2, 2.0, 15);
      setSweepData(res.sweep);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSweeping(false);
    }
  };

  const saveScenario = async () => {
    if (!result || !scenarioName.trim()) return;
    try {
      await apiQAOASaveScenario(scenarioName.trim(), result as unknown as Record<string, unknown>, portfolio);
      const updated = await apiQAOAGetScenarios();
      setScenarios(updated.scenarios);
      setScenarioName("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const statusColor = (result: QAOAOptimizeResponse | null) => {
    if (!result) return "";
    return result.expected_return > 0 ? "text-green-400" : "text-red-400";
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-primary" />
            QAOA Portfolio Optimization
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              Python · Qiskit
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-base font-medium">Runs a quantum algorithm to find the best mix of assets for your portfolio.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <Label>Portfolio</Label>
              <Select value={portfolio} onValueChange={setPortfolio}>
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(portfolios.length ? portfolios : ["Toy 3-asset tech portfolio", "Lachesis benchmark (equities + bond + gold)"]).map(p => (
                    <SelectItem key={p} value={p}>{p}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Backend</Label>
              <Select value={backend} onValueChange={setBackend}>
                <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {BACKENDS.map(b => <SelectItem key={b} value={b}>{b}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Regime Overlay</Label>
              <Select value={regime} onValueChange={setRegime}>
                <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {REGIMES.map(r => <SelectItem key={r} value={r}>{r}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Depth (QAOA layers): {depth}</Label>
              <Slider value={[depth]} onValueChange={([v]) => setDepth(v)} min={1} max={5} step={1} className="mt-2" />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label>λ (risk-return tradeoff): {lam.toFixed(2)}</Label>
              <Slider value={[lam]} onValueChange={([v]) => setLam(+v.toFixed(2))} min={0.1} max={2.0} step={0.05} className="mt-2" />
              <div className="flex gap-2 mt-2">
                {PERSONAS.map(p => (
                  <Button key={p.label} size="sm" variant={lam === p.lam ? "default" : "outline"} onClick={() => setLam(p.lam)}>
                    {p.label}
                  </Button>
                ))}
              </div>
            </div>
            <div>
              <Label htmlFor="shots-qaoa">Shots</Label>
              <Input
                id="shots-qaoa"
                type="number"
                value={shots}
                onChange={e => setShots(parseInt(e.target.value) || 1024)}
                min={64} max={8192} step={128}
              />
            </div>
          </div>

          <div className="flex gap-3">
            <Button onClick={runOptimize} disabled={isLoading} className="flex-1 h-12">
              {isLoading
                ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Optimizing...</>
                : <><Zap className="w-4 h-4 mr-2" />Run QAOA Optimization</>}
            </Button>
            <Button onClick={runSweep} disabled={isSweeping} variant="outline" className="h-12">
              {isSweeping
                ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Sweeping...</>
                : <><TrendingUp className="w-4 h-4 mr-2" />λ Sweep</>}
            </Button>
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

      {(result || sweepData.length > 0) && (
        <Tabs defaultValue="result" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="result">Result</TabsTrigger>
            <TabsTrigger value="frontier">Efficient Frontier</TabsTrigger>
            <TabsTrigger value="scenarios">Scenarios</TabsTrigger>
            <TabsTrigger value="log">Run Log</TabsTrigger>
          </TabsList>

          {/* Optimization result */}
          <TabsContent value="result">
            {result && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[
                    { label: "Selected Assets",   val: result.selected_assets.join(", ") || "None" },
                    { label: "Expected Return",    val: `${(result.expected_return * 100).toFixed(2)}%` },
                    { label: "Portfolio Risk",     val: `${(result.risk * 100).toFixed(2)}%` },
                    { label: "QAOA Energy",        val: result.energy.toFixed(4) },
                  ].map(({ label, val }) => (
                    <Card key={label} className="border-accent/20">
                      <CardContent className="pt-4">
                        <p className="text-xs text-muted-foreground">{label}</p>
                        <p className={`text-lg font-bold mt-1 ${label === "Expected Return" ? statusColor(result) : ""}`}>{val}</p>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                <Card className="border-accent/20">
                  <CardHeader><CardTitle className="text-sm">Bitstring</CardTitle></CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1">
                        {result.bitstring.split("").map((bit, i) => (
                          <div
                            key={i}
                            className={`w-10 h-10 rounded flex items-center justify-center text-lg font-bold border ${
                              bit === "1"
                                ? "bg-primary/20 border-primary text-primary"
                                : "bg-muted/30 border-accent/20 text-muted-foreground"
                            }`}
                          >
                            {bit}
                          </div>
                        ))}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {result.assets.map((a, i) => (
                          <span key={a} className={result.bitstring[i] === "1" ? "text-primary font-semibold" : ""}>
                            {a}{i < result.assets.length - 1 ? ", " : ""}
                          </span>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {result.narrative && (
                  <Card className="border-accent/20">
                    <CardHeader><CardTitle className="text-sm">Lachesis Analysis</CardTitle></CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground leading-relaxed">{result.narrative}</p>
                    </CardContent>
                  </Card>
                )}

                {/* Save scenario */}
                <Card className="border-accent/20">
                  <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Save className="w-4 h-4" />Save Scenario</CardTitle></CardHeader>
                  <CardContent className="flex gap-3">
                    <Input
                      value={scenarioName}
                      onChange={e => setScenarioName(e.target.value)}
                      placeholder="Scenario name…"
                      className="flex-1"
                    />
                    <Button onClick={saveScenario} disabled={!scenarioName.trim() || !result} size="sm">
                      Save
                    </Button>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          {/* λ sweep / efficient frontier */}
          <TabsContent value="frontier">
            {sweepData.length > 0 && (
              <Card className="border-accent/20">
                <CardHeader><CardTitle>Efficient Frontier (λ sweep)</CardTitle></CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={350}>
                    <LineChart data={sweepData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="lam" label={{ value: "λ", position: "insideBottomRight", offset: -5 }} />
                      <YAxis />
                      <Tooltip formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
                      <Line type="monotone" dataKey="expected_return" stroke="#22c55e" strokeWidth={2} name="Expected Return" dot={false} />
                      <Line type="monotone" dataKey="risk"            stroke="#ef4444" strokeWidth={2} name="Risk" dot={false} />
                      <Line type="monotone" dataKey="objective"       stroke="hsl(263 70% 50%)" strokeWidth={2} name="Objective" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Saved scenarios */}
          <TabsContent value="scenarios">
            <Card className="border-accent/20">
              <CardHeader><CardTitle className="flex items-center gap-2"><Save className="w-4 h-4" />Saved Scenarios</CardTitle></CardHeader>
              <CardContent>
                {scenarios.length === 0 ? (
                  <p className="text-muted-foreground text-sm">No saved scenarios yet.</p>
                ) : (
                  <div className="space-y-2">
                    {scenarios.map((s, i) => (
                      <div key={i} className="flex items-center justify-between p-3 border border-accent/20 rounded">
                        <span className="font-medium text-sm">{s.name}</span>
                        {s.timestamp && <span className="text-xs text-muted-foreground">{s.timestamp}</span>}
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Run history */}
          <TabsContent value="log">
            <Card className="border-accent/20">
              <CardHeader><CardTitle className="flex items-center gap-2"><History className="w-4 h-4" />Run History</CardTitle></CardHeader>
              <CardContent>
                {logRows.length === 0 ? (
                  <p className="text-muted-foreground text-sm">No runs logged yet.</p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-accent/20">
                          {["Timestamp", "Backend", "λ", "Assets", "Return", "Risk"].map(h => (
                            <th key={h} className="text-left py-2 pr-4 text-muted-foreground font-medium">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {logRows.slice(-20).reverse().map((row, i) => (
                          <tr key={i} className="border-b border-accent/10">
                            <td className="py-2 pr-4">{row.timestamp}</td>
                            <td className="py-2 pr-4">{row.backend}</td>
                            <td className="py-2 pr-4">{row.lambda}</td>
                            <td className="py-2 pr-4">{(row.selected_assets || "").replace(/\|/g, ", ")}</td>
                            <td className="py-2 pr-4 text-green-400">{row.expected_return ? `${(parseFloat(row.expected_return) * 100).toFixed(2)}%` : "—"}</td>
                            <td className="py-2 pr-4 text-red-400">{row.risk ? `${(parseFloat(row.risk) * 100).toFixed(2)}%` : "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};
