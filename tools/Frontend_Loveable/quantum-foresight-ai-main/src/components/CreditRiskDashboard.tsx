import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  TrendingDown, DollarSign, AlertCircle, CheckCircle,
  Activity, ExternalLink, Atom, RefreshCw,
} from "lucide-react";
import {
  apiCreditRiskPresets,
  apiCreditRiskAnalyze,
  CreditRiskObligorInput,
  CreditRiskResponse,
} from "@/lib/api";

const SP_RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
  "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"];

const HORIZON_OPTIONS = [
  { value: "0.25", label: "3 months" },
  { value: "0.5",  label: "6 months" },
  { value: "1.0",  label: "1 year" },
  { value: "2.0",  label: "2 years" },
  { value: "5.0",  label: "5 years" },
];

const fmt = (n: number) =>
  n >= 1000 ? `$${(n / 1000).toFixed(1)}k` : `$${n.toFixed(0)}`;

const fmtPct = (n: number) => `${(n * 100).toFixed(4)}%`;

const FALLBACK_SOURCES = [
  {
    title: "Qiskit Finance — Credit Risk Analysis Tutorial",
    url: "https://qiskit-community.github.io/qiskit-finance/tutorials/09_credit_risk_analysis.html",
    description: "GCI model, WeightedAdder, LinearAmplitudeFunction, IQAE for portfolio loss distribution",
  },
  {
    title: "S&P Global — 2025 Annual Global Corporate Default and Rating Transition Study",
    url: "https://www.spglobal.com/ratings/en/research/articles/250328-2025-annual-global-corporate-default-and-rating-transition-study-13473585",
    description: "Source for 1-year default rates: A+ ≈ 0.03%, BB+ ≈ 0.50%",
  },
  {
    title: "Basel II IRB Approach — BIS Working Paper",
    url: "https://www.bis.org/publ/work116.pdf",
    description: "Asset correlation formula ρ(PD): systemic factor model, basis for ρ = 0.28",
  },
  {
    title: "Woerner & Egger (2019) — Quantum Risk Analysis, npj Quantum Information",
    url: "https://www.nature.com/articles/s41534-019-0130-6",
    description: "Quantum amplitude estimation for financial risk; foundational paper for QAE-based VaR/CVaR",
  },
  {
    title: "Egger et al. (2020) — Credit Risk Analysis Using Quantum Computers, IEEE Transactions on Computers",
    url: "https://ieeexplore.ieee.org/document/9259208",
    description: "Direct application of GCI model + IQAE to credit portfolios; basis for the circuit design",
  },
];

export const CreditRiskDashboard = () => {
  const [obligors, setObligors] = useState<CreditRiskObligorInput[]>([]);
  const [confidence, setConfidence] = useState(0.95);
  const [horizonYears, setHorizonYears] = useState("1.0");
  const [stressMultiplier, setStressMultiplier] = useState(1.0);
  const [useQuantum, setUseQuantum] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CreditRiskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiCreditRiskPresets()
      .then(res => setObligors(res.presets.map(p => ({ ...p }))))
      .catch(() => {
        setObligors([
          { name: "Salesforce (CRM)", ticker: "CRM", sector: "SaaS / Software", sp_rating: "A+",  loan_usd: 100000, fico_score: 780 },
          { name: "Macy's (M)",        ticker: "M",   sector: "Retail",           sp_rating: "BB+", loan_usd: 100000, fico_score: 638 },
        ]);
      });
  }, []);

  const updateObligor = (idx: number, field: keyof CreditRiskObligorInput, val: string | number) => {
    setObligors(prev => prev.map((o, i) => i === idx ? { ...o, [field]: val } : o));
  };

  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiCreditRiskAnalyze({
        obligors,
        use_presets: false,
        confidence,
        horizon_years: parseFloat(horizonYears),
        stress_multiplier: stressMultiplier,
        use_quantum: useQuantum,
      });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const sources = result?.sources?.length ? result.sources : FALLBACK_SOURCES;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <TrendingDown className="w-6 h-6 text-primary" />
        <div>
          <h2 className="text-lg font-bold">Credit Risk Analysis</h2>
          <p className="text-xs text-muted-foreground">
            Gaussian Conditional Independence model · Iterative Quantum Amplitude Estimation · Monte Carlo fallback
          </p>
        </div>
        <Badge variant="outline" className="ml-auto border-primary/30 text-primary text-xs">
          <Atom className="w-3 h-3 mr-1" />Quantum-GCI
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ── Left panel: obligors + settings ────────────────────────────── */}
        <div className="lg:col-span-1 space-y-4">

          {/* Obligor cards */}
          {obligors.map((ob, idx) => (
            <Card key={idx} className="border-accent/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <span className="font-mono font-bold">{ob.ticker}</span>
                  <span className="text-muted-foreground font-normal truncate">{ob.name}</span>
                  <Badge variant="outline" className="ml-auto text-xs border-accent/30">{ob.sector}</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <Label className="text-xs">S&P Rating</Label>
                    <Select
                      value={ob.sp_rating}
                      onValueChange={v => updateObligor(idx, "sp_rating", v)}
                    >
                      <SelectTrigger className="h-7 text-xs mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="max-h-48 overflow-y-auto">
                        {SP_RATINGS.map(r => (
                          <SelectItem key={r} value={r} className="text-xs">{r}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-xs">FICO Score</Label>
                    <div className="flex items-center gap-1 mt-1">
                      <Badge variant="outline" className="h-7 px-2 text-xs font-mono w-full justify-center">
                        {ob.fico_score ?? 700}
                      </Badge>
                    </div>
                  </div>
                </div>
                <div>
                  <Label className="text-xs">Loan Exposure (USD)</Label>
                  <Input
                    type="number"
                    value={ob.loan_usd}
                    onChange={e => updateObligor(idx, "loan_usd", parseFloat(e.target.value) || 0)}
                    className="h-7 text-xs font-mono mt-1"
                    min={1000}
                    step={10000}
                  />
                </div>
              </CardContent>
            </Card>
          ))}

          {/* Settings card */}
          <Card className="border-accent/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Activity className="w-4 h-4 text-primary" />
                Analysis Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <Label className="text-xs">Confidence Level</Label>
                  <span className="text-xs font-mono text-primary">{(confidence * 100).toFixed(0)}%</span>
                </div>
                <Slider
                  min={80} max={99} step={1}
                  value={[confidence * 100]}
                  onValueChange={([v]) => setConfidence(v / 100)}
                  className="mt-1"
                />
              </div>

              <div>
                <Label className="text-xs">Time Horizon</Label>
                <Select value={horizonYears} onValueChange={setHorizonYears}>
                  <SelectTrigger className="h-7 text-xs mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {HORIZON_OPTIONS.map(o => (
                      <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <Label className="text-xs">Stress Multiplier</Label>
                  <span className="text-xs font-mono text-primary">{stressMultiplier.toFixed(1)}×</span>
                </div>
                <Slider
                  min={5} max={30} step={1}
                  value={[stressMultiplier * 10]}
                  onValueChange={([v]) => setStressMultiplier(v / 10)}
                  className="mt-1"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Scales PD upward — simulates stress scenarios
                </p>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-xs">Quantum IQAE</Label>
                  <p className="text-xs text-muted-foreground">Use Qiskit amplitude estimation</p>
                </div>
                <Switch checked={useQuantum} onCheckedChange={setUseQuantum} />
              </div>
            </CardContent>
          </Card>

          <Button
            onClick={runAnalysis}
            disabled={loading || obligors.length === 0}
            className="w-full"
          >
            {loading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Running…</>
              : <><TrendingDown className="w-4 h-4 mr-2" />Run Credit Risk Analysis</>
            }
          </Button>

          {error && (
            <div className="flex items-start gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded p-3">
              <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              {error}
            </div>
          )}
        </div>

        {/* ── Right panel: results ─────────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-4">
          {!result ? (
            <Card className="border-accent/20 h-64 flex items-center justify-center">
              <div className="text-center text-muted-foreground">
                <TrendingDown className="w-10 h-10 mx-auto mb-3 opacity-20" />
                <p className="text-sm">Configure obligors and click Run to see results</p>
                <p className="text-xs mt-1 opacity-60">Salesforce (CRM, A+) and Macy's (M, BB+) are pre-loaded</p>
              </div>
            </Card>
          ) : (
            <>
              {/* Risk metric cards */}
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: "Expected Loss", value: result.mc.expected_loss_usd, sub: "Monte Carlo EL", color: "text-blue-400" },
                  { label: `VaR ${(result.confidence * 100).toFixed(0)}%`, value: result.mc.var_usd, sub: `${(result.horizon_years * 12).toFixed(0)}-month horizon`, color: "text-amber-400" },
                  { label: `CVaR ${(result.confidence * 100).toFixed(0)}%`, value: result.mc.cvar_usd, sub: "Expected shortfall", color: "text-red-400" },
                ].map(({ label, value, sub, color }) => (
                  <Card key={label} className="border-accent/20">
                    <CardContent className="pt-4 pb-3">
                      <p className="text-xs text-muted-foreground">{label}</p>
                      <p className={`text-xl font-bold font-mono mt-1 ${color}`}>{fmt(value)}</p>
                      <p className="text-xs text-muted-foreground mt-1">{sub}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* Quantum comparison row */}
              {result.quantum.used && result.quantum.el_usd != null && (
                <Card className="border-primary/20 bg-primary/5">
                  <CardContent className="pt-3 pb-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Atom className="w-4 h-4 text-primary" />
                      <span className="text-xs font-medium text-primary">Quantum IQAE Estimates</span>
                      <Badge className="ml-auto text-xs bg-primary/20 text-primary border-primary/30">
                        {result.quantum.circuit_info
                          ? `${(result.quantum.circuit_info as Record<string, unknown>).num_qubits ?? "?"} qubits`
                          : "GCI circuit"}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-xs">
                      <div>
                        <span className="text-muted-foreground">Quantum EL: </span>
                        <span className="font-mono font-semibold">{fmt(result.quantum.el_usd)}</span>
                        {result.mc.expected_loss_usd > 0 && (
                          <span className="text-muted-foreground ml-1">
                            ({((result.quantum.el_usd / result.mc.expected_loss_usd - 1) * 100).toFixed(1)}% vs MC)
                          </span>
                        )}
                      </div>
                      {result.quantum.cvar_usd != null && (
                        <div>
                          <span className="text-muted-foreground">Quantum CVaR: </span>
                          <span className="font-mono font-semibold">{fmt(result.quantum.cvar_usd)}</span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
              {result.quantum.error && (
                <div className="flex items-start gap-2 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded p-3">
                  <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                  Quantum path unavailable — {result.quantum.error}. Showing Monte Carlo only.
                </div>
              )}

              {/* Loss distribution histogram */}
              {result.histogram.length > 0 && (
                <Card className="border-accent/20">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <BarChart className="w-4 h-4 text-primary" />
                      Portfolio Loss Distribution
                      <span className="text-xs text-muted-foreground font-normal ml-auto">
                        {result.mc.paths.toLocaleString()} MC paths
                      </span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={180}>
                      <BarChart data={result.histogram} margin={{ top: 4, right: 8, left: 0, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis
                          dataKey="label"
                          tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                          angle={-30}
                          textAnchor="end"
                          interval={1}
                        />
                        <YAxis
                          tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                          tickFormatter={v => `${(v * 100).toFixed(1)}%`}
                        />
                        <Tooltip
                          formatter={(v: number) => [`${(v * 100).toFixed(3)}%`, "Probability"]}
                          labelFormatter={(l: string) => `Loss: ${l}`}
                          contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", fontSize: 11 }}
                        />
                        <Bar dataKey="probability" fill="hsl(var(--primary))" opacity={0.8} radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* Per-obligor table + percentile table side by side */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="border-accent/20">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <DollarSign className="w-4 h-4 text-primary" />
                      Obligor Detail
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-muted-foreground border-b border-border/40">
                          <th className="text-left pb-1">Borrower</th>
                          <th className="text-right pb-1">PD adj.</th>
                          <th className="text-right pb-1">LGD</th>
                          <th className="text-right pb-1">EL</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.obligors.map((ob, i) => (
                          <tr key={i} className="border-b border-border/20">
                            <td className="py-1.5">
                              <div className="font-medium">{ob.ticker}</div>
                              <div className="text-muted-foreground">{ob.sp_rating}</div>
                            </td>
                            <td className="text-right font-mono">{fmtPct(ob.pd_adj_pct / 100)}</td>
                            <td className="text-right font-mono">{(ob.lgd_pct * 100).toFixed(0)}%</td>
                            <td className="text-right font-mono text-amber-400">{fmt(ob.el_own_usd)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="mt-3 pt-2 border-t border-border/40 flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">P(both default)</span>
                      <Badge variant="outline" className="text-xs font-mono border-red-500/30 text-red-400">
                        {(result.multi_default_prob * 100).toFixed(4)}%
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-accent/20">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Activity className="w-4 h-4 text-primary" />
                      Loss Percentiles
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-muted-foreground border-b border-border/40">
                          <th className="text-left pb-1">Percentile</th>
                          <th className="text-right pb-1">Portfolio Loss</th>
                          <th className="text-right pb-1">% of Exposure</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.percentile_table.map((row, i) => (
                          <tr key={i} className="border-b border-border/20">
                            <td className="py-1.5 font-medium">{row.label}</td>
                            <td className="text-right font-mono">{fmt(row.loss_usd)}</td>
                            <td className="text-right font-mono text-muted-foreground">
                              {result.total_exposure_usd > 0
                                ? `${(row.loss_usd / result.total_exposure_usd * 100).toFixed(2)}%`
                                : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="mt-3 pt-2 border-t border-border/40 flex items-center justify-between text-xs text-muted-foreground">
                      <span>Total exposure</span>
                      <span className="font-mono font-medium">{fmt(result.total_exposure_usd)}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Model status badges */}
              <div className="flex flex-wrap gap-2 text-xs">
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  <CheckCircle className="w-3 h-3 text-green-400" />
                  Monte Carlo ({result.mc.paths.toLocaleString()} paths)
                </div>
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  {result.quantum.used
                    ? <CheckCircle className="w-3 h-3 text-green-400" />
                    : <AlertCircle className="w-3 h-3 text-amber-400" />}
                  {result.quantum.used ? "Quantum IQAE active" : `Quantum not available${result.quantum.available ? "" : " (qiskit-finance not installed)"}`}
                </div>
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  Stress: {result.stress_multiplier.toFixed(1)}× · Horizon: {result.horizon_years}y · Conf: {(result.confidence * 100).toFixed(0)}%
                </div>
              </div>
            </>
          )}

          {/* Sources section — always visible */}
          <Card className="border-accent/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <ExternalLink className="w-4 h-4 text-primary" />
                Sources &amp; Methodology
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                {sources.map((src, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-primary font-mono text-xs mt-0.5 shrink-0">[{i + 1}]</span>
                    <div>
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs font-medium text-primary hover:underline flex items-center gap-1"
                      >
                        {src.title}
                        <ExternalLink className="w-2.5 h-2.5 opacity-60" />
                      </a>
                      <p className="text-xs text-muted-foreground mt-0.5">{src.description}</p>
                    </div>
                  </li>
                ))}
              </ul>
              <p className="text-xs text-muted-foreground/60 mt-4 pt-3 border-t border-border/40">
                Default probabilities from S&amp;P 2025 Annual Default Study. Correlations from Basel II IRB formula.
                Quantum path uses Qiskit Aer statevector simulator with GCI model circuits.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
