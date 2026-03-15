import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { apiFinancialAnalyze, FinancialAnalyzeResponse } from "@/lib/api";
import { useAppContext } from "@/contexts/AppContext";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer,
  AreaChart, Area, Tooltip
} from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, Shield, DollarSign, BarChart3, AlertCircle, Users, Atom, RefreshCw } from "lucide-react";
import { post } from "@/lib/api";

// ── Persona view helpers ──────────────────────────────────────────────────────
const PERSONAS = ["Chief Investment Officer", "Risk Officer", "Quant Researcher", "Client-Friendly Summary"] as const;
type PersonaKey = typeof PERSONAS[number];

function regimePhrase(regime: string): string {
  const r = regime.toLowerCase();
  if (r.includes("bull"))    return "a bullish / risk-on environment.";
  if (r.includes("bear"))    return "a bearish / risk-off environment.";
  if (r.includes("sideways"))return "a sideways or range-bound environment.";
  if (r.includes("low"))     return "a relatively calm, low-volatility environment.";
  if (r.includes("high") || r.includes("vol")) return "a high-volatility or uncertain environment.";
  return "a mixed or uncertain environment.";
}

interface PersonaViewProps {
  persona: PersonaKey;
  data: FinancialAnalyzeResponse;
  portfolioValue: number;
  confidence: number;
  lookbackDays: number;
  simulations: number;
}

function PersonaView({ persona, data, portfolioValue, confidence, lookbackDays, simulations }: PersonaViewProps) {
  const dollarVar  = data.var_mc  * portfolioValue;
  const dollarCVaR = data.cvar_mc * portfolioValue;
  const stressFactor = data.sentiment_multiplier ?? null;
  const stressedDollarVar  = data.var_mc_stressed  != null ? data.var_mc_stressed  * portfolioValue : null;
  const stressedDollarCVaR = data.cvar_mc_stressed != null ? data.cvar_mc_stressed * portfolioValue : null;
  const regimeSentence = regimePhrase(data.regime);

  const fmt$ = (v: number) => `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  const fmtPct = (v: number) => `${(v * 100).toFixed(4)}`;

  if (persona === "Chief Investment Officer") {
    return (
      <div className="space-y-3 text-sm">
        <p className="font-semibold text-primary text-base">CIO Lens</p>
        <ul className="space-y-1.5 list-disc list-inside text-muted-foreground leading-relaxed">
          <li>Market regime: <span className="font-medium text-foreground">{data.regime}</span> — {regimeSentence}</li>
          <li>Horizon: <span className="font-medium text-foreground">{lookbackDays} days</span>, α = <span className="font-medium text-foreground">{(confidence * 100).toFixed(0)}%</span></li>
          <li>Dollar VaR ≈ <span className="font-semibold text-red-400">{fmt$(dollarVar)}</span></li>
          <li>Dollar CVaR ≈ <span className="font-semibold text-red-500">{fmt$(dollarCVaR)}</span></li>
          {stressFactor != null ? (
            <li>Macro stress is active — factor <span className="font-medium text-orange-400">{stressFactor.toFixed(2)}</span>
              {stressedDollarVar != null && <> → stressed VaR ≈ <span className="text-orange-400">{fmt$(stressedDollarVar)}</span>, stressed CVaR ≈ <span className="text-orange-400">{stressedDollarCVaR != null ? fmt$(stressedDollarCVaR) : "—"}</span></>}
            </li>
          ) : (
            <li>Macro stress is not applied.</li>
          )}
          {data.use_qae
            ? <li>Method: <span className="text-primary font-medium">Quantum Amplitude Estimation</span> (sample-efficient tail estimation).</li>
            : <li>Method: classical Monte Carlo ({simulations.toLocaleString()} paths).</li>
          }
        </ul>
      </div>
    );
  }

  if (persona === "Risk Officer") {
    return (
      <div className="space-y-3 text-sm">
        <p className="font-semibold text-primary text-base">Risk Officer Lens</p>
        <ul className="space-y-1.5 list-disc list-inside text-muted-foreground leading-relaxed">
          <li>Regime: <span className="font-medium text-foreground">{data.regime}</span> — {regimeSentence}</li>
          <li>Dollar VaR ≈ <span className="font-semibold text-red-400">{fmt$(dollarVar)}</span></li>
          <li>Dollar CVaR ≈ <span className="font-semibold text-red-500">{fmt$(dollarCVaR)}</span></li>
          <li>Sentiment multiplier: <span className="font-medium text-foreground">{stressFactor != null ? stressFactor.toFixed(2) : "not applied"}</span></li>
          {stressFactor != null ? (
            <li>Macro stress factor: <span className="font-medium text-orange-400">{stressFactor.toFixed(2)}×</span></li>
          ) : (
            <li>Macro stress disabled.</li>
          )}
        </ul>
      </div>
    );
  }

  if (persona === "Quant Researcher") {
    return (
      <div className="space-y-3 text-sm font-mono">
        <p className="font-semibold text-primary text-base not-italic" style={{ fontFamily: "inherit" }}>Quant Research Lens</p>
        <ul className="space-y-1.5 list-disc list-inside text-muted-foreground leading-relaxed">
          <li>VaR(return) = <span className="text-red-400">{fmtPct(data.var_mc)}%</span>, CVaR(return) = <span className="text-red-500">{fmtPct(data.cvar_mc)}%</span></li>
          <li>Annualized σ = <span className="text-foreground">{(data.annualized_volatility * 100).toFixed(2)}%</span></li>
          <li>Sim paths: <span className="text-foreground">{simulations.toLocaleString()}</span></li>
          {stressFactor != null ? (
            <li>Macro stress multiplier = <span className="text-orange-400">{stressFactor.toFixed(4)}</span>
              {data.var_mc_stressed != null && <> → stressed VaR = <span className="text-orange-400">{fmtPct(data.var_mc_stressed)}%</span>, stressed CVaR = <span className="text-orange-400">{data.cvar_mc_stressed != null ? fmtPct(data.cvar_mc_stressed) : "—"}%</span></>}
            </li>
          ) : (
            <li>No macro overlay.</li>
          )}
          {data.use_qae
            ? <li>QAE: <span className="text-primary">on</span>{data.qae_tail_prob != null && <> — tail-prob proxy = <span className="text-primary">{(data.qae_tail_prob * 100).toFixed(2)}%</span></>}.</li>
            : <li>QAE: off — classical MC ({simulations.toLocaleString()} paths).</li>
          }
          <li>Sharpe = <span className="text-primary">{data.sharpe.toFixed(4)}</span>, Sortino = <span className="text-accent">{data.sortino.toFixed(4)}</span>, Max DD = <span className="text-orange-400">{(data.max_drawdown * 100).toFixed(2)}%</span></li>
        </ul>
      </div>
    );
  }

  // Client-Friendly Summary
  return (
    <div className="space-y-3 text-sm">
      <p className="font-semibold text-primary text-base">Client-Friendly Summary</p>
      <ul className="space-y-1.5 list-disc list-inside text-muted-foreground leading-relaxed">
        <li>Market looks <span className="font-medium text-foreground">{data.regime}</span> — {regimeSentence}</li>
        <li>Portfolio size: <span className="font-semibold text-foreground">{fmt$(portfolioValue)}</span></li>
        <li>Typical "bad case" short-horizon loss estimate: <span className="font-semibold text-red-400">{fmt$(dollarVar)}</span></li>
        <li>"Very bad tail" average loss estimate: <span className="font-semibold text-red-500">{fmt$(dollarCVaR)}</span></li>
        {stressedDollarVar != null && stressedDollarCVaR != null ? (
          <li>Under macro stress, losses rise to <span className="text-orange-400">~{fmt$(stressedDollarVar)}–{fmt$(stressedDollarCVaR)}</span>.</li>
        ) : (
          <li>Macro stress not applied.</li>
        )}
      </ul>
    </div>
  );
}

// ── Correlation matrix helpers ────────────────────────────────────────────────
function computeCorrelationMatrix(returns: Record<string, number[]>, tickers: string[]): number[][] {
  const means = tickers.map(t => {
    const r = returns[t] ?? [];
    return r.reduce((a, b) => a + b, 0) / (r.length || 1);
  });
  const stds = tickers.map((t, i) => {
    const r = returns[t] ?? [];
    const mean = means[i];
    return Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / (r.length || 1));
  });
  return tickers.map((ti, i) => tickers.map((tj, j) => {
    if (i === j) return 1;
    const ri = returns[ti] ?? [], rj = returns[tj] ?? [];
    const n = Math.min(ri.length, rj.length);
    if (n === 0 || stds[i] === 0 || stds[j] === 0) return 0;
    const cov = ri.slice(0, n).reduce((s, _, k) => s + (ri[k] - means[i]) * (rj[k] - means[j]), 0) / n;
    return cov / (stds[i] * stds[j]);
  }));
}

function corrColor(v: number): string {
  // +1 → purple, 0 → neutral, -1 → red
  if (v >= 0) {
    const t = v;
    return `rgba(139,92,246,${0.15 + t * 0.75})`;   // hsl(263) purple
  }
  const t = -v;
  return `rgba(239,68,68,${0.15 + t * 0.75})`;       // red
}

function CorrelationMatrix({ returns, tickers }: { returns: Record<string, number[]>; tickers: string[] }) {
  const matrix = computeCorrelationMatrix(returns, tickers);
  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse w-full">
        <thead>
          <tr>
            <th className="p-2 text-muted-foreground font-medium text-right w-16"></th>
            {tickers.map(t => (
              <th key={t} className="p-2 text-center font-semibold text-foreground">{t}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tickers.map((ti, i) => (
            <tr key={ti}>
              <td className="p-2 text-right font-semibold text-foreground pr-3">{ti}</td>
              {tickers.map((tj, j) => {
                const v = matrix[i][j];
                return (
                  <td
                    key={tj}
                    className="p-2 text-center font-mono rounded"
                    style={{ background: corrColor(v) }}
                    title={`${ti} × ${tj}: ${v.toFixed(4)}`}
                  >
                    {v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded" style={{ background: "rgba(139,92,246,0.9)" }} />
          Strong positive (+1)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded" style={{ background: "rgba(139,92,246,0.15)" }} />
          Weak positive
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded" style={{ background: "rgba(239,68,68,0.9)" }} />
          Strong negative (−1)
        </span>
      </div>
    </div>
  );
}

export const FinancialDashboard = () => {
  const { state } = useAppContext();
  const portfolioValue = state.finance.portfolio_value;

  const [tickers, setTickers] = useState("SPY,QQQ,AAPL");
  const [lookbackDays, setLookbackDays] = useState(365);
  const [confidence, setConfidence] = useState(0.95);
  const [simulations, setSimulations] = useState(50000);
  const [demoMode, setDemoMode] = useState(false);
  const [useQAE, setUseQAE] = useState(false);
  const [sentimentMult, setSentimentMult] = useState("");
  const [applyMacroStress, setApplyMacroStress] = useState(false);
  const [unemployment, setUnemployment] = useState(4.0);
  const [yield10y, setYield10y] = useState(4.0);
  const [fredLoading, setFredLoading] = useState(false);
  const [fredError, setFredError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState<FinancialAnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedPersona, setSelectedPersona] = useState<PersonaKey>("Chief Investment Officer");

  const macroStressMultiplier = (() => {
    const stress = 1.0 + 0.40 * ((unemployment - 4.0) / 4.0) + 0.20 * ((yield10y - 4.0) / 4.0);
    return Math.min(1.8, Math.max(0.8, stress));
  })();

  const fetchFredMacro = async () => {
    setFredError(null);
    try {
      const saved = localStorage.getItem("lachesis_admin_keys");
      const fredKey = saved ? (JSON.parse(saved) as Record<string, string>)["fred"] ?? "" : "";
      if (!fredKey.trim()) {
        setFredError("No FRED API key found — add it in the Admin tab first.");
        return;
      }
      setFredLoading(true);
      const res = await post<{ cpi: number | null; unemployment: number | null; yield_10y: number | null }>(
        "/api/fred/macro", { api_key: fredKey }
      );
      if (res.unemployment != null) setUnemployment(res.unemployment);
      if (res.yield_10y   != null) setYield10y(res.yield_10y);
    } catch (e) {
      setFredError(e instanceof Error ? e.message : "FRED fetch failed");
    } finally {
      setFredLoading(false);
    }
  };

  const analyzeMarket = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const tickerList = tickers.split(',').map(t => t.trim()).filter(Boolean);
      const manualMult = sentimentMult.trim() !== "" ? parseFloat(sentimentMult) : null;
      const mult = applyMacroStress ? macroStressMultiplier : (isNaN(manualMult as number) ? null : manualMult);
      const res = await apiFinancialAnalyze({
        tickers: tickerList,
        lookback_days: lookbackDays,
        confidence,
        simulations,
        demo_mode: demoMode,
        sentiment_multiplier: mult,
        use_qae: useQAE,
      });
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsLoading(false);
    }
  };

  const getPriceChartData = () => {
    if (!data) return [];
    return data.dates.map((date, i) => {
      const entry: Record<string, string | number> = { date };
      data.tickers.forEach(ticker => {
        entry[ticker] = data.prices[ticker]?.[i] ?? 0;
      });
      return entry;
    });
  };

  const getReturnsChartData = () => {
    if (!data) return [];
    let cumulative = 0;
    return data.portfolio_returns.map((ret, i) => {
      cumulative += ret;
      return { index: i, return: ret * 100, cumulative: cumulative * 100 };
    });
  };

  const getRegimeColor = (regime: string) => {
    if (regime === 'Low Volatility') return 'bg-green-500/20 text-green-400 border-green-500/30';
    if (regime === 'Medium Volatility') return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    if (regime === 'High Volatility') return 'bg-red-500/20 text-red-400 border-red-500/30';
    return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  };

  const priceData = getPriceChartData();
  const returnsData = getReturnsChartData();

  return (
    <div className="space-y-6">
      {/* Configuration */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-primary" />
            Financial Analytics Configuration
            {data && (
              <Badge variant="outline" className="ml-auto border-green-500/30 text-green-400 text-xs">
                {data.data_source === "yfinance" ? "Live yfinance" : "Synthetic Data"}
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="tickers">Tickers (comma-separated)</Label>
              <Input
                id="tickers"
                value={tickers}
                onChange={(e) => setTickers(e.target.value)}
                placeholder="SPY,QQQ,AAPL"
              />
            </div>
            <div>
              <Label htmlFor="lookback">Lookback Days</Label>
              <Input
                id="lookback"
                type="number"
                value={lookbackDays}
                onChange={(e) => setLookbackDays(parseInt(e.target.value) || 365)}
                min="60"
                max="2000"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label>Confidence Level: {(confidence * 100).toFixed(0)}%</Label>
              <Slider
                value={[confidence]}
                onValueChange={([value]) => setConfidence(value)}
                min={0.8} max={0.99} step={0.01}
                className="mt-2"
              />
            </div>
            <div>
              <Label htmlFor="sims">Monte Carlo Simulations</Label>
              <Input
                id="sims"
                type="number"
                value={simulations}
                onChange={(e) => setSimulations(parseInt(e.target.value) || 50000)}
                min="1000" max="200000" step="1000"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end">
            <div className="space-y-2 pt-2">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="demo"
                  checked={demoMode}
                  onCheckedChange={(checked) => setDemoMode(!!checked)}
                />
                <Label htmlFor="demo">Demo Mode (Synthetic Data) — uncheck for live yfinance</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="use-qae"
                  checked={useQAE}
                  onCheckedChange={(checked) => setUseQAE(!!checked)}
                />
                <Label htmlFor="use-qae" className="text-xs">
                  Quantum Amplitude Estimation (QAE)
                  <span className="text-muted-foreground ml-1">— requires qiskit-finance</span>
                </Label>
              </div>
            </div>
            <div>
              <Label htmlFor="sent-mult" className="text-xs">
                Sentiment Multiplier{" "}
                <span className="text-muted-foreground">(optional — overridden by Macro Stress)</span>
              </Label>
              <Input
                id="sent-mult"
                type="number"
                step={0.01}
                min={0.5} max={2.0}
                value={sentimentMult}
                onChange={e => setSentimentMult(e.target.value)}
                placeholder="e.g. 1.12 — stresses VaR"
                className="mt-1 h-8 text-xs"
                disabled={applyMacroStress}
              />
            </div>
          </div>

          {/* ── Macro Stress ──────────────────────────────────────────── */}
          <div className="border border-accent/30 rounded-lg p-4 space-y-4 bg-background/40">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="macro-stress"
                checked={applyMacroStress}
                onCheckedChange={(checked) => setApplyMacroStress(!!checked)}
              />
              <Label htmlFor="macro-stress" className="font-semibold cursor-pointer">
                Apply Macro Stress
              </Label>
              <span className="text-xs text-muted-foreground ml-1">
                — scales VaR/CVaR using unemployment & yield inputs
              </span>
            </div>

            {applyMacroStress && (
              <div className="space-y-4 pt-1">
                <div className="flex items-center gap-3">
                  <Button
                    size="sm" variant="outline"
                    className="h-7 text-xs border-primary/30 text-primary hover:bg-primary/10"
                    onClick={fetchFredMacro}
                    disabled={fredLoading}
                  >
                    <RefreshCw className={`w-3 h-3 mr-1 ${fredLoading ? "animate-spin" : ""}`} />
                    {fredLoading ? "Fetching..." : "Fetch Live from FRED"}
                  </Button>
                  <span className="text-xs text-muted-foreground">
                    Auto-fills unemployment & yield from your FRED key (Admin tab)
                  </span>
                </div>
                {fredError && (
                  <p className="text-xs text-red-400 flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" />{fredError}
                  </p>
                )}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label className="text-xs">
                      Unemployment Rate: <span className="text-primary font-semibold">{unemployment.toFixed(1)}%</span>
                    </Label>
                    <Slider
                      value={[unemployment]}
                      onValueChange={([v]) => setUnemployment(v)}
                      min={2.0} max={12.0} step={0.1}
                      className="mt-2"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>2%</span><span>Neutral: 4%</span><span>12%</span>
                    </div>
                  </div>
                  <div>
                    <Label className="text-xs">
                      10Y Treasury Yield: <span className="text-primary font-semibold">{yield10y.toFixed(1)}%</span>
                    </Label>
                    <Slider
                      value={[yield10y]}
                      onValueChange={([v]) => setYield10y(v)}
                      min={0.5} max={10.0} step={0.1}
                      className="mt-2"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>0.5%</span><span>Neutral: 4%</span><span>10%</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3 pt-1">
                  <div className={`px-3 py-1.5 rounded-md text-sm font-semibold border ${
                    macroStressMultiplier > 1.1 ? "bg-red-500/15 border-red-500/30 text-red-400"
                    : macroStressMultiplier < 0.95 ? "bg-green-500/15 border-green-500/30 text-green-400"
                    : "bg-yellow-500/15 border-yellow-500/30 text-yellow-400"
                  }`}>
                    Stress Multiplier: {macroStressMultiplier.toFixed(3)}×
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {macroStressMultiplier > 1.1
                      ? "Elevated macro risk — VaR/CVaR scaled up"
                      : macroStressMultiplier < 0.95
                      ? "Benign macro conditions — VaR/CVaR scaled down"
                      : "Near-neutral macro environment"}
                  </span>
                </div>
              </div>
            )}
          </div>

          <Button onClick={analyzeMarket} disabled={isLoading} className="w-full h-12">
            {isLoading ? (
              <><BarChart3 className="w-4 h-4 mr-2 animate-spin" />Analyzing Market (Python backend)...</>
            ) : (
              <><TrendingUp className="w-4 h-4 mr-2" />Fetch & Analyze Market Data</>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Error */}
      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{error}</span>
          </CardContent>
        </Card>
      )}

      {/* Risk Metrics */}
      {data && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <Card className="border-accent/20">
            <CardContent className="pt-4">
              <p className="text-xs font-medium text-muted-foreground">VaR (MC)</p>
              <p className="text-xl font-bold text-red-400">{(data.var_mc * 100).toFixed(2)}%</p>
              {data.var_mc_stressed != null && (
                <p className="text-xs text-orange-400 mt-0.5">
                  Stressed: {(data.var_mc_stressed * 100).toFixed(2)}%
                </p>
              )}
              <TrendingDown className="w-5 h-5 text-red-400 mt-1" />
            </CardContent>
          </Card>
          <Card className="border-accent/20">
            <CardContent className="pt-4">
              <p className="text-xs font-medium text-muted-foreground">CVaR (MC)</p>
              <p className="text-xl font-bold text-red-500">{(data.cvar_mc * 100).toFixed(2)}%</p>
              {data.cvar_mc_stressed != null && (
                <p className="text-xs text-orange-400 mt-0.5">
                  Stressed: {(data.cvar_mc_stressed * 100).toFixed(2)}%
                </p>
              )}
              <AlertTriangle className="w-5 h-5 text-red-500 mt-1" />
            </CardContent>
          </Card>
          <Card className="border-accent/20">
            <CardContent className="pt-4">
              <p className="text-xs font-medium text-muted-foreground">VaR (Hist)</p>
              <p className="text-xl font-bold text-orange-400">{(data.var_historical * 100).toFixed(2)}%</p>
              <TrendingDown className="w-5 h-5 text-orange-400 mt-1" />
            </CardContent>
          </Card>
          <Card className="border-accent/20">
            <CardContent className="pt-4">
              <p className="text-xs font-medium text-muted-foreground">Sharpe</p>
              <p className="text-xl font-bold text-primary">{data.sharpe.toFixed(3)}</p>
              <Shield className="w-5 h-5 text-primary mt-1" />
            </CardContent>
          </Card>
          <Card className="border-accent/20">
            <CardContent className="pt-4">
              <p className="text-xs font-medium text-muted-foreground">Sortino</p>
              <p className="text-xl font-bold text-accent">{data.sortino.toFixed(3)}</p>
              <Shield className="w-5 h-5 text-accent mt-1" />
            </CardContent>
          </Card>
          <Card className="border-accent/20">
            <CardContent className="pt-4">
              <p className="text-xs font-medium text-muted-foreground">Max Drawdown</p>
              <p className="text-xl font-bold text-orange-400">{(data.max_drawdown * 100).toFixed(2)}%</p>
              <TrendingDown className="w-5 h-5 text-orange-400 mt-1" />
            </CardContent>
          </Card>
        </div>
      )}

      {/* Market Regime */}
      {data && (
        <Card className="border-accent/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Market Regime Analysis
              {data.qae_active && (
                <Badge className="ml-auto bg-primary/20 text-primary border-primary/30 text-xs">
                  <Atom className="w-3 h-3 mr-1" />QAE Active
                </Badge>
              )}
              {data.use_qae && !data.qae_active && (
                <Badge variant="outline" className="ml-auto border-yellow-500/30 text-yellow-400 text-xs">
                  QAE unavailable — used MC
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4 flex-wrap">
              <Badge className={`px-4 py-2 ${getRegimeColor(data.regime)}`}>
                {data.regime}
              </Badge>
              <span className="text-sm text-muted-foreground">
                Annualized Volatility: {(data.annualized_volatility * 100).toFixed(2)}%
              </span>
              {data.use_qae && data.qae_tail_prob != null && (
                <span className="text-sm text-primary">
                  QAE tail-prob proxy: {(data.qae_tail_prob * 100).toFixed(2)}%
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Charts */}
      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="border-accent/20">
            <CardHeader><CardTitle>Price Performance</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={priceData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="date" className="text-muted-foreground" tick={false} />
                  <YAxis className="text-muted-foreground" />
                  <Tooltip />
                  {data.tickers.map((ticker, index) => (
                    <Line
                      key={ticker}
                      type="monotone"
                      dataKey={ticker}
                      stroke={`hsl(${263 + index * 50} 70% 50%)`}
                      strokeWidth={2}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="border-accent/20">
            <CardHeader><CardTitle>Cumulative Portfolio Returns</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={returnsData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="index" className="text-muted-foreground" tick={false} />
                  <YAxis className="text-muted-foreground" />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="cumulative"
                    stroke="hsl(263 70% 50%)"
                    fill="hsl(263 70% 50% / 0.2)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ── Correlation Matrix ───────────────────────────────────────────────── */}
      {data && data.tickers.length > 1 && (
        <Card className="border-accent/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-primary" />
              Return Correlation Matrix
              <span className="text-sm font-normal text-muted-foreground ml-1">
                — pairwise Pearson correlation of daily returns
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <CorrelationMatrix returns={data.returns} tickers={data.tickers} />
          </CardContent>
        </Card>
      )}

      {/* ── Persona Views ────────────────────────────────────────────────────── */}
      {data && (
        <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="w-5 h-5 text-primary" />
              Persona Views
              <span className="text-sm font-normal text-muted-foreground ml-1">
                — how different roles see this risk picture
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Persona selector */}
            <div className="flex flex-wrap gap-2">
              {PERSONAS.map(p => (
                <Button
                  key={p}
                  size="sm"
                  variant={selectedPersona === p ? "default" : "outline"}
                  className="text-xs h-8"
                  onClick={() => setSelectedPersona(p)}
                >
                  {p}
                </Button>
              ))}
            </div>

            {/* Persona content */}
            <div className="border border-accent/20 rounded-lg p-4 bg-background/40">
              <PersonaView
                persona={selectedPersona}
                data={data}
                portfolioValue={portfolioValue}
                confidence={confidence}
                lookbackDays={lookbackDays}
                simulations={simulations}
              />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default FinancialDashboard;
