/**
 * InsiderTradingDashboard.tsx
 * Two sections:
 *  1. SEC EDGAR Insider Filings (Forms 3/4/5) — real SEC data
 *     Single "Load Filings" button (ticker OR manual CIK) — matches Streamlit UX
 *  2. Portfolio Deep Analysis — per-asset stats via /api/financial/insider
 */
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import {
  AlertCircle, RefreshCw, Briefcase, TrendingUp,
  FileText, ExternalLink, Building2,
} from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import { post, apiEdgarLoadFilings, EdgarFiling, EdgarLoadResponse } from "@/lib/api";

// ─── Portfolio types ─────────────────────────────────────────────────────────
interface PerAsset {
  ticker: string;
  ann_return_pct: number;
  ann_vol_pct: number;
  sharpe: number;
  max_drawdown_pct: number;
  last_price: number;
}
interface Position { ticker: string; weight_pct: number; value_usd: number; }
interface InsiderResponse {
  tickers: string[];
  data_source: string;
  portfolio_value: number;
  var_1d_usd: number;
  cvar_1d_usd: number;
  regime: string;
  current_vol_ann_pct: number;
  per_asset: PerAsset[];
  positions: Position[];
}

function regimeColor(regime: string) {
  if (regime === "Calm")     return "text-green-400 border-green-500/30 bg-green-500/10";
  if (regime === "Moderate") return "text-yellow-400 border-yellow-500/30 bg-yellow-500/10";
  return "text-red-400 border-red-500/30 bg-red-500/10";
}

const FORM_OPTIONS = ["3", "4", "5"];

// ════════════════════════════════════════════════════════════════════════════
export const InsiderTradingDashboard = () => {
  const { state, setFinance } = useAppContext();
  const fin = state.finance;

  // ── SEC EDGAR state ─────────────────────────────────────────────────────
  const [edgarTicker,   setEdgarTicker]   = useState("AAPL");
  const [manualCIK,     setManualCIK]     = useState("");
  const [userAgent,     setUserAgent]     = useState("LachesisApp contact@lachesis.local");
  const [selectedForms, setSelectedForms] = useState<string[]>(["4"]);
  const [maxResults,    setMaxResults]    = useState(50);
  const [filings,       setFilings]       = useState<EdgarLoadResponse | null>(null);
  const [edgarLoading,  setEdgarLoading]  = useState(false);
  const [edgarError,    setEdgarError]    = useState<string | null>(null);

  // ── Portfolio analysis state ─────────────────────────────────────────────
  const [result,  setResult]  = useState<InsiderResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  // ── EDGAR: single "Load Filings" action (matches Streamlit) ─────────────
  const loadFilings = async () => {
    if (!edgarTicker.trim() && !manualCIK.trim()) {
      setEdgarError("Enter a ticker or a CIK number.");
      return;
    }
    if (selectedForms.length === 0) {
      setEdgarError("Select at least one form type (3, 4, or 5).");
      return;
    }
    if (!userAgent.trim()) {
      setEdgarError("SEC requires a User-Agent string with contact info.");
      return;
    }
    setEdgarLoading(true);
    setEdgarError(null);
    setFilings(null);
    try {
      const res = await apiEdgarLoadFilings(
        edgarTicker.trim(),
        manualCIK.trim(),
        selectedForms,
        userAgent.trim(),
        maxResults,
      );
      setFilings(res);
    } catch (e: unknown) {
      setEdgarError(e instanceof Error ? e.message : String(e));
    } finally {
      setEdgarLoading(false);
    }
  };

  const toggleForm = (f: string, checked: boolean) =>
    setSelectedForms(prev => checked ? [...prev, f] : prev.filter(x => x !== f));

  // ── Portfolio analysis ───────────────────────────────────────────────────
  const runPortfolio = async () => {
    setLoading(true); setError(null);
    try {
      const tickers = fin.tickers.split(",").map(t => t.trim()).filter(Boolean);
      const res = await post<InsiderResponse>("/api/financial/insider", {
        tickers,
        lookback_days: fin.lookback_days,
        portfolio_value: fin.portfolio_value,
        confidence: fin.confidence_level,
        simulations: Math.min(fin.mc_sims, 20000),
        demo_mode: fin.demo_mode,
      });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const sharpeData = result?.per_asset.map(a => ({
    ticker: a.ticker,
    sharpe: parseFloat(a.sharpe.toFixed(3)),
    color: a.sharpe > 1 ? "hsl(142,70%,45%)" : a.sharpe > 0 ? "hsl(48,80%,50%)" : "hsl(0,70%,50%)",
  })) ?? [];

  return (
    <div className="space-y-6">

      {/* ══ SEC EDGAR SECTION ══════════════════════════════════════════════ */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-primary" />
            SEC EDGAR — Insider Filings (Forms 3 / 4 / 5)
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              Live SEC Data
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-base font-medium mb-1">Looks up SEC filings to see when company executives bought or sold their own stock.</p>
          <p className="text-sm text-muted-foreground">
            Form 3 = initial ownership declaration. Form 4 = transaction changes (buys/sells). Form 5 = annual statement.
            Enter a ticker <span className="text-primary">or</span> a raw CIK. SEC requires a descriptive User-Agent (name + email).
          </p>

          {/* Row 1: ticker + optional CIK override */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="edgar-ticker" className="text-xs">
                Ticker Symbol <span className="text-muted-foreground">(auto-resolves to CIK)</span>
              </Label>
              <Input
                id="edgar-ticker"
                value={edgarTicker}
                onChange={e => setEdgarTicker(e.target.value.toUpperCase())}
                placeholder="AAPL"
                className="mt-1 h-8 text-xs font-mono"
              />
            </div>
            <div>
              <Label htmlFor="edgar-cik" className="text-xs">
                Manual CIK <span className="text-muted-foreground">(optional — overrides ticker lookup)</span>
              </Label>
              <Input
                id="edgar-cik"
                value={manualCIK}
                onChange={e => setManualCIK(e.target.value.replace(/\D/g, ""))}
                placeholder="0000320193"
                className="mt-1 h-8 text-xs font-mono"
              />
            </div>
          </div>

          {/* Row 2: user-agent */}
          <div>
            <Label htmlFor="edgar-ua" className="text-xs">
              User-Agent <span className="text-muted-foreground">(SEC requirement — include your name/email)</span>
            </Label>
            <Input
              id="edgar-ua"
              value={userAgent}
              onChange={e => setUserAgent(e.target.value)}
              placeholder="MyApp contact@example.com"
              className="mt-1 h-8 text-xs"
            />
          </div>

          {/* Row 3: form checkboxes + max results */}
          <div className="flex flex-wrap items-end gap-6">
            <div>
              <Label className="text-xs mb-2 block">Form Types</Label>
              <div className="flex gap-4">
                {FORM_OPTIONS.map(f => (
                  <div key={f} className="flex items-center gap-1.5">
                    <Checkbox
                      id={`form-${f}`}
                      checked={selectedForms.includes(f)}
                      onCheckedChange={v => toggleForm(f, !!v)}
                    />
                    <Label htmlFor={`form-${f}`} className="text-sm cursor-pointer">Form {f}</Label>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <Label htmlFor="edgar-max" className="text-xs">Max Results</Label>
              <Input
                id="edgar-max"
                type="number"
                value={maxResults}
                onChange={e => setMaxResults(parseInt(e.target.value) || 50)}
                min={1} max={200}
                className="mt-1 h-8 text-xs w-24"
              />
            </div>
          </div>

          {/* Single "Load Filings" button — mirrors Streamlit */}
          <Button
            onClick={loadFilings}
            disabled={edgarLoading || (!edgarTicker.trim() && !manualCIK.trim())}
            className="w-full h-11"
          >
            {edgarLoading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Loading filings from SEC EDGAR...</>
              : <><FileText className="w-4 h-4 mr-2" />Load Filings</>}
          </Button>
        </CardContent>
      </Card>

      {/* EDGAR error */}
      {edgarError && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{edgarError}</span>
          </CardContent>
        </Card>
      )}

      {/* EDGAR results */}
      {filings && (
        <Card className="border-accent/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Building2 className="w-4 h-4 text-primary" />
              {filings.company_name || filings.ticker}
              {filings.cik && (
                <span className="text-xs text-muted-foreground font-normal">
                  (CIK {parseInt(filings.cik)})
                </span>
              )}
              <Badge variant="outline" className="ml-auto text-xs">
                {filings.total_found} filing{filings.total_found !== 1 ? "s" : ""}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {filings.filings.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4 text-center">
                No filings found for the selected form types.
                Try selecting Form 3, 4, and 5 or checking the company's CIK.
              </p>
            ) : (
              <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-card">
                    <tr className="border-b border-accent/20 text-muted-foreground text-xs">
                      <th className="text-left py-2 pr-4">Date</th>
                      <th className="text-left py-2 pr-4">Form</th>
                      <th className="text-left py-2 pr-4">Description</th>
                      <th className="text-left py-2 pr-4">Accession #</th>
                      <th className="text-left py-2">Document</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filings.filings.map((f: EdgarFiling, i: number) => (
                      <tr
                        key={i}
                        className="border-b border-accent/10 hover:bg-accent/5 transition-colors"
                      >
                        <td className="py-2 pr-4 font-mono text-xs whitespace-nowrap">{f.filing_date}</td>
                        <td className="py-2 pr-4">
                          <Badge variant="outline" className="text-xs border-primary/30 text-primary">
                            Form {f.form}
                          </Badge>
                        </td>
                        <td className="py-2 pr-4 text-xs text-muted-foreground max-w-[180px] truncate">
                          {f.description || "—"}
                        </td>
                        <td className="py-2 pr-4 font-mono text-xs text-muted-foreground whitespace-nowrap">
                          {f.accession_number}
                        </td>
                        <td className="py-2">
                          <a
                            href={f.filing_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                          >
                            View <ExternalLink className="w-3 h-3" />
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ══ PORTFOLIO ANALYSIS SECTION ═══════════════════════════════════════ */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Briefcase className="w-5 h-5 text-primary" />
            Portfolio Deep Analysis
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              yfinance · Monte Carlo
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Per-asset annualised return, volatility, Sharpe, max drawdown and equal-weight position sizing.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <Label className="text-xs">Tickers</Label>
              <Input value={fin.tickers}
                onChange={e => setFinance({ ...fin, tickers: e.target.value })}
                className="mt-1 h-8 text-xs" placeholder="AAPL,MSFT,NVDA" />
            </div>
            <div>
              <Label className="text-xs">Portfolio Value ($)</Label>
              <Input type="number" step={10000} value={fin.portfolio_value}
                onChange={e => setFinance({ ...fin, portfolio_value: parseFloat(e.target.value) || 100000 })}
                className="mt-1 h-8 text-xs" />
            </div>
            <div>
              <Label className="text-xs">Lookback (days)</Label>
              <Input type="number" step={30} value={fin.lookback_days}
                onChange={e => setFinance({ ...fin, lookback_days: parseInt(e.target.value) || 252 })}
                className="mt-1 h-8 text-xs" />
            </div>
            <div className="flex items-end pb-1">
              <div className="flex items-center gap-2">
                <Switch checked={fin.demo_mode}
                  onCheckedChange={v => setFinance({ ...fin, demo_mode: v })} id="it-demo" />
                <Label htmlFor="it-demo" className="text-xs">Demo mode</Label>
              </div>
            </div>
          </div>
          <Button onClick={runPortfolio} disabled={loading} className="w-full h-10">
            {loading ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Analysing portfolio...</>
                     : <><TrendingUp className="w-4 h-4 mr-2" />Run Portfolio Analysis</>}
          </Button>
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

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Portfolio Value",  val: `$${result.portfolio_value.toLocaleString()}` },
              { label: "1-Day VaR (95%)", val: `$${result.var_1d_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, red: true },
              { label: "CVaR",            val: `$${result.cvar_1d_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, red: true },
              { label: "Ann. Volatility", val: `${result.current_vol_ann_pct.toFixed(1)}%` },
            ].map(({ label, val, red }) => (
              <div key={label} className={`p-4 border rounded-lg ${red ? "border-red-500/20 bg-red-500/5" : "border-accent/20"}`}>
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className={`text-xl font-bold font-mono mt-1 ${red ? "text-red-400" : "text-primary"}`}>{val}</p>
              </div>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <Badge className={`px-3 py-1 ${regimeColor(result.regime)}`}>{result.regime} Regime</Badge>
            <Badge variant="outline" className="text-xs">
              {result.data_source === "yfinance" ? "Live yfinance" : "Synthetic Data"}
            </Badge>
          </div>

          <Tabs defaultValue="assets">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="assets">Asset Stats</TabsTrigger>
              <TabsTrigger value="sharpe">Sharpe Chart</TabsTrigger>
              <TabsTrigger value="positions">Positions</TabsTrigger>
            </TabsList>

            <TabsContent value="assets">
              <Card className="border-accent/20">
                <CardContent className="pt-4 overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-accent/20 text-muted-foreground text-xs">
                        <th className="text-left py-2">Ticker</th>
                        <th className="text-right py-2">Ann Return</th>
                        <th className="text-right py-2">Ann Vol</th>
                        <th className="text-right py-2">Sharpe</th>
                        <th className="text-right py-2">Max DD</th>
                        <th className="text-right py-2">Last Price</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.per_asset.map(a => (
                        <tr key={a.ticker} className="border-b border-accent/10 hover:bg-accent/5">
                          <td className="py-2 font-semibold">{a.ticker}</td>
                          <td className={`py-2 text-right font-mono ${a.ann_return_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
                            {a.ann_return_pct >= 0 ? "+" : ""}{a.ann_return_pct.toFixed(2)}%
                          </td>
                          <td className="py-2 text-right font-mono">{a.ann_vol_pct.toFixed(2)}%</td>
                          <td className={`py-2 text-right font-mono ${a.sharpe > 1 ? "text-green-400" : a.sharpe > 0 ? "text-yellow-400" : "text-red-400"}`}>
                            {a.sharpe.toFixed(3)}
                          </td>
                          <td className="py-2 text-right font-mono text-red-400">{a.max_drawdown_pct.toFixed(2)}%</td>
                          <td className="py-2 text-right font-mono">${a.last_price.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="sharpe">
              <Card className="border-accent/20">
                <CardHeader><CardTitle className="text-base">Sharpe Ratios</CardTitle></CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={sharpeData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="ticker" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(v: number) => [v.toFixed(3), "Sharpe"]} />
                      <Bar dataKey="sharpe" name="Sharpe">
                        {sharpeData.map((d, i) => <Cell key={i} fill={d.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="positions">
              <Card className="border-accent/20">
                <CardHeader><CardTitle className="text-base">Equal-Weight Position Sizing</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                  {result.positions.map(pos => (
                    <div key={pos.ticker} className="flex items-center gap-3 p-3 border border-accent/20 rounded">
                      <span className="font-semibold w-16">{pos.ticker}</span>
                      <div className="flex-1 h-3 bg-muted/20 rounded overflow-hidden">
                        <div className="h-full bg-primary/60 rounded" style={{ width: `${pos.weight_pct}%` }} />
                      </div>
                      <span className="text-sm text-muted-foreground w-12 text-right">{pos.weight_pct.toFixed(1)}%</span>
                      <span className="text-sm font-mono w-24 text-right">
                        ${pos.value_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
};
