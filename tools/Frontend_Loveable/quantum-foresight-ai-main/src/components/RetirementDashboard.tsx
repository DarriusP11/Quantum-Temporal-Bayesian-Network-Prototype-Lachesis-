import { useState, useMemo, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import {
  ChevronDown, PiggyBank, TrendingUp, Clock, DollarSign, AlertCircle, CloudOff, RefreshCw,
  PlayCircle, Info, ShieldCheck,
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import {
  apiRetirementGet, apiRetirementSave, apiRetirementRiskAnalysis,
  type RetirementRiskAnalysisResponse,
} from "@/lib/api";

// Mirrors retirement.py's compute_growth() exactly — kept client-side too so
// sliders stay instantly responsive; the backend recomputes the same numbers
// server-side on every save, so the two never disagree.
function computeGrowth(
  currentAge: number,
  currentSavings: number,
  monthlyContribution: number,
  annualReturnPct: number,
  retirementAge: number
): { year: number; age: number; balance: number }[] {
  const r = annualReturnPct / 100 / 12;
  const years = Math.max(0, retirementAge - currentAge);
  const points: { year: number; age: number; balance: number }[] = [];
  let balance = currentSavings;
  for (let y = 0; y <= years; y++) {
    points.push({ year: y, age: currentAge + y, balance: Math.round(balance) });
    for (let m = 0; m < 12; m++) {
      balance = balance * (1 + r) + monthlyContribution;
    }
  }
  return points;
}

function formatMoney(n: number) {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

const STORAGE_KEY = "lachesis_retirement";

interface RetirementInputs {
  currentAge: number;
  currentSavings: number;
  monthlyContrib: number;
  annualReturn: number;
  retirementAge: number;
}

const DEFAULT_INPUTS: RetirementInputs = {
  currentAge: 22,
  currentSavings: 0,
  monthlyContrib: 200,
  annualReturn: 7,
  retirementAge: 65,
};

export function RetirementDashboard() {
  const { user } = useAuth();
  const [currentAge, setCurrentAge]         = useState(DEFAULT_INPUTS.currentAge);
  const [currentSavings, setCurrentSavings] = useState(DEFAULT_INPUTS.currentSavings);
  const [monthlyContrib, setMonthlyContrib] = useState(DEFAULT_INPUTS.monthlyContrib);
  const [annualReturn, setAnnualReturn]     = useState(DEFAULT_INPUTS.annualReturn);
  const [retirementAge, setRetirementAge]   = useState(DEFAULT_INPUTS.retirementAge);
  const [rothOpen, setRothOpen]             = useState(false);

  // Phase 4 — withdrawal / decumulation risk analysis
  const [rothPct, setRothPct]                     = useState(0);
  const [withdrawalRatePct, setWithdrawalRatePct] = useState(4);
  const [lifeExpectancy, setLifeExpectancy]       = useState(90);
  const [inflationRatePct, setInflationRatePct]   = useState(2.5);
  const [riskResult, setRiskResult] = useState<RetirementRiskAnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const didLoadRef = useRef(false);

  // Load: seed instantly from localStorage, then reconcile with the backend
  // (source of truth) once signed in.
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const p: Partial<RetirementInputs> = JSON.parse(saved);
        if (p.currentAge !== undefined) setCurrentAge(p.currentAge);
        if (p.currentSavings !== undefined) setCurrentSavings(p.currentSavings);
        if (p.monthlyContrib !== undefined) setMonthlyContrib(p.monthlyContrib);
        if (p.annualReturn !== undefined) setAnnualReturn(p.annualReturn);
        if (p.retirementAge !== undefined) setRetirementAge(p.retirementAge);
      }
    } catch {}

    if (!user) {
      setLoading(false);
      didLoadRef.current = true;
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const plan = await apiRetirementGet();
        if (cancelled) return;
        setCurrentAge(plan.current_age);
        setRetirementAge(plan.retirement_age);
        setCurrentSavings(plan.current_savings);
        setMonthlyContrib(plan.monthly_contribution);
        setAnnualReturn(plan.expected_return_rate);
        if (plan.roth_pct !== undefined) setRothPct(plan.roth_pct);
        if (plan.life_expectancy !== undefined) setLifeExpectancy(plan.life_expectancy);
        if (plan.withdrawal_rate_pct !== undefined) setWithdrawalRatePct(plan.withdrawal_rate_pct);
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) {
          setLoading(false);
          didLoadRef.current = true;
        }
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);

  // Save: always cache locally; debounce a backend sync while signed in.
  useEffect(() => {
    if (!didLoadRef.current) return;
    const inputs: RetirementInputs = { currentAge, currentSavings, monthlyContrib, annualReturn, retirementAge };
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(inputs)); } catch {}
    if (!user) return;

    const t = setTimeout(async () => {
      setSaving(true);
      try {
        await apiRetirementSave({
          current_age: currentAge,
          retirement_age: retirementAge,
          current_savings: currentSavings,
          monthly_contribution: monthlyContrib,
          expected_return_rate: annualReturn,
          roth_pct: rothPct,
          life_expectancy: lifeExpectancy,
          withdrawal_rate_pct: withdrawalRatePct,
        });
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setSaving(false);
      }
    }, 800);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentAge, currentSavings, monthlyContrib, annualReturn, retirementAge, rothPct, lifeExpectancy, withdrawalRatePct, user?.id]);

  const runRiskAnalysis = async () => {
    setIsAnalyzing(true);
    setAnalysisError(null);
    try {
      const res = await apiRetirementRiskAnalysis({
        starting_balance: projectedBalance,
        roth_pct: rothPct,
        withdrawal_rate_pct: withdrawalRatePct,
        inflation_rate_pct: inflationRatePct,
        horizon_years: Math.max(1, lifeExpectancy - retirementAge),
      });
      setRiskResult(res);
    } catch (e) {
      setAnalysisError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsAnalyzing(false);
    }
  };

  const years = Math.max(0, retirementAge - currentAge);

  // Start-now projection
  const nowData = useMemo(
    () => computeGrowth(currentAge, currentSavings, monthlyContrib, annualReturn, retirementAge),
    [currentAge, currentSavings, monthlyContrib, annualReturn, retirementAge]
  );

  // Start-at-30 projection (same monthly contribution, no current savings head start)
  const lateStartAge = Math.max(currentAge + 1, 30);
  const lateData = useMemo(
    () => computeGrowth(lateStartAge, 0, monthlyContrib, annualReturn, retirementAge),
    [lateStartAge, monthlyContrib, annualReturn, retirementAge]
  );

  const projectedBalance = nowData[nowData.length - 1]?.balance ?? 0;
  const lateBalance      = lateData[lateData.length - 1]?.balance ?? 0;
  const totalContributed = currentSavings + monthlyContrib * 12 * years;
  const totalGrowth      = projectedBalance - totalContributed;
  const growthMultiple   = totalContributed > 0 ? (projectedBalance / totalContributed).toFixed(1) : "—";

  // Merge chart data by aligning on age
  const chartData = nowData.map(point => {
    const latePoint = lateData.find(p => p.age === point.age);
    return {
      age: point.age,
      "Start Now": point.balance,
      "Start at 30": latePoint?.balance ?? null,
    };
  });

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Retirement Planner</h2>
          <p className="text-sm text-muted-foreground mt-1">
            See how compound interest works for you — starting early is the single biggest advantage you have.
          </p>
        </div>
        <div className="shrink-0">
          {!user ? (
            <Badge variant="outline" className="text-[10px] border-amber-500/30 text-amber-400 flex items-center gap-1">
              <CloudOff className="w-3 h-3" />Sign in to sync
            </Badge>
          ) : saving ? (
            <Badge variant="outline" className="text-[10px] border-border/40 text-muted-foreground flex items-center gap-1">
              <RefreshCw className="w-3 h-3 animate-spin" />Saving...
            </Badge>
          ) : (
            <Badge variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-400">Synced</Badge>
          )}
        </div>
      </div>

      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="p-3 flex items-center gap-2 text-xs text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" />
            Couldn't sync with the server — your changes are still saved in this browser. ({error})
          </CardContent>
        </Card>
      )}

      {loading ? (
        <Card><CardContent className="p-6 text-sm text-muted-foreground">Loading your plan...</CardContent></Card>
      ) : (
      <>
      {/* Inputs */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Your Numbers</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Current Age</Label>
            <div className="flex items-center gap-3">
              <Slider
                min={16} max={64} step={1}
                value={[currentAge]}
                onValueChange={([v]) => setCurrentAge(v)}
                className="flex-1"
              />
              <Badge variant="outline" className="w-12 text-center text-xs">{currentAge}</Badge>
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Retirement Age</Label>
            <div className="flex items-center gap-3">
              <Slider
                min={Math.max(currentAge + 1, 50)} max={75} step={1}
                value={[retirementAge]}
                onValueChange={([v]) => setRetirementAge(v)}
                className="flex-1"
              />
              <Badge variant="outline" className="w-12 text-center text-xs">{retirementAge}</Badge>
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Expected Annual Return (%)</Label>
            <div className="flex items-center gap-3">
              <Slider
                min={1} max={15} step={0.5}
                value={[annualReturn]}
                onValueChange={([v]) => setAnnualReturn(v)}
                className="flex-1"
              />
              <Badge variant="outline" className="w-14 text-center text-xs">{annualReturn}%</Badge>
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Current Savings ($)</Label>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground text-xs">$</span>
              <Input
                type="number"
                min={0}
                value={currentSavings}
                onChange={e => setCurrentSavings(Math.max(0, parseFloat(e.target.value) || 0))}
                className="h-8 text-sm"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Monthly Contribution ($)</Label>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground text-xs">$</span>
              <Input
                type="number"
                min={0}
                value={monthlyContrib}
                onChange={e => setMonthlyContrib(Math.max(0, parseFloat(e.target.value) || 0))}
                className="h-8 text-sm"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Card className="border-emerald-500/30 bg-emerald-500/5">
          <CardContent className="p-4">
            <div className="flex items-center gap-1 mb-1">
              <PiggyBank className="w-3 h-3 text-emerald-400" />
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Projected Balance</p>
            </div>
            <p className="text-xl font-bold text-emerald-400">{formatMoney(projectedBalance)}</p>
            <p className="text-[10px] text-muted-foreground">at age {retirementAge}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-1 mb-1">
              <DollarSign className="w-3 h-3 text-blue-400" />
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Total Contributed</p>
            </div>
            <p className="text-xl font-bold text-blue-400">{formatMoney(totalContributed)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-1 mb-1">
              <TrendingUp className="w-3 h-3 text-purple-400" />
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Compound Growth</p>
            </div>
            <p className="text-xl font-bold text-purple-400">{formatMoney(Math.max(0, totalGrowth))}</p>
            <p className="text-[10px] text-muted-foreground">{growthMultiple}× your money</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-1 mb-1">
              <Clock className="w-3 h-3 text-amber-400" />
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Years to Retire</p>
            </div>
            <p className="text-xl font-bold text-amber-400">{years}</p>
            <p className="text-[10px] text-muted-foreground">years from now</p>
          </CardContent>
        </Card>
      </div>

      {/* Growth Chart */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            Portfolio Growth Over Time
            {currentAge < 30 && (
              <Badge variant="outline" className="text-[10px] border-amber-500/30 text-amber-400">
                Starting now vs. at 30: +{formatMoney(Math.max(0, projectedBalance - lateBalance))} difference
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
              <XAxis
                dataKey="age"
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                label={{ value: "Age", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                tickFormatter={v => formatMoney(v)}
                width={60}
              />
              <Tooltip
                formatter={(v: number, name: string) => [formatMoney(v), name]}
                contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: "11px" }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Line type="monotone" dataKey="Start Now" stroke="#10b981" strokeWidth={2.5} dot={false} />
              {currentAge < 30 && (
                <Line type="monotone" dataKey="Start at 30" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 3" dot={false} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Start Now vs Start Late comparison */}
      {currentAge < 30 && (
        <div className="grid grid-cols-2 gap-4">
          <Card className="border-emerald-500/30 bg-emerald-500/5">
            <CardContent className="p-4 text-center">
              <p className="text-xs font-semibold text-muted-foreground mb-1">Start Now (age {currentAge})</p>
              <p className="text-2xl font-bold text-emerald-400">{formatMoney(projectedBalance)}</p>
              <p className="text-xs text-muted-foreground mt-1">{years} years of growth</p>
            </CardContent>
          </Card>
          <Card className="border-amber-500/30 bg-amber-500/5">
            <CardContent className="p-4 text-center">
              <p className="text-xs font-semibold text-muted-foreground mb-1">Wait Until Age 30</p>
              <p className="text-2xl font-bold text-amber-400">{formatMoney(lateBalance)}</p>
              <p className="text-xs text-red-400 font-medium mt-1">−{formatMoney(Math.max(0, projectedBalance - lateBalance))} by waiting</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Roth IRA tip */}
      <Collapsible open={rothOpen} onOpenChange={setRothOpen}>
        <CollapsibleTrigger className="flex w-full items-center justify-between p-4 rounded-xl border border-border/40 bg-card/60 hover:bg-card transition-colors text-sm font-semibold">
          <span>🏦 Why a Roth IRA is Perfect for Students</span>
          <ChevronDown className={`w-4 h-4 transition-transform ${rothOpen ? "rotate-180" : ""}`} />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-2 p-4 rounded-xl border border-border/40 bg-card/40 space-y-3 text-xs text-muted-foreground">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <p className="font-semibold text-foreground">What is a Roth IRA?</p>
                <p>A Roth IRA is a retirement account where you invest <span className="text-foreground font-medium">after-tax dollars</span> — meaning when you retire, all withdrawals are completely tax-free, including all the growth.</p>
                <p>For students in a low tax bracket, this is a huge deal. You pay taxes now (very little), and pay <strong>zero taxes</strong> on potentially hundreds of thousands in gains later.</p>
              </div>
              <div className="space-y-2">
                <p className="font-semibold text-foreground">2024 Key Numbers</p>
                <ul className="space-y-1">
                  <li className="flex justify-between"><span>Annual contribution limit:</span><span className="text-emerald-400 font-semibold">$7,000</span></li>
                  <li className="flex justify-between"><span>Monthly to max out:</span><span className="text-emerald-400 font-semibold">~$583/mo</span></li>
                  <li className="flex justify-between"><span>Income limit (single):</span><span className="text-foreground">$161,000</span></li>
                  <li className="flex justify-between"><span>Best time to open one:</span><span className="text-emerald-400 font-semibold">Right now</span></li>
                </ul>
                <p className="text-[10px] pt-1">You must have earned income to contribute. Even a part-time job qualifies.</p>
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>

      <p className="text-[10px] text-muted-foreground text-center px-4">
        The numbers above assume a single constant annual return with no volatility, taxes, or fees —
        a simple illustration of compounding, not a forecast. For a more realistic picture of what
        happens after you retire, see the risk analysis below.
      </p>

      {/* Phase 4 — Withdrawal / Decumulation Risk Analysis */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Retirement Risk & Withdrawal Analysis</CardTitle>
          <p className="text-xs text-muted-foreground">
            Starting from your projected balance of {formatMoney(projectedBalance)} at age {retirementAge}, this
            simulates {Math.max(1, lifeExpectancy - retirementAge)} years of withdrawals against hundreds of real
            historical market sequences — the "sequence of returns risk" that a single average-return number hides.
          </p>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-medium">% in Roth (tax-free withdrawals)</Label>
                <Badge variant="outline" className="text-xs">{rothPct}%</Badge>
              </div>
              <Slider min={0} max={100} step={5} value={[rothPct]} onValueChange={([v]) => setRothPct(v)} />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-medium">Withdrawal Rate (year 1)</Label>
                <Badge variant="outline" className="text-xs">{withdrawalRatePct}%</Badge>
              </div>
              <Slider min={2} max={6} step={0.25} value={[withdrawalRatePct]} onValueChange={([v]) => setWithdrawalRatePct(v)} />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-medium">Life Expectancy</Label>
                <Badge variant="outline" className="text-xs">{lifeExpectancy}</Badge>
              </div>
              <Slider min={Math.max(retirementAge + 1, 70)} max={100} step={1} value={[lifeExpectancy]} onValueChange={([v]) => setLifeExpectancy(v)} />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-medium">Inflation (grows your withdrawal $ each year)</Label>
                <Badge variant="outline" className="text-xs">{inflationRatePct}%</Badge>
              </div>
              <Slider min={0} max={6} step={0.25} value={[inflationRatePct]} onValueChange={([v]) => setInflationRatePct(v)} />
            </div>
          </div>

          <Button onClick={runRiskAnalysis} disabled={isAnalyzing || projectedBalance <= 0} className="w-full h-11">
            {isAnalyzing
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Running {500} historical simulations...</>
              : <><PlayCircle className="w-4 h-4 mr-2" />Run Retirement Risk Analysis</>}
          </Button>

          {analysisError && (
            <div className="flex items-center gap-2 text-xs text-red-400">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />{analysisError}
            </div>
          )}

          {riskResult && (
            <Tabs defaultValue="success">
              <TabsList className="grid grid-cols-3 w-full max-w-md">
                <TabsTrigger value="success" className="text-xs">Success Rate</TabsTrigger>
                <TabsTrigger value="balance" className="text-xs">Balance Over Time</TabsTrigger>
                <TabsTrigger value="taxes" className="text-xs">Taxes</TabsTrigger>
              </TabsList>

              <TabsContent value="success" className="mt-3 space-y-4">
                <Card className={riskResult.success_rate_pct >= 80 ? "border-emerald-500/30 bg-emerald-500/5" : "border-amber-500/30 bg-amber-500/5"}>
                  <CardContent className="p-6 flex flex-col items-center text-center gap-2">
                    <ShieldCheck className={`w-8 h-8 ${riskResult.success_rate_pct >= 80 ? "text-emerald-400" : "text-amber-400"}`} />
                    <p className={`text-4xl font-bold ${riskResult.success_rate_pct >= 80 ? "text-emerald-400" : "text-amber-400"}`}>
                      {riskResult.success_rate_pct}%
                    </p>
                    <p className="text-xs text-muted-foreground max-w-sm">
                      In {riskResult.success_rate_pct}% of {riskResult.n_simulations} simulated historical market sequences,
                      your money lasted all the way through age {lifeExpectancy} without running out.
                    </p>
                  </CardContent>
                </Card>
                <div className="grid grid-cols-3 gap-3">
                  <Card>
                    <CardContent className="p-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Worst Case</p>
                      <p className="text-sm font-bold text-red-400">{formatMoney(riskResult.worst_case_ending_balance)}</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Median</p>
                      <p className="text-sm font-bold text-foreground">{formatMoney(riskResult.median_ending_balance)}</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Best Case</p>
                      <p className="text-sm font-bold text-emerald-400">{formatMoney(riskResult.best_case_ending_balance)}</p>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="balance" className="mt-3">
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-sm">Balance Range Across Simulations</CardTitle></CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={riskResult.percentile_timeline} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                        <XAxis dataKey="year" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                          label={{ value: "Years into retirement", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
                        <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickFormatter={v => formatMoney(v)} width={60} />
                        <Tooltip formatter={(v: number, name: string) => [formatMoney(v), name]}
                          contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: "11px" }} />
                        <Legend wrapperStyle={{ fontSize: "11px" }} />
                        <Line type="monotone" dataKey="p90" name="Best 10%" stroke="#10b981" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
                        <Line type="monotone" dataKey="median" name="Median" stroke="#6366f1" strokeWidth={2.5} dot={false} />
                        <Line type="monotone" dataKey="p10" name="Worst 10%" stroke="#ef4444" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="taxes" className="mt-3 space-y-3">
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-sm">Year 1 Withdrawal Breakdown</CardTitle></CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Total withdrawal</span>
                      <span className="font-semibold text-foreground">{formatMoney(riskResult.initial_annual_withdrawal)}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Taxable (Traditional)</span>
                      <span className="font-semibold text-foreground">{formatMoney(riskResult.initial_taxable_withdrawal)}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Tax-free (Roth)</span>
                      <span className="font-semibold text-emerald-400">{formatMoney(riskResult.initial_tax_free_withdrawal)}</span>
                    </div>
                    <div className="border-t border-border/40 pt-2 flex justify-between text-xs font-semibold">
                      <span>Estimated federal tax (year 1)</span>
                      <span className="text-red-400">{formatMoney(riskResult.initial_year_tax)}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Median lifetime taxes paid</span>
                      <span className="font-medium text-foreground">{formatMoney(riskResult.median_lifetime_taxes_paid)}</span>
                    </div>
                  </CardContent>
                </Card>
                <p className="text-[10px] text-muted-foreground flex items-start gap-1 px-1">
                  <Info className="w-3 h-3 shrink-0 mt-0.5" />
                  Data source: {riskResult.data_source}.
                </p>
              </TabsContent>
            </Tabs>
          )}

          <p className="text-[10px] text-muted-foreground text-center px-4">
            Educational simulation only — 2024 federal single-filer tax brackets, no state tax; withdrawals are
            split proportionally between Traditional and Roth rather than optimized; historical returns are
            bootstrapped from real SPY data since 1993, a period that happens to include an unusually strong
            bull market, which may make results more optimistic than a longer market history would suggest.
            Past performance does not guarantee future results — this is not financial or tax advice.
          </p>
        </CardContent>
      </Card>
      </>
      )}
    </div>
  );
}
