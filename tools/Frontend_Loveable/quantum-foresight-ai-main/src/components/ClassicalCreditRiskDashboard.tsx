import { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  ShieldCheck, ShieldAlert, Shield, TrendingUp, AlertTriangle, AlertCircle, CheckCircle2, Info,
  Sparkles, RefreshCw, PlayCircle, CloudOff,
} from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import { useAuth } from "@/hooks/useAuth";
import {
  apiCreditSimSimulate, apiCreditSimGetProfile, apiCreditSimSaveProfile,
  type CreditSimPaymentBehavior, type CreditSimMonth, type CreditSimResponse,
} from "@/lib/api";

// ─── Loan-readiness snapshot (unchanged classical heuristic — separate from the
// month-by-month behavior simulator below; both share the same starting profile) ─

type RiskLevel = "Low" | "Medium" | "High";

interface RiskResult {
  level: RiskLevel;
  color: string;
  bgColor: string;
  borderColor: string;
  icon: typeof ShieldCheck;
  summary: string;
  tips: string[];
}

function assessRisk(
  fico: number,
  monthlyIncome: number,
  monthlyDebt: number,
  loanAmount: number,
  loanTermMonths: number,
  employment: string
): RiskResult {
  const newPayment = loanTermMonths > 0 ? loanAmount / loanTermMonths : 0;
  const totalDti = monthlyIncome > 0 ? (monthlyDebt + newPayment) / monthlyIncome : 1;
  const unemployed = employment === "unemployed";

  let riskScore = 0;
  if (fico < 580) riskScore += 3;
  else if (fico < 670) riskScore += 2;
  else if (fico < 740) riskScore += 1;

  if (totalDti > 0.5) riskScore += 3;
  else if (totalDti > 0.43) riskScore += 2;
  else if (totalDti > 0.36) riskScore += 1;

  if (unemployed) riskScore += 3;
  else if (employment === "part_time") riskScore += 1;

  const level: RiskLevel = riskScore >= 5 ? "High" : riskScore >= 2 ? "Medium" : "Low";

  const tips: string[] = [];
  if (fico < 670) tips.push("Pay down existing revolving debt to improve your credit utilization ratio — this is the fastest way to raise your FICO score.");
  if (fico < 740) tips.push("Make every payment on time for at least 6 months. Payment history is 35% of your FICO score.");
  if (totalDti > 0.43) tips.push("Reduce your monthly debt burden before taking on this loan. Aim for a total DTI below 36%.");
  if (loanAmount > 0 && loanTermMonths < 24) tips.push("Consider extending your loan term to reduce monthly payments and improve your DTI ratio.");
  if (unemployed) tips.push("Secure stable income before applying — lenders require demonstrated ability to repay.");
  if (employment === "part_time") tips.push("Moving to full-time employment or adding a co-signer can significantly strengthen your application.");
  if (tips.length === 0) tips.push("Your profile looks strong! Keep maintaining on-time payments and low credit utilization.");
  if (fico >= 740) tips.push("You may qualify for the best available interest rates — shop multiple lenders to compare offers.");

  const configs: Record<RiskLevel, Omit<RiskResult, "tips" | "level">> = {
    Low: {
      color: "text-emerald-400", bgColor: "bg-emerald-500/10", borderColor: "border-emerald-500/40",
      icon: ShieldCheck,
      summary: "Your profile indicates low credit risk. You're likely to qualify for competitive rates with most lenders.",
    },
    Medium: {
      color: "text-amber-400", bgColor: "bg-amber-500/10", borderColor: "border-amber-500/40",
      icon: Shield,
      summary: "Your profile shows moderate risk. Some lenders may approve you, but at higher rates. Strengthening your profile before applying is recommended.",
    },
    High: {
      color: "text-red-400", bgColor: "bg-red-500/10", borderColor: "border-red-500/40",
      icon: ShieldAlert,
      summary: "Your profile presents elevated risk factors. Consider improving your credit score and reducing debt before applying for this loan.",
    },
  };

  return { level, tips, ...configs[level] };
}

const FICO_TIERS = [
  { range: "300–579", min: 300, label: "Poor", color: "text-red-400" },
  { range: "580–669", min: 580, label: "Fair", color: "text-orange-400" },
  { range: "670–739", min: 670, label: "Good", color: "text-amber-400" },
  { range: "740–799", min: 740, label: "Very Good", color: "text-blue-400" },
  { range: "800–850", min: 800, label: "Exceptional", color: "text-emerald-400" },
];

function ficoTierFor(fico: number) {
  return [...FICO_TIERS].reverse().find(t => fico >= t.min) ?? FICO_TIERS[0];
}

const FACTOR_LABELS: Record<string, string> = {
  payment_history: "Payment History (35%)",
  utilization: "Credit Utilization (30%)",
  history_length: "Length of History (15%)",
  credit_mix: "Credit Mix (10%)",
  new_credit: "New Credit (10%)",
};

type PaymentPreset = "on_time" | "minimum_only" | "missed_sometimes";

function buildMonths(
  horizon: number,
  preset: PaymentPreset,
  missEveryN: number,
  utilizationPct: number,
  opensNewAccount: boolean,
  newAccountMonth: number
): CreditSimMonth[] {
  const months: CreditSimMonth[] = [];
  for (let m = 1; m <= horizon; m++) {
    let behavior: CreditSimPaymentBehavior = "on_time";
    if (preset === "minimum_only") behavior = "minimum_only";
    else if (preset === "missed_sometimes" && m % Math.max(1, missEveryN) === 0) behavior = "missed";
    months.push({
      payment_behavior: behavior,
      utilization_pct: utilizationPct,
      new_account_opened: opensNewAccount && m === newAccountMonth,
    });
  }
  return months;
}

export function ClassicalCreditRiskDashboard() {
  const { user } = useAuth();
  const { setClassicalCreditRiskSnapshot } = useAppContext();

  // Starting profile (shared by both the simulator and the loan-readiness card)
  const [fico, setFico] = useState(680);
  const [monthlyIncome, setMonthlyIncome] = useState(3000);
  const [monthlyDebt, setMonthlyDebt] = useState(400);
  const [employment, setEmployment] = useState("full_time");
  const [loanAmount, setLoanAmount] = useState(10000);
  const [loanTerm, setLoanTerm] = useState(36);

  // Behavior simulator controls
  const [horizon, setHorizon] = useState(12);
  const [preset, setPreset] = useState<PaymentPreset>("on_time");
  const [missEveryN, setMissEveryN] = useState(6);
  const [utilizationPct, setUtilizationPct] = useState(25);
  const [opensNewAccount, setOpensNewAccount] = useState(false);
  const [newAccountMonth, setNewAccountMonth] = useState(6);

  const [result, setResult] = useState<CreditSimResponse | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [exported, setExported] = useState(false);
  const [profileLoading, setProfileLoading] = useState(true);

  // Load a previously saved scenario (starting profile + last result) once signed in.
  useEffect(() => {
    if (!user) { setProfileLoading(false); return; }
    let cancelled = false;
    (async () => {
      try {
        const profile = await apiCreditSimGetProfile();
        if (cancelled) return;
        if (profile.starting_fico) setFico(profile.starting_fico);
        if (profile.monthly_income) setMonthlyIncome(profile.monthly_income);
        if (profile.monthly_debt) setMonthlyDebt(profile.monthly_debt);
        const assumptions = profile.behavior_assumptions as Partial<{
          horizon: number; preset: PaymentPreset; missEveryN: number;
          utilizationPct: number; opensNewAccount: boolean; newAccountMonth: number;
        }> | undefined;
        if (assumptions?.horizon) setHorizon(assumptions.horizon);
        if (assumptions?.preset) setPreset(assumptions.preset);
        if (assumptions?.missEveryN) setMissEveryN(assumptions.missEveryN);
        if (assumptions?.utilizationPct !== undefined) setUtilizationPct(assumptions.utilizationPct);
        if (assumptions?.opensNewAccount !== undefined) setOpensNewAccount(assumptions.opensNewAccount);
        if (assumptions?.newAccountMonth) setNewAccountMonth(assumptions.newAccountMonth);
        if (profile.last_trajectory) setResult(profile.last_trajectory);
      } catch {
        // No saved scenario yet, or not reachable — start fresh, not a hard error.
      } finally {
        if (!cancelled) setProfileLoading(false);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);

  const ficoTier = ficoTierFor(fico);

  const loanReadiness = useMemo(
    () => assessRisk(fico, monthlyIncome, monthlyDebt, loanAmount, loanTerm, employment),
    [fico, monthlyIncome, monthlyDebt, loanAmount, loanTerm, employment]
  );
  const newPayment = loanTerm > 0 ? loanAmount / loanTerm : 0;
  const dti = monthlyIncome > 0 ? monthlyDebt / monthlyIncome : 0;
  const totalDti = monthlyIncome > 0 ? (monthlyDebt + newPayment) / monthlyIncome : 0;
  const ReadinessIcon = loanReadiness.icon;

  const runSimulation = async () => {
    setIsSimulating(true);
    setSimError(null);
    try {
      const months = buildMonths(horizon, preset, missEveryN, utilizationPct, opensNewAccount, newAccountMonth);
      const res = await apiCreditSimSimulate({
        starting_fico: fico, monthly_income: monthlyIncome, monthly_debt: monthlyDebt, months,
      });
      setResult(res);
      if (user) {
        apiCreditSimSaveProfile({
          starting_fico: fico, monthly_income: monthlyIncome, monthly_debt: monthlyDebt,
          behavior_assumptions: { horizon, preset, missEveryN, utilizationPct, opensNewAccount, newAccountMonth },
          last_trajectory: res,
        }).catch(() => { /* best-effort save, not a blocking error for the run itself */ });
      }
    } catch (e) {
      setSimError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSimulating(false);
    }
  };

  const handleExport = () => {
    setClassicalCreditRiskSnapshot({
      timestamp: new Date().toISOString(),
      fico, fico_tier: ficoTier.label,
      employment,
      monthly_income: monthlyIncome, monthly_debt: monthlyDebt,
      loan_amount: loanAmount, loan_term_months: loanTerm,
      dti_pct: parseFloat((dti * 100).toFixed(1)),
      total_dti_pct: parseFloat((totalDti * 100).toFixed(1)),
      estimated_monthly_payment: parseFloat(newPayment.toFixed(2)),
      risk_level: loanReadiness.level,
      risk_summary: loanReadiness.summary,
      tips: loanReadiness.tips,
      simulated_starting_fico: result?.starting_fico,
      simulated_ending_fico: result?.ending_fico,
      simulated_months: result?.trajectory.length,
      simulation_tips: result?.tips,
      simulation_disclaimer: result?.disclaimer,
    });
    setExported(true);
    setTimeout(() => setExported(false), 3000);
  };

  const chartData = result?.trajectory.map(p => ({ month: p.month, score: p.score })) ?? [];
  const scoreDelta = result ? result.ending_fico - result.starting_fico : 0;

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Credit Behavior Simulator</h2>
          <p className="text-sm text-muted-foreground mt-1">
            See how everyday habits — paying on time, credit utilization, opening new accounts — shape a credit score over time. No quantum computing required.
          </p>
        </div>
        {!user && (
          <Badge variant="outline" className="text-[10px] border-amber-500/30 text-amber-400 flex items-center gap-1 shrink-0">
            <CloudOff className="w-3 h-3" />Sign in to save scenarios
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Controls */}
        <div className="space-y-5">
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm">Starting Profile</CardTitle></CardHeader>
            <CardContent className="space-y-5">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-medium">Starting FICO Score</Label>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className={`text-xs ${ficoTier.color} border-current/30`}>{ficoTier.label}</Badge>
                    <span className="text-sm font-bold text-foreground">{fico}</span>
                  </div>
                </div>
                <Slider min={300} max={850} step={1} value={[fico]} onValueChange={([v]) => setFico(v)} />
                <div className="flex justify-between text-[10px] text-muted-foreground">
                  {FICO_TIERS.map(t => (
                    <span key={t.range} className={fico >= t.min ? t.color : ""}>{t.range}</span>
                  ))}
                </div>
              </div>

              <div className="space-y-1.5">
                <Label className="text-xs font-medium">Employment Status</Label>
                <Select value={employment} onValueChange={setEmployment}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="full_time" className="text-xs">Full-Time Employed</SelectItem>
                    <SelectItem value="part_time" className="text-xs">Part-Time / Gig Work</SelectItem>
                    <SelectItem value="self_employed" className="text-xs">Self-Employed</SelectItem>
                    <SelectItem value="student" className="text-xs">Student (with income)</SelectItem>
                    <SelectItem value="unemployed" className="text-xs">Unemployed</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <Label className="text-xs font-medium">Gross Monthly Income ($)</Label>
                  <Input type="number" min={0} value={monthlyIncome}
                    onChange={e => setMonthlyIncome(Math.max(0, parseFloat(e.target.value) || 0))}
                    className="h-8 text-sm" />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs font-medium">Existing Monthly Debt ($)</Label>
                  <Input type="number" min={0} value={monthlyDebt}
                    onChange={e => setMonthlyDebt(Math.max(0, parseFloat(e.target.value) || 0))}
                    className="h-8 text-sm" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
            <CardHeader className="pb-3"><CardTitle className="text-sm">Simulate Your Behavior</CardTitle></CardHeader>
            <CardContent className="space-y-5">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-medium">Time Horizon</Label>
                  <Badge variant="outline" className="text-xs">{horizon} months</Badge>
                </div>
                <Slider min={3} max={36} step={1} value={[horizon]} onValueChange={([v]) => setHorizon(v)} />
              </div>

              <div className="space-y-1.5">
                <Label className="text-xs font-medium">Payment Behavior</Label>
                <Select value={preset} onValueChange={v => setPreset(v as PaymentPreset)}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="on_time" className="text-xs">Always pay on time</SelectItem>
                    <SelectItem value="minimum_only" className="text-xs">Minimum payments only</SelectItem>
                    <SelectItem value="missed_sometimes" className="text-xs">Occasionally miss a payment</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {preset === "missed_sometimes" && (
                <div className="space-y-2 pl-3 border-l-2 border-red-500/30">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs font-medium">Miss a payment every</Label>
                    <Badge variant="outline" className="text-xs">{missEveryN} months</Badge>
                  </div>
                  <Slider min={2} max={12} step={1} value={[missEveryN]} onValueChange={([v]) => setMissEveryN(v)} />
                </div>
              )}

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-medium">Credit Utilization</Label>
                  <Badge variant="outline" className="text-xs">{utilizationPct}%</Badge>
                </div>
                <Slider min={0} max={100} step={5} value={[utilizationPct]} onValueChange={([v]) => setUtilizationPct(v)} />
                <p className="text-[10px] text-muted-foreground">The % of your available credit limit you carry as a balance. Under 30% is recommended.</p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-medium">Open a new account partway through?</Label>
                  <Switch checked={opensNewAccount} onCheckedChange={setOpensNewAccount} />
                </div>
                {opensNewAccount && (
                  <div className="space-y-2 pl-3 border-l-2 border-border/40">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs font-medium">At month</Label>
                      <Badge variant="outline" className="text-xs">{Math.min(newAccountMonth, horizon)}</Badge>
                    </div>
                    <Slider min={1} max={horizon} step={1} value={[Math.min(newAccountMonth, horizon)]}
                      onValueChange={([v]) => setNewAccountMonth(v)} />
                  </div>
                )}
              </div>

              <Button onClick={runSimulation} disabled={isSimulating || profileLoading} className="w-full h-11">
                {isSimulating
                  ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Simulating...</>
                  : <><PlayCircle className="w-4 h-4 mr-2" />Run Simulation</>}
              </Button>

              {simError && (
                <div className="flex items-center gap-2 text-xs text-red-400">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />{simError}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {!result ? (
            <Card className="border-dashed border-border/40">
              <CardContent className="p-8 flex flex-col items-center text-center gap-2 text-muted-foreground">
                <PlayCircle className="w-8 h-8" />
                <p className="text-sm font-medium">Set your starting profile and behavior, then run the simulation.</p>
                <p className="text-xs">You'll see a month-by-month score trajectory and what's driving it.</p>
              </CardContent>
            </Card>
          ) : (
            <Tabs defaultValue="trajectory">
              <TabsList className="grid grid-cols-4 w-full">
                <TabsTrigger value="trajectory" className="text-xs">Trajectory</TabsTrigger>
                <TabsTrigger value="factors" className="text-xs">Factors</TabsTrigger>
                <TabsTrigger value="tips" className="text-xs">Tips</TabsTrigger>
                <TabsTrigger value="readiness" className="text-xs">Loan Readiness</TabsTrigger>
              </TabsList>

              <TabsContent value="trajectory" className="mt-3 space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <Card>
                    <CardContent className="p-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Start</p>
                      <p className="text-lg font-bold text-foreground">{result.starting_fico}</p>
                    </CardContent>
                  </Card>
                  <Card className={scoreDelta >= 0 ? "border-emerald-500/30 bg-emerald-500/5" : "border-red-500/30 bg-red-500/5"}>
                    <CardContent className="p-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Change</p>
                      <p className={`text-lg font-bold ${scoreDelta >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {scoreDelta >= 0 ? "+" : ""}{scoreDelta}
                      </p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">After {result.trajectory.length}mo</p>
                      <p className="text-lg font-bold text-foreground">{result.ending_fico}</p>
                    </CardContent>
                  </Card>
                </div>
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-sm">Simulated Score Trajectory</CardTitle></CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={240}>
                      <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                        <XAxis dataKey="month" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                          label={{ value: "Month", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
                        <YAxis domain={[300, 850]} tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} width={40} />
                        <ReferenceLine y={result.starting_fico} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
                        <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: "11px" }} />
                        <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={2.5} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="factors" className="mt-3">
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-sm">What's Driving Your Score</CardTitle></CardHeader>
                  <CardContent className="space-y-3">
                    {result.trajectory.length > 0 && Object.entries(result.trajectory[result.trajectory.length - 1].factor_breakdown).map(([factor, score]) => (
                      <div key={factor} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-medium text-foreground">{FACTOR_LABELS[factor] ?? factor}</span>
                          <span className="text-muted-foreground">{Math.round(score)}/100</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-muted/40 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${score >= 70 ? "bg-emerald-500" : score >= 40 ? "bg-amber-500" : "bg-red-500"}`}
                            style={{ width: `${Math.max(0, Math.min(100, score))}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="tips" className="mt-3 space-y-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-primary" />What Would Help Most
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {result.tips.map((tip, i) => (
                      <div key={i} className="flex gap-2 text-xs text-muted-foreground">
                        <span className="text-primary mt-0.5 shrink-0">•</span><span>{tip}</span>
                      </div>
                    ))}
                  </CardContent>
                </Card>
                <p className="text-[10px] text-muted-foreground text-center px-4 flex items-center justify-center gap-1">
                  <Info className="w-3 h-3 shrink-0" />{result.disclaimer}
                </p>
                <div className="flex justify-center">
                  {exported ? (
                    <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 text-xs gap-1">
                      <CheckCircle2 className="w-3 h-3" />Saved — switch to Lachesis AI to interpret
                    </Badge>
                  ) : (
                    <Button size="sm" variant="outline" className="gap-1.5 text-xs border-primary/40 text-primary hover:bg-primary/10" onClick={handleExport}>
                      <Sparkles className="w-3 h-3" />Send to Lachesis AI
                    </Button>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="readiness" className="mt-3 space-y-4">
                <Card className={`${loanReadiness.bgColor} ${loanReadiness.borderColor} border`}>
                  <CardContent className="p-6 flex flex-col items-center text-center gap-3">
                    <div className={`w-16 h-16 rounded-full ${loanReadiness.bgColor} ${loanReadiness.borderColor} border-2 flex items-center justify-center`}>
                      <ReadinessIcon className={`w-8 h-8 ${loanReadiness.color}`} />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold mb-1">Loan Readiness (current profile)</p>
                      <p className={`text-2xl font-bold ${loanReadiness.color}`}>{loanReadiness.level} Risk</p>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed max-w-xs">{loanReadiness.summary}</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-sm">Hypothetical Loan</CardTitle></CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-1.5">
                        <Label className="text-xs font-medium">Loan Amount ($)</Label>
                        <Input type="number" min={0} value={loanAmount}
                          onChange={e => setLoanAmount(Math.max(0, parseFloat(e.target.value) || 0))}
                          className="h-8 text-sm" />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label className="text-xs font-medium">Term</Label>
                          <Badge variant="outline" className="text-xs">{loanTerm} mo</Badge>
                        </div>
                        <Slider min={6} max={84} step={6} value={[loanTerm]} onValueChange={([v]) => setLoanTerm(v)} />
                      </div>
                    </div>
                    <div className="space-y-2">
                      {[
                        { label: "Current DTI (without new loan)", value: `${(dti * 100).toFixed(1)}%`, ok: dti < 0.36, warn: dti < 0.43, note: "Target: < 36%" },
                        { label: "Total DTI (with new loan)", value: `${(totalDti * 100).toFixed(1)}%`, ok: totalDti < 0.36, warn: totalDti < 0.43, note: "Max lenders accept: 43%" },
                        { label: "Estimated Monthly Payment", value: `$${newPayment.toFixed(2)}`, ok: true, warn: true, note: "Principal only (no interest)" },
                      ].map(({ label, value, ok, warn, note }) => (
                        <div key={label} className="flex items-center justify-between">
                          <div>
                            <p className="text-xs font-medium text-foreground">{label}</p>
                            <p className="text-[10px] text-muted-foreground">{note}</p>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className="text-sm font-bold text-foreground">{value}</span>
                            {ok ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                              : warn ? <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                              : <AlertTriangle className="w-3.5 h-3.5 text-red-400" />}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-sm flex items-center gap-2"><Info className="w-4 h-4 text-muted-foreground" />FICO Score Reference</CardTitle></CardHeader>
                  <CardContent>
                    <div className="space-y-1.5">
                      {FICO_TIERS.map(tier => (
                        <div key={tier.range} className={`flex justify-between items-center text-xs px-2 py-1 rounded ${ficoTier.range === tier.range ? "bg-muted/40" : ""}`}>
                          <span className={`font-medium ${tier.color}`}>{tier.label}</span>
                          <span className="text-muted-foreground">{tier.range}</span>
                          {ficoTier.range === tier.range && <Badge variant="outline" className="text-[9px]">You</Badge>}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          )}
        </div>
      </div>

      <p className="text-[10px] text-muted-foreground text-center px-4">
        This is an educational simulation based on general, publicly documented credit-scoring factors — not a real credit-bureau algorithm, financial advice, or a guarantee of any lender's decision.
      </p>
    </div>
  );
}
