import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ShieldCheck, ShieldAlert, Shield, TrendingUp, AlertTriangle, CheckCircle2, Info, Sparkles } from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";

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
  const dti = monthlyIncome > 0 ? monthlyDebt / monthlyIncome : 1;
  const newPayment = loanTermMonths > 0 ? loanAmount / loanTermMonths : 0;
  const totalDti = monthlyIncome > 0 ? (monthlyDebt + newPayment) / monthlyIncome : 1;
  const unemployed = employment === "unemployed";

  // Score risk factors
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
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
      borderColor: "border-emerald-500/40",
      icon: ShieldCheck,
      summary: "Your profile indicates low credit risk. You're likely to qualify for competitive rates with most lenders.",
    },
    Medium: {
      color: "text-amber-400",
      bgColor: "bg-amber-500/10",
      borderColor: "border-amber-500/40",
      icon: Shield,
      summary: "Your profile shows moderate risk. Some lenders may approve you, but at higher rates. Strengthening your profile before applying is recommended.",
    },
    High: {
      color: "text-red-400",
      bgColor: "bg-red-500/10",
      borderColor: "border-red-500/40",
      icon: ShieldAlert,
      summary: "Your profile presents elevated risk factors. Consider improving your credit score and reducing debt before applying for this loan.",
    },
  };

  return { level, tips, ...configs[level] };
}

const FICO_TIERS = [
  { range: "300–579", label: "Poor", color: "text-red-400" },
  { range: "580–669", label: "Fair", color: "text-orange-400" },
  { range: "670–739", label: "Good", color: "text-amber-400" },
  { range: "740–799", label: "Very Good", color: "text-blue-400" },
  { range: "800–850", label: "Exceptional", color: "text-emerald-400" },
];

export function ClassicalCreditRiskDashboard() {
  const [fico, setFico]                   = useState(680);
  const [monthlyIncome, setMonthlyIncome] = useState(3000);
  const [monthlyDebt, setMonthlyDebt]     = useState(400);
  const [loanAmount, setLoanAmount]       = useState(10000);
  const [loanTerm, setLoanTerm]           = useState(36);
  const [employment, setEmployment]       = useState("full_time");
  const [exported, setExported]           = useState(false);

  const { setClassicalCreditRiskSnapshot } = useAppContext();

  const dti = monthlyIncome > 0 ? monthlyDebt / monthlyIncome : 0;
  const newPayment = loanTerm > 0 ? loanAmount / loanTerm : 0;
  const totalDti = monthlyIncome > 0 ? (monthlyDebt + newPayment) / monthlyIncome : 0;

  const ficoTier = fico >= 800 ? FICO_TIERS[4]
    : fico >= 740 ? FICO_TIERS[3]
    : fico >= 670 ? FICO_TIERS[2]
    : fico >= 580 ? FICO_TIERS[1]
    : FICO_TIERS[0];

  const result = useMemo(
    () => assessRisk(fico, monthlyIncome, monthlyDebt, loanAmount, loanTerm, employment),
    [fico, monthlyIncome, monthlyDebt, loanAmount, loanTerm, employment]
  );

  const RiskIcon = result.icon;

  const EMPLOYMENT_LABELS: Record<string, string> = {
    full_time: "Full-Time Employed", part_time: "Part-Time / Gig Work",
    self_employed: "Self-Employed", student: "Student (with income)", unemployed: "Unemployed",
  };

  const handleExport = () => {
    setClassicalCreditRiskSnapshot({
      timestamp: new Date().toISOString(),
      fico,
      fico_tier: ficoTier.label,
      employment: EMPLOYMENT_LABELS[employment] ?? employment,
      monthly_income: monthlyIncome,
      monthly_debt: monthlyDebt,
      loan_amount: loanAmount,
      loan_term_months: loanTerm,
      dti_pct: parseFloat((dti * 100).toFixed(1)),
      total_dti_pct: parseFloat((totalDti * 100).toFixed(1)),
      estimated_monthly_payment: parseFloat(newPayment.toFixed(2)),
      risk_level: result.level,
      risk_summary: result.summary,
      tips: result.tips,
    });
    setExported(true);
    setTimeout(() => setExported(false), 3000);
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-foreground">Classical Credit Risk</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Understand your credit risk profile before applying for a loan — no quantum computing required.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Inputs */}
        <div className="space-y-5">
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm">Your Credit Profile</CardTitle></CardHeader>
            <CardContent className="space-y-5">
              {/* FICO */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-medium">FICO Credit Score</Label>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className={`text-xs ${ficoTier.color} border-current/30`}>{ficoTier.label}</Badge>
                    <span className="text-sm font-bold text-foreground">{fico}</span>
                  </div>
                </div>
                <Slider
                  min={300} max={850} step={1}
                  value={[fico]}
                  onValueChange={([v]) => setFico(v)}
                />
                <div className="flex justify-between text-[10px] text-muted-foreground">
                  {FICO_TIERS.map(t => (
                    <span key={t.range} className={fico >= parseInt(t.range) ? t.color : ""}>{t.range}</span>
                  ))}
                </div>
              </div>

              {/* Employment */}
              <div className="space-y-1.5">
                <Label className="text-xs font-medium">Employment Status</Label>
                <Select value={employment} onValueChange={setEmployment}>
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="full_time" className="text-xs">Full-Time Employed</SelectItem>
                    <SelectItem value="part_time" className="text-xs">Part-Time / Gig Work</SelectItem>
                    <SelectItem value="self_employed" className="text-xs">Self-Employed</SelectItem>
                    <SelectItem value="student" className="text-xs">Student (with income)</SelectItem>
                    <SelectItem value="unemployed" className="text-xs">Unemployed</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Income */}
              <div className="space-y-1.5">
                <Label className="text-xs font-medium">Gross Monthly Income ($)</Label>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground text-xs">$</span>
                  <Input
                    type="number" min={0}
                    value={monthlyIncome}
                    onChange={e => setMonthlyIncome(Math.max(0, parseFloat(e.target.value) || 0))}
                    className="h-8 text-sm"
                  />
                </div>
              </div>

              {/* Existing debt */}
              <div className="space-y-1.5">
                <Label className="text-xs font-medium">Existing Monthly Debt Payments ($)</Label>
                <p className="text-[10px] text-muted-foreground">Include credit cards, car loans, student loans — not rent</p>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground text-xs">$</span>
                  <Input
                    type="number" min={0}
                    value={monthlyDebt}
                    onChange={e => setMonthlyDebt(Math.max(0, parseFloat(e.target.value) || 0))}
                    className="h-8 text-sm"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm">Loan Details</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1.5">
                <Label className="text-xs font-medium">Loan Amount ($)</Label>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground text-xs">$</span>
                  <Input
                    type="number" min={0}
                    value={loanAmount}
                    onChange={e => setLoanAmount(Math.max(0, parseFloat(e.target.value) || 0))}
                    className="h-8 text-sm"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-medium">Loan Term</Label>
                  <Badge variant="outline" className="text-xs">{loanTerm} months</Badge>
                </div>
                <Slider
                  min={6} max={84} step={6}
                  value={[loanTerm]}
                  onValueChange={([v]) => setLoanTerm(v)}
                />
                <div className="flex justify-between text-[10px] text-muted-foreground">
                  <span>6 mo</span><span>2 yr</span><span>3 yr</span><span>5 yr</span><span>7 yr</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {/* Risk Verdict */}
          <Card className={`${result.bgColor} ${result.borderColor} border`}>
            <CardContent className="p-6 flex flex-col items-center text-center gap-3">
              <div className={`w-20 h-20 rounded-full ${result.bgColor} ${result.borderColor} border-2 flex items-center justify-center`}>
                <RiskIcon className={`w-10 h-10 ${result.color}`} />
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold mb-1">Risk Assessment</p>
                <p className={`text-3xl font-bold ${result.color}`}>{result.level} Risk</p>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed max-w-xs">{result.summary}</p>
              {exported ? (
                <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 text-xs gap-1">
                  <CheckCircle2 className="w-3 h-3" />Saved — switch to Lachesis AI to interpret
                </Badge>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  className="gap-1.5 text-xs border-primary/40 text-primary hover:bg-primary/10"
                  onClick={handleExport}
                >
                  <Sparkles className="w-3 h-3" />Send to Lachesis AI
                </Button>
              )}
            </CardContent>
          </Card>

          {/* Derived Metrics */}
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Key Metrics</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              {[
                {
                  label: "Current DTI (without new loan)",
                  value: `${(dti * 100).toFixed(1)}%`,
                  good: dti < 0.36,
                  warn: dti < 0.43,
                  note: "Target: < 36%",
                },
                {
                  label: "Total DTI (with new loan)",
                  value: `${(totalDti * 100).toFixed(1)}%`,
                  good: totalDti < 0.36,
                  warn: totalDti < 0.43,
                  note: "Max lenders accept: 43%",
                },
                {
                  label: "Estimated Monthly Payment",
                  value: `$${newPayment.toFixed(2)}`,
                  good: true,
                  warn: true,
                  note: "Principal only (no interest)",
                },
              ].map(({ label, value, good, warn, note }) => (
                <div key={label} className="flex items-center justify-between">
                  <div>
                    <p className="text-xs font-medium text-foreground">{label}</p>
                    <p className="text-[10px] text-muted-foreground">{note}</p>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm font-bold text-foreground">{value}</span>
                    {good
                      ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      : warn
                      ? <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                      : <AlertTriangle className="w-3.5 h-3.5 text-red-400" />}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Tips */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-primary" />How to Improve Your Profile
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {result.tips.map((tip, i) => (
                <div key={i} className="flex gap-2 text-xs text-muted-foreground">
                  <span className="text-primary mt-0.5 shrink-0">•</span>
                  <span>{tip}</span>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* FICO Reference */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Info className="w-4 h-4 text-muted-foreground" />FICO Score Reference
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1.5">
                {FICO_TIERS.map(tier => (
                  <div
                    key={tier.range}
                    className={`flex justify-between items-center text-xs px-2 py-1 rounded ${
                      ficoTier.range === tier.range ? "bg-muted/40" : ""
                    }`}
                  >
                    <span className={`font-medium ${tier.color}`}>{tier.label}</span>
                    <span className="text-muted-foreground">{tier.range}</span>
                    {ficoTier.range === tier.range && <Badge variant="outline" className="text-[9px]">You</Badge>}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Disclaimer */}
      <p className="text-[10px] text-muted-foreground text-center px-4">
        This is an educational estimate based on general lending guidelines, not financial advice. Actual loan decisions depend on many additional factors and lender policies.
      </p>
    </div>
  );
}
