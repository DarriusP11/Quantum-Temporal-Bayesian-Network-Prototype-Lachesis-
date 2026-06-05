import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { ChevronDown, PiggyBank, TrendingUp, Clock, DollarSign } from "lucide-react";

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

export function RetirementDashboard() {
  const [currentAge, setCurrentAge]         = useState(22);
  const [currentSavings, setCurrentSavings] = useState(0);
  const [monthlyContrib, setMonthlyContrib] = useState(200);
  const [annualReturn, setAnnualReturn]     = useState(7);
  const [retirementAge, setRetirementAge]   = useState(65);
  const [rothOpen, setRothOpen]             = useState(false);

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
      <div>
        <h2 className="text-2xl font-bold text-foreground">Retirement Planner</h2>
        <p className="text-sm text-muted-foreground mt-1">
          See how compound interest works for you — starting early is the single biggest advantage you have.
        </p>
      </div>

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
    </div>
  );
}
