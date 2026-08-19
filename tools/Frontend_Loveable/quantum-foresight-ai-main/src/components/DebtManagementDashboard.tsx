import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import {
  GraduationCap, CreditCard, Landmark, Briefcase, Plus, Trash2, RefreshCw, PlayCircle,
  AlertCircle, CloudOff, Wand2, Trophy,
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import {
  apiDebtManagementSimulate, apiDebtManagementGet, apiDebtManagementSave, apiDebtManagementMinimumPaymentHint,
  type DebtEntry, type DebtType, type DebtStrategy, type DebtPlanSimulateResponse,
} from "@/lib/api";

const DEBT_TYPE_LABELS: Record<DebtType, string> = {
  student_loan: "Student Loan",
  credit_card: "Credit Card",
  personal_loan: "Personal / Bank Loan",
  business_loan: "Business Loan",
};

const DEBT_TYPE_ICONS: Record<DebtType, typeof GraduationCap> = {
  student_loan: GraduationCap, credit_card: CreditCard, personal_loan: Landmark, business_loan: Briefcase,
};

const STRATEGY_LABELS: Record<DebtStrategy, string> = {
  minimum_only: "Minimum Only",
  snowball: "Snowball",
  avalanche: "Avalanche",
};

const STRATEGY_COLORS: Record<DebtStrategy, string> = {
  minimum_only: "#94a3b8",
  snowball: "#f59e0b",
  avalanche: "#10b981",
};

const STORAGE_KEY = "lachesis_debt_plan";

function uid() {
  return Math.random().toString(36).slice(2);
}

function newDebt(): DebtEntry {
  return { id: uid(), name: "", type: "credit_card", balance: 0, apr_pct: 20, minimum_payment: 25 };
}

function formatMoney(n: number) {
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

function formatMonths(m: number | null) {
  if (m === null) return "—";
  const years = Math.floor(m / 12);
  const months = m % 12;
  if (years === 0) return `${months}mo`;
  if (months === 0) return `${years}yr`;
  return `${years}yr ${months}mo`;
}

export function DebtManagementDashboard() {
  const { user } = useAuth();
  const [debts, setDebts] = useState<DebtEntry[]>([newDebt()]);
  const [extraMonthlyPayment, setExtraMonthlyPayment] = useState(200);

  const [result, setResult] = useState<DebtPlanSimulateResponse | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [hintLoadingId, setHintLoadingId] = useState<string | null>(null);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const p = JSON.parse(saved);
        if (p.debts?.length) setDebts(p.debts);
        if (p.extraMonthlyPayment !== undefined) setExtraMonthlyPayment(p.extraMonthlyPayment);
      }
    } catch {}

    if (!user) { setProfileLoading(false); return; }
    let cancelled = false;
    (async () => {
      try {
        const plan = await apiDebtManagementGet();
        if (cancelled) return;
        if (plan.debts?.length) setDebts(plan.debts);
        if (plan.extra_monthly_payment !== undefined) setExtraMonthlyPayment(plan.extra_monthly_payment);
      } catch {
        // No saved plan yet — start fresh.
      } finally {
        if (!cancelled) setProfileLoading(false);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ debts, extraMonthlyPayment })); } catch {}
  }, [debts, extraMonthlyPayment]);

  const updateDebt = (id: string, patch: Partial<DebtEntry>) => {
    setDebts(prev => prev.map(d => d.id === id ? { ...d, ...patch } : d));
  };
  const addDebt = () => setDebts(prev => [...prev, newDebt()]);
  const removeDebt = (id: string) => setDebts(prev => prev.filter(d => d.id !== id));

  const suggestMinimum = async (debt: DebtEntry) => {
    setHintLoadingId(debt.id);
    try {
      const res = await apiDebtManagementMinimumPaymentHint(debt.type, debt.balance, debt.apr_pct);
      updateDebt(debt.id, { minimum_payment: res.minimum_payment_hint });
    } catch {
      // Non-critical — just leave the current value.
    } finally {
      setHintLoadingId(null);
    }
  };

  const runSimulation = async () => {
    const validDebts = debts.filter(d => d.balance > 0);
    if (validDebts.length === 0) {
      setSimError("Add at least one debt with a balance greater than $0.");
      return;
    }
    setIsSimulating(true);
    setSimError(null);
    try {
      const res = await apiDebtManagementSimulate({ debts: validDebts, extra_monthly_payment: extraMonthlyPayment });
      setResult(res);
      if (user) {
        apiDebtManagementSave({ debts: validDebts, extra_monthly_payment: extraMonthlyPayment })
          .catch(() => { /* best-effort save */ });
      }
    } catch (e) {
      setSimError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSimulating(false);
    }
  };

  const strategyOrder: DebtStrategy[] = ["minimum_only", "snowball", "avalanche"];

  const chartData = (() => {
    if (!result) return [];
    const maxMonths = Math.max(...strategyOrder.map(s => result.strategies[s].timeline.length), 0);
    return Array.from({ length: maxMonths }, (_, i) => {
      const row: Record<string, number> = { month: i + 1 };
      strategyOrder.forEach(s => {
        const point = result.strategies[s].timeline[i];
        row[STRATEGY_LABELS[s]] = point ? point.total_remaining_balance : 0;
      });
      return row;
    });
  })();

  const anyHitCap = result ? strategyOrder.some(s => result.strategies[s].hit_cap) : false;

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Debt Management Simulator</h2>
          <p className="text-sm text-muted-foreground mt-1">
            List out your debts — student loans, credit cards, personal or business loans — and see how fast Snowball or Avalanche gets you debt-free compared to paying minimums only.
          </p>
        </div>
        {!user && (
          <Badge variant="outline" className="text-[10px] border-amber-500/30 text-amber-400 flex items-center gap-1 shrink-0">
            <CloudOff className="w-3 h-3" />Sign in to sync
          </Badge>
        )}
      </div>

      <Card>
        <CardHeader className="pb-3 flex flex-row items-center justify-between">
          <CardTitle className="text-sm">Your Debts</CardTitle>
          <Button size="sm" variant="outline" className="h-7 text-xs gap-1" onClick={addDebt}>
            <Plus className="w-3 h-3" />Add Debt
          </Button>
        </CardHeader>
        <CardContent className="space-y-3">
          {debts.map(debt => {
            const Icon = DEBT_TYPE_ICONS[debt.type];
            return (
              <div key={debt.id} className="p-3 rounded-lg border border-border/40 bg-card/40 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <Icon className="w-4 h-4 text-muted-foreground shrink-0" />
                    <Input
                      placeholder="e.g. Chase Sapphire, Federal Student Loan..."
                      value={debt.name}
                      onChange={e => updateDebt(debt.id, { name: e.target.value })}
                      className="h-8 text-sm flex-1"
                    />
                  </div>
                  <Button variant="ghost" size="icon" className="h-7 w-7 text-red-400 hover:text-red-300 hover:bg-red-500/10 shrink-0" onClick={() => removeDebt(debt.id)}>
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                  <div className="space-y-1">
                    <Label className="text-[10px] text-muted-foreground">Type</Label>
                    <Select value={debt.type} onValueChange={v => updateDebt(debt.id, { type: v as DebtType })}>
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {(Object.keys(DEBT_TYPE_LABELS) as DebtType[]).map(t => (
                          <SelectItem key={t} value={t} className="text-xs">{DEBT_TYPE_LABELS[t]}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[10px] text-muted-foreground">Balance ($)</Label>
                    <Input type="number" min={0} value={debt.balance}
                      onChange={e => updateDebt(debt.id, { balance: Math.max(0, parseFloat(e.target.value) || 0) })}
                      className="h-8 text-xs" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[10px] text-muted-foreground">APR (%)</Label>
                    <Input type="number" min={0} step={0.1} value={debt.apr_pct}
                      onChange={e => updateDebt(debt.id, { apr_pct: Math.max(0, parseFloat(e.target.value) || 0) })}
                      className="h-8 text-xs" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[10px] text-muted-foreground">Min. Payment ($)</Label>
                    <Input type="number" min={0} value={debt.minimum_payment}
                      onChange={e => updateDebt(debt.id, { minimum_payment: Math.max(0, parseFloat(e.target.value) || 0) })}
                      className="h-8 text-xs" />
                  </div>
                  <div className="space-y-1 flex flex-col justify-end">
                    <Button
                      size="sm" variant="ghost" className="h-8 text-[10px] gap-1"
                      disabled={hintLoadingId === debt.id}
                      onClick={() => suggestMinimum(debt)}
                    >
                      {hintLoadingId === debt.id ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Wand2 className="w-3 h-3" />}
                      Suggest
                    </Button>
                  </div>
                </div>
              </div>
            );
          })}
          {debts.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-4">No debts yet — click "Add Debt" to get started.</p>
          )}
        </CardContent>
      </Card>

      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader className="pb-3"><CardTitle className="text-sm">Extra Monthly Payment</CardTitle></CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs font-medium">Budget above the minimums, applied by Snowball / Avalanche</Label>
            <Badge variant="outline" className="text-xs">${extraMonthlyPayment}/mo</Badge>
          </div>
          <Slider min={0} max={2000} step={25} value={[extraMonthlyPayment]} onValueChange={([v]) => setExtraMonthlyPayment(v)} />
          <p className="text-[10px] text-muted-foreground">"Minimum Only" never uses this — it's the no-extra-effort baseline.</p>
        </CardContent>
      </Card>

      <Button onClick={runSimulation} disabled={isSimulating || profileLoading} className="w-full h-11">
        {isSimulating
          ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Simulating...</>
          : <><PlayCircle className="w-4 h-4 mr-2" />Run Comparison</>}
      </Button>

      {simError && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="p-3 flex items-center gap-2 text-xs text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" />{simError}
          </CardContent>
        </Card>
      )}

      {result && (
        <Tabs defaultValue="timeline">
          <TabsList className="grid grid-cols-3 w-full max-w-md">
            <TabsTrigger value="timeline" className="text-xs">Payoff Timeline</TabsTrigger>
            <TabsTrigger value="summary" className="text-xs">Strategy Summary</TabsTrigger>
            <TabsTrigger value="order" className="text-xs">Payoff Order</TabsTrigger>
          </TabsList>

          {anyHitCap && (
            <div className="flex items-center gap-2 text-xs text-amber-400 mt-3">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              One or more strategies didn't finish within 50 years — your minimum payments may not be covering the interest on some debts.
            </div>
          )}

          <TabsContent value="timeline" className="mt-3">
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Total Remaining Balance Over Time</CardTitle></CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                    <XAxis dataKey="month" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                      label={{ value: "Month", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
                    <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickFormatter={v => formatMoney(v)} width={60} />
                    <Tooltip formatter={(v: number, name: string) => [formatMoney(v), name]}
                      contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: "11px" }} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    {strategyOrder.map(s => (
                      <Line key={s} type="monotone" dataKey={STRATEGY_LABELS[s]} stroke={STRATEGY_COLORS[s]}
                        strokeWidth={s === "minimum_only" ? 1.5 : 2.5} strokeDasharray={s === "minimum_only" ? "5 3" : undefined} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="summary" className="mt-3 space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {strategyOrder.map(s => {
                const r = result.strategies[s];
                return (
                  <Card key={s} className={s === "avalanche" ? "border-emerald-500/30 bg-emerald-500/5" : undefined}>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-1.5">
                        {s === "avalanche" && <Trophy className="w-3.5 h-3.5 text-emerald-400" />}{STRATEGY_LABELS[s]}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">Debt-free in</span>
                        <span className="font-semibold text-foreground">{formatMonths(r.months_to_debt_free)}</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">Total interest paid</span>
                        <span className="font-semibold text-foreground">{formatMoney(r.total_interest_paid)}</span>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">What You'd Save vs. Minimum Only</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Snowball</span>
                  <span className="font-semibold text-amber-400">
                    {formatMoney(result.summary.snowball_vs_minimum.interest_saved)} saved
                    {result.summary.snowball_vs_minimum.months_saved !== null && `, ${result.summary.snowball_vs_minimum.months_saved} months faster`}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Avalanche</span>
                  <span className="font-semibold text-emerald-400">
                    {formatMoney(result.summary.avalanche_vs_minimum.interest_saved)} saved
                    {result.summary.avalanche_vs_minimum.months_saved !== null && `, ${result.summary.avalanche_vs_minimum.months_saved} months faster`}
                  </span>
                </div>
                <div className="border-t border-border/40 pt-2 flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Avalanche vs. Snowball</span>
                  <span className="font-medium text-foreground">
                    {formatMoney(result.summary.avalanche_vs_snowball.interest_saved)} less interest
                  </span>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="order" className="mt-3">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {strategyOrder.map(s => {
                const r = result.strategies[s];
                const rows = [...r.payoff_order]
                  .map(o => ({ ...o, debt: debts.find(d => d.id === o.id) }))
                  .sort((a, b) => (a.payoff_month ?? Infinity) - (b.payoff_month ?? Infinity));
                return (
                  <Card key={s}>
                    <CardHeader className="pb-2"><CardTitle className="text-sm">{STRATEGY_LABELS[s]}</CardTitle></CardHeader>
                    <CardContent className="space-y-1.5">
                      {rows.map(row => (
                        <div key={row.id} className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground truncate">{row.debt?.name || "Unnamed debt"}</span>
                          <span className="font-medium text-foreground shrink-0 ml-2">{formatMonths(row.payoff_month)}</span>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>
        </Tabs>
      )}

      <p className="text-[10px] text-muted-foreground text-center px-4">
        Educational estimate only — assumes fixed APRs and payments with no new charges, missed payments, or promotional-rate changes.
      </p>
    </div>
  );
}
