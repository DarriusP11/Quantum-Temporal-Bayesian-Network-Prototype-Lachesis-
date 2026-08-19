import { useState, useEffect } from "react";
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
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from "recharts";
import {
  Home, Building2, GraduationCap, Warehouse, RefreshCw, PlayCircle, AlertCircle, CloudOff, RotateCcw, Info,
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import {
  apiHomePlanningDefaults, apiHomePlanningSimulate, apiHomePlanningGet, apiHomePlanningSave,
  type HomePlanningType, type HomePlanningEvaluation, type HomePlanningSimulateResponse,
  type HomePlanningUtilityDefault,
} from "@/lib/api";

const TYPE_LABELS: Record<HomePlanningType, string> = {
  home: "Home (Buy)",
  apartment: "Apartment (Rent)",
  dorm: "Dorm",
  mobile_home: "Mobile Home",
};

const TYPE_ICONS: Record<HomePlanningType, typeof Home> = {
  home: Home, apartment: Building2, dorm: GraduationCap, mobile_home: Warehouse,
};

const DEFAULT_INPUTS_BY_TYPE: Record<HomePlanningType, Record<string, number | boolean>> = {
  home: {
    purchase_price: 300000, down_payment_pct: 20, mortgage_rate_pct: 7, term_years: 30,
    property_tax_rate_pct: 1.1, annual_insurance: 1500, hoa_monthly: 0, closing_costs_pct: 3,
  },
  apartment: { monthly_rent: 1500, deposit_multiplier: 1, renters_insurance_monthly: 15 },
  dorm: { cost_per_semester: 6000, semesters_per_year: 2, meal_plan_included: true },
  mobile_home: {
    purchase_price: 80000, down_payment_pct: 10, loan_rate_pct: 9, term_years: 20,
    lot_rent_monthly: 450, annual_insurance: 800,
  },
};

const FALLBACK_UTILITIES: Record<string, HomePlanningUtilityDefault> = {
  electricity: { label: "Electricity", default_monthly: 135 },
  water_sewer: { label: "Water & Sewer", default_monthly: 45 },
  gas_heating: { label: "Gas / Heating", default_monthly: 70 },
  internet: { label: "Internet", default_monthly: 65 },
  trash: { label: "Trash & Recycling", default_monthly: 25 },
};

const FIELD_LABELS: Record<string, string> = {
  monthly_pi: "Principal & Interest",
  monthly_tax: "Property Tax",
  monthly_insurance: "Insurance",
  monthly_hoa: "HOA",
  monthly_pmi: "PMI",
  monthly_rent: "Rent",
  monthly_lot_rent: "Lot Rent",
};

interface OptionState {
  type: HomePlanningType;
  inputs: Record<string, number | boolean>;
}

const STORAGE_KEY = "lachesis_home_planning";

function formatMoney(n: number) {
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

function evaluationLineItems(ev: HomePlanningEvaluation): { label: string; value: number }[] {
  const items = Object.entries(ev)
    .filter(([k, v]) => k.startsWith("monthly_") && k !== "monthly_total" && typeof v === "number" && v !== 0)
    .map(([k, v]) => ({ label: FIELD_LABELS[k] ?? k.replace(/^monthly_/, "").replace(/_/g, " "), value: v as number }));
  if (typeof ev.utilities_total === "number" && ev.utilities_total > 0) {
    items.push({ label: "Utilities", value: ev.utilities_total as number });
  }
  return items;
}

function OptionEditor({
  title, option, setOption,
}: {
  title: string;
  option: OptionState;
  setOption: (o: OptionState) => void;
}) {
  const Icon = TYPE_ICONS[option.type];
  const updateInput = (key: string, value: number | boolean) => {
    setOption({ ...option, inputs: { ...option.inputs, [key]: value } });
  };
  const num = (key: string) => (option.inputs[key] as number) ?? 0;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2"><Icon className="w-4 h-4" />{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-1.5">
          <Label className="text-xs font-medium">Housing Type</Label>
          <Select
            value={option.type}
            onValueChange={v => setOption({ type: v as HomePlanningType, inputs: DEFAULT_INPUTS_BY_TYPE[v as HomePlanningType] })}
          >
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {(Object.keys(TYPE_LABELS) as HomePlanningType[]).map(t => (
                <SelectItem key={t} value={t} className="text-xs">{TYPE_LABELS[t]}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {option.type === "home" && (
          <div className="grid grid-cols-2 gap-3">
            <Field label="Purchase Price ($)" value={num("purchase_price")} onChange={v => updateInput("purchase_price", v)} />
            <Field label="Down Payment (%)" value={num("down_payment_pct")} onChange={v => updateInput("down_payment_pct", v)} />
            <Field label="Mortgage Rate (%)" value={num("mortgage_rate_pct")} step={0.1} onChange={v => updateInput("mortgage_rate_pct", v)} />
            <Field label="Term (years)" value={num("term_years")} onChange={v => updateInput("term_years", v)} />
            <Field label="Property Tax Rate (%/yr)" value={num("property_tax_rate_pct")} step={0.1} onChange={v => updateInput("property_tax_rate_pct", v)} />
            <Field label="Insurance ($/yr)" value={num("annual_insurance")} onChange={v => updateInput("annual_insurance", v)} />
            <Field label="HOA ($/mo)" value={num("hoa_monthly")} onChange={v => updateInput("hoa_monthly", v)} />
            <Field label="Closing Costs (%)" value={num("closing_costs_pct")} step={0.5} onChange={v => updateInput("closing_costs_pct", v)} />
          </div>
        )}

        {option.type === "apartment" && (
          <div className="grid grid-cols-2 gap-3">
            <Field label="Monthly Rent ($)" value={num("monthly_rent")} onChange={v => updateInput("monthly_rent", v)} />
            <Field label="Deposit (× rent)" value={num("deposit_multiplier")} step={0.5} onChange={v => updateInput("deposit_multiplier", v)} />
            <Field label="Renters Insurance ($/mo)" value={num("renters_insurance_monthly")} onChange={v => updateInput("renters_insurance_monthly", v)} />
          </div>
        )}

        {option.type === "dorm" && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Cost per Semester ($)" value={num("cost_per_semester")} onChange={v => updateInput("cost_per_semester", v)} />
              <Field label="Semesters / Year" value={num("semesters_per_year")} onChange={v => updateInput("semesters_per_year", v)} />
            </div>
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium">Meal plan included</Label>
              <Switch checked={Boolean(option.inputs.meal_plan_included)} onCheckedChange={v => updateInput("meal_plan_included", v)} />
            </div>
            <p className="text-[10px] text-muted-foreground">Utilities are usually bundled into dorm fees, so they're excluded from this option's totals.</p>
          </div>
        )}

        {option.type === "mobile_home" && (
          <div className="grid grid-cols-2 gap-3">
            <Field label="Purchase Price ($)" value={num("purchase_price")} onChange={v => updateInput("purchase_price", v)} />
            <Field label="Down Payment (%)" value={num("down_payment_pct")} onChange={v => updateInput("down_payment_pct", v)} />
            <Field label="Loan Rate (%)" value={num("loan_rate_pct")} step={0.1} onChange={v => updateInput("loan_rate_pct", v)} />
            <Field label="Term (years)" value={num("term_years")} onChange={v => updateInput("term_years", v)} />
            <Field label="Lot Rent ($/mo)" value={num("lot_rent_monthly")} onChange={v => updateInput("lot_rent_monthly", v)} />
            <Field label="Insurance ($/yr)" value={num("annual_insurance")} onChange={v => updateInput("annual_insurance", v)} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Field({ label, value, onChange, step = 1 }: { label: string; value: number; onChange: (v: number) => void; step?: number }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs font-medium">{label}</Label>
      <Input type="number" step={step} value={value} onChange={e => onChange(parseFloat(e.target.value) || 0)} className="h-8 text-sm" />
    </div>
  );
}

export function HomePlanningDashboard() {
  const { user } = useAuth();
  const [optionA, setOptionA] = useState<OptionState>({ type: "home", inputs: DEFAULT_INPUTS_BY_TYPE.home });
  const [optionB, setOptionB] = useState<OptionState>({ type: "apartment", inputs: DEFAULT_INPUTS_BY_TYPE.apartment });
  const [utilityDefs, setUtilityDefs] = useState<Record<string, HomePlanningUtilityDefault>>(FALLBACK_UTILITIES);
  const [utilities, setUtilities] = useState<Record<string, number>>(
    Object.fromEntries(Object.entries(FALLBACK_UTILITIES).map(([k, v]) => [k, v.default_monthly]))
  );
  const [horizonYears, setHorizonYears] = useState(10);
  const [appreciationPct, setAppreciationPct] = useState(3);
  const [sellingCostPct, setSellingCostPct] = useState(6);

  const [result, setResult] = useState<HomePlanningSimulateResponse | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [profileLoading, setProfileLoading] = useState(true);

  useEffect(() => {
    apiHomePlanningDefaults()
      .then(res => {
        setUtilityDefs(res.utilities);
        setUtilities(prev => {
          const seeded = Object.fromEntries(Object.entries(res.utilities).map(([k, v]) => [k, v.default_monthly]));
          return { ...seeded, ...prev };
        });
      })
      .catch(() => { /* fall back to hardcoded defaults, already set */ });
  }, []);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const p = JSON.parse(saved);
        if (p.optionA) setOptionA(p.optionA);
        if (p.optionB) setOptionB(p.optionB);
        if (p.utilities) setUtilities(p.utilities);
        if (p.horizonYears) setHorizonYears(p.horizonYears);
        if (p.appreciationPct !== undefined) setAppreciationPct(p.appreciationPct);
        if (p.sellingCostPct !== undefined) setSellingCostPct(p.sellingCostPct);
      }
    } catch {}

    if (!user) { setProfileLoading(false); return; }
    let cancelled = false;
    (async () => {
      try {
        const plan = await apiHomePlanningGet();
        if (cancelled) return;
        if (plan.option_a) setOptionA({ type: plan.option_a.type, inputs: plan.option_a.inputs as Record<string, number | boolean> });
        if (plan.option_b) setOptionB({ type: plan.option_b.type, inputs: plan.option_b.inputs as Record<string, number | boolean> });
        if (plan.utilities && Object.keys(plan.utilities).length) setUtilities(plan.utilities);
        const cs = plan.comparison_settings as Partial<{ horizon_years: number; appreciation_rate_pct: number; selling_cost_pct: number }>;
        if (cs?.horizon_years) setHorizonYears(cs.horizon_years);
        if (cs?.appreciation_rate_pct !== undefined) setAppreciationPct(cs.appreciation_rate_pct);
        if (cs?.selling_cost_pct !== undefined) setSellingCostPct(cs.selling_cost_pct);
      } catch {
        // No saved plan yet — start fresh, not a hard error.
      } finally {
        if (!cancelled) setProfileLoading(false);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ optionA, optionB, utilities, horizonYears, appreciationPct, sellingCostPct }));
    } catch {}
  }, [optionA, optionB, utilities, horizonYears, appreciationPct, sellingCostPct]);

  const resetUtilities = () => {
    setUtilities(Object.fromEntries(Object.entries(utilityDefs).map(([k, v]) => [k, v.default_monthly])));
  };

  const runSimulation = async () => {
    setIsSimulating(true);
    setSimError(null);
    try {
      const res = await apiHomePlanningSimulate({
        option_a: optionA, option_b: optionB, utilities,
        horizon_years: horizonYears, appreciation_rate_pct: appreciationPct, selling_cost_pct: sellingCostPct,
      });
      setResult(res);
      if (user) {
        apiHomePlanningSave({
          option_a: optionA, option_b: optionB, utilities,
          comparison_settings: { horizon_years: horizonYears, appreciation_rate_pct: appreciationPct, selling_cost_pct: sellingCostPct },
        }).catch(() => { /* best-effort save */ });
      }
    } catch (e) {
      setSimError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSimulating(false);
    }
  };

  const chartData = result?.comparison?.timeline.map(p => ({
    year: p.year, "Buy (net)": p.net_buy_cost, "Rent (net)": p.net_rent_cost,
  })) ?? [];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Home Planning</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Compare what it actually costs to live somewhere — buy a home, rent an apartment, live in a dorm, or own a mobile home — including utilities.
          </p>
        </div>
        {!user && (
          <Badge variant="outline" className="text-[10px] border-amber-500/30 text-amber-400 flex items-center gap-1 shrink-0">
            <CloudOff className="w-3 h-3" />Sign in to sync
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <OptionEditor title="Option A" option={optionA} setOption={setOptionA} />
        <OptionEditor title="Option B" option={optionB} setOption={setOptionB} />
      </div>

      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader className="pb-3 flex flex-row items-center justify-between">
          <CardTitle className="text-sm">Monthly Utilities (shared, editable estimates)</CardTitle>
          <Button size="sm" variant="ghost" className="h-7 text-xs gap-1" onClick={resetUtilities}>
            <RotateCcw className="w-3 h-3" />Reset to average
          </Button>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {Object.entries(utilityDefs).map(([key, def]) => (
              <div key={key} className="space-y-1.5">
                <Label className="text-xs font-medium">{def.label}</Label>
                <div className="flex items-center gap-1">
                  <span className="text-muted-foreground text-xs">$</span>
                  <Input
                    type="number" min={0}
                    value={utilities[key] ?? def.default_monthly}
                    onChange={e => setUtilities(prev => ({ ...prev, [key]: parseFloat(e.target.value) || 0 }))}
                    className="h-8 text-sm"
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground mt-2">Applied to Home, Apartment, and Mobile Home options. Dorm costs are usually bundled, so utilities aren't added there.</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3"><CardTitle className="text-sm">Rent-vs-Buy Comparison Assumptions</CardTitle></CardHeader>
        <CardContent className="grid grid-cols-1 sm:grid-cols-3 gap-5">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium">Horizon</Label>
              <Badge variant="outline" className="text-xs">{horizonYears} yr</Badge>
            </div>
            <Slider min={1} max={30} step={1} value={[horizonYears]} onValueChange={([v]) => setHorizonYears(v)} />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium">Home Appreciation</Label>
              <Badge variant="outline" className="text-xs">{appreciationPct}%/yr</Badge>
            </div>
            <Slider min={0} max={8} step={0.5} value={[appreciationPct]} onValueChange={([v]) => setAppreciationPct(v)} />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium">Selling Costs (if sold)</Label>
              <Badge variant="outline" className="text-xs">{sellingCostPct}%</Badge>
            </div>
            <Slider min={0} max={10} step={0.5} value={[sellingCostPct]} onValueChange={([v]) => setSellingCostPct(v)} />
          </div>
        </CardContent>
      </Card>

      <Button onClick={runSimulation} disabled={isSimulating || profileLoading} className="w-full h-11">
        {isSimulating
          ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Comparing...</>
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
        <Tabs defaultValue="breakdown">
          <TabsList className="grid grid-cols-3 w-full max-w-md">
            <TabsTrigger value="breakdown" className="text-xs">Monthly Breakdown</TabsTrigger>
            <TabsTrigger value="timeline" className="text-xs">Rent vs Buy</TabsTrigger>
            <TabsTrigger value="upfront" className="text-xs">Upfront Costs</TabsTrigger>
          </TabsList>

          <TabsContent value="breakdown" className="mt-3">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {[{ label: "Option A", ev: result.option_a }, { label: "Option B", ev: result.option_b }].map(({ label, ev }) => ev && (
                <Card key={label}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center justify-between">
                      <span>{label} — {TYPE_LABELS[ev.type]}</span>
                      <Badge variant="outline" className="text-emerald-400 border-emerald-500/30">{formatMoney(ev.grand_total_monthly)}/mo</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-1.5">
                    {evaluationLineItems(ev).map(item => (
                      <div key={item.label} className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground capitalize">{item.label}</span>
                        <span className="font-medium text-foreground">{formatMoney(item.value)}</span>
                      </div>
                    ))}
                    <div className="border-t border-border/40 pt-1.5 mt-1.5 flex justify-between text-xs font-semibold">
                      <span>Total / month</span><span>{formatMoney(ev.grand_total_monthly)}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="timeline" className="mt-3 space-y-4">
            {!result.comparison ? (
              <Card className="border-dashed border-border/40">
                <CardContent className="p-6 text-center text-xs text-muted-foreground flex flex-col items-center gap-2">
                  <Info className="w-5 h-5" />
                  A rent-vs-buy timeline needs one Buy option (Home or Mobile Home) and one Rent option (Apartment or Dorm). Both options are currently the same category, so only the monthly breakdown applies.
                </CardContent>
              </Card>
            ) : (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <Card>
                    <CardContent className="p-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Upfront: Buy</p>
                      <p className="text-lg font-bold text-foreground">{formatMoney(result.comparison.upfront_comparison.buy)}</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Upfront: Rent</p>
                      <p className="text-lg font-bold text-foreground">{formatMoney(result.comparison.upfront_comparison.rent)}</p>
                    </CardContent>
                  </Card>
                  <Card className="border-emerald-500/30 bg-emerald-500/5">
                    <CardContent className="p-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Breakeven</p>
                      <p className="text-lg font-bold text-emerald-400">
                        {result.comparison.breakeven_year ? `Year ${result.comparison.breakeven_year}` : `Not within ${horizonYears}yr`}
                      </p>
                    </CardContent>
                  </Card>
                </div>
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-sm">Net Cost Over Time (net of a hypothetical home sale)</CardTitle></CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                        <XAxis dataKey="year" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                          label={{ value: "Year", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
                        <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickFormatter={v => formatMoney(v)} width={60} />
                        <Tooltip formatter={(v: number, name: string) => [formatMoney(v), name]}
                          contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: "11px" }} />
                        <Legend wrapperStyle={{ fontSize: "11px" }} />
                        {result.comparison.breakeven_year && (
                          <ReferenceLine x={result.comparison.breakeven_year} stroke="#10b981" strokeDasharray="3 3" />
                        )}
                        <Line type="monotone" dataKey="Buy (net)" stroke="#10b981" strokeWidth={2.5} dot={false} />
                        <Line type="monotone" dataKey="Rent (net)" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 3" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>

          <TabsContent value="upfront" className="mt-3">
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Upfront Cost Comparison</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                {[{ label: `Option A — ${TYPE_LABELS[optionA.type]}`, ev: result.option_a }, { label: `Option B — ${TYPE_LABELS[optionB.type]}`, ev: result.option_b }].map(({ label, ev }) => ev && (
                  <div key={label} className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">{label}</span>
                    <span className="font-semibold text-foreground">{formatMoney(ev.upfront_cost)}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      <p className="text-[10px] text-muted-foreground text-center px-4">
        Educational estimate only — utility, tax, and insurance figures are rough national averages, and the rent-vs-buy comparison is a simplified cash-flow model (it doesn't account for investing the down payment elsewhere or mortgage-interest tax deductions).
      </p>
    </div>
  );
}
