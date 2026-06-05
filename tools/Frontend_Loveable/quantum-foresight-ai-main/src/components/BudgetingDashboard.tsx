import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Plus, Trash2, ChevronDown, ChevronUp, DollarSign, TrendingUp, AlertCircle } from "lucide-react";

interface BudgetItem {
  id: string;
  name: string;
  amount: string;
}

interface BudgetCategory {
  id: string;
  emoji: string;
  label: string;
  items: BudgetItem[];
}

const DEFAULT_CATEGORIES: BudgetCategory[] = [
  { id: "housing",        emoji: "🏠", label: "Housing",        items: [] },
  { id: "food",           emoji: "🍕", label: "Food",           items: [] },
  { id: "transportation", emoji: "🚗", label: "Transportation", items: [] },
  { id: "entertainment",  emoji: "🏀", label: "Entertainment",  items: [] },
  { id: "education",      emoji: "📚", label: "Education",      items: [] },
  { id: "healthcare",     emoji: "💊", label: "Healthcare",     items: [] },
  { id: "clothing",       emoji: "👕", label: "Clothing",       items: [] },
  { id: "subscriptions",  emoji: "📱", label: "Subscriptions",  items: [] },
  { id: "savings",        emoji: "🐷", label: "Savings",        items: [] },
];

const CHART_COLORS = [
  "#6366f1", "#10b981", "#f59e0b", "#ef4444",
  "#8b5cf6", "#06b6d4", "#f97316", "#14b8a6", "#ec4899",
];

const STORAGE_KEY = "lachesis_budget";

function uid() {
  return Math.random().toString(36).slice(2);
}

function categoryTotal(cat: BudgetCategory): number {
  return cat.items.reduce((sum, i) => sum + (parseFloat(i.amount) || 0), 0);
}

export function BudgetingDashboard() {
  const [income, setIncome] = useState("");
  const [categories, setCategories] = useState<BudgetCategory[]>(DEFAULT_CATEGORIES);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [newCatEmoji, setNewCatEmoji] = useState("✨");
  const [newCatLabel, setNewCatLabel] = useState("");
  const [addingCat, setAddingCat] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);

  // Persist to localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const { income: si, categories: sc } = JSON.parse(saved);
        if (si !== undefined) setIncome(si);
        if (sc) setCategories(sc);
      }
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ income, categories }));
    } catch {}
  }, [income, categories]);

  const totalSpend = categories.reduce((sum, c) => sum + categoryTotal(c), 0);
  const incomeNum = parseFloat(income) || 0;
  const surplus = incomeNum - totalSpend;
  const surplusPositive = surplus >= 0;

  // Add item to category
  const addItem = (catId: string) => {
    setCategories(prev => prev.map(c =>
      c.id === catId
        ? { ...c, items: [...c.items, { id: uid(), name: "", amount: "" }] }
        : c
    ));
  };

  const updateItem = (catId: string, itemId: string, field: "name" | "amount", value: string) => {
    setCategories(prev => prev.map(c =>
      c.id === catId
        ? { ...c, items: c.items.map(i => i.id === itemId ? { ...i, [field]: value } : i) }
        : c
    ));
  };

  const deleteItem = (catId: string, itemId: string) => {
    setCategories(prev => prev.map(c =>
      c.id === catId ? { ...c, items: c.items.filter(i => i.id !== itemId) } : c
    ));
  };

  const addCategory = () => {
    if (!newCatLabel.trim()) return;
    setCategories(prev => [...prev, {
      id: uid(), emoji: newCatEmoji, label: newCatLabel.trim(), items: [],
    }]);
    setNewCatLabel("");
    setNewCatEmoji("✨");
    setAddingCat(false);
  };

  const deleteCategory = (catId: string) => {
    setCategories(prev => prev.filter(c => c.id !== catId));
    if (expanded === catId) setExpanded(null);
  };

  // Pie chart data — only non-zero categories
  const pieData = categories
    .map((c, i) => ({ name: c.label, value: categoryTotal(c), color: CHART_COLORS[i % CHART_COLORS.length] }))
    .filter(d => d.value > 0);

  // 50/30/20 rule
  const needsPct = incomeNum > 0 ? Math.round(((categoryTotal(categories.find(c => c.id === "housing")!) +
    categoryTotal(categories.find(c => c.id === "food")!) +
    categoryTotal(categories.find(c => c.id === "transportation")!) +
    categoryTotal(categories.find(c => c.id === "healthcare")!)) / incomeNum) * 100) : 0;
  const savingsPct = incomeNum > 0 ? Math.round((categoryTotal(categories.find(c => c.id === "savings")!) / incomeNum) * 100) : 0;
  const wantsPct = incomeNum > 0 ? Math.max(0, Math.round((totalSpend / incomeNum) * 100) - needsPct) : 0;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-foreground">Budget Planner</h2>
        <p className="text-muted-foreground text-sm mt-1">Track your monthly spending, set goals, and stay on top of your finances.</p>
      </div>

      {/* Income + Summary Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="border-emerald-500/30 bg-emerald-500/5">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground mb-1 font-medium flex items-center gap-1"><DollarSign className="w-3 h-3" />Monthly Take-Home</p>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground text-sm">$</span>
              <Input
                type="number"
                placeholder="e.g. 2500"
                value={income}
                onChange={e => setIncome(e.target.value)}
                className="h-8 text-sm border-emerald-500/30 bg-background/60"
              />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground mb-1 font-medium flex items-center gap-1"><TrendingUp className="w-3 h-3" />Total Budgeted</p>
            <p className="text-xl font-bold text-foreground">${totalSpend.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
          </CardContent>
        </Card>
        <Card className={surplusPositive ? "border-emerald-500/30 bg-emerald-500/5" : "border-red-500/30 bg-red-500/5"}>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground mb-1 font-medium flex items-center gap-1">
              <AlertCircle className="w-3 h-3" />{surplusPositive ? "Surplus" : "Deficit"}
            </p>
            <p className={`text-xl font-bold ${surplusPositive ? "text-emerald-400" : "text-red-400"}`}>
              {surplusPositive ? "+" : ""}${Math.abs(surplus).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Category Grid */}
      <div>
        <h3 className="text-sm font-semibold text-foreground mb-3">Categories — click to add items</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {categories.map((cat, idx) => {
            const total = categoryTotal(cat);
            const isExpanded = expanded === cat.id;
            return (
              <div key={cat.id} className="col-span-1">
                <button
                  onClick={() => setExpanded(isExpanded ? null : cat.id)}
                  className={`w-full p-4 rounded-xl border text-left transition-all hover:border-emerald-500/50 ${
                    isExpanded
                      ? "border-emerald-500/60 bg-emerald-500/10"
                      : "border-border/40 bg-card/60 hover:bg-card"
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <span className="text-2xl">{cat.emoji}</span>
                    <div className="flex items-center gap-1">
                      {isExpanded ? <ChevronUp className="w-3 h-3 text-muted-foreground" /> : <ChevronDown className="w-3 h-3 text-muted-foreground" />}
                    </div>
                  </div>
                  <p className="text-xs font-semibold text-foreground mt-2">{cat.label}</p>
                  {total > 0
                    ? <p className="text-xs text-emerald-400 font-medium">${total.toFixed(2)}/mo</p>
                    : <p className="text-xs text-muted-foreground">No items yet</p>}
                </button>
              </div>
            );
          })}

          {/* Add Category button */}
          <div className="col-span-1">
            {addingCat ? (
              <div className="p-3 rounded-xl border border-dashed border-emerald-500/50 bg-emerald-500/5 space-y-2">
                <div className="flex gap-1">
                  <Input value={newCatEmoji} onChange={e => setNewCatEmoji(e.target.value)} className="w-12 h-7 text-center text-sm p-1" maxLength={2} />
                  <Input value={newCatLabel} onChange={e => setNewCatLabel(e.target.value)} placeholder="Label" className="flex-1 h-7 text-xs" />
                </div>
                <div className="flex gap-1">
                  <Button size="sm" className="flex-1 h-6 text-xs bg-emerald-600 hover:bg-emerald-700" onClick={addCategory}>Add</Button>
                  <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => setAddingCat(false)}>Cancel</Button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setAddingCat(true)}
                className="w-full h-full min-h-[90px] rounded-xl border border-dashed border-border/40 flex flex-col items-center justify-center gap-1 text-muted-foreground hover:text-foreground hover:border-emerald-500/40 transition-all"
              >
                <Plus className="w-5 h-5" />
                <span className="text-xs font-medium">Add Category</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Expanded Category Detail */}
      {expanded && (() => {
        const cat = categories.find(c => c.id === expanded)!;
        return (
          <Card className="border-emerald-500/30 bg-emerald-500/5">
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-sm flex items-center gap-2">
                <span className="text-xl">{cat.emoji}</span> {cat.label}
                <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 text-xs">
                  ${categoryTotal(cat).toFixed(2)}/mo
                </Badge>
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                className="text-red-400 hover:text-red-300 hover:bg-red-500/10 h-7 text-xs"
                onClick={() => deleteCategory(cat.id)}
              >
                <Trash2 className="w-3 h-3 mr-1" />Delete Category
              </Button>
            </CardHeader>
            <CardContent className="space-y-2">
              {cat.items.length === 0 && (
                <p className="text-xs text-muted-foreground">No items yet. Add your first one below!</p>
              )}
              {cat.items.map(item => (
                <div key={item.id} className="flex items-center gap-2">
                  <Input
                    placeholder="e.g. Netflix, Rent, Groceries..."
                    value={item.name}
                    onChange={e => updateItem(cat.id, item.id, "name", e.target.value)}
                    className="flex-1 h-8 text-xs border-border/40 bg-background/60"
                  />
                  <span className="text-muted-foreground text-xs">$</span>
                  <Input
                    type="number"
                    placeholder="0.00"
                    value={item.amount}
                    onChange={e => updateItem(cat.id, item.id, "amount", e.target.value)}
                    className="w-24 h-8 text-xs border-border/40 bg-background/60"
                  />
                  <span className="text-xs text-muted-foreground">/mo</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                    onClick={() => deleteItem(cat.id, item.id)}
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              ))}
              <Button
                variant="outline"
                size="sm"
                className="w-full h-7 text-xs border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/10 mt-1"
                onClick={() => addItem(cat.id)}
              >
                <Plus className="w-3 h-3 mr-1" />Add Item
              </Button>
            </CardContent>
          </Card>
        );
      })()}

      {/* Summary: Chart + Table */}
      {pieData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Spending Breakdown</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: number) => [`$${value.toFixed(2)}`, "Monthly"]}
                    contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: "11px" }}
                  />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: "11px" }} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Monthly Summary</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-1.5">
                {categories.filter(c => categoryTotal(c) > 0).map((cat, i) => (
                  <div key={cat.id} className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: CHART_COLORS[categories.indexOf(cat) % CHART_COLORS.length] }} />
                      {cat.emoji} {cat.label}
                    </span>
                    <span className="font-medium">${categoryTotal(cat).toFixed(2)}</span>
                  </div>
                ))}
                <div className="border-t border-border/40 pt-1.5 mt-1.5 flex justify-between text-xs font-semibold">
                  <span>Total</span><span>${totalSpend.toFixed(2)}</span>
                </div>
                {incomeNum > 0 && (
                  <div className={`flex justify-between text-xs font-semibold ${surplusPositive ? "text-emerald-400" : "text-red-400"}`}>
                    <span>{surplusPositive ? "Surplus" : "Deficit"}</span>
                    <span>{surplusPositive ? "+" : "-"}${Math.abs(surplus).toFixed(2)}</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* 50/30/20 Guide */}
      <Collapsible open={guideOpen} onOpenChange={setGuideOpen}>
        <CollapsibleTrigger className="flex w-full items-center justify-between p-4 rounded-xl border border-border/40 bg-card/60 hover:bg-card transition-colors text-sm font-semibold">
          <span>💡 The 50/30/20 Budgeting Rule</span>
          <ChevronDown className={`w-4 h-4 transition-transform ${guideOpen ? "rotate-180" : ""}`} />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-2 p-4 rounded-xl border border-border/40 bg-card/40 space-y-3">
            <p className="text-xs text-muted-foreground">A simple framework for budgeting — spend 50% on needs, 30% on wants, and save 20%.</p>
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Needs", target: 50, actual: needsPct, color: "text-blue-400", bg: "bg-blue-500/10 border-blue-500/30", desc: "Housing, food, transport, healthcare" },
                { label: "Wants", target: 30, actual: wantsPct, color: "text-purple-400", bg: "bg-purple-500/10 border-purple-500/30", desc: "Entertainment, clothing, subscriptions" },
                { label: "Savings", target: 20, actual: savingsPct, color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/30", desc: "Emergency fund, retirement, investments" },
              ].map(({ label, target, actual, color, bg, desc }) => (
                <div key={label} className={`p-3 rounded-lg border ${bg} text-center`}>
                  <p className="text-xs font-semibold text-foreground">{label}</p>
                  <p className="text-xs text-muted-foreground">{desc}</p>
                  <p className={`text-lg font-bold mt-1 ${color}`}>
                    {incomeNum > 0 ? `${actual}%` : "—"}
                  </p>
                  <p className="text-[10px] text-muted-foreground">target: {target}%</p>
                  {incomeNum > 0 && (
                    <Badge variant="outline" className={`text-[10px] mt-1 ${actual <= target ? "border-emerald-500/30 text-emerald-400" : "border-red-500/30 text-red-400"}`}>
                      {actual <= target ? "On track" : "Over target"}
                    </Badge>
                  )}
                </div>
              ))}
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
