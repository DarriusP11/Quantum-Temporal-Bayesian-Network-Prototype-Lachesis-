import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

// ── Context & Sidebar ────────────────────────────────────────────────────────
import { AppProvider, useAppContext, SUPPORTED_LANGUAGES } from "@/contexts/AppContext";
import { AppSidebar } from "@/components/AppSidebar";
import AuthGuard from "@/components/AuthGuard";
import { useAuth } from "@/hooks/useAuth";

// ── Subscription ─────────────────────────────────────────────────────────────
import { useSubscription } from "@/hooks/useSubscription";
import { LockedTabOverlay } from "@/components/LockedTabOverlay";
import { SubscriptionBadge } from "@/components/SubscriptionBadge";
import { PricingModal } from "@/components/PricingModal";
import { post } from "@/lib/api";

// ── Quantum tabs ─────────────────────────────────────────────────────────────
import { CircuitInspectorDashboard } from "@/components/CircuitInspectorDashboard";
import { ReducedStatesDashboard }    from "@/components/ReducedStatesDashboard";
import { FidelityExportDashboard }   from "@/components/FidelityExportDashboard";
import { PresetsDashboard }          from "@/components/PresetsDashboard";
import { PresentScenariosDashboard } from "@/components/PresentScenariosDashboard";
import { AdvancedQuantumDashboard }  from "@/components/AdvancedQuantumDashboard";

// ── Finance / AI tabs ────────────────────────────────────────────────────────
import { ForesightDashboard }        from "@/components/ForesightDashboard";
import { FinancialDashboard }        from "@/components/FinancialDashboard";
import { InsiderTradingDashboard }   from "@/components/InsiderTradingDashboard";
import { CreditRiskDashboard }       from "@/components/CreditRiskDashboard";
import { QTBNDashboard }             from "@/components/QTBNDashboard";
import { QAOADashboard }             from "@/components/QAOADashboard";
import { SentimentDashboard }        from "@/components/SentimentDashboard";
import { PromptStudioDashboard }     from "@/components/PromptStudioDashboard";
import { VQEDashboard }              from "@/components/VQEDashboard";
import { LachesisAssistant }         from "@/components/LachesisAssistant";
import { AdminDashboard }            from "@/components/AdminDashboard";
import { QuantumHardwareTab }        from "@/components/QuantumHardwareTab";

import {
  Atom, Layers, Shield, BookOpen, BarChart2,
  TrendingUp, TrendingDown, Briefcase, Sparkles, Brain, Zap, Newspaper,
  Wand2, Gauge, KeyRound, Thermometer, LineChart, Cpu,
} from "lucide-react";

const OWNER_EMAIL = "darriusperson@gmail.com";

const TABS = [
  // ── AI / Classical ───────────────────────────────────────────────────
  { value: "assistant",      label: "Lachesis AI",        icon: Sparkles },
  { value: "finance",        label: "Financial Analytics",icon: TrendingUp },
  { value: "insider",        label: "Insider Trading",    icon: Briefcase },
  { value: "sentiment",      label: "Sentiment Analysis", icon: Newspaper },
  { value: "credit-risk",    label: "Credit Risk",        icon: TrendingDown },
  { value: "prompt-studio",  label: "Prompt Studio",      icon: Wand2 },
  // ── Quantum / Qiskit ─────────────────────────────────────────────────
  { value: "qtbn",           label: "Q-TBN",              icon: Brain },
  { value: "foresight",      label: "Foresight",          icon: Thermometer },
  { value: "circuit-inspector", label: "Circuit Inspector", icon: Atom },
  { value: "reduced",           label: "Reduced States",    icon: Layers },
  { value: "fidelity",       label: "Fidelity & Export",  icon: Shield },
  { value: "presets",        label: "Presets",            icon: BookOpen },
  { value: "scenarios",      label: "Present Scenarios",  icon: BarChart2 },
  { value: "advanced",       label: "Advanced Quantum",   icon: Gauge },
  { value: "qaoa",             label: "Toy QAOA",            icon: Zap },
  { value: "vqe",             label: "VQE",                 icon: LineChart },
  { value: "quantum-hardware",label: "Quantum Hardware",    icon: Cpu },
] as const;

// ── Subscription gating ──────────────────────────────────────────────────────
const PRO_TABS        = new Set(["qaoa", "vqe", "credit-risk"]);
const ENTERPRISE_TABS = new Set(["quantum-hardware"]);

function LanguageSelector() {
  const { state, setLanguage } = useAppContext();
  const current = SUPPORTED_LANGUAGES.find(l => l.code === state.language) ?? SUPPORTED_LANGUAGES[0];
  return (
    <Select value={state.language} onValueChange={setLanguage}>
      <SelectTrigger className="h-7 w-[130px] text-xs border-accent/30 bg-background/60">
        <SelectValue>
          <span className="font-medium">{current.native}</span>
        </SelectValue>
      </SelectTrigger>
      <SelectContent className="max-h-72 overflow-y-auto">
        {SUPPORTED_LANGUAGES.map(lang => (
          <SelectItem key={lang.code} value={lang.code} className="text-xs">
            <span className="font-medium">{lang.native}</span>
            <span className="text-muted-foreground ml-1.5">· {lang.label}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function AppLayout() {
  const { user } = useAuth();
  const isOwner = user?.email?.toLowerCase() === OWNER_EMAIL;

  // ── Subscription ────────────────────────────────────────────────────────
  const { subscription, refresh } = useSubscription();
  const [pricingOpen, setPricingOpen]   = useState(false);
  const [pricingTier, setPricingTier]   = useState<"pro" | "enterprise">("pro");

  const openUpgrade = (tier: "pro" | "enterprise" = "pro") => {
    setPricingTier(tier);
    setPricingOpen(true);
  };

  const handleManage = async () => {
    try {
      if (!user?.id) return;
      const { url } = await post<{ url: string }>("/api/billing/portal-session", { user_id: user.id });
      window.open(url, "_blank");
    } catch {
      openUpgrade(subscription.plan === "enterprise" ? "enterprise" : "pro");
    }
  };

  return (
    <>
    <PricingModal
      open={pricingOpen}
      onClose={() => setPricingOpen(false)}
      onSuccess={refresh}
      defaultTier={pricingTier}
    />
    <div className="flex h-screen overflow-hidden bg-background">
      {/* ── Left sidebar ─────────────────────────────────────────────────── */}
      <AppSidebar isOwner={isOwner} />

      {/* ── Main area ────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header bar */}
        <div className="border-b border-border/40 bg-card/80 backdrop-blur px-6 py-3 shrink-0">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Atom className="w-8 h-8 text-primary animate-pulse" />
              <div className="absolute inset-0 w-8 h-8 border-2 border-primary/30 rounded-full animate-ping" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                Lachesis
              </h1>
              <p className="text-xs text-muted-foreground hidden sm:block">
                Quantum-Enhanced Financial Analytics &amp; Foresight Platform
              </p>
            </div>
            <div className="ml-auto flex gap-2 items-center">
              <Badge variant="outline" className="border-primary/30 bg-primary/10 text-xs hidden lg:flex">
                <Brain className="w-3 h-3 mr-1" />AI-Powered
              </Badge>
              <Badge variant="outline" className="border-accent/30 bg-accent/10 text-xs hidden lg:flex">
                <Atom className="w-3 h-3 mr-1" />Quantum-Ready
              </Badge>
              {isOwner && (
                <Badge className="bg-primary/20 text-primary border-primary/30 text-xs">Owner</Badge>
              )}
              <SubscriptionBadge
                plan={subscription.plan}
                loading={subscription.loading}
                onUpgrade={() => openUpgrade("pro")}
                onManage={handleManage}
              />
              <LanguageSelector />
            </div>
          </div>
        </div>

        {/* Tab system */}
        <Tabs defaultValue="assistant" className="flex-1 flex flex-col overflow-hidden">
          {/* Scrollable tab strip */}
          <div className="border-b border-border/40 bg-card shrink-0 overflow-x-auto">
            <TabsList className="flex w-max min-w-full bg-transparent rounded-none h-10 px-2 gap-0.5">
              {TABS.map(({ value, label, icon: Icon }) => (
                <TabsTrigger
                  key={value}
                  value={value}
                  className="flex items-center gap-1.5 whitespace-nowrap px-3 h-9 text-xs rounded-sm data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-[inset_0_-2px_0_hsl(var(--primary))]"
                >
                  <Icon className="w-3 h-3" />{label}
                </TabsTrigger>
              ))}
              {isOwner && (
                <TabsTrigger
                  value="admin"
                  className="flex items-center gap-1.5 whitespace-nowrap px-3 h-9 text-xs rounded-sm data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-[inset_0_-2px_0_hsl(var(--primary))]"
                >
                  <KeyRound className="w-3 h-3" />Admin
                </TabsTrigger>
              )}
            </TabsList>
          </div>

          {/* Scrollable content area */}
          <div className="flex-1 overflow-y-auto px-6 py-6">
            <TabsContent value="circuit-inspector" className="mt-0"><CircuitInspectorDashboard /></TabsContent>
            <TabsContent value="reduced"          className="mt-0"><ReducedStatesDashboard /></TabsContent>
            <TabsContent value="fidelity"       className="mt-0"><FidelityExportDashboard /></TabsContent>
            <TabsContent value="presets"        className="mt-0"><PresetsDashboard /></TabsContent>
            <TabsContent value="scenarios"      className="mt-0"><PresentScenariosDashboard /></TabsContent>
            <TabsContent value="foresight"      className="mt-0"><ForesightDashboard /></TabsContent>
            <TabsContent value="advanced"       className="mt-0"><AdvancedQuantumDashboard /></TabsContent>
            <TabsContent value="finance"        className="mt-0"><FinancialDashboard /></TabsContent>
            <TabsContent value="insider"        className="mt-0"><InsiderTradingDashboard /></TabsContent>
            <TabsContent value="qtbn"           className="mt-0"><QTBNDashboard /></TabsContent>
            <TabsContent value="qaoa" className="mt-0">
              {subscription.is_pro || subscription.is_enterprise
                ? <QAOADashboard />
                : <LockedTabOverlay requiredPlan="pro" tabName="Toy QAOA" onUpgrade={() => openUpgrade("pro")} />}
            </TabsContent>
            <TabsContent value="sentiment"      className="mt-0"><SentimentDashboard /></TabsContent>
            <TabsContent value="credit-risk" className="mt-0">
              {subscription.is_pro || subscription.is_enterprise
                ? <CreditRiskDashboard />
                : <LockedTabOverlay requiredPlan="pro" tabName="Credit Risk" onUpgrade={() => openUpgrade("pro")} />}
            </TabsContent>
            <TabsContent value="prompt-studio"  className="mt-0"><PromptStudioDashboard /></TabsContent>
            <TabsContent value="vqe" className="mt-0">
              {subscription.is_pro || subscription.is_enterprise
                ? <VQEDashboard />
                : <LockedTabOverlay requiredPlan="pro" tabName="VQE" onUpgrade={() => openUpgrade("pro")} />}
            </TabsContent>
            <TabsContent value="quantum-hardware" className="mt-0">
              {subscription.is_enterprise
                ? <QuantumHardwareTab />
                : <LockedTabOverlay requiredPlan="enterprise" tabName="Quantum Hardware" onUpgrade={() => openUpgrade("enterprise")} />}
            </TabsContent>
            <TabsContent value="assistant"      className="mt-0"><LachesisAssistant /></TabsContent>
            {isOwner && (
              <TabsContent value="admin"        className="mt-0"><AdminDashboard /></TabsContent>
            )}
          </div>
        </Tabs>
      </div>
    </div>
    </>
  );
}

const Index = () => (
  <AuthGuard>
    <AppProvider>
      <AppLayout />
    </AppProvider>
  </AuthGuard>
);

export default Index;
