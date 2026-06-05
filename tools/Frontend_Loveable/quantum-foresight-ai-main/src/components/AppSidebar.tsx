/**
 * AppSidebar.tsx — Full Lachesis sidebar matching the Streamlit sidebar.
 * Quantum controls · Noise config · Finance/Data settings · Persistence · Account
 */
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/hooks/useAuth";
import { useAppContext } from "@/contexts/AppContext";
import { ChevronDown, Save, RotateCcw, User, LogOut, HelpCircle, Atom, BarChart2 } from "lucide-react";

const OWNER_EMAIL = "darriusperson@gmail.com";

// ── Tab Guide data ─────────────────────────────────────────────────────────────
const TAB_GUIDE = [
  // AI / Classical
  { name: "Lachesis AI",         section: "ai",      description: "Chat with your AI financial assistant. Ask about your portfolio, get market insights, and receive plain-English explanations of complex financial concepts." },
  { name: "Financial Analytics", section: "ai",      description: "Analyze your stock portfolio with real market data. Configure tickers, portfolio value, lookback period, and confidence level. View VaR/CVaR, Sharpe/Sortino ratios, return charts, a correlation matrix, and Persona Views." },
  { name: "Insider Trading",     section: "ai",      description: "Track stock purchases and sales made by company executives via SEC EDGAR filings. When insiders buy their own stock heavily, it's often a bullish signal." },
  { name: "Sentiment Analysis",  section: "ai",      description: "Measures the market's mood about your stocks by scanning financial news headlines using VADER sentiment scoring. Positive coverage = bullish; negative coverage = bearish." },
  { name: "Credit Risk",         section: "ai",      description: "Pro — Quantum-powered credit risk analysis using Qiskit's Gaussian Conditional Independence model. Estimates portfolio Expected Loss, VaR, and CVaR via Iterative Quantum Amplitude Estimation or Monte Carlo simulation across borrower portfolios." },
  // Quantum / Qiskit
  { name: "Foresight",           section: "quantum", description: "Sweeps depolarizing and amplitude-damping noise parameters across a quantum circuit and measures KL-divergence from the ideal output — shows how sensitive your circuit is to hardware noise." },
  { name: "Circuit Inspector",   section: "quantum", description: "Main quantum circuit workbench with three sub-tabs: Statevector (configure qubits, gates, shots & visualize amplitude/phase), Measurement (compare ideal vs. noisy shot counts), and Noise (tune depolarizing, amplitude-damping, phase-damping, and CNOT error channels)." },
  { name: "Reduced States",      section: "quantum", description: "Computes the reduced density matrix for each qubit via partial trace and renders interactive 3D Bloch spheres. Purity near 0.5 on a multi-qubit circuit indicates entanglement." },
  { name: "Fidelity & Export",   section: "quantum", description: "Benchmarks how accurately a noisy circuit reproduces the ideal output using quantum fidelity F = (Σ√p·q)². Also lets you download simulation results as JSON for external analysis." },
  { name: "Presets",             section: "quantum", description: "Load pre-built quantum circuit configurations — Bell state, GHZ, QFT, and more — to instantly explore standard quantum algorithms without manual gate setup." },
  { name: "Present Scenarios",   section: "quantum", description: "Run and compare multiple named quantum simulation scenarios side-by-side to understand how different gate configurations or noise levels affect the output distribution." },
  { name: "Advanced Quantum",    section: "quantum", description: "Deep-dive diagnostics: state tomography (reconstruct density matrix from Pauli measurements), randomized benchmarking (measure error-per-gate decay), Bayesian noise calibration, and gate process fidelity." },
  { name: "Toy QAOA",            section: "quantum", description: "Pro — Demo of the Quantum Approximate Optimization Algorithm applied to portfolio allocation. Uses Qiskit's QAOA with a COBYLA optimizer to find optimal asset weights across configurable portfolios including Magnificent 7." },
  { name: "VQE",                 section: "quantum", description: "Pro — Variational Quantum Eigensolver used as a financial risk gate. Solves custom Hamiltonians (MaxCut, Ising, Pauli) with RealAmplitudes or EfficientSU2 ansätze, then maps the ground-state energy to a trade approval policy." },
  { name: "Quantum Hardware",    section: "quantum", description: "Enterprise — Connect to real quantum processing units via IBM Quantum, Google Quantum AI, or other cloud providers. Run circuits on actual QPUs and benchmark against simulation. Coming soon." },
] as const;

// ═══════════════════════════════════════════════════════════════════════════════
export function AppSidebar({ isOwner = false }: { isOwner?: boolean }) {
  const { user, signOut } = useAuth();
  const { state, setNumQubits, resetToDefaults, activeSection, setActiveSection } = useAppContext();

  const [openGuide, setOpenGuide] = useState(false);

  const displayName = user?.user_metadata?.display_name || user?.email || "Guest";
  const role = user?.email?.toLowerCase() === OWNER_EMAIL ? "Owner" : "User";

  return (
    <div className="flex flex-col h-full overflow-y-auto bg-card/60 backdrop-blur border-r border-accent/20 w-72 shrink-0">
      {/* ── Section Toggle ───────────────────────────────────────────────── */}
      <div className="p-4 border-b border-accent/20">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Section</p>
        <div className="flex gap-1 p-1 bg-muted/40 rounded-lg border border-accent/10">
          <button
            onClick={() => setActiveSection('classical')}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2 rounded-md text-xs font-semibold transition-all ${
              activeSection === 'classical'
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <BarChart2 className="w-3 h-3" />Classical
          </button>
          <button
            onClick={() => setActiveSection('quantum')}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2 rounded-md text-xs font-semibold transition-all ${
              activeSection === 'quantum'
                ? 'bg-primary text-primary-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <Atom className="w-3 h-3" />Quantum
          </button>
        </div>
        <p className="text-[10px] text-muted-foreground mt-1.5">
          {activeSection === 'quantum'
            ? 'Advanced quantum simulation & analytics'
            : 'Budgeting, retirement & credit tools'}
        </p>
      </div>

      {/* ── Account ──────────────────────────────────────────────────────── */}
      <div className="p-4 border-b border-accent/20">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
            <User className="w-4 h-4 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{displayName}</p>
            <Badge variant="outline" className={`text-xs ${role === "Owner" ? "border-primary/50 text-primary" : "border-accent/50"}`}>
              {role}
            </Badge>
          </div>
          <Button variant="ghost" size="icon" onClick={signOut} title="Sign out">
            <LogOut className="w-4 h-4 text-muted-foreground" />
          </Button>
        </div>
      </div>

      {/* ── Tab Guide ────────────────────────────────────────────────────── */}
      <Collapsible open={openGuide} onOpenChange={setOpenGuide} className="px-3 py-2">
        <CollapsibleTrigger className="flex w-full items-center justify-between py-1 text-sm font-semibold hover:text-primary transition-colors">
          <div className="flex items-center gap-2">
            <HelpCircle className="w-4 h-4 text-primary" />
            Tab Guide
          </div>
          <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${openGuide ? "rotate-180" : ""}`} />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-2 max-h-80 overflow-y-auto space-y-3 pr-1 pb-1">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">AI / Classical</p>
            {TAB_GUIDE.filter(t => t.section === "ai").map(tab => (
              <div key={tab.name} className="space-y-0.5">
                <p className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                  <span className="text-primary text-[8px]">●</span>{tab.name}
                </p>
                <p className="text-[11px] text-muted-foreground leading-relaxed pl-3">{tab.description}</p>
              </div>
            ))}
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground pt-1">Quantum / Qiskit</p>
            {TAB_GUIDE.filter(t => t.section === "quantum").map(tab => (
              <div key={tab.name} className="space-y-0.5">
                <p className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                  <span className="text-accent text-[8px]">●</span>{tab.name}
                </p>
                <p className="text-[11px] text-muted-foreground leading-relaxed pl-3">{tab.description}</p>
              </div>
            ))}
          </div>
        </CollapsibleContent>
      </Collapsible>

      <Separator />

      {/* ── Persistence ──────────────────────────────────────────────────── */}
      <div className="p-4 flex gap-2">
        <Button
          variant="outline" size="sm" className="flex-1 text-xs"
          onClick={() => {
            try {
              localStorage.setItem("lachesis_settings", JSON.stringify(state));
            } catch {}
          }}
        >
          <Save className="w-3 h-3 mr-1" />Save
        </Button>
        <Button
          variant="outline" size="sm" className="flex-1 text-xs"
          onClick={() => {
            try {
              const saved = localStorage.getItem("lachesis_settings");
              if (saved) {
                const parsed = JSON.parse(saved);
                if (parsed.num_qubits) setNumQubits(parsed.num_qubits);
              }
            } catch {}
          }}
        >
          <Save className="w-3 h-3 mr-1" />Load
        </Button>
        <Button
          variant="ghost" size="sm" className="text-xs"
          onClick={resetToDefaults}
          title="Reset to defaults"
        >
          <RotateCcw className="w-3 h-3" />
        </Button>
      </div>
    </div>
  );
}
