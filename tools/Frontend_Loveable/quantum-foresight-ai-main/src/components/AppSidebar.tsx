/**
 * AppSidebar.tsx — Full Lachesis sidebar matching the Streamlit sidebar.
 * Quantum controls · Noise config · Finance/Data settings · Persistence · Account
 */
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/hooks/useAuth";
import { useAppContext, GateChoice, GateStepConfig, NoiseConfig } from "@/contexts/AppContext";
import { ChevronDown, ChevronRight, Atom, Settings, TrendingUp, Save, RotateCcw, User, LogOut, Cpu } from "lucide-react";

// ── Client-side ASCII circuit preview (updates live as controls change) ───────
function buildAsciiCircuit(state: ReturnType<typeof import("@/contexts/AppContext").useAppContext>["state"]): string {
  const { num_qubits, step0, step1, step2 } = state;
  const steps = [step0, step1, step2];
  const qKeys = (["q0","q1","q2","q3"] as const).slice(0, num_qubits);
  const angleKeys = (["q0_angle","q1_angle","q2_angle","q3_angle"] as const);

  // Build rows per qubit
  const rows: string[] = qKeys.map((_, qi) => `q${qi}: `);
  const cnotRows: string[] = steps.map(() => "    ");

  steps.forEach((step) => {
    const cnotActive = (step.cnot_01 && num_qubits >= 2) ||
                       (step.cnot_12 && num_qubits >= 3) ||
                       (step.cnot_23 && num_qubits >= 4);
    qKeys.forEach((qk, qi) => {
      const g = step[qk] as string;
      const ang = step[angleKeys[qi]];
      let label = g === "None" ? "─────" :
                  ["RX","RY","RZ"].includes(g) ? `${g}(${ang.toFixed(2)})`.padEnd(8,"─") :
                  `──${g}───`;
      rows[qi] += `──[${label}]──`;
    });
    // CNOT indicator
    if (cnotActive) {
      let cnotStr = "    ";
      if (step.cnot_01 && num_qubits >= 2) cnotStr += " CNOT(q0→q1)";
      if (step.cnot_12 && num_qubits >= 3) cnotStr += " CNOT(q1→q2)";
      if (step.cnot_23 && num_qubits >= 4) cnotStr += " CNOT(q2→q3)";
      cnotRows.push(cnotStr);
    }
  });

  return rows.map(r => r + "─|").join("\n");
}

const GATES: GateChoice[] = ["None","H","X","Y","Z","RX","RY","RZ","S","T"];
const OWNER_EMAIL = "darriusperson@gmail.com";

// ── Tiny reusable noise slider ────────────────────────────────────────────────
function NoiseSlider({ label, value, onChange, min = 0, max = 0.2, step = 0.005 }: {
  label: string; value: number; onChange: (v: number) => void;
  min?: number; max?: number; step?: number;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-primary">{value.toFixed(3)}</span>
      </div>
      <Slider value={[value]} onValueChange={([v]) => onChange(v)} min={min} max={max} step={step} />
    </div>
  );
}

// ── Single step gate row ──────────────────────────────────────────────────────
function StepRow({ stepIdx, nq, step, onChange }: {
  stepIdx: number; nq: number; step: GateStepConfig;
  onChange: (s: GateStepConfig) => void;
}) {
  const qKeys = (["q0","q1","q2","q3"] as const).slice(0, nq);
  const angleKeys = (["q0_angle","q1_angle","q2_angle","q3_angle"] as const).slice(0, nq);
  const cnotPairs: [keyof GateStepConfig, number, number][] = [
    ["cnot_01",0,1], ["cnot_12",1,2], ["cnot_23",2,3]
  ];
  return (
    <div className="space-y-2 pl-2 border-l border-accent/30">
      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">T{stepIdx}</p>
      {qKeys.map((qk, qi) => (
        <div key={qk} className="flex items-center gap-2">
          <span className="text-xs w-5 text-muted-foreground">q{qi}</span>
          <Select value={step[qk] as string}
            onValueChange={v => onChange({ ...step, [qk]: v as GateChoice })}>
            <SelectTrigger className="h-7 text-xs w-20">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {GATES.map(g => <SelectItem key={g} value={g}>{g}</SelectItem>)}
            </SelectContent>
          </Select>
          {(step[qk] as string) !== "None" && ["RX","RY","RZ"].includes(step[qk] as string) && (
            <div className="flex items-center gap-1 flex-1">
              <span className="text-xs text-muted-foreground">×π</span>
              <Input
                type="number" step={0.05} min={0} max={2}
                value={step[angleKeys[qi]]}
                onChange={e => onChange({ ...step, [angleKeys[qi]]: parseFloat(e.target.value) || 0 })}
                className="h-7 text-xs w-16"
              />
            </div>
          )}
        </div>
      ))}
      {nq >= 2 && cnotPairs.filter(([, , t]) => t < nq).map(([key, c, t]) => (
        <div key={key} className="flex items-center gap-2">
          <Switch
            checked={step[key] as boolean}
            onCheckedChange={v => onChange({ ...step, [key]: v })}
            id={`cnot-${stepIdx}-${c}${t}`}
          />
          <Label htmlFor={`cnot-${stepIdx}-${c}${t}`} className="text-xs cursor-pointer">
            CNOT q{c}→q{t}
          </Label>
        </div>
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
export function AppSidebar({ isOwner = false }: { isOwner?: boolean }) {
  const { user, signOut } = useAuth();
  const { state, setNumQubits, setShots, setUseSeed, setSeedVal, setStep, setNoise, setFinance, resetToDefaults } = useAppContext();
  const { num_qubits, shots, use_seed, seed_val, step0, step1, step2, noise, finance } = state;

  const [openQuantum, setOpenQuantum]   = useState(true);
  const [openNoise, setOpenNoise]       = useState(false);
  const [openFinance, setOpenFinance]   = useState(false);

  const displayName = user?.user_metadata?.display_name || user?.email || "Guest";
  const role = user?.email?.toLowerCase() === OWNER_EMAIL ? "Owner" : "User";

  return (
    <div className="flex flex-col h-full overflow-y-auto bg-card/60 backdrop-blur border-r border-accent/20 w-72 shrink-0">
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

      {/* ── Quantum Controls ─────────────────────────────────────────────── */}
      <Collapsible open={openQuantum} onOpenChange={setOpenQuantum}>
        <CollapsibleTrigger className="flex items-center gap-2 w-full p-3 hover:bg-accent/10 text-sm font-semibold">
          <Atom className="w-4 h-4 text-primary" />
          Quantum Controls
          {openQuantum ? <ChevronDown className="w-3 h-3 ml-auto" /> : <ChevronRight className="w-3 h-3 ml-auto" />}
        </CollapsibleTrigger>
        <CollapsibleContent className="px-4 pb-4 space-y-4">
          {/* Qubits */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <Label>Qubits</Label>
              <span className="font-mono text-primary">{num_qubits}</span>
            </div>
            <Slider value={[num_qubits]} onValueChange={([v]) => setNumQubits(v)} min={1} max={4} step={1} />
          </div>
          {/* Shots */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <Label>Shots</Label>
              <span className="font-mono text-primary">{shots.toLocaleString()}</span>
            </div>
            <Slider value={[shots]} onValueChange={([v]) => setShots(v)} min={128} max={16384} step={128} />
          </div>
          {/* Seed */}
          <div className="flex items-center gap-2">
            <Switch checked={use_seed} onCheckedChange={setUseSeed} id="use-seed" />
            <Label htmlFor="use-seed" className="text-xs cursor-pointer">Fixed seed</Label>
          </div>
          {use_seed && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <Label>Seed value</Label>
                <span className="font-mono text-primary">{seed_val}</span>
              </div>
              <Slider value={[seed_val]} onValueChange={([v]) => setSeedVal(v)} min={0} max={9999} step={1} />
            </div>
          )}
          <Separator />
          {/* Gate steps */}
          <div className="space-y-3">
            <StepRow stepIdx={0} nq={num_qubits} step={step0} onChange={s => setStep(0, s)} />
            <StepRow stepIdx={1} nq={num_qubits} step={step1} onChange={s => setStep(1, s)} />
            <StepRow stepIdx={2} nq={num_qubits} step={step2} onChange={s => setStep(2, s)} />
          </div>
        </CollapsibleContent>
      </Collapsible>

      <Separator />

      {/* ── Noise Configuration ──────────────────────────────────────────── */}
      <Collapsible open={openNoise} onOpenChange={setOpenNoise}>
        <CollapsibleTrigger className="flex items-center gap-2 w-full p-3 hover:bg-accent/10 text-sm font-semibold">
          <Settings className="w-4 h-4 text-primary" />
          Noise Configuration
          {openNoise ? <ChevronDown className="w-3 h-3 ml-auto" /> : <ChevronRight className="w-3 h-3 ml-auto" />}
        </CollapsibleTrigger>
        <CollapsibleContent className="px-4 pb-4 space-y-4">
          {/* Depolarizing */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Switch checked={noise.enable_depolarizing}
                onCheckedChange={v => setNoise({ ...noise, enable_depolarizing: v })} id="dep" />
              <Label htmlFor="dep" className="text-xs cursor-pointer">Depolarizing</Label>
            </div>
            {noise.enable_depolarizing && (
              <div className="pl-2 space-y-2">
                <NoiseSlider label="T0" value={noise.pdep0} onChange={v => setNoise({ ...noise, pdep0: v })} />
                <NoiseSlider label="T1" value={noise.pdep1} onChange={v => setNoise({ ...noise, pdep1: v })} />
                <NoiseSlider label="T2" value={noise.pdep2} onChange={v => setNoise({ ...noise, pdep2: v })} />
              </div>
            )}
          </div>
          {/* Amplitude damping */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Switch checked={noise.enable_amplitude_damping}
                onCheckedChange={v => setNoise({ ...noise, enable_amplitude_damping: v })} id="amp" />
              <Label htmlFor="amp" className="text-xs cursor-pointer">Amplitude Damping</Label>
            </div>
            {noise.enable_amplitude_damping && (
              <div className="pl-2 space-y-2">
                <NoiseSlider label="T0" value={noise.pamp0} onChange={v => setNoise({ ...noise, pamp0: v })} />
                <NoiseSlider label="T1" value={noise.pamp1} onChange={v => setNoise({ ...noise, pamp1: v })} />
                <NoiseSlider label="T2" value={noise.pamp2} onChange={v => setNoise({ ...noise, pamp2: v })} />
              </div>
            )}
          </div>
          {/* Phase damping */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Switch checked={noise.enable_phase_damping}
                onCheckedChange={v => setNoise({ ...noise, enable_phase_damping: v })} id="phs" />
              <Label htmlFor="phs" className="text-xs cursor-pointer">Phase Damping</Label>
            </div>
            {noise.enable_phase_damping && (
              <div className="pl-2 space-y-2">
                <NoiseSlider label="T0" value={noise.pph0} onChange={v => setNoise({ ...noise, pph0: v })} />
                <NoiseSlider label="T1" value={noise.pph1} onChange={v => setNoise({ ...noise, pph1: v })} />
                <NoiseSlider label="T2" value={noise.pph2} onChange={v => setNoise({ ...noise, pph2: v })} />
              </div>
            )}
          </div>
          {/* CNOT noise */}
          {num_qubits >= 2 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Switch checked={noise.enable_cnot_noise}
                  onCheckedChange={v => setNoise({ ...noise, enable_cnot_noise: v })} id="cnot" />
                <Label htmlFor="cnot" className="text-xs cursor-pointer">CNOT Depolarizing</Label>
              </div>
              {noise.enable_cnot_noise && (
                <div className="pl-2 space-y-2">
                  <NoiseSlider label="T0" value={noise.pcnot0} onChange={v => setNoise({ ...noise, pcnot0: v })} />
                  <NoiseSlider label="T1" value={noise.pcnot1} onChange={v => setNoise({ ...noise, pcnot1: v })} />
                  <NoiseSlider label="T2" value={noise.pcnot2} onChange={v => setNoise({ ...noise, pcnot2: v })} />
                </div>
              )}
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>

      <Separator />

      {/* ── Finance / Data ───────────────────────────────────────────────── */}
      <Collapsible open={openFinance} onOpenChange={setOpenFinance}>
        <CollapsibleTrigger className="flex items-center gap-2 w-full p-3 hover:bg-accent/10 text-sm font-semibold">
          <TrendingUp className="w-4 h-4 text-primary" />
          Finance / Data
          {openFinance ? <ChevronDown className="w-3 h-3 ml-auto" /> : <ChevronRight className="w-3 h-3 ml-auto" />}
        </CollapsibleTrigger>
        <CollapsibleContent className="px-4 pb-4 space-y-3">
          <div>
            <Label className="text-xs">Tickers (comma-separated)</Label>
            <Input
              value={finance.tickers}
              onChange={e => setFinance({ ...finance, tickers: e.target.value })}
              placeholder="SPY,QQQ,AAPL"
              className="mt-1 h-8 text-xs"
            />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <Label>Lookback days</Label>
              <span className="font-mono text-primary">{finance.lookback_days}</span>
            </div>
            <Slider value={[finance.lookback_days]}
              onValueChange={([v]) => setFinance({ ...finance, lookback_days: v })}
              min={30} max={1000} step={30} />
          </div>
          <div>
            <Label className="text-xs">Portfolio value ($)</Label>
            <Input
              type="number" step={10000}
              value={finance.portfolio_value}
              onChange={e => setFinance({ ...finance, portfolio_value: parseFloat(e.target.value) || 0 })}
              className="mt-1 h-8 text-xs"
            />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <Label>Confidence</Label>
              <span className="font-mono text-primary">{(finance.confidence_level * 100).toFixed(0)}%</span>
            </div>
            <Slider value={[finance.confidence_level]}
              onValueChange={([v]) => setFinance({ ...finance, confidence_level: v })}
              min={0.80} max={0.99} step={0.01} />
          </div>
          <div>
            <Label className="text-xs">MC simulations</Label>
            <Input
              type="number" step={5000}
              value={finance.mc_sims}
              onChange={e => setFinance({ ...finance, mc_sims: parseInt(e.target.value) || 10000 })}
              className="mt-1 h-8 text-xs"
            />
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={finance.demo_mode}
              onCheckedChange={v => setFinance({ ...finance, demo_mode: v })} id="demo" />
            <Label htmlFor="demo" className="text-xs cursor-pointer">Demo mode (synthetic data)</Label>
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={finance.per_share}
              onCheckedChange={v => setFinance({ ...finance, per_share: v })} id="per-share" />
            <Label htmlFor="per-share" className="text-xs cursor-pointer">Per share</Label>
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={finance.show_position}
              onCheckedChange={v => setFinance({ ...finance, show_position: v })} id="show-position" />
            <Label htmlFor="show-position" className="text-xs cursor-pointer">Position ($ = shares × price)</Label>
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
                if (parsed.shots)      setShots(parsed.shots);
                if (parsed.noise)      setNoise(parsed.noise);
                if (parsed.finance)    setFinance(parsed.finance);
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
