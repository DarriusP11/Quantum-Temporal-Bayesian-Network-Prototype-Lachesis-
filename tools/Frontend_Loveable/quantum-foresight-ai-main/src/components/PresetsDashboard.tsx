/**
 * PresetsDashboard.tsx — Circuit presets (Bell, dephasing, GHZ, etc.) + scenario library.
 * Loads a preset into the global context then runs simulation.
 */
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { AlertCircle, RefreshCw, BookOpen, Zap } from "lucide-react";
import { useAppContext, GateStepConfig } from "@/contexts/AppContext";
import { get, post, apiQuantumSimulate, QuantumSimulateResponse } from "@/lib/api";

interface PresetMeta { key: string; label: string; }
interface PresetConfig {
  label: string;
  num_qubits: number;
  step0: Partial<GateStepConfig>;
  step1: Partial<GateStepConfig>;
  step2: Partial<GateStepConfig>;
  noise: Record<string, unknown>;
}

const defaultStep = (): GateStepConfig => ({
  q0:"None", q0_angle:0, q1:"None", q1_angle:0,
  q2:"None", q2_angle:0, q3:"None", q3_angle:0,
  cnot_01:false, cnot_12:false, cnot_23:false,
});

export const PresetsDashboard = () => {
  const { setNumQubits, setStep, setNoise, state, buildQuantumRequest } = useAppContext();
  const [presets, setPresets]     = useState<PresetMeta[]>([]);
  const [selected, setSelected]   = useState<string | null>(null);
  const [result, setResult]       = useState<QuantumSimulateResponse | null>(null);
  const [loading, setLoading]     = useState(false);
  const [loadingPreset, setLoadingPreset] = useState<string | null>(null);
  const [error, setError]         = useState<string | null>(null);

  useEffect(() => {
    get<{ presets: PresetMeta[] }>("/api/quantum/presets")
      .then(r => setPresets(r.presets))
      .catch(() => {});
  }, []);

  const applyPreset = async (key: string) => {
    setLoadingPreset(key);
    try {
      const cfg = await get<PresetConfig>(`/api/quantum/presets/${key}`);
      setNumQubits(cfg.num_qubits);
      const mergeStep = (partial: Partial<GateStepConfig>): GateStepConfig => ({
        ...defaultStep(), ...partial,
      });
      setStep(0, mergeStep(cfg.step0));
      setStep(1, mergeStep(cfg.step1));
      setStep(2, mergeStep(cfg.step2));
      if (cfg.noise) {
        setNoise({
          ...state.noise,
          enable_depolarizing: !!(cfg.noise.enable_depolarizing),
          pdep0: Number(cfg.noise.depolarizing_prob ?? state.noise.pdep0),
          enable_amplitude_damping: !!(cfg.noise.enable_amplitude_damping),
          pamp0: Number(cfg.noise.amplitude_damping_prob ?? state.noise.pamp0),
          enable_phase_damping: !!(cfg.noise.enable_phase_damping),
          pph0: Number(cfg.noise.phase_damping_prob ?? state.noise.pph0),
          enable_cnot_noise: !!(cfg.noise.enable_cnot_noise),
          pcnot0: Number(cfg.noise.cnot_noise_prob ?? state.noise.pcnot0),
          pdep1: state.noise.pdep1, pdep2: state.noise.pdep2,
          pamp1: state.noise.pamp1, pamp2: state.noise.pamp2,
          pph1: state.noise.pph1,  pph2: state.noise.pph2,
          pcnot1: state.noise.pcnot1, pcnot2: state.noise.pcnot2,
        });
      }
      setSelected(key);
      setResult(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingPreset(null);
    }
  };

  const runSimulation = async () => {
    setLoading(true); setError(null);
    try {
      const req = buildQuantumRequest() as Parameters<typeof apiQuantumSimulate>[0];
      const res = await apiQuantumSimulate(req);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const chartData = result ? result.probabilities.map((p, i) => ({
    state: `|${i.toString(2).padStart(state.num_qubits, "0")}⟩`,
    probability: parseFloat((p * 100).toFixed(2)),
  })) : [];

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-primary" />
            Circuit Presets
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              {presets.length} presets
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-base font-medium mb-1">Ready-made circuit setups — click one to instantly load a famous quantum configuration.</p>
          <p className="text-sm text-muted-foreground mb-4">
            Click a preset to load it into the sidebar controls, then run the simulation.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {presets.map(p => (
              <Button
                key={p.key}
                variant={selected === p.key ? "default" : "outline"}
                onClick={() => applyPreset(p.key)}
                disabled={loadingPreset === p.key}
                className="h-auto py-3 text-left flex flex-col items-start"
              >
                {loadingPreset === p.key
                  ? <><RefreshCw className="w-3 h-3 animate-spin mb-1" /><span className="text-xs">Loading...</span></>
                  : <><Zap className="w-3 h-3 mb-1" /><span className="text-xs font-semibold">{p.label}</span></>}
              </Button>
            ))}
          </div>

          {selected && (
            <Button onClick={runSimulation} disabled={loading} className="w-full h-10 mt-4">
              {loading ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Simulating...</>
                       : <><Zap className="w-4 h-4 mr-2" />Run Preset Circuit</>}
            </Button>
          )}
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{error}</span>
          </CardContent>
        </Card>
      )}

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Qubits",   val: result.num_qubits },
              { label: "Shots",    val: state.shots.toLocaleString() },
              { label: "States",   val: result.probabilities.length },
              { label: "Fidelity", val: `${(result.fidelity * 100).toFixed(2)}%` },
            ].map(({ label, val }) => (
              <div key={label} className="p-4 border border-accent/20 rounded-lg">
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className="text-xl font-bold font-mono mt-1 text-primary">{val}</p>
              </div>
            ))}
          </div>

          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-base">Measurement Distribution</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, "Probability"]} />
                  <Bar dataKey="probability" name="Probability (%)">
                    {chartData.map((_, i) => (
                      <Cell key={i} fill={`hsl(${200 + i * 50}, 70%, 55%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {result.circuit_lines.length > 0 && (
            <Card className="border-accent/20">
              <CardHeader><CardTitle className="text-base">Circuit Diagram</CardTitle></CardHeader>
              <CardContent>
                <pre className="text-xs font-mono bg-black/20 p-3 rounded overflow-x-auto text-green-400">
                  {result.circuit_lines.join("\n")}
                </pre>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
};
