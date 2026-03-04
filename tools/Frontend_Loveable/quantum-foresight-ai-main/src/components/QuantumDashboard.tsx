import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { QuantumConfig } from "@/types/quantum";
import { apiQuantumSimulate, QuantumSimulateResponse } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';
import { Activity, Zap, TrendingUp, AlertCircle } from "lucide-react";

interface QuantumDashboardProps {
  config: QuantumConfig;
}

function configToApiRequest(config: QuantumConfig) {
  const gateKeys = ["step0", "step1", "step2"] as const;
  const steps = gateKeys.map((_, i) => {
    const step = (config.gates as Record<string, { q0: { type: string; angle: number }; q1: { type: string; angle: number }; cnot: boolean }>)[`step${i}`];
    return {
      q0: step?.q0?.type ?? "None",
      q0_angle: step?.q0?.angle ?? 0,
      q1: step?.q1?.type ?? "None",
      q1_angle: step?.q1?.angle ?? 0,
      cnot_01: step?.cnot ?? false,
    };
  });

  return {
    num_qubits: config.numQubits,
    shots: config.shots,
    seed: config.seed,
    step0: steps[0],
    step1: steps[1],
    step2: steps[2],
    noise: {
      enable_depolarizing: config.enableDepolarizing,
      depolarizing_prob: config.depolarizingProbs?.[0] ?? 0.01,
      enable_amplitude_damping: config.enableAmplitudeDamping,
      amplitude_damping_prob: config.amplitudeDampingProbs?.[1] ?? 0.02,
      enable_phase_damping: config.enablePhaseDamping,
      phase_damping_prob: config.phaseDampingProbs?.[0] ?? 0.02,
      enable_cnot_noise: config.enableCNOTNoise,
      cnot_noise_prob: config.cnotNoiseProbs?.[0] ?? 0.02,
    },
  };
}

export const QuantumDashboard = ({ config }: QuantumDashboardProps) => {
  const [result, setResult] = useState<QuantumSimulateResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeSimulation, setActiveSimulation] = useState<'statevector' | 'counts' | 'fidelity' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSimulation = async (mode: 'statevector' | 'counts' | 'fidelity') => {
    setIsLoading(true);
    setActiveSimulation(mode);
    setError(null);
    try {
      const req = configToApiRequest(config);
      const res = await apiQuantumSimulate(req);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsLoading(false);
      setActiveSimulation(null);
    }
  };

  const getChartData = () => {
    if (!result) return [];
    const probs = result.probabilities;
    return probs.map((p, i) => ({
      state: `|${i.toString(2).padStart(result.num_qubits, "0")}⟩`,
      probability: p,
    }));
  };

  const chartData = getChartData();

  return (
    <div className="space-y-6">
      {/* Circuit Visualization */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary" />
            Quantum Circuit (Qiskit)
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              Python Backend
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {result?.circuit_lines ? (
            <div className="bg-muted/30 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              {result.circuit_lines.map((line, i) => (
                <div key={i} className="whitespace-pre">{line}</div>
              ))}
            </div>
          ) : (
            <div className="bg-muted/20 rounded-lg p-4 font-mono text-sm text-muted-foreground text-center py-8">
              Run a simulation to see the Qiskit circuit diagram
            </div>
          )}
        </CardContent>
      </Card>

      {/* Control Buttons */}
      <div className="grid grid-cols-3 gap-4">
        <Button
          onClick={() => runSimulation('statevector')}
          disabled={isLoading}
          className="h-16 flex flex-col gap-1"
          variant={activeSimulation === 'statevector' ? 'default' : 'outline'}
        >
          <Zap className="w-5 h-5" />
          <span>Statevector</span>
          {isLoading && activeSimulation === 'statevector' && (
            <div className="w-full mt-1"><Progress className="h-1" /></div>
          )}
        </Button>

        <Button
          onClick={() => runSimulation('counts')}
          disabled={isLoading}
          className="h-16 flex flex-col gap-1"
          variant={activeSimulation === 'counts' ? 'default' : 'outline'}
        >
          <TrendingUp className="w-5 h-5" />
          <span>Measurements</span>
          {isLoading && activeSimulation === 'counts' && (
            <div className="w-full mt-1"><Progress className="h-1" /></div>
          )}
        </Button>

        <Button
          onClick={() => runSimulation('fidelity')}
          disabled={isLoading}
          className="h-16 flex flex-col gap-1"
          variant={activeSimulation === 'fidelity' ? 'default' : 'outline'}
        >
          <Activity className="w-5 h-5" />
          <span>Fidelity</span>
          {isLoading && activeSimulation === 'fidelity' && (
            <div className="w-full mt-1"><Progress className="h-1" /></div>
          )}
        </Button>
      </div>

      {/* Error */}
      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{error}</span>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle>Quantum Metrics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span>Fidelity (noisy vs ideal)</span>
                <Badge variant={result.fidelity > 0.9 ? 'default' : 'secondary'}>
                  {result.fidelity.toFixed(6)}
                </Badge>
              </div>
              <div className="flex justify-between items-center text-sm text-muted-foreground">
                <span>Shots</span>
                <span>{config.shots.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center text-sm text-muted-foreground">
                <span>Qubits</span>
                <span>{result.num_qubits}</span>
              </div>

              <div>
                <h4 className="font-semibold mb-2 text-sm">State Probabilities</h4>
                {result.probabilities.map((prob, i) => {
                  const label = `|${i.toString(2).padStart(result.num_qubits, "0")}⟩`;
                  return (
                    <div key={label} className="flex justify-between items-center mb-1 text-sm">
                      <span className="font-mono">{label}</span>
                      <span>{(prob * 100).toFixed(2)}%</span>
                    </div>
                  );
                })}
              </div>

              <div>
                <h4 className="font-semibold mb-2 text-sm">Measurement Counts</h4>
                {(Object.entries(result.counts) as [string, number][])
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 8)
                  .map(([state, count]) => (
                    <div key={state} className="flex justify-between items-center mb-1 text-sm">
                      <span className="font-mono">|{state}⟩</span>
                      <span>{count}</span>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>

          {chartData.length > 0 && (
            <Card className="border-accent/20">
              <CardHeader>
                <CardTitle>Quantum State Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="state" className="text-muted-foreground" />
                    <YAxis className="text-muted-foreground" domain={[0, 1]} />
                    <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                      {chartData.map((_, index) => (
                        <Cell key={index} fill={`hsl(${263 + index * 30} 70% 50%)`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
};
