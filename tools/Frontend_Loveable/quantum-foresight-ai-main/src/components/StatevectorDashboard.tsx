/**
 * StatevectorDashboard.tsx — Statevector amplitude & phase visualization.
 * Driven by the global AppContext (sidebar controls).
 */
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { AlertCircle, RefreshCw, Atom } from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import { apiQuantumSimulate, QuantumSimulateResponse } from "@/lib/api";

function complexAngle(re: number, im: number) {
  return Math.atan2(im, re) * (180 / Math.PI);
}

function phaseColor(deg: number) {
  const h = ((deg % 360) + 360) % 360;
  return `hsl(${h}, 80%, 55%)`;
}

export const StatevectorDashboard = () => {
  const { state, buildQuantumRequest } = useAppContext();
  const [result, setResult]   = useState<QuantumSimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const run = async () => {
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

  const nq = state.num_qubits;
  const dim = result ? result.statevector_real.length : Math.pow(2, nq);

  const chartData = result ? result.statevector_real.map((re, i) => {
    const im = result.statevector_imag[i];
    const amp = Math.sqrt(re * re + im * im);
    const prob = result.probabilities[i];
    const phase = complexAngle(re, im);
    return {
      state: `|${i.toString(2).padStart(nq, "0")}⟩`,
      amplitude: parseFloat(amp.toFixed(4)),
      probability: parseFloat((prob * 100).toFixed(2)),
      phase: parseFloat(phase.toFixed(1)),
      phaseColor: phaseColor(phase),
    };
  }) : [];

  return (
    <div className="space-y-6">
      {/* Header card */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Atom className="w-5 h-5 text-primary" />
            Statevector Analysis
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              {dim}-dim Hilbert space
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            Displays ideal (noise-free) statevector amplitudes and phases from the sidebar circuit.
            Sidebar controls set qubits, shots, and gate steps.
          </p>
          <Button onClick={run} disabled={loading} className="w-full h-10">
            {loading ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Simulating...</>
                     : <><Atom className="w-4 h-4 mr-2" />Run Statevector Simulation</>}
          </Button>
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
          {/* Probability amplitudes */}
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="text-base">Probability Amplitudes |ψ|²</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(4)}`, "Prob (%)"]} />
                  <Bar dataKey="probability" name="Probability (%)">
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={`hsl(${210 + i * 30}, 70%, 55%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Phase visualization */}
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="text-base">Phase (degrees) per Basis State</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                  <YAxis domain={[-180, 180]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(1)}°`, "Phase"]} />
                  <Bar dataKey="phase" name="Phase (°)">
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={d.phaseColor} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Amplitude table */}
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="text-base">Amplitude Table</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-accent/20 text-muted-foreground text-xs">
                      <th className="text-left py-2 pr-4">State</th>
                      <th className="text-right py-2 pr-4">Re(α)</th>
                      <th className="text-right py-2 pr-4">Im(α)</th>
                      <th className="text-right py-2 pr-4">|α|</th>
                      <th className="text-right py-2 pr-4">|α|²</th>
                      <th className="text-right py-2">Phase (°)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {chartData.map((d, i) => (
                      <tr key={i} className="border-b border-accent/10 hover:bg-accent/5">
                        <td className="py-1.5 pr-4 font-mono">{d.state}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{result.statevector_real[i].toFixed(4)}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{result.statevector_imag[i].toFixed(4)}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{d.amplitude.toFixed(4)}</td>
                        <td className="py-1.5 pr-4 text-right font-mono text-primary">{(d.probability / 100).toFixed(4)}</td>
                        <td className="py-1.5 text-right font-mono" style={{ color: d.phaseColor }}>{d.phase.toFixed(1)}°</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Circuit diagram */}
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
