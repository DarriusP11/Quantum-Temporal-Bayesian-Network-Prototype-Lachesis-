/**
 * PresentScenariosDashboard.tsx — Analyse the current circuit scenario
 * (robustness metrics: fidelity, TV distance, regime).
 */
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from "recharts";
import { AlertCircle, RefreshCw, BarChart2 } from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import { apiQuantumSimulate, post, QuantumSimulateResponse } from "@/lib/api";

interface MeasurementResponse {
  ideal_probs: Record<string, number>;
  noisy_probs: Record<string, number>;
  tv_distance: number;
  all_states: string[];
  num_qubits: number;
}

export const PresentScenariosDashboard = () => {
  const { buildQuantumRequest, state } = useAppContext();
  const [svResult, setSvResult]     = useState<QuantumSimulateResponse | null>(null);
  const [measResult, setMeasResult] = useState<MeasurementResponse | null>(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState<string | null>(null);

  const noiseEnabled = state.noise.enable_depolarizing || state.noise.enable_amplitude_damping
    || state.noise.enable_phase_damping || state.noise.enable_cnot_noise;

  const run = async () => {
    setLoading(true); setError(null);
    try {
      const req = buildQuantumRequest() as Parameters<typeof apiQuantumSimulate>[0];
      const [sv, meas] = await Promise.all([
        apiQuantumSimulate(req),
        post<MeasurementResponse>("/api/quantum/measurement", req),
      ]);
      setSvResult(sv);
      setMeasResult(meas);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // Robustness score (0-100)
  const fidelity  = svResult?.fidelity ?? 1.0;
  const tv        = measResult?.tv_distance ?? 0.0;
  const robustness = Math.max(0, Math.min(100, fidelity * 100 - tv * 50));

  const radarData = [
    { metric: "Fidelity",    value: (fidelity * 100).toFixed(1) },
    { metric: "TV Robustness", value: ((1 - tv) * 100).toFixed(1) },
    { metric: "Shot density",  value: Math.min(100, (state.shots / 200)).toFixed(1) },
    { metric: "Qubit count", value: (state.num_qubits * 25).toString() },
    { metric: "Noise level", value: noiseEnabled ? "40" : "100" },
  ].map(d => ({ ...d, value: parseFloat(d.value) }));

  const barData = svResult ? svResult.probabilities.map((p, i) => ({
    state: `|${i.toString(2).padStart(state.num_qubits, "0")}⟩`,
    ideal: parseFloat((p * 100).toFixed(2)),
    noisy: measResult ? parseFloat(((measResult.noisy_probs[i.toString(2).padStart(state.num_qubits, "0")] ?? p) * 100).toFixed(2)) : parseFloat((p * 100).toFixed(2)),
  })) : [];

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-primary" />
            Present Scenarios
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              Robustness analysis
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-base font-medium mb-1">Grades your current circuit for reliability under noise — like a health check for your quantum setup.</p>
          <p className="text-sm text-muted-foreground mb-4">
            Analyses the currently configured circuit scenario — combines fidelity and
            TV distance into a robustness score. Enable noise channels in the sidebar for meaningful analysis.
          </p>
          <Button onClick={run} disabled={loading} className="w-full h-10">
            {loading ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Analysing scenario...</>
                     : <><BarChart2 className="w-4 h-4 mr-2" />Analyse Current Scenario</>}
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

      {(svResult || measResult) && (
        <>
          {/* Metric summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Fidelity",     val: `${(fidelity * 100).toFixed(2)}%`,
                color: fidelity > 0.95 ? "text-green-400" : "text-yellow-400" },
              { label: "TV Distance",  val: (tv).toFixed(4),
                color: tv < 0.05 ? "text-green-400" : tv < 0.15 ? "text-yellow-400" : "text-red-400" },
              { label: "Robustness",   val: `${robustness.toFixed(1)}/100`,
                color: robustness > 80 ? "text-green-400" : robustness > 60 ? "text-yellow-400" : "text-red-400" },
              { label: "Noise active", val: noiseEnabled ? "YES" : "NO",
                color: noiseEnabled ? "text-yellow-400" : "text-green-400" },
            ].map(({ label, val, color }) => (
              <div key={label} className="p-4 border border-accent/20 rounded-lg">
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className={`text-xl font-bold font-mono mt-1 ${color}`}>{val}</p>
              </div>
            ))}
          </div>

          <Tabs defaultValue="radar">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="radar">Radar Overview</TabsTrigger>
              <TabsTrigger value="counts">Count Comparison</TabsTrigger>
            </TabsList>

            <TabsContent value="radar">
              <Card className="border-accent/20">
                <CardHeader><CardTitle className="text-base">Scenario Quality Radar</CardTitle></CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={280}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="rgba(255,255,255,0.1)" />
                      <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} />
                      <Radar name="Score" dataKey="value" stroke="hsl(210,80%,60%)" fill="hsl(210,80%,60%)" fillOpacity={0.3} />
                    </RadarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="counts">
              <Card className="border-accent/20">
                <CardHeader><CardTitle className="text-base">Ideal vs Noisy Probabilities</CardTitle></CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={barData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`]} />
                      <Bar dataKey="ideal" name="Ideal (%)" fill="hsl(210,80%,60%)" />
                      <Bar dataKey="noisy" name="Noisy (%)" fill="hsl(30,80%,55%)" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
};
