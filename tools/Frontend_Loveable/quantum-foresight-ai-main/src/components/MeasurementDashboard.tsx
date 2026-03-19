/**
 * MeasurementDashboard.tsx — Ideal vs noisy counts comparison + TV distance.
 */
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { AlertCircle, RefreshCw, Activity } from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import { post } from "@/lib/api";

interface MeasurementResponse {
  ideal_counts: Record<string, number>;
  noisy_counts: Record<string, number>;
  ideal_probs: Record<string, number>;
  noisy_probs: Record<string, number>;
  tv_distance: number;
  all_states: string[];
  num_qubits: number;
}

function tvColor(tv: number) {
  if (tv < 0.05)  return "text-green-400 border-green-500/30 bg-green-500/10";
  if (tv < 0.15)  return "text-yellow-400 border-yellow-500/30 bg-yellow-500/10";
  return "text-red-400 border-red-500/30 bg-red-500/10";
}

export const MeasurementDashboard = () => {
  const { buildQuantumRequest, state } = useAppContext();
  const [result, setResult]   = useState<MeasurementResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const noiseEnabled = state.noise.enable_depolarizing || state.noise.enable_amplitude_damping
    || state.noise.enable_phase_damping || state.noise.enable_cnot_noise;

  const run = async () => {
    setLoading(true); setError(null);
    try {
      const req = buildQuantumRequest();
      const res = await post<MeasurementResponse>("/api/quantum/measurement", req);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const chartData = result ? result.all_states.map(st => ({
    state: `|${st}⟩`,
    ideal: parseFloat(((result.ideal_probs[st] ?? 0) * 100).toFixed(2)),
    noisy: parseFloat(((result.noisy_probs[st] ?? 0) * 100).toFixed(2)),
  })) : [];

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary" />
            Measurement Comparison
            <Badge variant="outline" className={`ml-auto text-xs ${noiseEnabled ? "border-yellow-500/30 bg-yellow-500/10 text-yellow-400" : "border-accent/30"}`}>
              {noiseEnabled ? "Noise ON" : "Noise OFF"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-base font-medium mb-1">Compares what a perfect circuit outputs vs. what the real noisy one produces.</p>
          <p className="text-sm text-muted-foreground mb-4">
            Compares ideal (noiseless) vs noisy measurement distributions.
            TV distance measures how much noise corrupts the output — enable noise channels in the sidebar.
          </p>
          <Button onClick={run} disabled={loading} className="w-full h-10">
            {loading ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Running measurements...</>
                     : <><Activity className="w-4 h-4 mr-2" />Run Measurement</>}
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
          {/* TV distance */}
          <Card className={`border ${tvColor(result.tv_distance)}`}>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-muted-foreground">Total Variation Distance (TV)</p>
                  <p className="text-3xl font-bold font-mono mt-1">{result.tv_distance.toFixed(4)}</p>
                </div>
                <Badge className={`text-sm px-3 py-1 ${tvColor(result.tv_distance)}`}>
                  {result.tv_distance < 0.05 ? "Robust" : result.tv_distance < 0.15 ? "Moderate noise" : "High noise"}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                TV ∈ [0,1] — 0 means perfect match, 1 means completely different distributions.
              </p>
            </CardContent>
          </Card>

          {/* Comparison chart */}
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="text-base">Ideal vs Noisy Probabilities (%)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="state" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`]} />
                  <Legend />
                  <Bar dataKey="ideal" name="Ideal (%)" fill="hsl(210,80%,60%)" />
                  <Bar dataKey="noisy" name="Noisy (%)" fill="hsl(30,80%,55%)" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Count table */}
          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-base">Raw Counts</CardTitle></CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-accent/20 text-muted-foreground text-xs">
                      <th className="text-left py-2">State</th>
                      <th className="text-right py-2">Ideal count</th>
                      <th className="text-right py-2">Noisy count</th>
                      <th className="text-right py-2">Ideal %</th>
                      <th className="text-right py-2">Noisy %</th>
                      <th className="text-right py-2">|Δ| %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.all_states.map(st => {
                      const ip = result.ideal_probs[st] ?? 0;
                      const np_ = result.noisy_probs[st] ?? 0;
                      const diff = Math.abs(ip - np_) * 100;
                      return (
                        <tr key={st} className="border-b border-accent/10 hover:bg-accent/5">
                          <td className="py-1.5 font-mono">{`|${st}⟩`}</td>
                          <td className="py-1.5 text-right font-mono">{result.ideal_counts[st] ?? 0}</td>
                          <td className="py-1.5 text-right font-mono">{result.noisy_counts[st] ?? 0}</td>
                          <td className="py-1.5 text-right font-mono">{(ip * 100).toFixed(2)}%</td>
                          <td className="py-1.5 text-right font-mono">{(np_ * 100).toFixed(2)}%</td>
                          <td className={`py-1.5 text-right font-mono ${diff > 5 ? "text-red-400" : "text-green-400"}`}>{diff.toFixed(2)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
};
