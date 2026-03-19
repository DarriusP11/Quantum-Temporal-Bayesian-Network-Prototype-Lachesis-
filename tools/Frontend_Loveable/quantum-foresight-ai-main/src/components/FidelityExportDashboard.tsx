/**
 * FidelityExportDashboard.tsx — State fidelity display + PDF export placeholder.
 */
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, RefreshCw, Shield, Download } from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import { apiQuantumSimulate, QuantumSimulateResponse } from "@/lib/api";

function fidelityColor(f: number) {
  if (f >= 0.99) return "text-green-400 border-green-500/30 bg-green-500/10";
  if (f >= 0.90) return "text-yellow-400 border-yellow-500/30 bg-yellow-500/10";
  return "text-red-400 border-red-500/30 bg-red-500/10";
}

function fidelityLabel(f: number) {
  if (f >= 0.99) return "Excellent";
  if (f >= 0.95) return "Good";
  if (f >= 0.90) return "Fair";
  return "Degraded";
}

export const FidelityExportDashboard = () => {
  const { buildQuantumRequest, state } = useAppContext();
  const [result, setResult]   = useState<QuantumSimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

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

  const exportReport = () => {
    setExporting(true);
    // Build a plain-text report and trigger download
    try {
      const lines: string[] = [
        "LACHESIS EXECUTIVE REPORT",
        "=".repeat(60),
        `Generated: ${new Date().toISOString()}`,
        "",
        "QUANTUM CIRCUIT CONFIGURATION",
        "-".repeat(40),
        `Qubits: ${state.num_qubits}`,
        `Shots:  ${state.shots}`,
        `Seed:   ${state.use_seed ? state.seed_val : "random"}`,
        "",
      ];
      if (result) {
        lines.push("STATEVECTOR RESULTS");
        lines.push("-".repeat(40));
        result.probabilities.forEach((p, i) => {
          const label = `|${i.toString(2).padStart(state.num_qubits, "0")}⟩`;
          lines.push(`${label}  p=${p.toFixed(6)}`);
        });
        lines.push("");
        lines.push("FIDELITY");
        lines.push("-".repeat(40));
        lines.push(`State fidelity (ideal vs noisy): ${(result.fidelity * 100).toFixed(2)}%`);
        lines.push(`Quality: ${fidelityLabel(result.fidelity)}`);
        lines.push("");
        if (result.circuit_lines.length > 0) {
          lines.push("CIRCUIT DIAGRAM");
          lines.push("-".repeat(40));
          lines.push(...result.circuit_lines);
        }
      }
      lines.push("", "=".repeat(60), "Lachesis — Quantum-Enhanced Financial Analytics Platform");

      const blob = new Blob([lines.join("\n")], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `lachesis_report_${Date.now()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary" />
            Fidelity &amp; Export
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              Qiskit · Helstrom bound
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-base font-medium mb-1">Scores how accurate your circuit is (0–1) and lets you download a full report.</p>
          <p className="text-sm text-muted-foreground mb-4">
            Measures how faithfully the noisy circuit reproduces the ideal state via
            Bhattacharyya fidelity on measurement distributions. Export a full report.
          </p>
          <div className="flex gap-3">
            <Button onClick={run} disabled={loading} className="flex-1 h-10">
              {loading ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Calculating...</>
                       : <><Shield className="w-4 h-4 mr-2" />Calculate Fidelity</>}
            </Button>
            {result && (
              <Button variant="outline" onClick={exportReport} disabled={exporting} className="h-10">
                <Download className="w-4 h-4 mr-2" />Export Report
              </Button>
            )}
          </div>
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
          {/* Main fidelity card */}
          <Card className={`border ${fidelityColor(result.fidelity)}`}>
            <CardContent className="pt-6">
              <div className="text-center space-y-3">
                <p className="text-xs text-muted-foreground uppercase tracking-wide">State Fidelity</p>
                <p className="text-6xl font-bold font-mono">{(result.fidelity * 100).toFixed(2)}%</p>
                <Badge className={`text-sm px-4 py-1 ${fidelityColor(result.fidelity)}`}>
                  {fidelityLabel(result.fidelity)}
                </Badge>
              </div>
              <div className="mt-4">
                <Progress value={result.fidelity * 100} className="h-3" />
              </div>
              <p className="text-xs text-muted-foreground text-center mt-2">
                Bhattacharyya fidelity between ideal and noisy distributions: F = (Σ √(p_ideal · p_noisy))²
              </p>
            </CardContent>
          </Card>

          {/* Stats grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Qubits", val: result.num_qubits.toString() },
              { label: "Shots", val: state.shots.toLocaleString() },
              { label: "Basis states", val: result.probabilities.length.toString() },
              { label: "Fidelity", val: `${(result.fidelity * 100).toFixed(3)}%` },
            ].map(({ label, val }) => (
              <div key={label} className="p-4 border border-accent/20 rounded-lg">
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className="text-xl font-bold font-mono mt-1 text-primary">{val}</p>
              </div>
            ))}
          </div>

          {/* Probability table */}
          <Card className="border-accent/20">
            <CardHeader><CardTitle className="text-base">Measurement Distribution</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-2">
                {result.probabilities.map((p, i) => {
                  const label = `|${i.toString(2).padStart(state.num_qubits, "0")}⟩`;
                  return (
                    <div key={i} className="flex items-center gap-3">
                      <span className="font-mono text-xs w-12 shrink-0">{label}</span>
                      <div className="flex-1 h-4 bg-muted/20 rounded overflow-hidden">
                        <div
                          className="h-full bg-primary/60 rounded transition-all"
                          style={{ width: `${p * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs w-16 text-right text-primary">{(p * 100).toFixed(2)}%</span>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
};
