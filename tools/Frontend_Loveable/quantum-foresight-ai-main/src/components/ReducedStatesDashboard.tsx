/**
 * ReducedStatesDashboard.tsx — Partial trace per qubit → 3D Bloch sphere.
 * Each qubit gets its own independent Plotly 3D scene so that aspectmode:"cube"
 * always renders a true sphere regardless of qubit count.
 */
import { useState, lazy, Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, RefreshCw, Layers } from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import { post } from "@/lib/api";

// Lazy-load Plotly to keep initial bundle small
const Plot = lazy(() => import("react-plotly.js"));

interface ReducedState {
  qubit: number;
  bloch_x: number;
  bloch_y: number;
  bloch_z: number;
  purity: number;
  rho_real: number[][];
  rho_imag: number[][];
}
interface ReducedStatesResponse {
  num_qubits: number;
  reduced_states: ReducedState[];
  noise_applied?: boolean;
}

const COLORS = ["#7c3aed", "#059669", "#dc2626", "#d97706"];

// ── Sphere surface mesh centred at origin ────────────────────────────────────
function sphereMesh() {
  const u: number[] = [], v: number[] = [];
  const nU = 20, nV = 10;
  for (let i = 0; i <= nU; i++) u.push((i / nU) * 2 * Math.PI);
  for (let j = 0; j <= nV; j++) v.push((j / nV) * Math.PI);

  const x: number[][] = [], y: number[][] = [], z: number[][] = [];
  for (const ui of u) {
    const xRow: number[] = [], yRow: number[] = [], zRow: number[] = [];
    for (const vi of v) {
      xRow.push(Math.cos(ui) * Math.sin(vi));
      yRow.push(Math.sin(ui) * Math.sin(vi));
      zRow.push(Math.cos(vi));
    }
    x.push(xRow); y.push(yRow); z.push(zRow);
  }
  return { x, y, z };
}

// ── Build Plotly traces for a single qubit Bloch sphere ──────────────────────
function buildSingleBlochFigure(rs: ReducedState, qi: number, color: string) {
  const { x, y, z } = sphereMesh();

  const traces: Plotly.Data[] = [
    // Wireframe sphere surface
    {
      type: "surface" as const,
      x, y, z,
      opacity: 0.18,
      colorscale: [[0, color], [1, color]],
      showscale: false,
      name: `q${qi} sphere`,
      hoverinfo: "skip" as const,
    } as Plotly.Data,

    // Bloch vector: origin → (bx, by, bz)
    {
      type: "scatter3d" as const,
      x: [0, rs.bloch_x],
      y: [0, rs.bloch_y],
      z: [0, rs.bloch_z],
      mode: "lines+markers" as const,
      line: { color, width: 8 },
      marker: { size: [2, 8], color },
      name: `|r|=${Math.sqrt(rs.bloch_x ** 2 + rs.bloch_y ** 2 + rs.bloch_z ** 2).toFixed(3)}`,
    } as Plotly.Data,

    // Axis labels
    {
      type: "scatter3d" as const,
      x: [0, 0, 1.3, -1.3, 0, 0],
      y: [0, 0, 0, 0, 1.3, -1.3],
      z: [1.3, -1.3, 0, 0, 0, 0],
      mode: "text" as const,
      text: ["|0⟩", "|1⟩", "+X", "-X", "+Y", "-Y"],
      textfont: { size: 10, color: "#94a3b8" },
      showlegend: false,
      hoverinfo: "skip" as const,
    } as Plotly.Data,
  ];

  const layout: Partial<Plotly.Layout> = {
    title: { text: `Qubit ${qi}`, font: { color: "#e2e8f0", size: 13 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    scene: {
      bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        color: "#94a3b8", gridcolor: "#334155", zerolinecolor: "#475569",
        range: [-1.5, 1.5], title: { text: "X" },
      },
      yaxis: {
        color: "#94a3b8", gridcolor: "#334155", zerolinecolor: "#475569",
        range: [-1.5, 1.5], title: { text: "Y" },
      },
      zaxis: {
        color: "#94a3b8", gridcolor: "#334155", zerolinecolor: "#475569",
        range: [-1.5, 1.5], title: { text: "Z" },
      },
      aspectmode: "cube" as const,
      camera: { eye: { x: 1.5, y: 1.5, z: 0.8 } },
    },
    legend: { font: { color: "#e2e8f0" }, bgcolor: "rgba(0,0,0,0)" },
    margin: { l: 0, r: 0, t: 35, b: 0 },
    height: 340,
  };

  return { traces, layout };
}

// ── Bloch compass bar ─────────────────────────────────────────────────────────
function BlochBar({ label, value }: { label: string; value: number }) {
  const pct = ((value + 1) / 2) * 100;
  const col = value > 0.5 ? "bg-green-500" : value < -0.5 ? "bg-red-500" : "bg-yellow-500";
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{value >= 0 ? "+" : ""}{value.toFixed(4)}</span>
      </div>
      <div className="relative h-2 bg-muted rounded">
        <div className="absolute left-1/2 top-0 w-px h-2 bg-border z-10" />
        <div className={`absolute h-2 rounded ${col}`}
          style={{ left: value >= 0 ? "50%" : `${pct}%`, width: `${Math.abs(value) / 2 * 100}%` }} />
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════════
export const ReducedStatesDashboard = () => {
  const { buildQuantumRequest, state } = useAppContext();
  const [result, setResult]   = useState<ReducedStatesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const run = async () => {
    setLoading(true); setError(null);
    try {
      const req = buildQuantumRequest();
      const res = await post<ReducedStatesResponse>("/api/quantum/reduced-states", req);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // Grid: 1 col for single qubit, 2 cols for 2+
  const gridCols = result && result.num_qubits === 1
    ? "grid-cols-1 max-w-sm mx-auto"
    : "grid-cols-1 sm:grid-cols-2";

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary" />
            Reduced States — 3D Bloch Sphere
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              {state.num_qubits} qubit{state.num_qubits > 1 ? "s" : ""} · Plotly 3D
            </Badge>
            {result?.noise_applied && (
              <Badge variant="outline" className="border-orange-500/30 bg-orange-500/10 text-orange-400 text-xs">
                Noise Applied
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-base font-medium mb-1">Shows how each individual qubit behaves when connected to others — reveals quantum entanglement.</p>
          <p className="text-sm text-muted-foreground mb-4">
            Computes the reduced density matrix for each qubit via partial trace, then projects
            onto the Bloch sphere. For entangled qubits (e.g. Bell state), each reduced state
            will be maximally mixed (|r|=0, purity=0.5). Use 2+ qubits with CNOT to see entanglement.
          </p>
          <Button onClick={run} disabled={loading} className="w-full h-10">
            {loading ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Computing partial traces...</>
                     : <><Layers className="w-4 h-4 mr-2" />Compute Bloch Spheres</>}
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
          {/* One independent 3D Plot per qubit — each has its own scene so
              aspectmode:"cube" always produces a true sphere */}
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="text-base">
                3D Bloch Sphere{result.num_qubits > 1 ? "s" : ""} (Interactive — drag to rotate)
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <div className={`grid gap-2 ${gridCols}`}>
                {result.reduced_states.map((rs, qi) => {
                  const color = COLORS[qi % COLORS.length];
                  const fig = buildSingleBlochFigure(rs, qi, color);
                  return (
                    <Suspense
                      key={qi}
                      fallback={
                        <div className="h-[340px] flex items-center justify-center text-muted-foreground text-sm">
                          Loading 3D renderer...
                        </div>
                      }
                    >
                      <Plot
                        data={fig.traces as Plotly.Data[]}
                        layout={fig.layout}
                        config={{ responsive: true, displayModeBar: true, displaylogo: false }}
                        style={{ width: "100%", height: "340px" }}
                      />
                    </Suspense>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Per-qubit metric cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {result.reduced_states.map((rs) => {
              const blochLen = Math.sqrt(rs.bloch_x**2 + rs.bloch_y**2 + rs.bloch_z**2);
              const isMixed = rs.purity < 0.51;
              const isPure  = rs.purity > 0.99;
              return (
                <Card key={rs.qubit} className="border-accent/20">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      Qubit {rs.qubit} — Reduced State
                      <Badge variant="outline" className={`text-xs ml-auto ${
                        isPure  ? "border-green-500/50 text-green-400" :
                        isMixed ? "border-red-500/50 text-red-400" :
                                  "border-yellow-500/50 text-yellow-400"}`}>
                        {isPure ? "Pure" : isMixed ? "Maximally Mixed" : "Near-Pure"}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-3">
                      <BlochBar label="⟨X⟩ (Bloch X)" value={rs.bloch_x} />
                      <BlochBar label="⟨Y⟩ (Bloch Y)" value={rs.bloch_y} />
                      <BlochBar label="⟨Z⟩ (Bloch Z)" value={rs.bloch_z} />
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="p-2 bg-muted/20 rounded">
                        <p className="text-xs text-muted-foreground">|r| Bloch length</p>
                        <p className="font-mono font-semibold">{blochLen.toFixed(4)}</p>
                      </div>
                      <div className="p-2 bg-muted/20 rounded">
                        <p className="text-xs text-muted-foreground">Purity Tr(ρ²)</p>
                        <p className="font-mono font-semibold">{rs.purity.toFixed(4)}</p>
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground mb-1">Purity</p>
                      <Progress value={rs.purity * 100} className="h-2" />
                      <p className="text-xs text-right text-muted-foreground mt-0.5">
                        {(rs.purity * 100).toFixed(1)}%
                        {isMixed ? " — entangled with rest of system" : ""}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground mb-1">ρ (reduced density matrix, real part)</p>
                      <div className="font-mono text-xs space-y-0.5 bg-black/20 p-2 rounded">
                        {rs.rho_real.map((row, ri) => (
                          <div key={ri} className="flex gap-4">
                            {row.map((v, ci) => (
                              <span key={ci} className="w-16 text-right text-primary/80">{v.toFixed(4)}</span>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
};
