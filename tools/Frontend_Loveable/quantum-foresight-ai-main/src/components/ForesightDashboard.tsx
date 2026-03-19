import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiForesightSweep, apiForesightGetScenarios, apiForesightSaveScenario, ForesightSweepResponse } from "@/lib/api";
import { AlertCircle, RefreshCw, Save, Layers, Thermometer } from "lucide-react";

// Simple KL-divergence heatmap using a div grid (no extra chart libs needed)
const HeatmapCell = ({ value, max }: { value: number; max: number }) => {
  const ratio = max > 0 ? value / max : 0;
  const r = Math.round(239 * ratio);
  const g = Math.round(68 * (1 - ratio) + 100 * ratio);
  const b = Math.round(68 * (1 - ratio));
  return (
    <div
      className="flex items-center justify-center text-xs font-mono rounded border border-black/20"
      style={{
        backgroundColor: `rgb(${r},${g},${b})`,
        color: ratio > 0.5 ? "white" : "black",
        minWidth: 60,
        minHeight: 40,
      }}
    >
      {value.toFixed(3)}
    </div>
  );
};

export const ForesightDashboard = () => {
  const [shots, setShots]               = useState(1024);
  const [seeds, setSeeds]               = useState("17,42,99");
  const [pdepValues, setPdepValues]     = useState("0.0,0.01,0.03,0.05");
  const [pampValues, setPampValues]     = useState("0.0,0.02");
  const [isLoading, setIsLoading]       = useState(false);
  const [result, setResult]             = useState<ForesightSweepResponse | null>(null);
  const [scenarios, setScenarios]       = useState<Record<string, unknown>>({});
  const [scenarioName, setScenarioName] = useState("");
  const [error, setError]               = useState<string | null>(null);

  useEffect(() => {
    apiForesightGetScenarios().then(r => setScenarios(r.scenarios)).catch(() => {});
  }, []);

  const parseFloatList = (s: string) =>
    s.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v));

  const parseIntList = (s: string) =>
    s.split(",").map(v => parseInt(v.trim())).filter(v => !isNaN(v));

  const runSweep = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const pdep = parseFloatList(pdepValues);
      const pamp = parseFloatList(pampValues);
      const seedList = parseIntList(seeds);
      const res = await apiForesightSweep({
        shots,
        seeds: seedList,
        pdep_values: pdep,
        pamp_values: pamp,
        circuit: {
          num_qubits: 1,
          shots,
          seed: seedList[0] ?? 17,
          step0: { q0: "H", q0_angle: 0.5 },
        },
      });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsLoading(false);
    }
  };

  const saveScenario = async () => {
    if (!result || !scenarioName.trim()) return;
    try {
      await apiForesightSaveScenario(scenarioName.trim(), {
        pdep_values: result.pdep_values,
        pamp_values: result.pamp_values,
        grid_shape: [result.pdep_values.length, result.pamp_values.length],
      });
      const updated = await apiForesightGetScenarios();
      setScenarios(updated.scenarios);
      setScenarioName("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  // Find max KL for normalizing the heatmap colours
  const maxKL = result
    ? Math.max(...result.grid.flatMap(row => row.map(cell => cell.kl_divergence)), 0.001)
    : 0.001;

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary" />
            Foresight Parameter Sweep
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              Python · Qiskit
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-base font-medium mb-1">Maps how increasing noise levels affect your circuit — like a weather forecast for quantum errors.</p>
          <p className="text-sm text-muted-foreground">
            Sweeps depolarizing and amplitude damping noise parameters across a grid, computing KL divergence
            from the ideal (no-noise) distribution at each point.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <Label htmlFor="fs-shots">Shots per run</Label>
              <Input
                id="fs-shots"
                type="number"
                value={shots}
                onChange={e => setShots(parseInt(e.target.value) || 1024)}
                min={64} max={8192}
              />
            </div>
            <div>
              <Label htmlFor="fs-seeds">Seeds (comma-separated)</Label>
              <Input
                id="fs-seeds"
                value={seeds}
                onChange={e => setSeeds(e.target.value)}
                placeholder="17,42,99"
              />
            </div>
            <div>
              <Label htmlFor="fs-pdep">Depolarizing values</Label>
              <Input
                id="fs-pdep"
                value={pdepValues}
                onChange={e => setPdepValues(e.target.value)}
                placeholder="0.0,0.01,0.03,0.05"
              />
            </div>
            <div>
              <Label htmlFor="fs-pamp">Amplitude damping values</Label>
              <Input
                id="fs-pamp"
                value={pampValues}
                onChange={e => setPampValues(e.target.value)}
                placeholder="0.0,0.02"
              />
            </div>
          </div>

          <Button onClick={runSweep} disabled={isLoading} className="w-full h-12">
            {isLoading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Running Sweep...</>
              : <><Thermometer className="w-4 h-4 mr-2" />Run Parameter Sweep</>}
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
        <Tabs defaultValue="heatmap" className="space-y-4">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="heatmap">KL Divergence Heatmap</TabsTrigger>
            <TabsTrigger value="scenarios">Saved Scenarios</TabsTrigger>
          </TabsList>

          <TabsContent value="heatmap">
            <Card className="border-accent/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Thermometer className="w-5 h-5 text-primary" />
                  KL Divergence: Depolarizing × Amplitude Damping
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="overflow-x-auto">
                  {/* Header row: pamp axis */}
                  <div className="flex gap-2 mb-1 ml-20">
                    {result.pamp_values.map(p => (
                      <div key={p} className="text-xs text-center text-muted-foreground min-w-[60px]">
                        pamp={p}
                      </div>
                    ))}
                  </div>
                  {result.grid.map((row, ri) => (
                    <div key={ri} className="flex items-center gap-2 mb-1">
                      <div className="text-xs text-muted-foreground w-20 shrink-0 text-right pr-2">
                        pdep={result.pdep_values[ri]}
                      </div>
                      {row.map((cell, ci) => (
                        <HeatmapCell key={ci} value={cell.kl_divergence} max={maxKL} />
                      ))}
                    </div>
                  ))}
                </div>

                <div className="flex items-center gap-2 mt-3">
                  <div className="h-3 w-24 rounded" style={{ background: "linear-gradient(to right, rgb(0,100,68), rgb(239,68,68))" }} />
                  <span className="text-xs text-muted-foreground">Low → High KL divergence from ideal</span>
                </div>

                {/* Save scenario */}
                <div className="flex gap-3 pt-2 border-t border-accent/20">
                  <Input
                    value={scenarioName}
                    onChange={e => setScenarioName(e.target.value)}
                    placeholder="Save scenario as…"
                    className="flex-1"
                  />
                  <Button onClick={saveScenario} disabled={!scenarioName.trim()} size="sm">
                    <Save className="w-4 h-4 mr-1" />Save
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="scenarios">
            <Card className="border-accent/20">
              <CardHeader><CardTitle>Saved Scenarios</CardTitle></CardHeader>
              <CardContent>
                {Object.keys(scenarios).length === 0 ? (
                  <p className="text-sm text-muted-foreground">No saved scenarios.</p>
                ) : (
                  <div className="space-y-2">
                    {Object.entries(scenarios).map(([name, data]) => (
                      <div key={name} className="flex items-center justify-between p-3 border border-accent/20 rounded">
                        <span className="font-medium text-sm">{name}</span>
                        <span className="text-xs text-muted-foreground">
                          {(data as Record<string, unknown>).saved_at as string ?? ""}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};
