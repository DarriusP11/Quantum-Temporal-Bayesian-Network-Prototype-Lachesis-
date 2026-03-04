import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { QuantumConfig, GateType } from "@/types/quantum";
import { QUANTUM_SCENARIOS } from "@/lib/quantum-simulator";

interface QuantumControlsProps {
  config: QuantumConfig;
  onConfigChange: (config: QuantumConfig) => void;
}

const GATE_OPTIONS: GateType[] = ['None', 'H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'S', 'T'];

export const QuantumControls = ({ config, onConfigChange }: QuantumControlsProps) => {
  const updateConfig = (updates: Partial<QuantumConfig>) => {
    onConfigChange({ ...config, ...updates });
  };

  const applyScenario = (scenarioName: string) => {
    const scenario = QUANTUM_SCENARIOS[scenarioName];
    if (scenario) {
      onConfigChange({ ...config, ...scenario } as QuantumConfig);
    }
  };

  return (
    <div className="space-y-4">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <span className="w-2 h-2 bg-primary rounded-full animate-pulse"></span>
            Quantum Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="qubits">Number of Qubits</Label>
              <Select value={config.numQubits.toString()} onValueChange={(v) => updateConfig({ numQubits: parseInt(v) })}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 Qubit</SelectItem>
                  <SelectItem value="2">2 Qubits</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label htmlFor="shots">Shots</Label>
              <Input
                id="shots"
                type="number"
                value={config.shots}
                onChange={(e) => updateConfig({ shots: parseInt(e.target.value) || 2048 })}
                min="128"
                max="100000"
                step="128"
              />
            </div>
          </div>

          <div>
            <Label htmlFor="seed">Random Seed</Label>
            <Input
              id="seed"
              type="number"
              value={config.seed}
              onChange={(e) => updateConfig({ seed: parseInt(e.target.value) || 17 })}
              min="0"
              max="10000"
            />
          </div>
        </CardContent>
      </Card>

      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="text-lg">Noise Models</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="depolarizing"
                checked={config.enableDepolarizing}
                onCheckedChange={(checked) => updateConfig({ enableDepolarizing: !!checked })}
              />
              <Label htmlFor="depolarizing">Depolarizing</Label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Checkbox
                id="amplitude"
                checked={config.enableAmplitudeDamping}
                onCheckedChange={(checked) => updateConfig({ enableAmplitudeDamping: !!checked })}
              />
              <Label htmlFor="amplitude">Amplitude Damping</Label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Checkbox
                id="phase"
                checked={config.enablePhaseDamping}
                onCheckedChange={(checked) => updateConfig({ enablePhaseDamping: !!checked })}
              />
              <Label htmlFor="phase">Phase Damping</Label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Checkbox
                id="cnot"
                checked={config.enableCNOTNoise}
                onCheckedChange={(checked) => updateConfig({ enableCNOTNoise: !!checked })}
              />
              <Label htmlFor="cnot">CNOT Noise</Label>
            </div>
          </div>

          {config.enableDepolarizing && (
            <div>
              <Label>Depolarizing Error Rates</Label>
              {config.depolarizingProbs.map((prob, i) => (
                <div key={i} className="flex items-center space-x-2 mt-2">
                  <Label className="w-16">Step {i}:</Label>
                  <Slider
                    value={[prob]}
                    onValueChange={([value]) => {
                      const newProbs = [...config.depolarizingProbs];
                      newProbs[i] = value;
                      updateConfig({ depolarizingProbs: newProbs as [number, number, number] });
                    }}
                    max={0.2}
                    step={0.001}
                    className="flex-1"
                  />
                  <span className="w-12 text-sm">{prob.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}

          {config.enableAmplitudeDamping && (
            <div>
              <Label>Amplitude Damping Rates</Label>
              {config.amplitudeDampingProbs.map((prob, i) => (
                <div key={i} className="flex items-center space-x-2 mt-2">
                  <Label className="w-16">Step {i}:</Label>
                  <Slider
                    value={[prob]}
                    onValueChange={([value]) => {
                      const newProbs = [...config.amplitudeDampingProbs];
                      newProbs[i] = value;
                      updateConfig({ amplitudeDampingProbs: newProbs as [number, number, number] });
                    }}
                    max={0.4}
                    step={0.001}
                    className="flex-1"
                  />
                  <span className="w-12 text-sm">{prob.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="text-lg">Gate Configuration - Step 0</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>Qubit 0 Gate</Label>
              <Select 
                value={config.gates.step0.q0.type} 
                onValueChange={(value: GateType) => 
                  updateConfig({
                    gates: {
                      ...config.gates,
                      step0: {
                        ...config.gates.step0,
                        q0: { ...config.gates.step0.q0, type: value }
                      }
                    }
                  })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {GATE_OPTIONS.map(gate => (
                    <SelectItem key={gate} value={gate}>{gate}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            {config.numQubits > 1 && (
              <div>
                <Label>Qubit 1 Gate</Label>
                <Select 
                  value={config.gates.step0.q1.type} 
                  onValueChange={(value: GateType) => 
                    updateConfig({
                      gates: {
                        ...config.gates,
                        step0: {
                          ...config.gates.step0,
                          q1: { ...config.gates.step0.q1, type: value }
                        }
                      }
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {GATE_OPTIONS.map(gate => (
                      <SelectItem key={gate} value={gate}>{gate}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="cnot0"
              checked={config.gates.step0.cnot}
              onCheckedChange={(checked) => 
                updateConfig({
                  gates: {
                    ...config.gates,
                    step0: { ...config.gates.step0, cnot: !!checked }
                  }
                })
              }
            />
            <Label htmlFor="cnot0">Apply CNOT</Label>
          </div>
        </CardContent>
      </Card>

      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="text-lg">Quantum Scenarios</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-2">
            {Object.keys(QUANTUM_SCENARIOS).map(scenario => (
              <Button
                key={scenario}
                variant="outline"
                size="sm"
                onClick={() => applyScenario(scenario)}
                className="text-xs"
              >
                {scenario}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};