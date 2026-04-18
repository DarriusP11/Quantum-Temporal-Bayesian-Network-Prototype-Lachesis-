import { Atom, Clock, Server, Cpu, Wifi } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { useToast } from "@/hooks/use-toast";

export function QuantumHardwareTab() {
  const [email, setEmail] = useState("");
  const { toast } = useToast();

  const handleNotify = () => {
    if (!email.trim()) return;
    toast({ title: "You're on the list!", description: "We'll notify you when real quantum hardware access launches." });
    setEmail("");
  };

  return (
    <div className="space-y-6 max-w-3xl mx-auto py-6">
      {/* Coming Soon Hero */}
      <div className="text-center space-y-4 py-8">
        <div className="relative inline-flex">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-amber-500/20 to-orange-500/20 border border-amber-500/30 flex items-center justify-center">
            <Atom className="w-10 h-10 text-amber-500 animate-spin" style={{ animationDuration: "6s" }} />
          </div>
          <Badge className="absolute -top-2 -right-2 bg-amber-500 text-white text-xs px-2 py-0.5">
            Enterprise
          </Badge>
        </div>

        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-amber-400 to-orange-400 bg-clip-text text-transparent">
            Real Quantum Hardware
          </h1>
          <p className="text-muted-foreground mt-2 text-base max-w-lg mx-auto">
            Execute your quantum circuits on actual quantum processors — not simulators.
            Direct API access to leading quantum computing platforms is coming to Lachesis Enterprise.
          </p>
        </div>

        <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
          <Clock className="w-4 h-4 text-amber-500" />
          <span className="font-medium text-amber-400">Coming Soon</span>
        </div>
      </div>

      {/* Planned Integrations */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          {
            name: "IBM Quantum",
            desc: "Access IBM Eagle, Heron, and Falcon processors via qiskit-ibm-runtime. Up to 127 qubits.",
            icon: Cpu,
            color: "border-blue-500/30 bg-blue-500/5",
          },
          {
            name: "Google Quantum AI",
            desc: "Run circuits on Sycamore and Willow processors via Google Cirq Cloud integration.",
            icon: Server,
            color: "border-green-500/30 bg-green-500/5",
          },
          {
            name: "IonQ / Other",
            desc: "Connect to trapped-ion and photonic quantum systems via cloud provider APIs.",
            icon: Wifi,
            color: "border-purple-500/30 bg-purple-500/5",
          },
        ].map(({ name, desc, icon: Icon, color }) => (
          <Card key={name} className={`border ${color}`}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <Icon className="w-4 h-4" />
                {name}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground">{desc}</p>
              <Badge variant="outline" className="mt-3 text-xs border-border/50">
                Planned
              </Badge>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* What to Expect */}
      <Card className="border-amber-500/20 bg-amber-500/5">
        <CardHeader>
          <CardTitle className="text-base">What to expect</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            {[
              "Run your QAOA and VQE circuits on real superconducting qubits",
              "Automatic transpilation to hardware-native gate sets",
              "Job queue management and result retrieval",
              "Error mitigation (ZNE, readout calibration) applied automatically",
              "Side-by-side comparison: simulator vs. hardware fidelity",
              "Usage quotas per Enterprise plan with job history",
            ].map((item) => (
              <li key={item} className="flex items-start gap-2">
                <span className="text-amber-500 mt-0.5">•</span>
                {item}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Notify Me */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Get notified when it launches</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Input
              placeholder="your@email.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleNotify()}
              className="flex-1"
            />
            <Button onClick={handleNotify} disabled={!email.trim()} className="bg-amber-500 hover:bg-amber-600 text-white">
              Notify Me
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Enterprise subscribers get early access. No spam — one email when it's ready.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
