import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiQTBNForecast, QTBNForecastResponse } from "@/lib/api";
import {
  PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar
} from 'recharts';
import { Brain, TrendingUp, Zap, Target, Clock, Network, AlertCircle } from "lucide-react";

const REGIME_COLORS: Record<string, string> = {
  calm:     "#22c55e",
  stressed: "#f59e0b",
  crisis:   "#ef4444",
};

export const QTBNDashboard = () => {
  const [priorRegime, setPriorRegime] = useState("calm");
  const [riskOnPrior, setRiskOnPrior] = useState(0.5);
  const [driftMu, setDriftMu] = useState(0.08);
  const [horizonDays, setHorizonDays] = useState(10);
  const [steps, setSteps] = useState(5);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<QTBNForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = async () => {
    setIsRunning(true);
    setError(null);
    try {
      const res = await apiQTBNForecast({
        prior_regime: priorRegime,
        risk_on_prior: riskOnPrior,
        drift_mu: driftMu,
        horizon_days: horizonDays,
        steps,
      });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsRunning(false);
    }
  };

  // Forecast buckets pie chart
  const getForecastPieData = () => {
    if (!result) return [];
    return [
      { name: "Gain",       value: +(result.P_gain * 100).toFixed(1),        fill: "#22c55e" },
      { name: "Flat",       value: +(result.P_flat * 100).toFixed(1),        fill: "#64748b" },
      { name: "Loss",       value: +(result.P_loss * 100).toFixed(1),        fill: "#f59e0b" },
      { name: "Severe Loss",value: +(result.P_severe_loss * 100).toFixed(1), fill: "#ef4444" },
    ];
  };

  // Regime timeline line chart
  const getTimelineData = () => {
    if (!result) return [];
    return result.regime_timeline.map((step, i) => ({
      t: `T+${i + 1}`,
      calm:     +(step.calm * 100).toFixed(1),
      stressed: +(step.stressed * 100).toFixed(1),
      crisis:   +(step.crisis * 100).toFixed(1),
    }));
  };

  // Drift & risk-on path
  const getPathData = () => {
    if (!result) return [];
    return result.drift_path.map((d, i) => ({
      t: `T+${i}`,
      drift: +(d * 100).toFixed(2),
      risk_on: +((result.risk_on_path[i] ?? 0) * 100).toFixed(1),
    }));
  };

  const pieData = getForecastPieData();
  const timelineData = getTimelineData();
  const pathData = getPathData();

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <div className="relative">
              <Network className="w-8 h-8 text-primary" />
              <Zap className="w-4 h-4 text-accent absolute -top-1 -right-1 animate-pulse" />
            </div>
            <div>
              <h3 className="text-xl font-bold">Quantum Temporal Bayesian Network</h3>
              <p className="text-base font-medium">Uses a quantum-powered model to forecast market conditions and portfolio risk over time.</p>
              <p className="text-sm text-muted-foreground font-normal">
                Python QTBN engine with regime-aware drift forecasting
              </p>
            </div>
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              Python Backend
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <Label>Prior Regime</Label>
              <Select value={priorRegime} onValueChange={setPriorRegime}>
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="calm">Calm</SelectItem>
                  <SelectItem value="stressed">Stressed</SelectItem>
                  <SelectItem value="crisis">Crisis</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Risk-On Prior: {(riskOnPrior * 100).toFixed(0)}%</Label>
              <Slider
                value={[riskOnPrior]}
                onValueChange={([v]) => setRiskOnPrior(v)}
                min={0} max={1} step={0.05}
                className="mt-2"
              />
            </div>
            <div>
              <Label>Drift μ (annual): {(driftMu * 100).toFixed(1)}%</Label>
              <Slider
                value={[driftMu]}
                onValueChange={([v]) => setDriftMu(v)}
                min={-0.2} max={0.3} step={0.01}
                className="mt-2"
              />
            </div>
            <div>
              <Label>Horizon (days): {horizonDays}</Label>
              <Slider
                value={[horizonDays]}
                onValueChange={([v]) => setHorizonDays(v)}
                min={1} max={63} step={1}
                className="mt-2"
              />
            </div>
          </div>

          <div className="flex gap-4">
            <Button onClick={runAnalysis} disabled={isRunning} className="flex-1">
              {isRunning ? (
                <><Brain className="w-4 h-4 mr-2 animate-spin" />Running QTBN Inference...</>
              ) : (
                <><Zap className="w-4 h-4 mr-2" />Execute Q-TBN Analysis</>
              )}
            </Button>
            <Button variant="outline" onClick={() => setResult(null)}>Reset</Button>
          </div>

          {isRunning && <Progress className="h-2" />}
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
        <Tabs defaultValue="forecast" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="forecast">Forecast</TabsTrigger>
            <TabsTrigger value="regime">Regime Timeline</TabsTrigger>
            <TabsTrigger value="path">Drift & Risk Path</TabsTrigger>
            <TabsTrigger value="summary">Summary</TabsTrigger>
          </TabsList>

          {/* Forecast buckets */}
          <TabsContent value="forecast" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="border-accent/20">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5 text-primary" />
                    {horizonDays}-Day Forecast Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}%`}
                      >
                        {pieData.map((entry, i) => (
                          <Cell key={i} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="border-accent/20">
                <CardHeader>
                  <CardTitle>Probability Buckets</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 pt-2">
                  {pieData.map((d) => (
                    <div key={d.name}>
                      <div className="flex justify-between text-sm mb-1">
                        <span>{d.name}</span>
                        <span style={{ color: d.fill }}>{d.value}%</span>
                      </div>
                      <Progress value={d.value} className="h-2" style={{ "--progress-fill": d.fill } as React.CSSProperties} />
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Regime timeline */}
          <TabsContent value="regime">
            <Card className="border-accent/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-primary" />
                  Regime Probability Evolution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={timelineData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Line type="monotone" dataKey="calm"     stroke={REGIME_COLORS.calm}     strokeWidth={2} name="Calm" />
                    <Line type="monotone" dataKey="stressed" stroke={REGIME_COLORS.stressed} strokeWidth={2} name="Stressed" />
                    <Line type="monotone" dataKey="crisis"   stroke={REGIME_COLORS.crisis}   strokeWidth={2} name="Crisis" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Drift & risk path */}
          <TabsContent value="path">
            <Card className="border-accent/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-primary" />
                  Expected Drift & Risk-On Path
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={pathData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="drift"   stroke="hsl(263 70% 50%)" strokeWidth={2} name="Drift %" />
                    <Line type="monotone" dataKey="risk_on" stroke="#22c55e"           strokeWidth={2} name="Risk-On %" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Summary */}
          <TabsContent value="summary">
            <Card className="border-accent/20">
              <CardHeader><CardTitle>Analysis Summary</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: "Prior Regime", val: result.prior_regime.toUpperCase() },
                    { label: "Horizon",      val: `${result.horizon_days} days` },
                    { label: "P(Gain)",      val: `${(result.P_gain * 100).toFixed(1)}%` },
                    { label: "P(Flat)",      val: `${(result.P_flat * 100).toFixed(1)}%` },
                    { label: "P(Loss)",      val: `${(result.P_loss * 100).toFixed(1)}%` },
                    { label: "P(Severe Loss)",val:`${(result.P_severe_loss * 100).toFixed(1)}%` },
                  ].map(({ label, val }) => (
                    <div key={label} className="flex justify-between items-center p-2 border border-accent/20 rounded text-sm">
                      <span className="text-muted-foreground">{label}</span>
                      <span className="font-semibold">{val}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};
