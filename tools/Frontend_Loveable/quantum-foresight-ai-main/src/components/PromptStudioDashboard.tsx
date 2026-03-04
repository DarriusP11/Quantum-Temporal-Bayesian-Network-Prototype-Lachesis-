/**
 * PromptStudioDashboard.tsx — LLM-powered scenario generation with template rendering.
 * Mirrors the "Prompt Studio" tab in qtbn_simulator_clean.py.
 */
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertCircle, RefreshCw, Wand2, Copy, CheckCheck } from "lucide-react";
import { get, post } from "@/lib/api";
import { useAppContext } from "@/contexts/AppContext";

interface Template { key: string; template: string; }
interface GenerateResponse { prompt: string; result: string; template: string; tokens_requested: number; }

export const PromptStudioDashboard = () => {
  const { state } = useAppContext();
  const fin = state.finance;

  const [templates, setTemplates]     = useState<Template[]>([]);
  const [selectedTpl, setSelectedTpl] = useState("risk_scenario");
  const [customPrompt, setCustomPrompt] = useState("");
  const [useCustom, setUseCustom]     = useState(false);
  const [openaiKey, setOpenaiKey]     = useState("");
  const [maxTokens, setMaxTokens]     = useState(500);
  const [result, setResult]           = useState<GenerateResponse | null>(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [copied, setCopied]           = useState(false);

  useEffect(() => {
    get<{ templates: Template[] }>("/api/prompt-studio/templates").then(r => {
      setTemplates(r.templates);
    }).catch(() => {});
  }, []);

  const tickers = fin.tickers.split(",").map(t => t.trim()).filter(Boolean);
  const regime = "Current";

  const generate = async () => {
    setLoading(true); setError(null);
    try {
      const variables: Record<string, unknown> = {
        tickers: tickers.join(", "),
        regime,
        portfolio_value: fin.portfolio_value,
        p_gain: 0.35, p_flat: 0.30, p_loss: 0.25, p_severe: 0.10,
        var_usd: Math.round(fin.portfolio_value * 0.05),
        num_qubits: state.num_qubits,
        gates: `T0:${state.step0.q0}`,
        noise: state.noise.enable_depolarizing ? `dep=${state.noise.pdep0}` : "none",
        ticker: tickers[0] ?? "SPY",
        notional: fin.portfolio_value,
        policy: "Moderate",
        status: "APPROVED",
      };
      const res = await post<GenerateResponse>("/api/prompt-studio/generate", {
        template: useCustom ? "custom" : selectedTpl,
        variables,
        custom_prompt: useCustom ? customPrompt : null,
        openai_api_key: openaiKey || null,
        max_tokens: maxTokens,
        language: state.language ?? "English",
      });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const copyResult = async () => {
    if (!result) return;
    await navigator.clipboard.writeText(result.result);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const selectedTemplate = templates.find(t => t.key === selectedTpl);

  return (
    <div className="space-y-6">
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wand2 className="w-5 h-5 text-primary" />
            Prompt Studio
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              LLM scenario generation
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Generate AI-driven financial scenario narratives using pre-built templates.
            Variables are auto-filled from sidebar controls. Add an OpenAI key for GPT-4o-mini output.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Template selector */}
            <div className="space-y-2">
              <Label className="text-xs">Template</Label>
              <Select value={selectedTpl} onValueChange={v => { setSelectedTpl(v); setUseCustom(false); }}>
                <SelectTrigger className="h-9 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {templates.map(t => (
                    <SelectItem key={t.key} value={t.key} className="text-xs">{t.key.replace(/_/g," ")}</SelectItem>
                  ))}
                  <SelectItem value="custom" className="text-xs">Custom prompt</SelectItem>
                </SelectContent>
              </Select>
              {selectedTemplate && !useCustom && (
                <div className="text-xs text-muted-foreground bg-muted/20 p-2 rounded font-mono break-all">
                  {selectedTemplate.template}
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label className="text-xs">OpenAI API key (optional)</Label>
              <Input type="password" value={openaiKey} onChange={e => setOpenaiKey(e.target.value)}
                placeholder="sk-..." className="h-9 text-xs font-mono" />
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <Label>Max tokens</Label>
                  <span className="font-mono text-primary">{maxTokens}</span>
                </div>
                <Slider value={[maxTokens]} onValueChange={([v]) => setMaxTokens(v)} min={100} max={2000} step={100} />
              </div>
            </div>
          </div>

          {/* Custom prompt */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input type="checkbox" id="use-custom" checked={useCustom}
                onChange={e => setUseCustom(e.target.checked)} />
              <Label htmlFor="use-custom" className="text-xs cursor-pointer">Use custom prompt instead</Label>
            </div>
            {useCustom && (
              <Textarea value={customPrompt} onChange={e => setCustomPrompt(e.target.value)}
                placeholder="Write your custom prompt here..." className="min-h-[100px] text-sm" />
            )}
          </div>

          <Button onClick={generate} disabled={loading} className="w-full h-10">
            {loading ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Generating...</>
                     : <><Wand2 className="w-4 h-4 mr-2" />Generate Scenario</>}
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
          {/* Rendered prompt */}
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                Rendered Prompt
                <Badge variant="outline" className="text-xs">{result.template}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="text-xs bg-muted/20 p-3 rounded font-mono whitespace-pre-wrap">
                {result.prompt}
              </pre>
            </CardContent>
          </Card>

          {/* Generated output */}
          <Card className="border-primary/20 bg-primary/5">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Wand2 className="w-4 h-4 text-primary" />
                Generated Scenario
                <Button variant="ghost" size="sm" className="ml-auto" onClick={copyResult}>
                  {copied ? <CheckCheck className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm leading-relaxed whitespace-pre-wrap">{result.result}</div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
};
