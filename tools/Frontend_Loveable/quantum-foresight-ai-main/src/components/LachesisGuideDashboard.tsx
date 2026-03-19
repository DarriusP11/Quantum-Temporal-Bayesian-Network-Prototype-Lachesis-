/**
 * LachesisGuideDashboard.tsx — AI-powered Lachesis narrative generation.
 * Mirrors the "Lachesis Guide" tab in qtbn_simulator_clean.py.
 */
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { AlertCircle, RefreshCw, Sparkles, MessageSquare, Send, Mic, MicOff, Volume2, VolumeX } from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import { post } from "@/lib/api";
import { useVoice } from "@/hooks/useVoice";

interface GuideResponse {
  narrative: string;
  context: string;
  question: string;
}

const QUICK_QUESTIONS = [
  "What is the current market risk level?",
  "Should I reduce position size?",
  "What hedges are appropriate for this regime?",
  "Explain the VaR metrics in plain English",
  "What does high volatility mean for my portfolio?",
  "How does the QTBN forecast impact my strategy?",
];

export const LachesisGuideDashboard = () => {
  const { state } = useAppContext();
  const fin = state.finance;
  const { isListening, isSpeaking, micSupported, ttsSupported, startListening, stopListening, speak, stopSpeaking } = useVoice(state.language ?? "English");

  const [question, setQuestion]   = useState("");
  const [openaiKey, setOpenaiKey] = useState("");
  const [showKey, setShowKey]     = useState(false);
  const [regime, setRegime]       = useState("Unknown");
  const [varUsd, setVarUsd]       = useState("");
  const [history, setHistory]     = useState<Array<{ q: string; a: string }>>([]);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [autoSpeak, setAutoSpeak] = useState(false);

  const ask = async (q?: string) => {
    const finalQ = q ?? question;
    if (!finalQ.trim()) return;
    setLoading(true); setError(null);
    try {
      const tickers = fin.tickers.split(",").map(t => t.trim()).filter(Boolean);
      const res = await post<GuideResponse>("/api/financial/lachesis-guide", {
        question: finalQ,
        tickers,
        regime,
        var_usd: varUsd ? parseFloat(varUsd) : null,
        cvar_usd: varUsd ? parseFloat(varUsd) * 1.5 : null,
        portfolio_value: fin.portfolio_value,
        openai_api_key: openaiKey || null,
        language: state.language ?? "English",
      });
      setHistory(prev => [{ q: finalQ, a: res.narrative }, ...prev]);
      setQuestion("");
      if (autoSpeak && ttsSupported) speak(res.narrative);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Config */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            Lachesis Guide
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              AI-powered risk narrative
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-base font-medium mb-1">A chat assistant that explains your risk results and answers portfolio questions.</p>
          <p className="text-sm text-muted-foreground">
            Ask Lachesis anything about your portfolio risk. Provide an OpenAI key for GPT-4.1-mini powered answers,
            or use the built-in rule-based guidance.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div>
              <Label className="text-xs">Market Regime</Label>
              <select
                className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
                value={regime}
                onChange={e => setRegime(e.target.value)}
              >
                {["Unknown","Calm","Moderate","High Volatility","Stress","Crisis"].map(r => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </div>
            <div>
              <Label className="text-xs">Est. VaR ($, optional)</Label>
              <Input value={varUsd} onChange={e => setVarUsd(e.target.value)}
                placeholder="e.g. 5000" className="mt-1 h-8 text-xs" />
            </div>
            <div>
              <Label className="text-xs">OpenAI API key (optional)</Label>
              <Input
                type={showKey ? "text" : "password"}
                value={openaiKey}
                onChange={e => setOpenaiKey(e.target.value)}
                placeholder="sk-..."
                className="mt-1 h-8 text-xs font-mono"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick questions */}
      <Card className="border-accent/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <MessageSquare className="w-4 h-4 text-primary" />
            Quick Questions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {QUICK_QUESTIONS.map(q => (
              <Button
                key={q} variant="outline" size="sm"
                className="text-xs h-auto py-1.5"
                onClick={() => ask(q)}
                disabled={loading}
              >
                {q}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Custom question */}
      <Card className="border-accent/20">
        <CardContent className="pt-4">
          <div className="flex gap-3">
            <Textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="Ask Lachesis anything about your portfolio risk..."
              className="flex-1 text-sm min-h-[80px]"
              onKeyDown={e => { if (e.key === "Enter" && e.ctrlKey) ask(); }}
            />
            <div className="flex flex-col gap-2 self-end">
              {micSupported && (
                <Button
                  variant="outline"
                  size="icon"
                  className={`h-9 w-9 ${isListening ? "bg-red-500/20 border-red-500/50 animate-pulse" : ""}`}
                  onClick={() => isListening
                    ? stopListening()
                    : startListening(text => setQuestion(prev => prev ? `${prev} ${text}` : text))
                  }
                  disabled={loading}
                  title={isListening ? "Stop listening" : "Speak your question"}
                >
                  {isListening ? <MicOff className="w-4 h-4 text-red-400" /> : <Mic className="w-4 h-4" />}
                </Button>
              )}
              {ttsSupported && (
                <Button
                  variant="outline"
                  size="icon"
                  className={`h-9 w-9 ${autoSpeak ? "bg-primary/10 border-primary/40" : ""}`}
                  onClick={() => { if (isSpeaking) stopSpeaking(); else setAutoSpeak(v => !v); }}
                  title={isSpeaking ? "Stop speaking" : autoSpeak ? "Auto-speak on" : "Toggle auto-speak"}
                >
                  {isSpeaking ? <VolumeX className="w-4 h-4 text-primary" /> : <Volume2 className="w-4 h-4" />}
                </Button>
              )}
              <Button onClick={() => ask()} disabled={loading || !question.trim()} className="h-9 w-9" size="icon">
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              </Button>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-1">Ctrl+Enter to send{micSupported ? " · Mic button to speak" : ""}</p>
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

      {/* Conversation history */}
      {history.length > 0 && (
        <div className="space-y-4">
          {history.map((h, i) => (
            <Card key={i} className="border-accent/20">
              <CardContent className="pt-4 space-y-3">
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="text-xs shrink-0">You</Badge>
                  <p className="text-sm text-muted-foreground">{h.q}</p>
                </div>
                <div className="flex items-start gap-2">
                  <Badge className="text-xs shrink-0 bg-primary/20 text-primary border-primary/30">Lachesis</Badge>
                  <div className="flex-1 text-sm leading-relaxed whitespace-pre-wrap">{h.a}</div>
                  {ttsSupported && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0 opacity-60 hover:opacity-100"
                      onClick={() => isSpeaking ? stopSpeaking() : speak(h.a)}
                      title="Read aloud"
                    >
                      {isSpeaking ? <VolumeX className="w-3 h-3" /> : <Volume2 className="w-3 h-3" />}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};
