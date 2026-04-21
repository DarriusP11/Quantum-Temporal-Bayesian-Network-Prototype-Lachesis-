/**
 * AdminDashboard.tsx — API key management (owner-only tab).
 * Only rendered when user email === "darriusperson@gmail.com".
 */
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Activity, AlertCircle, CheckCircle, Eye, EyeOff, KeyRound, Trash2 } from "lucide-react";
import { post, get } from "@/lib/api";

interface KeySpec {
  service: string;
  label: string;
  placeholder: string;
  hint: string;
}

const KEY_SPECS: KeySpec[] = [
  { service: "openai",      label: "OpenAI API Key",          placeholder: "sk-...",        hint: "GPT-4o-mini for Lachesis Guide & Prompt Studio" },
  { service: "fred",        label: "FRED API Key",            placeholder: "32-char key...", hint: "Federal Reserve macro data (CPI, yields)" },
  { service: "perplexity",  label: "Perplexity API Key",      placeholder: "pplx-...",      hint: "Real-time sentiment via Perplexity search" },
  { service: "voice_openai",label: "Voice OpenAI Key",        placeholder: "sk-...",        hint: "TTS/STT for Lachesis voice panel" },
  { service: "voice_elevenlabs", label: "ElevenLabs Key",     placeholder: "xi-...",        hint: "High-quality voice synthesis" },
  { service: "serpapi",     label: "SerpAPI Key",             placeholder: "serpapi key...", hint: "Google search for Lachesis AI real-time market data" },
  { service: "ibm_quantum", label: "IBM Quantum API Token",   placeholder: "IBM token...",  hint: "Used per-request for Circuit Inspector → IBM Hardware tab" },
];

type KeyStatus = "unconfigured" | "valid" | "invalid";

interface BillingHealth {
  stripe_connected:       boolean;
  pro_price_valid:        boolean;
  enterprise_price_valid: boolean;
  supabase_ok:            boolean;
  webhook_secret_set:     boolean;
}

export const AdminDashboard = () => {
  const [keys, setKeys]     = useState<Record<string, string>>(
    () => {
      try {
        const saved = localStorage.getItem("lachesis_admin_keys");
        return saved ? JSON.parse(saved) : {};
      } catch { return {}; }
    }
  );
  const [show, setShow]     = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState<Record<string, KeyStatus>>({});
  const [validating, setValidating] = useState<string | null>(null);

  // ── Billing health ──────────────────────────────────────────────────────────
  const [health, setHealth]             = useState<BillingHealth | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError]   = useState<string | null>(null);

  const fetchHealth = async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const data = await get<BillingHealth>("/api/billing/health");
      setHealth(data);
    } catch (e: unknown) {
      setHealthError(e instanceof Error ? e.message : "Could not reach backend");
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => { fetchHealth(); }, []);

  const setKey = (service: string, val: string) => {
    const updated = { ...keys, [service]: val };
    setKeys(updated);
    // Persist session-only (localStorage cleared on browser close if desired)
    try { localStorage.setItem("lachesis_admin_keys", JSON.stringify(updated)); } catch {}
    setStatus(s => ({ ...s, [service]: "unconfigured" }));
  };

  const validate = async (service: string) => {
    const key = keys[service];
    if (!key?.trim()) return;
    setValidating(service);
    try {
      const res = await post<{ service: string; valid: boolean; hint: string }>("/api/admin/validate-key", {
        service, api_key: key,
      });
      setStatus(s => ({ ...s, [service]: res.valid ? "valid" : "invalid" }));
    } catch {
      setStatus(s => ({ ...s, [service]: "invalid" }));
    } finally {
      setValidating(null);
    }
  };

  const clearAll = () => {
    setKeys({});
    setStatus({});
    try { localStorage.removeItem("lachesis_admin_keys"); } catch {}
  };

  return (
    <div className="space-y-6">
      <Card className="border-primary/30 bg-gradient-to-br from-card to-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <KeyRound className="w-5 h-5 text-primary" />
            Admin — API Key Management
            <Badge className="ml-auto bg-primary/20 text-primary border-primary/30 text-xs">Owner Only</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-sm text-muted-foreground bg-muted/20 p-3 rounded mb-4">
            <AlertCircle className="w-4 h-4 shrink-0" />
            Keys are stored in browser localStorage only — they never leave your device.
            Session keys override any environment-level keys.
          </div>
          <Button variant="outline" size="sm" onClick={clearAll} className="border-red-500/30 text-red-400 hover:bg-red-500/10">
            <Trash2 className="w-3 h-3 mr-1" />Clear all session keys
          </Button>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {KEY_SPECS.map(spec => {
          const val = keys[spec.service] ?? "";
          const st  = status[spec.service] ?? "unconfigured";
          const isShown = show[spec.service] ?? false;
          return (
            <Card key={spec.service} className={`border ${st === "valid" ? "border-green-500/30" : st === "invalid" ? "border-red-500/30" : "border-accent/20"}`}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  {spec.label}
                  <Badge variant="outline" className={`text-xs ml-auto ${
                    st === "valid" ? "border-green-500/30 text-green-400" :
                    st === "invalid" ? "border-red-500/30 text-red-400" :
                    "border-accent/30 text-muted-foreground"
                  }`}>
                    {st === "valid" ? "✓ Valid" : st === "invalid" ? "✗ Invalid" : "Not set"}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-xs text-muted-foreground">{spec.hint}</p>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Input
                      type={isShown ? "text" : "password"}
                      value={val}
                      onChange={e => setKey(spec.service, e.target.value)}
                      placeholder={spec.placeholder}
                      className="h-8 text-xs font-mono pr-8"
                    />
                    <button
                      type="button"
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      onClick={() => setShow(s => ({ ...s, [spec.service]: !isShown }))}
                    >
                      {isShown ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                    </button>
                  </div>
                  <Button
                    size="sm" variant="outline" className="h-8 text-xs px-2"
                    onClick={() => validate(spec.service)}
                    disabled={!val.trim() || validating === spec.service}
                  >
                    {validating === spec.service ? "..." : "Validate"}
                  </Button>
                </div>
                {st === "valid" && (
                  <div className="flex items-center gap-1 text-xs text-green-400">
                    <CheckCircle className="w-3 h-3" />Key format looks valid
                  </div>
                )}
                {st === "invalid" && (
                  <div className="flex items-center gap-1 text-xs text-red-400">
                    <AlertCircle className="w-3 h-3" />Key format invalid — {spec.hint.split("(")[0].trim()}
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* ── Billing System Health ───────────────────────────────────────────── */}
      <Card className="border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Activity className="w-4 h-4 text-primary" />
            Billing System Health
            <Button
              size="sm" variant="outline"
              className="ml-auto h-6 text-xs px-2"
              onClick={fetchHealth}
              disabled={healthLoading}
            >
              {healthLoading ? "Checking…" : "Refresh"}
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {healthError && (
            <div className="flex items-center gap-2 text-xs text-red-400 mb-3">
              <AlertCircle className="w-3 h-3 shrink-0" />
              {healthError}
            </div>
          )}
          {!health && !healthError && (
            <p className="text-xs text-muted-foreground">Loading…</p>
          )}
          {health && (() => {
            const checks: { key: keyof BillingHealth; label: string; hint: string }[] = [
              { key: "stripe_connected",       label: "Stripe API key valid",        hint: "STRIPE_SECRET_KEY on Railway" },
              { key: "pro_price_valid",        label: "Pro price ID valid",          hint: "STRIPE_PRO_MONTHLY_PRICE_ID in Stripe" },
              { key: "enterprise_price_valid", label: "Enterprise price ID valid",   hint: "STRIPE_ENTERPRISE_MONTHLY_PRICE_ID in Stripe" },
              { key: "supabase_ok",            label: "Supabase billing columns",    hint: "Run ALTER TABLE migration in Supabase SQL editor" },
              { key: "webhook_secret_set",     label: "Webhook secret configured",   hint: "STRIPE_WEBHOOK_SECRET on Railway" },
            ];
            return (
              <ul className="space-y-2">
                {checks.map(({ key, label, hint }) => {
                  const ok = health[key];
                  return (
                    <li key={key} className="flex items-start gap-2 text-xs">
                      {ok
                        ? <CheckCircle className="w-3.5 h-3.5 text-green-400 mt-0.5 shrink-0" />
                        : <AlertCircle className="w-3.5 h-3.5 text-red-400 mt-0.5 shrink-0" />
                      }
                      <span className={ok ? "text-green-400 font-medium" : "text-red-400 font-medium"}>{label}</span>
                      <span className="text-muted-foreground">— {hint}</span>
                    </li>
                  );
                })}
              </ul>
            );
          })()}
        </CardContent>
      </Card>

      <Card className="border-accent/20">
        <CardHeader><CardTitle className="text-sm">How to use your keys</CardTitle></CardHeader>
        <CardContent className="text-xs text-muted-foreground space-y-2">
          <p>• <strong>OpenAI key</strong> — paste into the Lachesis Guide or Prompt Studio "API key" field</p>
          <p>• <strong>Perplexity key</strong> — used by Sentiment Analysis for real-time web search</p>
          <p>• <strong>FRED key</strong> — unlocks macro stress testing (CPI, Fed Funds, unemployment)</p>
          <p>• <strong>ElevenLabs key</strong> — higher-quality voice synthesis in the AI Assistant tab</p>
          <p className="pt-1 text-muted-foreground/60">Keys validated here are format-checked only. Actual API calls are made from the backend at request time.</p>
        </CardContent>
      </Card>
    </div>
  );
};
