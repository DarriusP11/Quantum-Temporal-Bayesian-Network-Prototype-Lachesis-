import { useState } from "react";
import { loadStripe } from "@stripe/stripe-js";
import {
  Elements,
  CardElement,
  useStripe,
  useElements,
} from "@stripe/react-stripe-js";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Check, Zap, Crown, Star, Loader2 } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { post } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY ?? "");

// ── Tier definitions ────────────────────────────────────────────────────────

const TIERS = [
  {
    id:      "free",
    label:   "Free",
    icon:    Star,
    price:   0,
    priceId: null as string | null,
    color:   "border-border",
    badge:   "",
    features: [
      "Lachesis AI copilot",
      "Financial Analytics (Monte Carlo)",
      "Insider Trading + SEC EDGAR",
      "Sentiment Analysis",
      "Prompt Studio",
      "Q-TBN regime forecasting",
      "Foresight noise sweeps",
      "Full quantum circuit simulator",
      "Advanced Quantum diagnostics",
    ],
  },
  {
    id:      "pro",
    label:   "Pro",
    icon:    Zap,
    price:   29.99,
    priceId: import.meta.env.VITE_STRIPE_PRO_MONTHLY_PRICE_ID ?? "",
    color:   "border-primary/60 shadow-[0_0_0_1px_hsl(var(--primary)/0.3)]",
    badge:   "Most Popular",
    features: [
      "Everything in Free",
      "Toy QAOA portfolio optimization",
      "VQE Risk Gate + Hamiltonian solver",
      "Quantum Amplitude Estimation (QAE) for VaR",
      "Priority support",
    ],
  },
  {
    id:      "enterprise",
    label:   "Enterprise",
    icon:    Crown,
    price:   99.99,
    priceId: import.meta.env.VITE_STRIPE_ENTERPRISE_MONTHLY_PRICE_ID ?? "",
    color:   "border-amber-500/60 shadow-[0_0_0_1px_hsl(40_90%_50%/0.3)]",
    badge:   "Coming: Real Hardware",
    features: [
      "Everything in Pro",
      "Real Quantum Hardware access (coming soon)",
      "IBM Quantum + Google Quantum AI",
      "Hardware-native transpilation",
      "Dedicated support + SLA",
    ],
  },
];

// ── Card form inside Elements context ───────────────────────────────────────

function CheckoutForm({
  priceId,
  onSuccess,
  onCancel,
}: {
  priceId: string;
  onSuccess: () => void;
  onCancel: () => void;
}) {
  const stripe   = useStripe();
  const elements = useElements();
  const { user } = useAuth();
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!stripe || !elements || !user) return;

    setLoading(true);
    setError(null);

    try {
      // 1. Create SetupIntent to collect card
      const { client_secret, customer_id } = await post<{ client_secret: string; customer_id: string }>(
        "/api/billing/create-setup-intent",
        { user_id: user.id, email: user.email }
      );

      // 2. Confirm card setup (3D Secure handled by Stripe if needed)
      const cardEl = elements.getElement(CardElement);
      if (!cardEl) throw new Error("Card element not found");

      const { error: setupError, setupIntent } = await stripe.confirmCardSetup(client_secret, {
        payment_method: { card: cardEl },
      });
      if (setupError) throw new Error(setupError.message ?? "Card setup failed");

      const paymentMethodId = typeof setupIntent.payment_method === "string"
        ? setupIntent.payment_method
        : setupIntent.payment_method?.id ?? "";

      // 3. Create subscription with the confirmed payment method
      await post("/api/billing/create-subscription", {
        user_id:           user.id,
        price_id:          priceId,
        payment_method_id: paymentMethodId,
      });

      toast({ title: "Subscription activated!", description: "Your plan has been upgraded. Enjoy!" });
      onSuccess();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Payment failed. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 pt-4">
      <div className="p-3 rounded-lg border border-border bg-background">
        <CardElement
          options={{
            style: {
              base: {
                fontSize:   "14px",
                color:      "#ffffff",
                "::placeholder": { color: "#6b7280" },
              },
            },
          }}
        />
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex gap-2">
        <Button type="submit" disabled={!stripe || loading} className="flex-1 gap-2">
          {loading && <Loader2 className="w-4 h-4 animate-spin" />}
          {loading ? "Processing…" : "Subscribe Now"}
        </Button>
        <Button type="button" variant="outline" onClick={onCancel} disabled={loading}>
          Cancel
        </Button>
      </div>

      <p className="text-xs text-muted-foreground text-center">
        Secured by Stripe · Cancel anytime · No hidden fees
      </p>
    </form>
  );
}

// ── Main modal ───────────────────────────────────────────────────────────────

interface PricingModalProps {
  open:     boolean;
  onClose:  () => void;
  onSuccess: () => void;
  /** Pre-select the tier to highlight */
  defaultTier?: "pro" | "enterprise";
}

export function PricingModal({ open, onClose, onSuccess, defaultTier = "pro" }: PricingModalProps) {
  const [selectedTier, setSelectedTier] = useState<string | null>(null);

  const handleSuccess = () => {
    setSelectedTier(null);
    onSuccess();
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-center">
            Choose your Lachesis plan
          </DialogTitle>
          <p className="text-sm text-muted-foreground text-center">Billed monthly · Cancel anytime</p>
        </DialogHeader>

        {/* Tier cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          {TIERS.map((tier) => {
            const Icon          = tier.icon;
            const isHighlighted = tier.id === defaultTier && !selectedTier;
            const isSelected    = selectedTier === tier.id;

            return (
              <div
                key={tier.id}
                className={`relative rounded-xl border p-5 flex flex-col gap-4 transition-all
                  ${tier.color}
                  ${isHighlighted ? "ring-2 ring-primary/50" : ""}
                  ${isSelected    ? "ring-2 ring-primary"    : ""}
                `}
              >
                {tier.badge && (
                  <Badge className={`absolute -top-2.5 left-1/2 -translate-x-1/2 text-xs px-2 ${
                    tier.id === "enterprise"
                      ? "bg-amber-500 text-white"
                      : "bg-primary text-primary-foreground"
                  }`}>
                    {tier.badge}
                  </Badge>
                )}

                <div className="flex items-center gap-2">
                  <Icon className={`w-5 h-5 ${
                    tier.id === "enterprise" ? "text-amber-400"
                    : tier.id === "pro"      ? "text-primary"
                    : "text-muted-foreground"
                  }`} />
                  <span className="font-semibold">{tier.label}</span>
                </div>

                <div>
                  {tier.price === 0 ? (
                    <span className="text-2xl font-bold">Free</span>
                  ) : (
                    <>
                      <span className="text-2xl font-bold">${tier.price.toFixed(2)}</span>
                      <span className="text-sm text-muted-foreground">/mo</span>
                    </>
                  )}
                </div>

                <ul className="space-y-1.5 flex-1">
                  {tier.features.map((f) => (
                    <li key={f} className="flex items-start gap-2 text-xs text-muted-foreground">
                      <Check className="w-3.5 h-3.5 text-green-400 mt-0.5 shrink-0" />
                      {f}
                    </li>
                  ))}
                </ul>

                {tier.id !== "free" && (
                  <Button
                    size="sm"
                    variant={isSelected ? "default" : "outline"}
                    className={`w-full mt-2 ${
                      tier.id === "enterprise" && !isSelected
                        ? "border-amber-500/40 text-amber-400 hover:bg-amber-500/10"
                        : ""
                    }`}
                    onClick={() => setSelectedTier(isSelected ? null : tier.id)}
                  >
                    {isSelected ? "Hide form" : `Get ${tier.label}`}
                  </Button>
                )}

                {/* Inline checkout form */}
                {isSelected && tier.priceId && (
                  <Elements stripe={stripePromise}>
                    <CheckoutForm
                      priceId={tier.priceId}
                      onSuccess={handleSuccess}
                      onCancel={() => setSelectedTier(null)}
                    />
                  </Elements>
                )}
              </div>
            );
          })}
        </div>

        <p className="text-xs text-muted-foreground text-center pt-2">
          All plans include a secure Stripe checkout. Upgrade or cancel anytime from your billing portal.
        </p>
      </DialogContent>
    </Dialog>
  );
}
