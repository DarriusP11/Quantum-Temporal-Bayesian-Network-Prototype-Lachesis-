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
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Check, Loader2 } from "lucide-react";
import { authPost } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY ?? "");

export const PRICE_LABEL = "$8.99";

const FEATURES = [
  "Lachesis AI copilot",
  "Budgeting, Retirement & Debt planning tools",
  "Credit Behavior Simulator",
  "Home Planning cost simulator",
  "Financial Analytics (Monte Carlo)",
  "Insider Trading + SEC EDGAR",
  "Sentiment Analysis",
];

// ── Card form inside Elements context — shared by PricingModal and the
// blocking SubscriptionGate so the Stripe Elements logic lives in one place ──

export function CheckoutForm({
  onSuccess,
  onCancel,
  submitLabel = "Subscribe Now",
}: {
  onSuccess: () => void;
  onCancel?: () => void;
  submitLabel?: string;
}) {
  const stripe   = useStripe();
  const elements = useElements();
  const { user }  = useAuth();
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!stripe || !elements) return;

    setLoading(true);
    setError(null);

    try {
      // 1. Create SetupIntent to collect card. The backend derives *who* from
      // the auth token; email is passed along for the Stripe Customer record.
      const { client_secret } = await authPost<{ client_secret: string }>(
        "/api/billing/create-setup-intent",
        { email: user?.email ?? "" }
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

      // 3. Create the subscription — the server always uses its one configured price
      const sub = await authPost<{
        subscription_id: string;
        status: string;
        plan: string;
        client_secret: string | null;
      }>("/api/billing/create-subscription", { payment_method_id: paymentMethodId });

      // 4. Stripe often can't activate the subscription synchronously — if it
      // didn't, confirm the initial invoice's PaymentIntent here, then have
      // the server re-sync the real status rather than assuming success.
      if (sub.status !== "active" && sub.status !== "trialing") {
        if (!sub.client_secret) {
          throw new Error("Subscription requires payment confirmation, but no client secret was returned.");
        }
        const { error: paymentError, paymentIntent } = await stripe.confirmCardPayment(sub.client_secret);
        if (paymentError) throw new Error(paymentError.message ?? "Payment confirmation failed");
        if (paymentIntent?.status !== "succeeded") {
          throw new Error(`Payment was not completed (status: ${paymentIntent?.status ?? "unknown"}).`);
        }

        const confirmed = await authPost<{ is_subscribed: boolean; status: string }>(
          "/api/billing/confirm-subscription",
          {}
        );
        if (!confirmed.is_subscribed) {
          throw new Error(`Subscription isn't active yet (status: ${confirmed.status}). Please try again in a moment.`);
        }
      }

      toast({ title: "Subscription activated!", description: "You're all set — welcome to Lachesis." });
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
          {loading ? "Processing…" : submitLabel}
        </Button>
        {onCancel && (
          <Button type="button" variant="outline" onClick={onCancel} disabled={loading}>
            Cancel
          </Button>
        )}
      </div>

      <p className="text-xs text-muted-foreground text-center">
        Secured by Stripe · Cancel anytime · No hidden fees
      </p>
    </form>
  );
}

// ── Feature list card, shared visual between the modal and the blocking gate ─

export function PlanFeatureList() {
  return (
    <ul className="space-y-1.5">
      {FEATURES.map((f) => (
        <li key={f} className="flex items-start gap-2 text-xs text-muted-foreground">
          <Check className="w-3.5 h-3.5 text-green-400 mt-0.5 shrink-0" />
          {f}
        </li>
      ))}
    </ul>
  );
}

// ── Main modal (re-subscribe / manage flow for existing users) ───────────────

interface PricingModalProps {
  open:      boolean;
  onClose:   () => void;
  onSuccess: () => void;
}

export function PricingModal({ open, onClose, onSuccess }: PricingModalProps) {
  const [showForm, setShowForm] = useState(false);

  const handleSuccess = () => {
    setShowForm(false);
    onSuccess();
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) { setShowForm(false); onClose(); } }}>
      <DialogContent className="max-w-md w-full">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-center">Lachesis Premium</DialogTitle>
          <p className="text-sm text-muted-foreground text-center">Billed monthly · Cancel anytime</p>
        </DialogHeader>

        <div className="text-center py-2">
          <span className="text-3xl font-bold">{PRICE_LABEL}</span>
          <span className="text-sm text-muted-foreground">/mo</span>
        </div>

        <PlanFeatureList />

        {!showForm ? (
          <Button className="w-full mt-2" onClick={() => setShowForm(true)}>Subscribe</Button>
        ) : (
          <Elements stripe={stripePromise}>
            <CheckoutForm onSuccess={handleSuccess} onCancel={() => setShowForm(false)} />
          </Elements>
        )}
      </DialogContent>
    </Dialog>
  );
}
