import { loadStripe } from "@stripe/stripe-js";
import { Elements } from "@stripe/react-stripe-js";
import { Loader2 } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useSubscription } from "@/hooks/useSubscription";
import { CheckoutForm, PlanFeatureList, PRICE_LABEL } from "@/components/PricingModal";

const OWNER_EMAIL = "darriusperson@gmail.com";
const DEV_BYPASS = import.meta.env.DEV; // unlocks the paywall on localhost, matching the rest of the app's dev convenience

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY ?? "");

interface SubscriptionGateProps {
  children: React.ReactNode;
}

/**
 * Blocking payment gate — sits between AuthGuard and the app itself. A signed-in
 * user who hasn't paid sees this instead of the app, full stop; there is no
 * per-tab gating anymore now that this exists.
 */
export function SubscriptionGate({ children }: SubscriptionGateProps) {
  const { user } = useAuth();
  const { subscription, refresh } = useSubscription();

  if (DEV_BYPASS || user?.email === OWNER_EMAIL) {
    return <>{children}</>;
  }

  if (subscription.loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="relative">
            <img src="/apple-touch-icon.png" alt="Lachesis" className="w-12 h-12 object-contain animate-pulse mx-auto" />
            <div className="absolute inset-0 w-12 h-12 border-2 border-primary/30 rounded-full animate-ping"></div>
          </div>
          <div className="space-y-2">
            <Loader2 className="w-6 h-6 animate-spin mx-auto text-primary" />
            <p className="text-muted-foreground">Loading...</p>
          </div>
        </div>
      </div>
    );
  }

  if (subscription.is_subscribed) {
    return <>{children}</>;
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center space-y-2">
          <img src="/apple-touch-icon.png" alt="Lachesis" className="w-12 h-12 object-contain mx-auto" />
          <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Subscribe to Lachesis
          </h1>
          <p className="text-sm text-muted-foreground">
            A subscription is required to use Lachesis — your financial literacy tutor.
          </p>
        </div>

        <div className="rounded-xl border border-primary/20 bg-gradient-to-br from-card to-primary/5 p-6 space-y-4">
          <div className="text-center">
            <span className="text-3xl font-bold">{PRICE_LABEL}</span>
            <span className="text-sm text-muted-foreground">/mo</span>
          </div>
          <PlanFeatureList />
          <Elements stripe={stripePromise}>
            <CheckoutForm onSuccess={refresh} submitLabel="Subscribe & Continue" />
          </Elements>
        </div>
      </div>
    </div>
  );
}
