import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/hooks/useAuth";
import { authGet } from "@/lib/api";

export type Plan = "free" | "premium";

export interface SubscriptionState {
  plan: Plan;
  status: string;
  period_end: string | null;
  is_subscribed: boolean;
  loading: boolean;
}

const DEFAULT_STATE: SubscriptionState = {
  plan: "free",
  status: "active",
  period_end: null,
  is_subscribed: false,
  loading: true,
};

export function useSubscription() {
  const { user } = useAuth();
  const [subscription, setSubscription] = useState<SubscriptionState>(DEFAULT_STATE);

  const refresh = useCallback(async () => {
    if (!user?.id) {
      setSubscription({ ...DEFAULT_STATE, loading: false });
      return;
    }
    try {
      const data = await authGet<Omit<SubscriptionState, "loading">>("/api/billing/subscription-status");
      setSubscription({ ...data, loading: false });
    } catch {
      setSubscription({ ...DEFAULT_STATE, loading: false });
    }
  }, [user?.id]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { subscription, refresh };
}
