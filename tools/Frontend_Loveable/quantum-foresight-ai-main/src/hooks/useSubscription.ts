export type Plan = "free" | "basic" | "pro" | "enterprise";

export interface SubscriptionState {
  plan: Plan;
  status: string;
  period_end: string | null;
  is_basic: boolean;
  is_pro: boolean;
  is_enterprise: boolean;
  loading: boolean;
}

/* DEFAULT_STATE — used by real implementation below
const DEFAULT_STATE: SubscriptionState = {
  plan: "free",
  status: "active",
  period_end: null,
  is_basic: false,
  is_pro: false,
  is_enterprise: false,
  loading: true,
};
*/

// TESTING BYPASS — remove this block and uncomment the real implementation below to re-enable paywall
const BYPASS_STATE: SubscriptionState = {
  plan: "enterprise",
  status: "active",
  period_end: null,
  is_basic: true,
  is_pro: true,
  is_enterprise: true,
  loading: false,
};

export function useSubscription() {
  return { subscription: BYPASS_STATE, refresh: async () => {} };
}

/* REAL IMPLEMENTATION — uncomment to re-enable paywall (restore imports and DEFAULT_STATE too)
import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/hooks/useAuth";
import { get } from "@/lib/api";

export function useSubscription() {
  const { user } = useAuth();
  const [subscription, setSubscription] = useState<SubscriptionState>(DEFAULT_STATE);

  const refresh = useCallback(async () => {
    if (!user?.id) {
      setSubscription({ ...DEFAULT_STATE, loading: false });
      return;
    }
    try {
      const data = await get<SubscriptionState>(
        `/api/billing/subscription-status?user_id=${user.id}`
      );
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
*/
