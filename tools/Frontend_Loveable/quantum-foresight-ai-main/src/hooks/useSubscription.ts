import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/hooks/useAuth";
import { get } from "@/lib/api";

export type Plan = "free" | "pro" | "enterprise";

export interface SubscriptionState {
  plan: Plan;
  status: string;
  period_end: string | null;
  is_pro: boolean;
  is_enterprise: boolean;
  loading: boolean;
}

const DEFAULT_STATE: SubscriptionState = {
  plan: "free",
  status: "active",
  period_end: null,
  is_pro: false,
  is_enterprise: false,
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
      const data = await get<SubscriptionState>(
        `/api/billing/subscription-status?user_id=${user.id}`
      );
      setSubscription({ ...data, loading: false });
    } catch {
      // If billing endpoint not yet live, default to free
      setSubscription({ ...DEFAULT_STATE, loading: false });
    }
  }, [user?.id]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { subscription, refresh };
}
