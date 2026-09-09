import { Button } from "@/components/ui/button";
import { type Plan } from "@/hooks/useSubscription";

interface SubscriptionBadgeProps {
  plan: Plan;
  loading: boolean;
  onManage: () => void;
}

export function SubscriptionBadge({ plan, loading, onManage }: SubscriptionBadgeProps) {
  if (loading || plan === "free") return null;

  return (
    <Button
      size="sm"
      variant="ghost"
      className="h-6 text-xs px-2 text-muted-foreground hover:text-foreground"
      onClick={onManage}
    >
      Manage
    </Button>
  );
}
