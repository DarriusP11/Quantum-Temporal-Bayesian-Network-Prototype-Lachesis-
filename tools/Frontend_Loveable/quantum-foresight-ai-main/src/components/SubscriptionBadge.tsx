import { Zap, Crown, Star } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type Plan } from "@/hooks/useSubscription";

interface SubscriptionBadgeProps {
  plan: Plan;
  loading: boolean;
  onUpgrade: () => void;
  onManage: () => void;
}

const PLAN_CONFIG: Record<Plan, { label: string; icon: typeof Zap; className: string }> = {
  free:       { label: "Free",       icon: Star,  className: "bg-muted text-muted-foreground border-border" },
  pro:        { label: "Pro",        icon: Zap,   className: "bg-primary/20 text-primary border-primary/40" },
  enterprise: { label: "Enterprise", icon: Crown, className: "bg-amber-500/20 text-amber-400 border-amber-500/40" },
};

export function SubscriptionBadge({ plan, loading, onUpgrade, onManage }: SubscriptionBadgeProps) {
  if (loading) {
    return <div className="h-6 w-16 rounded-full bg-muted animate-pulse" />;
  }

  const config = PLAN_CONFIG[plan];
  const Icon   = config.icon;

  return (
    <div className="flex items-center gap-2">
      <Badge variant="outline" className={`text-xs px-2 py-0.5 flex items-center gap-1 ${config.className}`}>
        <Icon className="w-3 h-3" />
        {config.label}
      </Badge>

      {plan === "free" ? (
        <Button
          size="sm"
          variant="outline"
          className="h-6 text-xs px-2 border-primary/30 text-primary hover:bg-primary/10"
          onClick={onUpgrade}
        >
          Upgrade
        </Button>
      ) : (
        <Button
          size="sm"
          variant="ghost"
          className="h-6 text-xs px-2 text-muted-foreground hover:text-foreground"
          onClick={onManage}
        >
          Manage
        </Button>
      )}
    </div>
  );
}
