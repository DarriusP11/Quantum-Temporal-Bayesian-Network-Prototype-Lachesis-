import { Lock, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

interface LockedTabOverlayProps {
  requiredPlan: "pro" | "enterprise";
  tabName: string;
  onUpgrade: () => void;
}

export function LockedTabOverlay({ requiredPlan, tabName, onUpgrade }: LockedTabOverlayProps) {
  const planLabel = requiredPlan === "enterprise" ? "Enterprise" : "Pro";
  const planColor = requiredPlan === "enterprise"
    ? "from-amber-500/20 to-orange-500/20 border-amber-500/30"
    : "from-primary/20 to-accent/20 border-primary/30";
  const btnColor  = requiredPlan === "enterprise"
    ? "bg-amber-500 hover:bg-amber-600 text-white"
    : "";

  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[400px] gap-6 p-8">
      <div className={`flex flex-col items-center gap-4 p-8 rounded-2xl border bg-gradient-to-br ${planColor} max-w-md w-full text-center`}>
        <div className="w-16 h-16 rounded-full bg-background/60 flex items-center justify-center border border-border/40">
          <Lock className="w-7 h-7 text-primary" />
        </div>

        <div>
          <h2 className="text-xl font-bold text-foreground mb-1">
            {tabName} requires {planLabel}
          </h2>
          <p className="text-sm text-muted-foreground">
            Upgrade your plan to unlock{" "}
            <span className="font-medium text-foreground">{tabName}</span> and all{" "}
            {planLabel} features.
          </p>
        </div>

        <Button onClick={onUpgrade} className={`gap-2 ${btnColor}`}>
          <Zap className="w-4 h-4" />
          Upgrade to {planLabel}
        </Button>
      </div>
    </div>
  );
}
