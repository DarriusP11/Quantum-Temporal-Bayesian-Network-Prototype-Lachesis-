import { useAuth } from '@/hooks/useAuth';
import { Loader2 } from 'lucide-react';
import Auth from '@/pages/Auth';

interface AuthGuardProps {
  children: React.ReactNode;
}

const AuthGuard = ({ children }: AuthGuardProps) => {
  const { user, loading } = useAuth();

  if (loading) {
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

  if (!user) {
    return <Auth />;
  }

  return <>{children}</>;
};

export default AuthGuard;
