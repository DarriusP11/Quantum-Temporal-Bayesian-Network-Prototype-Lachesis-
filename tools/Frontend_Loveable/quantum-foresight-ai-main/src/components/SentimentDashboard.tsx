import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { apiSentimentAnalyze, SentimentResponse, SentimentItem } from "@/lib/api";
import { Brain, TrendingUp, TrendingDown, AlertCircle, Newspaper, RefreshCw, Rss, Zap } from "lucide-react";

type Provider = "google_rss" | "perplexity";

const scoreColor = (score: number) => {
  if (score > 0.2)  return "text-green-400";
  if (score < -0.2) return "text-red-400";
  return "text-yellow-400";
};

const scoreBadgeClass = (score: number) => {
  if (score > 0.2)  return "border-green-500/30 bg-green-500/10 text-green-400";
  if (score < -0.2) return "border-red-500/30 bg-red-500/10 text-red-400";
  return "border-yellow-500/30 bg-yellow-500/10 text-yellow-400";
};

const ScoreIcon = ({ score }: { score: number }) => {
  if (score > 0.2)  return <TrendingUp className="w-3 h-3" />;
  if (score < -0.2) return <TrendingDown className="w-3 h-3" />;
  return <span className="w-3 h-3 inline-block">–</span>;
};

export const SentimentDashboard = () => {
  const [provider, setProvider]         = useState<Provider>("google_rss");
  const [tickers, setTickers]           = useState("AAPL,MSFT,NVDA");
  const [keywords, setKeywords]         = useState("");
  const [maxItems, setMaxItems]         = useState(30);
  const [pplxKey, setPplxKey]           = useState("");
  const [pplxModel, setPplxModel]       = useState("sonar");
  const [isLoading, setIsLoading]       = useState(false);
  const [result, setResult]             = useState<SentimentResponse | null>(null);
  const [error, setError]               = useState<string | null>(null);

  const analyze = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const tickerList  = tickers.split(",").map(t => t.trim()).filter(Boolean);
      const keywordList = keywords.split(",").map(k => k.trim()).filter(Boolean);
      const res = await apiSentimentAnalyze({
        tickers: tickerList,
        keywords: keywordList,
        max_items: maxItems,
        provider,
        ...(provider === "perplexity" && {
          perplexity_api_key: pplxKey,
          perplexity_model: pplxModel,
        }),
      });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsLoading(false);
    }
  };

  const multiplierColor = (m: number) => {
    if (m > 1.1)  return "text-red-400";
    if (m < 0.9)  return "text-green-400";
    return "text-yellow-400";
  };

  return (
    <div className="space-y-6">
      {/* Config */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-accent/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary" />
            Market Sentiment Analysis
            <Badge variant="outline" className="ml-auto border-primary/30 bg-primary/10 text-xs">
              {provider === "google_rss" ? "Google News RSS · VADER" : "Perplexity Live Web"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">

          {/* Provider toggle */}
          <div>
            <Label className="text-xs text-muted-foreground mb-2 block">Sentiment Source</Label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setProvider("google_rss")}
                className={`flex items-center gap-2 p-3 rounded-lg border text-sm transition-all ${
                  provider === "google_rss"
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-accent/30 text-muted-foreground hover:border-accent/60"
                }`}
              >
                <Rss className="w-4 h-4 flex-shrink-0" />
                <div className="text-left">
                  <p className="font-medium">Google News RSS</p>
                  <p className="text-xs opacity-70">VADER sentiment · free</p>
                </div>
              </button>
              <button
                type="button"
                onClick={() => setProvider("perplexity")}
                className={`flex items-center gap-2 p-3 rounded-lg border text-sm transition-all ${
                  provider === "perplexity"
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-accent/30 text-muted-foreground hover:border-accent/60"
                }`}
              >
                <Zap className="w-4 h-4 flex-shrink-0" />
                <div className="text-left">
                  <p className="font-medium">Perplexity AI</p>
                  <p className="text-xs opacity-70">Live web search · API key required</p>
                </div>
              </button>
            </div>
          </div>

          {/* Perplexity-only fields */}
          {provider === "perplexity" && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-3 border border-primary/20 rounded-lg bg-primary/5">
              <div>
                <Label htmlFor="pplx-key" className="text-xs">Perplexity API Key <span className="text-red-400">*</span></Label>
                <Input
                  id="pplx-key"
                  type="password"
                  value={pplxKey}
                  onChange={e => setPplxKey(e.target.value)}
                  placeholder="pplx-..."
                  className="mt-1 h-8 text-xs font-mono"
                />
              </div>
              <div>
                <Label htmlFor="pplx-model" className="text-xs">Model</Label>
                <Input
                  id="pplx-model"
                  value={pplxModel}
                  onChange={e => setPplxModel(e.target.value)}
                  placeholder="sonar"
                  className="mt-1 h-8 text-xs"
                />
                <p className="text-xs text-muted-foreground mt-1">e.g. sonar, sonar-pro, sonar-reasoning</p>
              </div>
            </div>
          )}

          {/* Common fields */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="sent-tickers">Tickers (comma-separated)</Label>
              <Input
                id="sent-tickers"
                value={tickers}
                onChange={e => setTickers(e.target.value)}
                placeholder="AAPL,MSFT,NVDA"
              />
            </div>
            <div>
              <Label htmlFor="sent-keywords">Boost Keywords (optional)</Label>
              <Input
                id="sent-keywords"
                value={keywords}
                onChange={e => setKeywords(e.target.value)}
                placeholder="AI,earnings,guidance"
              />
            </div>
            <div>
              <Label htmlFor="sent-max">Max Headlines</Label>
              <Input
                id="sent-max"
                type="number"
                value={maxItems}
                onChange={e => setMaxItems(parseInt(e.target.value) || 30)}
                min={5} max={200}
              />
            </div>
          </div>

          <Button onClick={analyze} disabled={isLoading} className="w-full h-12">
            {isLoading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Fetching Sentiment...</>
              : <><Newspaper className="w-4 h-4 mr-2" />Analyze Sentiment</>}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="pt-4 flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{error}</span>
          </CardContent>
        </Card>
      )}

      {result && (
        <>
          {/* Provider badge */}
          {result.provider && (
            <p className="text-xs text-muted-foreground px-1">
              Source: <span className="text-primary font-medium">{result.provider}</span>
            </p>
          )}

          {/* Summary metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="border-accent/20">
              <CardContent className="pt-6 text-center">
                <p className="text-sm text-muted-foreground mb-1">Avg Sentiment Score</p>
                <p className={`text-3xl font-bold ${scoreColor(result.avg_score)}`}>
                  {result.avg_score > 0 ? "+" : ""}{result.avg_score.toFixed(3)}
                </p>
                <p className="text-xs text-muted-foreground mt-1">–1 bearish → +1 bullish</p>
              </CardContent>
            </Card>

            <Card className="border-accent/20">
              <CardContent className="pt-6 text-center">
                <p className="text-sm text-muted-foreground mb-1">VaR Stress Multiplier</p>
                <p className={`text-3xl font-bold ${multiplierColor(result.multiplier)}`}>
                  {result.multiplier.toFixed(3)}×
                </p>
                <p className="text-xs text-muted-foreground mt-1">Applied to drift μ in QTBN</p>
              </CardContent>
            </Card>

            <Card className="border-accent/20">
              <CardContent className="pt-6 text-center">
                <p className="text-sm text-muted-foreground mb-1">Headlines Processed</p>
                <p className="text-3xl font-bold text-primary">{result.total_items}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {result.tickers.join(", ")}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Headlines table */}
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Newspaper className="w-5 h-5 text-primary" />
                News Headlines & Scores
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
                {result.items.map((item: SentimentItem, i: number) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 p-3 rounded-lg border border-accent/10 hover:border-accent/30 transition-colors"
                  >
                    <Badge variant="outline" className="shrink-0 text-xs mt-0.5">
                      {item.ticker}
                    </Badge>
                    <div className="flex-1 min-w-0">
                      {item.link ? (
                        <a
                          href={item.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm hover:text-primary transition-colors line-clamp-2"
                        >
                          {item.title}
                        </a>
                      ) : (
                        <p className="text-sm line-clamp-2">{item.title}</p>
                      )}
                      {item.published && (
                        <p className="text-xs text-muted-foreground mt-0.5">{item.published}</p>
                      )}
                    </div>
                    <Badge variant="outline" className={`shrink-0 flex items-center gap-1 ${scoreBadgeClass(item.score)}`}>
                      <ScoreIcon score={item.score} />
                      {item.score > 0 ? "+" : ""}{item.score.toFixed(2)}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
};
