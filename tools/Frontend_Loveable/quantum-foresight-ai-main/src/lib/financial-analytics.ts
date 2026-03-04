import { FinancialData, RiskMetrics } from "@/types/quantum";

export class FinancialAnalytics {
  
  // Generate synthetic financial data (fallback for demo)
  static generateSyntheticData(tickers: string[], days: number = 365): FinancialData {
    const dates = Array.from({ length: days }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (days - 1 - i));
      return date.toISOString().split('T')[0];
    });

    const prices: Record<string, number[]> = {};
    const returns: Record<string, number[]> = {};

    tickers.forEach(ticker => {
      const startPrice = 100 + Math.random() * 50;
      const drift = 0.0004 + Math.random() * 0.0006;
      const volatility = 0.012 + Math.random() * 0.006;
      
      const tickerPrices: number[] = [startPrice];
      const tickerReturns: number[] = [];

      for (let i = 1; i < days; i++) {
        const randomReturn = drift + volatility * this.normalRandom();
        const newPrice = tickerPrices[i - 1] * Math.exp(randomReturn);
        tickerPrices.push(newPrice);
        tickerReturns.push(randomReturn);
      }

      prices[ticker] = tickerPrices;
      returns[ticker] = tickerReturns;
    });

    return { tickers, prices, dates, returns };
  }

  // Box-Muller transform for normal random numbers
  private static normalRandom(): number {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); // Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // Calculate log returns from prices
  static calculateLogReturns(prices: number[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push(Math.log(prices[i] / prices[i - 1]));
    }
    return returns;
  }

  // Monte Carlo VaR/CVaR calculation
  static calculateVaRCVaR(
    returns: number[], 
    horizonDays: number = 10, 
    simulations: number = 50000, 
    alpha: number = 0.95
  ): { var: number; cvar: number } {
    if (returns.length === 0) return { var: NaN, cvar: NaN };

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    const std = Math.sqrt(variance);

    const portfolioReturns: number[] = [];
    
    for (let sim = 0; sim < simulations; sim++) {
      let portfolioReturn = 0;
      for (let day = 0; day < horizonDays; day++) {
        portfolioReturn += mean + std * this.normalRandom();
      }
      portfolioReturns.push(portfolioReturn);
    }

    portfolioReturns.sort((a, b) => a - b);
    const varIndex = Math.floor((1 - alpha) * simulations);
    const var95 = portfolioReturns[varIndex];
    
    const tailReturns = portfolioReturns.slice(0, varIndex + 1);
    const cvar = tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length;

    return { var: var95, cvar };
  }

  // Detect market regime based on volatility
  static detectRegime(returns: number[], threshold: number = 0.30): string {
    if (returns.length < 21) return "Insufficient Data";

    // Calculate rolling 21-day volatility
    const rollingVols: number[] = [];
    for (let i = 20; i < returns.length; i++) {
      const window = returns.slice(i - 20, i + 1);
      const mean = window.reduce((sum, r) => sum + r, 0) / window.length;
      const variance = window.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (window.length - 1);
      const annualizedVol = Math.sqrt(variance * 252);
      rollingVols.push(annualizedVol);
    }

    const currentVol = rollingVols[rollingVols.length - 1];
    
    if (currentVol > threshold * 1.25) return "High Volatility";
    if (currentVol > threshold * 0.75) return "Medium Volatility";
    return "Low Volatility";
  }

  // Calculate basic portfolio metrics
  static calculateBasicMetrics(returns: number[]): Partial<RiskMetrics> {
    if (returns.length === 0) return {};

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    const std = Math.sqrt(variance);
    
    // Annualized metrics
    const annualizedReturn = mean * 252;
    const annualizedVol = std * Math.sqrt(252);
    const riskFreeRate = 0.04; // 4% risk-free rate
    
    const sharpe = (annualizedReturn - riskFreeRate) / annualizedVol;

    // Maximum drawdown calculation
    const cumulativeReturns = returns.reduce((acc, r, i) => {
      acc.push(i === 0 ? Math.exp(r) : acc[i - 1] * Math.exp(r));
      return acc;
    }, [] as number[]);

    let maxDrawdown = 0;
    let peak = cumulativeReturns[0];
    
    for (let i = 1; i < cumulativeReturns.length; i++) {
      if (cumulativeReturns[i] > peak) {
        peak = cumulativeReturns[i];
      }
      const drawdown = (peak - cumulativeReturns[i]) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return {
      sharpe: isFinite(sharpe) ? sharpe : 0,
      maxDrawdown,
      volatility: annualizedVol
    };
  }

  // Calculate portfolio returns from multiple assets
  static calculatePortfolioReturns(financialData: FinancialData, weights?: number[]): number[] {
    const { tickers, returns } = financialData;
    const numAssets = tickers.length;
    const equalWeights = weights || new Array(numAssets).fill(1 / numAssets);
    
    if (returns[tickers[0]].length === 0) return [];
    
    const portfolioReturns: number[] = [];
    const periods = returns[tickers[0]].length;
    
    for (let i = 0; i < periods; i++) {
      let portfolioReturn = 0;
      for (let j = 0; j < numAssets; j++) {
        portfolioReturn += equalWeights[j] * returns[tickers[j]][i];
      }
      portfolioReturns.push(portfolioReturn);
    }
    
    return portfolioReturns;
  }

  // Generate comprehensive risk analysis
  static analyzeRisk(
    financialData: FinancialData, 
    horizonDays: number = 10, 
    confidence: number = 0.95
  ): RiskMetrics {
    const portfolioReturns = this.calculatePortfolioReturns(financialData);
    const { var: varValue, cvar } = this.calculateVaRCVaR(portfolioReturns, horizonDays, 50000, confidence);
    const basicMetrics = this.calculateBasicMetrics(portfolioReturns);
    const regime = this.detectRegime(portfolioReturns);
    
    return {
      var: varValue,
      cvar,
      regime,
      ...basicMetrics,
      volatility: basicMetrics.volatility || 0,
      sharpe: basicMetrics.sharpe || 0,
      maxDrawdown: basicMetrics.maxDrawdown || 0
    };
  }
}