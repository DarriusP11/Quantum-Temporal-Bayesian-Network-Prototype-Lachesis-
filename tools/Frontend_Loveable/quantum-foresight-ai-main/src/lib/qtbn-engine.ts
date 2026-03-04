import { QTBNConfig, QTBNState, FinancialGraph, MarketRegimeNode, QTBNInferenceResult } from "@/types/qtbn";
import { FinancialData } from "@/types/quantum";

export class QuantumTemporalBayesianNetwork {
  private config: QTBNConfig;
  private currentState: QTBNState;
  private graph: FinancialGraph;

  constructor(config: QTBNConfig) {
    this.config = config;
    this.currentState = this.initializeState();
    this.graph = this.buildFinancialGraph();
  }

  private initializeState(): QTBNState {
    return {
      timeStep: 0,
      hiddenStates: new Array(this.config.stateSpaceSize).fill(0),
      observations: [],
      beliefs: {},
      confidenceScore: 0.5
    };
  }

  private buildFinancialGraph(): FinancialGraph {
    const nodes: MarketRegimeNode[] = [
      {
        name: "MarketRegime",
        states: ["Bull", "Bear", "Sideways", "Volatile"],
        parents: [],
        priorProbabilities: [0.3, 0.2, 0.35, 0.15]
      },
      {
        name: "VolatilityLevel",
        states: ["Low", "Medium", "High", "Extreme"],
        parents: ["MarketRegime"],
        priorProbabilities: [0.4, 0.35, 0.2, 0.05],
        conditionalProbabilities: {
          "Bull": [0.5, 0.35, 0.12, 0.03],
          "Bear": [0.1, 0.25, 0.45, 0.2],
          "Sideways": [0.6, 0.3, 0.08, 0.02],
          "Volatile": [0.05, 0.15, 0.4, 0.4]
        }
      },
      {
        name: "SentimentScore",
        states: ["VeryBearish", "Bearish", "Neutral", "Bullish", "VeryBullish"],
        parents: ["MarketRegime"],
        priorProbabilities: [0.1, 0.2, 0.4, 0.2, 0.1],
        conditionalProbabilities: {
          "Bull": [0.02, 0.08, 0.25, 0.45, 0.2],
          "Bear": [0.25, 0.45, 0.25, 0.04, 0.01],
          "Sideways": [0.05, 0.15, 0.6, 0.15, 0.05],
          "Volatile": [0.15, 0.25, 0.4, 0.15, 0.05]
        }
      },
      {
        name: "RiskLevel",
        states: ["VeryLow", "Low", "Medium", "High", "VeryHigh"],
        parents: ["VolatilityLevel", "SentimentScore"],
        priorProbabilities: [0.15, 0.25, 0.3, 0.2, 0.1]
      }
    ];

    return {
      nodes,
      timeSteps: this.config.timeHorizon,
      currentBelief: this.initializeBeliefs(nodes)
    };
  }

  private initializeBeliefs(nodes: MarketRegimeNode[]): Record<string, number> {
    const beliefs: Record<string, number> = {};
    
    nodes.forEach(node => {
      node.states.forEach((state, index) => {
        const key = `${node.name}_${state}`;
        beliefs[key] = node.priorProbabilities[index] || 0.25;
      });
    });

    return beliefs;
  }

  // Quantum-enhanced posterior sampling using amplitude estimation
  private quantumPosteriorSampling(priors: number[], observations: number[]): number[] {
    // Educational quantum simulation - amplitude-based sampling
    const quantumAmplitudes = priors.map((p, i) => {
      const observationWeight = observations[i] || 1.0;
      return Math.sqrt(p * observationWeight);
    });

    // Normalize amplitudes
    const normalization = Math.sqrt(quantumAmplitudes.reduce((sum, amp) => sum + amp * amp, 0));
    const normalizedAmplitudes = quantumAmplitudes.map(amp => amp / normalization);

    // Convert back to probabilities (|amplitude|^2)
    return normalizedAmplitudes.map(amp => amp * amp);
  }

  // QAOA-inspired optimization for MAP estimation
  private quantumMAPEstimation(beliefs: Record<string, number>): string[] {
    // Simulate quantum approximate optimization
    const stateBeliefs = Object.entries(beliefs);
    
    // Apply quantum-inspired mixing and cost operators
    const optimizedBeliefs = stateBeliefs.map(([state, belief]) => {
      // Quantum mixing operator (rotation in probability space)
      const mixingAngle = Math.PI / 4;
      const mixedBelief = belief * Math.cos(mixingAngle) + (1 - belief) * Math.sin(mixingAngle);
      
      // Cost operator (favors higher probability states)
      const costFactor = Math.exp(mixedBelief);
      
      return [state, mixedBelief * costFactor];
    });

    // Return top states (MAP sequence)
    return optimizedBeliefs
      .sort(([, a], [, b]) => (b as number) - (a as number))
      .slice(0, 3)
      .map(([state]) => state as string);
  }

  // Hybrid inference pipeline as described in the research
  public performInference(financialData: FinancialData, observations: number[]): QTBNInferenceResult {
    const results: QTBNInferenceResult = {
      posteriorBeliefs: {},
      maxAPosterior: [],
      confidence: 0,
      temporalPredictions: []
    };

    // Step 1: Encode priors
    const priors = Object.values(this.graph.currentBelief);

    // Step 2: Quantum-enhanced posterior estimation
    const posteriorProbs = this.quantumPosteriorSampling(priors, observations);
    
    // Step 3: Update beliefs using Bayesian rule
    const updatedBeliefs: Record<string, number> = {};
    const beliefKeys = Object.keys(this.graph.currentBelief);
    
    beliefKeys.forEach((key, index) => {
      updatedBeliefs[key] = posteriorProbs[index] || this.graph.currentBelief[key];
    });

    // Step 4: Quantum MAP estimation
    const mapSequence = this.quantumMAPEstimation(updatedBeliefs);

    // Step 5: Temporal predictions across horizon
    const predictions = this.generateTemporalPredictions(updatedBeliefs, financialData);

    // Calculate confidence based on entropy of beliefs
    const entropy = Object.values(updatedBeliefs).reduce((h, p) => {
      return h - (p > 0 ? p * Math.log2(p) : 0);
    }, 0);
    const maxEntropy = Math.log2(Object.keys(updatedBeliefs).length);
    const confidence = 1 - (entropy / maxEntropy);

    results.posteriorBeliefs = updatedBeliefs;
    results.maxAPosterior = mapSequence;
    results.confidence = confidence;
    results.temporalPredictions = predictions;

    // Update internal state
    this.graph.currentBelief = updatedBeliefs;
    this.currentState.timeStep += 1;
    this.currentState.beliefs = updatedBeliefs;
    this.currentState.confidenceScore = confidence;

    return results;
  }

  private generateTemporalPredictions(beliefs: Record<string, number>, financialData: FinancialData) {
    const predictions = [];
    
    for (let t = 1; t <= Math.min(this.config.timeHorizon, 10); t++) {
      // Temporal transition using first-order Markov assumption
      const futureBeliefs: Record<string, number> = {};
      
      Object.entries(beliefs).forEach(([state, prob]) => {
        // Apply temporal decay and regime persistence
        const persistence = 0.8; // Regime persistence factor
        const noise = 0.1 * Math.random(); // Market noise
        futureBeliefs[state] = prob * persistence + noise;
      });

      // Normalize future beliefs
      const total = Object.values(futureBeliefs).reduce((sum, p) => sum + p, 0);
      Object.keys(futureBeliefs).forEach(key => {
        futureBeliefs[key] /= total;
      });

      // Determine most likely regime
      const topRegime = Object.entries(futureBeliefs)
        .filter(([key]) => key.includes('MarketRegime'))
        .sort(([, a], [, b]) => b - a)[0];

      predictions.push({
        timeStep: t,
        predictions: futureBeliefs,
        regime: topRegime ? topRegime[0].split('_')[1] : 'Unknown'
      });
    }

    return predictions;
  }

  // Market regime detection using Q-TBN
  public detectMarketRegime(financialData: FinancialData): {
    currentRegime: string;
    confidence: number;
    regimeProbabilities: Record<string, number>;
    transitions: Array<{ from: string; to: string; probability: number }>;
  } {
    // Extract market features for observation
    const observations = this.extractMarketFeatures(financialData);
    
    // Perform Q-TBN inference
    const inference = this.performInference(financialData, observations);
    
    // Extract regime probabilities
    const regimeBeliefs = Object.entries(inference.posteriorBeliefs)
      .filter(([key]) => key.includes('MarketRegime'))
      .reduce((acc, [key, prob]) => {
        const regime = key.split('_')[1];
        acc[regime] = prob;
        return acc;
      }, {} as Record<string, number>);

    // Determine current regime
    const currentRegime = Object.entries(regimeBeliefs)
      .sort(([, a], [, b]) => b - a)[0][0];

    // Calculate transition probabilities
    const transitions = this.calculateRegimeTransitions(regimeBeliefs);

    return {
      currentRegime,
      confidence: inference.confidence,
      regimeProbabilities: regimeBeliefs,
      transitions
    };
  }

  private extractMarketFeatures(financialData: FinancialData): number[] {
    // Convert financial data to observation vector for Q-TBN
    const features: number[] = [];
    
    if (financialData.returns && Object.keys(financialData.returns).length > 0) {
      const allReturns = Object.values(financialData.returns).flat();
      
      // Volatility feature
      const volatility = Math.sqrt(allReturns.reduce((sum, r) => sum + r * r, 0) / allReturns.length);
      features.push(volatility);
      
      // Momentum feature
      const momentum = allReturns.slice(-10).reduce((sum, r) => sum + r, 0) / 10;
      features.push(momentum + 0.5); // Normalize to positive
      
      // Trend consistency
      const trendChanges = allReturns.slice(1).reduce((count, r, i) => {
        return count + (Math.sign(r) !== Math.sign(allReturns[i]) ? 1 : 0);
      }, 0);
      const consistency = 1 - (trendChanges / Math.max(allReturns.length - 1, 1));
      features.push(consistency);
      
      // Mean reversion indicator
      const meanReturn = allReturns.reduce((sum, r) => sum + r, 0) / allReturns.length;
      const reversion = Math.exp(-Math.abs(momentum - meanReturn));
      features.push(reversion);
    } else {
      // Default features if no data
      features.push(0.15, 0.5, 0.7, 0.6);
    }

    return features;
  }

  private calculateRegimeTransitions(regimeBeliefs: Record<string, number>) {
    const regimes = Object.keys(regimeBeliefs);
    const transitions = [];

    for (const from of regimes) {
      for (const to of regimes) {
        if (from !== to) {
          // Simplified transition probability based on regime similarity
          const fromProb = regimeBeliefs[from];
          const toProb = regimeBeliefs[to];
          const similarity = this.calculateRegimeSimilarity(from, to);
          const transitionProb = fromProb * similarity * 0.1; // Base transition rate
          
          transitions.push({ from, to, probability: transitionProb });
        }
      }
    }

    return transitions.sort((a, b) => b.probability - a.probability).slice(0, 6);
  }

  private calculateRegimeSimilarity(regime1: string, regime2: string): number {
    // Regime similarity matrix based on financial theory
    const similarities: Record<string, Record<string, number>> = {
      "Bull": { "Sideways": 0.6, "Bear": 0.1, "Volatile": 0.3 },
      "Bear": { "Bull": 0.1, "Sideways": 0.4, "Volatile": 0.5 },
      "Sideways": { "Bull": 0.6, "Bear": 0.4, "Volatile": 0.3 },
      "Volatile": { "Bull": 0.3, "Bear": 0.5, "Sideways": 0.3 }
    };

    return similarities[regime1]?.[regime2] || 0.1;
  }

  // Get current QTBN state for visualization
  public getCurrentState(): QTBNState {
    return { ...this.currentState };
  }

  // Reset QTBN to initial state
  public reset(): void {
    this.currentState = this.initializeState();
    this.graph.currentBelief = this.initializeBeliefs(this.graph.nodes);
  }
}