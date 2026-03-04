export interface QuantumConfig {
  numQubits: number;
  shots: number;
  seed: number;
  
  // Noise parameters
  enableDepolarizing: boolean;
  enableAmplitudeDamping: boolean;
  enablePhaseDamping: boolean;
  enableCNOTNoise: boolean;
  
  depolarizingProbs: [number, number, number];
  amplitudeDampingProbs: [number, number, number];
  phaseDampingProbs: [number, number, number];
  cnotNoiseProbs: [number, number, number];
  
  // Gate configurations
  gates: {
    step0: { q0: GateConfig; q1: GateConfig; cnot: boolean };
    step1: { q0: GateConfig; q1: GateConfig; cnot: boolean };
    step2: { q0: GateConfig; q1: GateConfig; cnot: boolean };
  };
}

export interface GateConfig {
  type: GateType;
  angle: number; // in π units
}

export type GateType = 'None' | 'H' | 'X' | 'Y' | 'Z' | 'RX' | 'RY' | 'RZ' | 'S' | 'T';

export interface QuantumResult {
  statevector?: Complex[];
  counts?: Record<string, number>;
  fidelity?: number;
  probabilities?: Record<string, number>;
}

export interface Complex {
  real: number;
  imag: number;
}

export interface FinancialData {
  tickers: string[];
  prices: Record<string, number[]>;
  dates: string[];
  returns: Record<string, number[]>;
}

export interface RiskMetrics {
  var: number;
  cvar: number;
  sharpe: number;
  maxDrawdown: number;
  regime: string;
  volatility: number;
}

export interface SentimentData {
  score: number;
  confidence: number;
  emotions: {
    fear: number;
    optimism: number;
    uncertainty: number;
  };
  topics: string[];
  alerts: Array<{
    type: string;
    zscore: number;
    direction: 'up' | 'down';
  }>;
}