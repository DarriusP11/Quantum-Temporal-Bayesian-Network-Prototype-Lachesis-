export interface QTBNState {
  timeStep: number;
  hiddenStates: number[];
  observations: number[];
  beliefs: Record<string, number>;
  confidenceScore: number;
}

export interface QTBNConfig {
  timeHorizon: number;
  stateSpaceSize: number;
  observationSpace: string[];
  transitionModel: 'markov' | 'dynamic' | 'hidden_markov';
  quantumBackend: 'simulator' | 'qiskit' | 'pennylane';
  inferenceMethod: 'qaoa' | 'grover' | 'amplitude_estimation';
}

export interface MarketRegimeNode {
  name: string;
  states: string[];
  parents: string[];
  priorProbabilities: number[];
  conditionalProbabilities?: Record<string, number[]>;
}

export interface FinancialGraph {
  nodes: MarketRegimeNode[];
  timeSteps: number;
  currentBelief: Record<string, number>;
}

export interface QTBNInferenceResult {
  posteriorBeliefs: Record<string, number>;
  maxAPosterior: string[];
  confidence: number;
  temporalPredictions: Array<{
    timeStep: number;
    predictions: Record<string, number>;
    regime: string;
  }>;
}