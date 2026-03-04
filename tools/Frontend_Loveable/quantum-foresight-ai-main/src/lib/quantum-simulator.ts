import { QuantumConfig, QuantumResult, Complex, GateType } from "@/types/quantum";

export class QuantumSimulator {
  private config: QuantumConfig;

  constructor(config: QuantumConfig) {
    this.config = config;
  }

  // Simplified quantum state simulation (educational approximation)
  simulateStatevector(): Complex[] {
    const { numQubits } = this.config;
    const dim = Math.pow(2, numQubits);
    let state = new Array(dim).fill(0).map((_, i) => ({ real: i === 0 ? 1 : 0, imag: 0 }));

    // Apply gates step by step
    state = this.applyGateStep(state, this.config.gates.step0);
    state = this.applyGateStep(state, this.config.gates.step1);
    state = this.applyGateStep(state, this.config.gates.step2);

    // Apply noise if enabled
    if (this.hasNoise()) {
      state = this.applyNoise(state);
    }

    return state;
  }

  simulateCounts(): Record<string, number> {
    const statevector = this.simulateStatevector();
    const probabilities = statevector.map(c => c.real * c.real + c.imag * c.imag);
    const counts: Record<string, number> = {};
    
    // Monte Carlo sampling
    for (let shot = 0; shot < this.config.shots; shot++) {
      const rand = Math.random();
      let cumProb = 0;
      for (let i = 0; i < probabilities.length; i++) {
        cumProb += probabilities[i];
        if (rand < cumProb) {
          const bitString = i.toString(2).padStart(this.config.numQubits, '0');
          counts[bitString] = (counts[bitString] || 0) + 1;
          break;
        }
      }
    }
    
    return counts;
  }

  calculateFidelity(): number {
    // Simplified fidelity calculation
    const idealState = this.simulateIdealStatevector();
    const noisyState = this.simulateStatevector();
    
    let fidelity = 0;
    for (let i = 0; i < idealState.length; i++) {
      const ideal = idealState[i];
      const noisy = noisyState[i];
      fidelity += (ideal.real * noisy.real + ideal.imag * noisy.imag);
    }
    
    return Math.abs(fidelity);
  }

  private simulateIdealStatevector(): Complex[] {
    const originalConfig = { ...this.config };
    // Temporarily disable noise
    this.config.enableDepolarizing = false;
    this.config.enableAmplitudeDamping = false;
    this.config.enablePhaseDamping = false;
    this.config.enableCNOTNoise = false;
    
    const result = this.simulateStatevector();
    
    // Restore original config
    this.config = originalConfig;
    return result;
  }

  private applyGateStep(state: Complex[], gateStep: any): Complex[] {
    let newState = [...state];
    
    // Apply single qubit gates
    if (gateStep.q0.type !== 'None') {
      newState = this.applySingleQubitGate(newState, 0, gateStep.q0.type, gateStep.q0.angle);
    }
    
    if (this.config.numQubits > 1 && gateStep.q1.type !== 'None') {
      newState = this.applySingleQubitGate(newState, 1, gateStep.q1.type, gateStep.q1.angle);
    }
    
    // Apply CNOT if specified
    if (gateStep.cnot && this.config.numQubits > 1) {
      newState = this.applyCNOT(newState);
    }
    
    return newState;
  }

  private applySingleQubitGate(state: Complex[], qubit: number, gate: GateType, angle: number): Complex[] {
    const newState = [...state];
    const theta = angle * Math.PI;
    
    for (let i = 0; i < state.length; i++) {
      const bit = (i >> qubit) & 1;
      const j = i ^ (1 << qubit); // Flip the target qubit
      
      if (bit === 0) { // Only process |0⟩ states to avoid double processing
        const amp0 = state[i];
        const amp1 = state[j];
        
        switch (gate) {
          case 'H':
            newState[i] = { 
              real: (amp0.real + amp1.real) / Math.sqrt(2), 
              imag: (amp0.imag + amp1.imag) / Math.sqrt(2) 
            };
            newState[j] = { 
              real: (amp0.real - amp1.real) / Math.sqrt(2), 
              imag: (amp0.imag - amp1.imag) / Math.sqrt(2) 
            };
            break;
          case 'X':
            newState[i] = amp1;
            newState[j] = amp0;
            break;
          case 'Y':
            newState[i] = { real: -amp1.imag, imag: amp1.real };
            newState[j] = { real: amp0.imag, imag: -amp0.real };
            break;
          case 'Z':
            newState[j] = { real: -amp1.real, imag: -amp1.imag };
            break;
          case 'RX':
            const cosHalf = Math.cos(theta / 2);
            const sinHalf = Math.sin(theta / 2);
            newState[i] = {
              real: cosHalf * amp0.real + sinHalf * amp1.imag,
              imag: cosHalf * amp0.imag - sinHalf * amp1.real
            };
            newState[j] = {
              real: cosHalf * amp1.real + sinHalf * amp0.imag,
              imag: cosHalf * amp1.imag - sinHalf * amp0.real
            };
            break;
          case 'RY':
            const cosHalfY = Math.cos(theta / 2);
            const sinHalfY = Math.sin(theta / 2);
            newState[i] = {
              real: cosHalfY * amp0.real - sinHalfY * amp1.real,
              imag: cosHalfY * amp0.imag - sinHalfY * amp1.imag
            };
            newState[j] = {
              real: sinHalfY * amp0.real + cosHalfY * amp1.real,
              imag: sinHalfY * amp0.imag + cosHalfY * amp1.imag
            };
            break;
          case 'RZ':
            const expNeg = { real: Math.cos(-theta/2), imag: Math.sin(-theta/2) };
            const expPos = { real: Math.cos(theta/2), imag: Math.sin(theta/2) };
            newState[i] = {
              real: amp0.real * expNeg.real - amp0.imag * expNeg.imag,
              imag: amp0.real * expNeg.imag + amp0.imag * expNeg.real
            };
            newState[j] = {
              real: amp1.real * expPos.real - amp1.imag * expPos.imag,
              imag: amp1.real * expPos.imag + amp1.imag * expPos.real
            };
            break;
        }
      }
    }
    
    return newState;
  }

  private applyCNOT(state: Complex[]): Complex[] {
    const newState = [...state];
    
    for (let i = 0; i < state.length; i++) {
      const controlBit = (i >> 0) & 1; // qubit 0 is control
      const targetBit = (i >> 1) & 1;  // qubit 1 is target
      
      if (controlBit === 1) {
        const j = i ^ (1 << 1); // Flip target qubit
        newState[i] = state[j];
        newState[j] = state[i];
      }
    }
    
    return newState;
  }

  private hasNoise(): boolean {
    return this.config.enableDepolarizing || 
           this.config.enableAmplitudeDamping || 
           this.config.enablePhaseDamping || 
           this.config.enableCNOTNoise;
  }

  private applyNoise(state: Complex[]): Complex[] {
    // Simplified noise model - adds random phase and amplitude errors
    return state.map(amp => {
      let { real, imag } = amp;
      
      if (this.config.enableDepolarizing) {
        const prob = this.config.depolarizingProbs.reduce((a, b) => a + b, 0) / 3;
        if (Math.random() < prob) {
          // Random Pauli error
          const error = Math.floor(Math.random() * 3);
          if (error === 0) { real = -real; imag = -imag; } // Z error
          else if (error === 1) { [real, imag] = [imag, -real]; } // Y error
          else { [real, imag] = [real, imag]; } // X error handled in bit flip
        }
      }
      
      if (this.config.enableAmplitudeDamping) {
        const prob = this.config.amplitudeDampingProbs.reduce((a, b) => a + b, 0) / 3;
        const factor = Math.sqrt(1 - prob);
        real *= factor;
        imag *= factor;
      }
      
      if (this.config.enablePhaseDamping) {
        const prob = this.config.phaseDampingProbs.reduce((a, b) => a + b, 0) / 3;
        const phase = prob * Math.random() * 2 * Math.PI;
        const cosPhase = Math.cos(phase);
        const sinPhase = Math.sin(phase);
        [real, imag] = [real * cosPhase - imag * sinPhase, real * sinPhase + imag * cosPhase];
      }
      
      return { real, imag };
    });
  }

  generateCircuitVisualization(): string[] {
    const lines: string[] = [];
    const { numQubits } = this.config;
    
    lines.push("Quantum Circuit Visualization:");
    lines.push("═══════════════════════════════");
    
    for (let q = 0; q < numQubits; q++) {
      let line = `q${q}: |0⟩──`;
      
      // Step 0
      const step0Gate = q === 0 ? this.config.gates.step0.q0 : this.config.gates.step0.q1;
      if (step0Gate.type !== 'None') {
        line += `[${step0Gate.type}]──`;
      } else {
        line += "──────";
      }
      
      if (this.config.gates.step0.cnot && numQubits > 1) {
        if (q === 0) line += "●──";
        else line += "⊕──";
      } else {
        line += "───";
      }
      
      // Step 1
      const step1Gate = q === 0 ? this.config.gates.step1.q0 : this.config.gates.step1.q1;
      if (step1Gate.type !== 'None') {
        line += `[${step1Gate.type}]──`;
      } else {
        line += "──────";
      }
      
      if (this.config.gates.step1.cnot && numQubits > 1) {
        if (q === 0) line += "●──";
        else line += "⊕──";
      } else {
        line += "───";
      }
      
      // Step 2
      const step2Gate = q === 0 ? this.config.gates.step2.q0 : this.config.gates.step2.q1;
      if (step2Gate.type !== 'None') {
        line += `[${step2Gate.type}]──`;
      } else {
        line += "──────";
      }
      
      if (this.config.gates.step2.cnot && numQubits > 1) {
        if (q === 0) line += "●";
        else line += "⊕";
      } else {
        line += "──";
      }
      
      lines.push(line);
    }
    
    return lines;
  }
}

export const QUANTUM_SCENARIOS: Record<string, Partial<QuantumConfig>> = {
  Bell: {
    numQubits: 2,
    shots: 2048,
    seed: 17,
    enableDepolarizing: true,
    enableAmplitudeDamping: false,
    enablePhaseDamping: false,
    enableCNOTNoise: true,
    depolarizingProbs: [0.01, 0.02, 0.02],
    amplitudeDampingProbs: [0.0, 0.0, 0.0],
    phaseDampingProbs: [0.0, 0.0, 0.0],
    cnotNoiseProbs: [0.02, 0.02, 0.02],
    gates: {
      step0: { 
        q0: { type: 'H' as GateType, angle: 0.5 }, 
        q1: { type: 'None' as GateType, angle: 0.0 }, 
        cnot: true 
      },
      step1: { 
        q0: { type: 'None' as GateType, angle: 0.0 }, 
        q1: { type: 'None' as GateType, angle: 0.0 }, 
        cnot: false 
      },
      step2: { 
        q0: { type: 'None' as GateType, angle: 0.0 }, 
        q1: { type: 'None' as GateType, angle: 0.0 }, 
        cnot: false 
      }
    }
  },
  "Dephase-1q": {
    numQubits: 1,
    shots: 2048,
    seed: 17,
    enableDepolarizing: false,
    enableAmplitudeDamping: false,
    enablePhaseDamping: true,
    enableCNOTNoise: false,
    depolarizingProbs: [0.0, 0.0, 0.0],
    amplitudeDampingProbs: [0.0, 0.0, 0.0],
    phaseDampingProbs: [0.04, 0.05, 0.05],
    cnotNoiseProbs: [0.0, 0.0, 0.0],
    gates: {
      step0: { 
        q0: { type: 'H' as GateType, angle: 0.5 }, 
        q1: { type: 'None' as GateType, angle: 0.0 }, 
        cnot: false 
      },
      step1: { 
        q0: { type: 'None' as GateType, angle: 0.0 }, 
        q1: { type: 'None' as GateType, angle: 0.0 }, 
        cnot: false 
      },
      step2: { 
        q0: { type: 'None' as GateType, angle: 0.0 }, 
        q1: { type: 'None' as GateType, angle: 0.0 }, 
        cnot: false 
      }
    }
  },
  "Damp-1q": {
    numQubits: 1,
    shots: 2048,
    seed: 17,
    enableDepolarizing: false,
    enableAmplitudeDamping: true,
    enablePhaseDamping: false,
    enableCNOTNoise: false,
    depolarizingProbs: [0.0, 0.0, 0.0],
    amplitudeDampingProbs: [0.0, 0.20, 0.20],
    phaseDampingProbs: [0.0, 0.0, 0.0],
    cnotNoiseProbs: [0.0, 0.0, 0.0],
    gates: {
      step0: { 
        q0: { type: 'X' as GateType, angle: 0.0 }, 
        q1: { type: 'None' as GateType, angle: 0.0 }, 
        cnot: false 
      },
      step1: { 
        q0: { type: 'None' as GateType, angle: 0.0 }, 
        q1: { type: 'None' as GateType, angle: 0.0 }, 
        cnot: false 
      },
      step2: { 
        q0: { type: 'None' as GateType, angle: 0.0 }, 
        q1: { type: 'None' as GateType, angle: 0.0 }, 
        cnot: false 
      }
    }
  }
};