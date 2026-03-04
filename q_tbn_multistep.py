from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

# Convert probability to Ry rotation angle
def ry_angle(prob):
    return 2 * np.arcsin(np.sqrt(prob))

# Controlled Ry gate for conditional probabilities
def apply_conditional_ry(qc, control, target, p_if_zero, p_if_one):
    qc.x(control)
    qc.cry(ry_angle(p_if_zero), control, target)
    qc.x(control)
    qc.cry(ry_angle(p_if_one), control, target)

# Build Q-TBN circuit for multiple time steps
def build_qtbn_circuit(T, priors, transitions):
    num_qubits = 2 * T  # One hidden + one observed per step
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Encode P(X_0)
    qc.ry(ry_angle(priors[0]), 0)

    # Encode P(Y_0 | X_0)
    apply_conditional_ry(qc, 0, 1, p_if_zero=priors[1][0], p_if_one=priors[1][1])

    # Loop through time steps
    for t in range(1, T):
        x_prev = 2 * (t - 1)
        x_curr = 2 * t
        y_curr = x_curr + 1

        # Encode P(X_t | X_{t-1})
        apply_conditional_ry(qc, x_prev, x_curr,
                             p_if_zero=transitions[t][0],
                             p_if_one=transitions[t][1])

        # Encode P(Y_t | X_t)
        apply_conditional_ry(qc, x_curr, y_curr,
                             p_if_zero=priors[1][0],
                             p_if_one=priors[1][1])

    qc.measure(range(num_qubits), range(num_qubits))
    return qc

# Simulation using modern AerSimulator
def simulate(circuit, shots=1024):
    backend = AerSimulator()
    transpiled = transpile(circuit, backend)
    result = backend.run(transpiled, shots=shots).result()
    return result.get_counts()

# Main execution
if __name__ == "__main__":
    T = 3  # Number of time steps
    priors = [0.6, [0.2, 0.8]]  # P(X0=1), P(Yt=1 | Xt=0), P(Yt=1 | Xt=1)
    transitions = {
        1: [0.3, 0.7],  # P(X1=1 | X0=0), P(X1=1 | X0=1)
        2: [0.4, 0.9]   # P(X2=1 | X1=0), P(X2=1 | X1=1)
    }

    qc = build_qtbn_circuit(T, priors, transitions)
    print(qc.draw())
    counts = simulate(qc)
    print("\nMeasurement Results:")
    for outcome, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(outcome, "→", count)
