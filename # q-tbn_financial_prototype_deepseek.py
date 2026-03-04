# q-tbn_financial_prototype.py
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import Aer 
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter
from qiskit.algorithms import AmplitudeEstimation
from qiskit_finance.applications import PortfolioOptimization
import pandas as pd
from datetime import datetime, timedelta

# =====================
# QUANTUM CORE ENGINE
# =====================

class QuantumTBN:
    def __init__(self, time_steps=3, risk_prior=0.2, volatility_prior=0.1):
        self.T = time_steps
        self.risk_prior = risk_prior
        self.volatility_prior = volatility_prior
        self.simulator = Aer.get_backend('aer_simulator')
        self.transition_history = []
        self.belief_history = []
        
        # Default financial transition models
        self.risk_transitions = {
            0: [0.85, 0.15],  # Low-risk persistence
            1: [0.25, 0.75]   # High-risk persistence
        }
        
        self.volatility_model = {
            0: [0.9, 0.1],   # When risk=0 (low)
            1: [0.1, 0.9]     # When risk=1 (high)
        }
        
        self.position_model = {
            0: [0.8, 0.2],   # When volatility=0 (low)
            1: [0.3, 0.7]    # When volatility=1 (high)
        }
    
    def encode_prior(self, qc, target, probability):
        """Encode prior probability using RY rotation"""
        theta = 2 * np.arccos(np.sqrt(1 - probability))
        qc.ry(theta, target)
    
    def conditional_gate(self, qc, control, target, probs_0, probs_1):
        """Apply controlled rotation for conditional probabilities"""
        # For state when control=0
        theta_0 = 2 * np.arccos(np.sqrt(probs_0[0]))
        qc.cry(theta_0, control, target)
        
        # For state when control=1
        qc.x(control)
        theta_1 = 2 * np.arccos(np.sqrt(probs_1[0]))
        qc.cry(theta_1, control, target)
        qc.x(control)
    
    def build_circuit(self, evidence=None):
        """Construct quantum circuit for T time steps"""
        # 3 variables per time step: Risk, Volatility, Position
        num_qubits = 3 * self.T
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # Encode initial priors
        self.encode_prior(qc, 0, self.risk_prior)          # Risk_0
        self.conditional_gate(qc, 0, 1, 
                             self.volatility_model[0],
                             self.volatility_model[1])      # Volatility_0
        self.conditional_gate(qc, 1, 2, 
                             self.position_model[0],
                             self.position_model[1])        # Position_0
        
        # Apply transitions for subsequent time steps
        for t in range(1, self.T):
            prev_risk = 3*(t-1)
            curr_risk = 3*t
            curr_vol = 3*t + 1
            curr_pos = 3*t + 2
            
            # Risk_t depends on Risk_{t-1}
            self.conditional_gate(qc, prev_risk, curr_risk,
                                 self.risk_transitions[0],
                                 self.risk_transitions[1])
            
            # Volatility_t depends on Risk_t
            self.conditional_gate(qc, curr_risk, curr_vol,
                                 self.volatility_model[0],
                                 self.volatility_model[1])
            
            # Position_t depends on Volatility_t
            self.conditional_gate(qc, curr_vol, curr_pos,
                                 self.position_model[0],
                                 self.position_model[1])
            
            # Apply evidence if provided
            if evidence and t in evidence:
                if 'risk' in evidence[t]:
                    qc.reset(curr_risk)
                    if evidence[t]['risk'] == 1:
                        qc.x(curr_risk)
                if 'volatility' in evidence[t]:
                    qc.reset(curr_vol)
                    if evidence[t]['volatility'] == 1:
                        qc.x(curr_vol)
        
        qc.measure(range(num_qubits), range(num_qubits))
        return qc
    
    def simulate(self, qc, shots=2048):
        """Execute quantum circuit"""
        transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled, shots=shots)
        return job.result().get_counts()
    
    def compute_marginals(self, counts):
        """Calculate marginal probabilities for all variables"""
        marginals = {'risk': [], 'volatility': [], 'position': []}
        total = sum(counts.values())
        
        for t in range(self.T):
            risk_idx = 3*t
            vol_idx = 3*t + 1
            pos_idx = 3*t + 2
            
            risk_count = vol_count = pos_count = 0
            
            for bitstring, freq in counts.items():
                reversed_str = bitstring[::-1]  # Qiskit uses little-endian
                if reversed_str[risk_idx] == '1':
                    risk_count += freq
                if reversed_str[vol_idx] == '1':
                    vol_count += freq
                if reversed_str[pos_idx] == '1':
                    pos_count += freq
            
            marginals['risk'].append(risk_count / total)
            marginals['volatility'].append(vol_count / total)
            marginals['position'].append(pos_count / total)
        
        self.belief_history.append(marginals)
        return marginals
    
    def hybrid_inference(self, evidence=None, shots=2048):
        """Perform hybrid quantum-classical inference"""
        qc = self.build_circuit(evidence)
        counts = self.simulate(qc, shots)
        return self.compute_marginals(counts)
    
    def quantum_amplitude_estimation(self, state_index, precision=3):
        """Use amplitude estimation for more precise probabilities"""
        num_qubits = 3 * self.T
        qc = self.build_circuit()
        qc.remove_final_measurements()
        
        # Define estimation problem
        ae = AmplitudeEstimation(num_eval_qubits=precision)
        result = ae.estimate(quantum_circuit=qc, 
                             state_preparation=None,
                             objective_qubits=[state_index])
        return result.estimation
    
    def update_model_from_data(self, new_data):
        """Adapt transition models based on new market data"""
        # Placeholder for actual machine learning logic
        print(f"Updating model with new data: {len(new_data)} records")
        
        # Simple adjustment for demonstration
        avg_volatility = np.mean(new_data['volatility'])
        if avg_volatility > 0.6:
            self.risk_transitions[0] = [0.75, 0.25]
            self.risk_transitions[1] = [0.15, 0.85]
            print("Transition model updated for high volatility regime")
        elif avg_volatility < 0.3:
            self.risk_transitions[0] = [0.9, 0.1]
            self.risk_transitions[1] = [0.35, 0.65]
            print("Transition model updated for low volatility regime")
        
        self.transition_history.append({
            'timestamp': datetime.now(),
            'risk_transitions': dict(self.risk_transitions),
            'volatility_model': dict(self.volatility_model)
        })
    
    def detect_regime_shift(self, beliefs, threshold=0.7):
        """Identify significant market regime changes"""
        risk_diff = np.diff(beliefs['risk'])
        vol_diff = np.diff(beliefs['volatility'])
        
        max_risk_change = np.max(np.abs(risk_diff))
        max_vol_change = np.max(np.abs(vol_diff))
        
        if max_risk_change > threshold or max_vol_change > threshold:
            shift_time = np.argmax(np.abs(risk_diff)) + 1
            return True, shift_time
        return False, None

# =====================
# FINANCIAL APPLICATION
# =====================

class FinancialForecaster:
    def __init__(self, tbn_engine):
        self.engine = tbn_engine
        self.portfolio = {
            'stocks': 100000,
            'bonds': 50000,
            'cash': 20000
        }
        self.history = []
    
    def generate_market_scenario(self, days=30):
        """Generate synthetic market data for demonstration"""
        dates = [datetime.now() + timedelta(days=i) for i in range(days)]
        volatility = np.clip(np.cumsum(np.random.normal(0, 0.1, days)), 0, 1)
        risk = 1 / (1 + np.exp(-np.cumsum(np.random.normal(0, 0.2, days))))
        
        return pd.DataFrame({
            'date': dates,
            'volatility': volatility,
            'risk': risk,
            'sp500': np.cumprod(1 + np.random.normal(0, volatility/10))
        })
    
    def portfolio_optimization(self, risk_belief, volatility_belief):
        """Quantum-enhanced portfolio optimization"""
        # Simplified portfolio optimization
        stock_allocation = 0.7 - 0.4 * risk_belief - 0.2 * volatility_belief
        bond_allocation = 0.2 + 0.3 * risk_belief - 0.1 * volatility_belief
        cash_allocation = 0.1 + 0.1 * risk_belief + 0.3 * volatility_belief
        
        # Normalize
        total = stock_allocation + bond_allocation + cash_allocation
        stock_allocation /= total
        bond_allocation /= total
        cash_allocation /= total
        
        return {
            'stocks': stock_allocation,
            'bonds': bond_allocation,
            'cash': cash_allocation
        }
    
    def trade_recommendation(self, beliefs):
        """Generate trading recommendations based on beliefs"""
        recommendations = []
        
        for t in range(self.engine.T):
            risk = beliefs['risk'][t]
            vol = beliefs['volatility'][t]
            
            if risk > 0.7:
                rec = {
                    'time': t,
                    'action': 'SELL',
                    'asset': 'high-risk stocks',
                    'confidence': risk,
                    'reason': 'Extreme risk probability'
                }
            elif risk < 0.3 and vol < 0.4:
                rec = {
                    'time': t,
                    'action': 'BUY',
                    'asset': 'leveraged ETFs',
                    'confidence': 1 - risk,
                    'reason': 'Low risk with moderate volatility'
                }
            elif vol > 0.6:
                rec = {
                    'time': t,
                    'action': 'HEDGE',
                    'asset': 'VIX futures',
                    'confidence': vol,
                    'reason': 'High volatility expected'
                }
            else:
                rec = {
                    'time': t,
                    'action': 'HOLD',
                    'asset': 'current positions',
                    'confidence': (1 - risk) * (1 - vol),
                    'reason': 'Moderate market conditions'
                }
            
            recommendations.append(rec)
        
        return recommendations
    
    def run_forecast(self, days=5, evidence=None):
        """Run full forecasting pipeline"""
        print(f"\n=== RUNNING FINANCIAL FORECAST FOR {days} DAYS ===")
        
        # Generate synthetic market data
        market_data = self.generate_market_scenario(days)
        print("\nGenerated market data:")
        print(market_data[['date', 'risk', 'volatility']].head())
        
        # Update model with recent data
        self.engine.update_model_from_data(market_data)
        
        # Run quantum inference
        beliefs = self.engine.hybrid_inference(evidence)
        
        # Generate recommendations
        recommendations = self.trade_recommendation(beliefs)
        
        # Portfolio optimization
        current_risk = beliefs['risk'][-1]
        current_vol = beliefs['volatility'][-1]
        allocation = self.portfolio_optimization(current_risk, current_vol)
        
        # Save results
        result = {
            'timestamp': datetime.now(),
            'horizon_days': days,
            'beliefs': beliefs,
            'recommendations': recommendations,
            'allocation': allocation,
            'evidence': evidence
        }
        
        self.history.append(result)
        return result

# =====================
# VISUALIZATION & REPORTING
# =====================

def plot_forecast_results(results):
    """Visualize forecasting results"""
    beliefs = results['beliefs']
    
    plt.figure(figsize=(14, 10))
    
    # Plot risk beliefs
    plt.subplot(3, 1, 1)
    plt.plot(beliefs['risk'], 'ro-', linewidth=2)
    plt.title('Risk Probability Forecast')
    plt.ylabel('P(Risk=High)')
    plt.grid(True)
    plt.ylim(0, 1)
    
    # Plot volatility beliefs
    plt.subplot(3, 1, 2)
    plt.plot(beliefs['volatility'], 'bo-', linewidth=2)
    plt.title('Volatility Probability Forecast')
    plt.ylabel('P(Volatility=High)')
    plt.grid(True)
    plt.ylim(0, 1)
    
    # Plot position beliefs
    plt.subplot(3, 1, 3)
    plt.plot(beliefs['position'], 'go-', linewidth=2)
    plt.title('Position Probability Forecast')
    plt.ylabel('P(Position=Aggressive)')
    plt.xlabel('Time Steps (Days)')
    plt.grid(True)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('forecast_plot.png')
    plt.show()
    
    # Plot portfolio allocation
    allocation = results['allocation']
    labels = list(allocation.keys())
    sizes = [allocation[k] for k in labels]
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Recommended Portfolio Allocation')
    plt.savefig('portfolio_allocation.png')
    plt.show()

def generate_report(results):
    """Generate textual report of forecasting results"""
    report = f"Quantum Financial Forecast Report\n"
    report += f"Generated at: {datetime.now()}\n"
    report += f"Forecast Horizon: {results['horizon_days']} days\n\n"
    
    # Belief summary
    beliefs = results['beliefs']
    report += "BELIEF PROBABILITIES:\n"
    report += "Time\tRisk\tVolatility\tPosition\n"
    for t in range(len(beliefs['risk'])):
        report += f"{t}\t{beliefs['risk'][t]:.2f}\t{beliefs['volatility'][t]:.2f}\t\t{beliefs['position'][t]:.2f}\n"
    
    # Recommendations
    report += "\nTRADE RECOMMENDATIONS:\n"
    for rec in results['recommendations']:
        report += (f"Day {rec['time']}: {rec['action']} {rec['asset']} "
                  f"(Confidence: {rec['confidence']:.2f})\n")
        report += f"Reason: {rec['reason']}\n\n"
    
    # Portfolio allocation
    allocation = results['allocation']
    report += "OPTIMAL PORTFOLIO ALLOCATION:\n"
    for asset, percent in allocation.items():
        report += f"- {asset.upper()}: {percent*100:.1f}%\n"
    
    # Save to file
    with open('forecast_report.txt', 'w') as f:
        f.write(report)
    
    return report

# =====================
# MAIN EXECUTION
# =====================

if __name__ == "__main__":
    print("Starting Quantum Temporal Bayesian Network for Financial Foresight")
    
    # Initialize quantum engine
    qtbn = QuantumTBN(time_steps=5, risk_prior=0.25, volatility_prior=0.15)
    
    # Initialize financial forecaster
    forecaster = FinancialForecaster(qtbn)
    
    # Add evidence (e.g., from real-time market feeds)
    evidence = {
        0: {'risk': 0},    # Day 0: Low risk observed
        2: {'volatility': 1}  # Day 2: High volatility observed
    }
    
    # Run forecasting pipeline
    results = forecaster.run_forecast(days=5, evidence=evidence)
    
    # Visualize results
    plot_forecast_results(results)
    
    # Generate report
    report = generate_report(results)
    print("\n=== FORECAST REPORT ===")
    print(report)
    
    print("\nForecast complete. Results saved to:")
    print("- forecast_plot.png")
    print("- portfolio_allocation.png")
    print("- forecast_report.txt")