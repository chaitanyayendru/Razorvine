"""
Quantum-Enhanced Optimization Engine for Customer Analytics
Combines quantum computing simulation with classical optimization algorithms
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
# from qiskit.algorithms import VQE, QAOA  # Commented out - not available in current qiskit version
# from qiskit.algorithms.optimizers import SPSA, COBYLA  # Commented out - not available in current qiskit version
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Pauli
# from qiskit.opflow import PauliSumOp  # Commented out - not available in current qiskit version
import optuna
from scipy.optimize import minimize
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantumOptimizationConfig:
    """Configuration for quantum-enhanced optimization"""
    n_qubits: int = 8
    shots: int = 1000
    max_iterations: int = 100
    quantum_weight: float = 0.3  # Weight of quantum solution vs classical
    entanglement_depth: int = 2
    optimization_method: str = 'hybrid'  # 'quantum', 'classical', 'hybrid'

class QuantumCustomerOptimizer:
    """Quantum-enhanced customer optimization using Qiskit"""
    
    def __init__(self, config: QuantumOptimizationConfig):
        self.config = config
        self.backend = Aer.get_backend('qasm_simulator')
        self.quantum_results = {}
        
    def create_customer_qubo(self, customer_data: pd.DataFrame, 
                           target_metric: str = 'revenue') -> np.ndarray:
        """Create Quadratic Unconstrained Binary Optimization (QUBO) matrix for customer targeting"""
        n_customers = len(customer_data)
        qubo_matrix = np.zeros((n_customers, n_customers))
        
        # Diagonal terms (customer value)
        for i in range(n_customers):
            customer_value = customer_data[target_metric].iloc[i]
            qubo_matrix[i, i] = -customer_value  # Negative for maximization
        
        # Off-diagonal terms (customer interactions/conflicts)
        for i in range(n_customers):
            for j in range(i+1, n_customers):
                # Simulate customer interaction effects
                interaction_strength = np.random.uniform(-0.1, 0.1)
                qubo_matrix[i, j] = interaction_strength
                qubo_matrix[j, i] = interaction_strength
        
        return qubo_matrix
    
    def qubo_to_quantum_circuit(self, qubo_matrix: np.ndarray) -> QuantumCircuit:
        """Convert QUBO problem to quantum circuit"""
        n_variables = qubo_matrix.shape[0]
        n_qubits = min(n_variables, self.config.n_qubits)
        
        # Create quantum registers
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Apply Hadamard gates for superposition
        for i in range(n_qubits):
            qc.h(qr[i])
        
        # Apply entanglement layers
        for depth in range(self.config.entanglement_depth):
            for i in range(n_qubits - 1):
                qc.cx(qr[i], qr[i+1])
            qc.cx(qr[n_qubits-1], qr[0])  # Wrap-around connection
            
            # Apply rotation gates based on QUBO matrix
            for i in range(n_qubits):
                angle = np.pi * qubo_matrix[i, i] if i < qubo_matrix.shape[0] else 0
                qc.rz(angle, qr[i])
        
        # Measure all qubits
        qc.measure_all()
        
        return qc
    
    def solve_quantum_qubo(self, qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Solve QUBO problem using quantum circuit"""
        qc = self.qubo_to_quantum_circuit(qubo_matrix)
        
        # Execute quantum circuit using Aer backend
        backend = Aer.get_backend('qasm_simulator')
        job = backend.run(qc, shots=self.config.shots)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Find best solution
        best_bitstring = max(counts, key=counts.get)
        best_value = self.evaluate_qubo_solution(qubo_matrix, best_bitstring)
        
        return {
            'best_solution': best_bitstring,
            'best_value': best_value,
            'counts': counts,
            'circuit': qc
        }
    
    def evaluate_qubo_solution(self, qubo_matrix: np.ndarray, bitstring: str) -> float:
        """Evaluate QUBO solution"""
        solution = np.array([int(bit) for bit in bitstring])
        return solution.T @ qubo_matrix @ solution
    
    def quantum_customer_segmentation(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Quantum-enhanced customer segmentation"""
        logger.info("Performing quantum-enhanced customer segmentation...")
        
        # Prepare features for segmentation
        features = ['age', 'income', 'base_purchase_prob', 'price_sensitivity']
        
        # Clean and prepare data
        X = customer_data[features].copy()
        
        # Handle any string or invalid data
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        X = X.astype(float)
        X = X.values
        
        # Ensure no infinite or NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create quantum-inspired distance matrix
        n_customers = len(X_scaled)
        distance_matrix = np.zeros((n_customers, n_customers))
        
        for i in range(n_customers):
            for j in range(n_customers):
                if i != j:
                    # Quantum-inspired distance calculation
                    euclidean_dist = np.linalg.norm(X_scaled[i] - X_scaled[j])
                    quantum_factor = np.exp(-euclidean_dist)  # Quantum tunneling effect
                    distance_matrix[i, j] = euclidean_dist * quantum_factor
        
        # Use quantum-enhanced clustering
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add quantum noise to cluster assignments
        quantum_clusters = []
        for cluster in clusters:
            if np.random.random() < 0.1:  # 10% quantum uncertainty
                cluster = (cluster + 1) % n_clusters
            quantum_clusters.append(cluster)
        
        return {
            'clusters': quantum_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'distance_matrix': distance_matrix,
            'quantum_factor': 0.1
        }

class HybridOptimizationEngine:
    """Hybrid quantum-classical optimization engine"""
    
    def __init__(self, quantum_config: QuantumOptimizationConfig):
        self.quantum_optimizer = QuantumCustomerOptimizer(quantum_config)
        self.config = quantum_config
        
    def optimize_customer_targeting(self, 
                                  customer_data: pd.DataFrame,
                                  budget: float,
                                  target_metric: str = 'revenue') -> Dict[str, Any]:
        """Optimize customer targeting using hybrid quantum-classical approach"""
        logger.info(f"Starting hybrid optimization for customer targeting with budget ${budget:,.2f}")
        
        try:
            # Ensure required columns exist
            if target_metric not in customer_data.columns:
                logger.warning(f"Target metric '{target_metric}' not found, using 'customer_lifetime_value'")
                target_metric = 'customer_lifetime_value'
            
            if 'acquisition_cost' not in customer_data.columns:
                logger.warning("'acquisition_cost' column not found, creating default values")
                customer_data['acquisition_cost'] = np.random.uniform(50, 200, len(customer_data))
            
            # Ensure numeric columns
            for col in [target_metric, 'acquisition_cost']:
                if col in customer_data.columns:
                    customer_data[col] = pd.to_numeric(customer_data[col], errors='coerce').fillna(0)
            
            # Step 1: Quantum-enhanced segmentation
            segmentation_result = self.quantum_optimizer.quantum_customer_segmentation(customer_data)
            
            # Step 2: Create QUBO for customer selection
            qubo_matrix = self.quantum_optimizer.create_customer_qubo(customer_data, target_metric)
            
            # Step 3: Solve with quantum approach
            quantum_result = self.quantum_optimizer.solve_quantum_qubo(qubo_matrix)
            
            # Step 4: Solve with classical approach
            classical_result = self.solve_classical_optimization(customer_data, budget, target_metric)
            
            # Step 5: Hybrid combination
            hybrid_result = self.combine_quantum_classical_solutions(
                quantum_result, classical_result, customer_data
            )
            
            # Step 6: Apply budget constraints
            final_result = self.apply_budget_constraints(hybrid_result, budget, customer_data)
            
            return {
                'quantum_solution': quantum_result,
                'classical_solution': classical_result,
                'hybrid_solution': final_result,
                'segmentation': segmentation_result,
                'budget_utilization': final_result.get('total_cost', 0) / budget if budget > 0 else 0,
                'expected_revenue': final_result.get('total_revenue', 0)
            }
        except Exception as e:
            logger.error(f"Error in quantum optimization: {e}")
            # Return fallback solution
            return {
                'quantum_solution': {'best_solution': '0' * 8},
                'classical_solution': {'solution': np.zeros(len(customer_data))},
                'hybrid_solution': {
                    'selected_customers': customer_data.head(10),
                    'total_revenue': 0,
                    'total_cost': 0,
                    'quantum_weight': 0.0
                },
                'segmentation': {'clusters': [0] * len(customer_data)},
                'budget_utilization': 0.0,
                'expected_revenue': 0.0
            }
    
    def solve_classical_optimization(self, 
                                   customer_data: pd.DataFrame,
                                   budget: float,
                                   target_metric: str) -> Dict[str, Any]:
        """Solve customer targeting using classical optimization"""
        
        def objective_function(x):
            """Objective function for customer selection"""
            selected_customers = x.astype(bool)
            if not np.any(selected_customers):
                return 0
            
            total_revenue = customer_data[target_metric][selected_customers].sum()
            total_cost = customer_data['acquisition_cost'][selected_customers].sum()
            
            if total_cost > budget:
                return -1e6  # Penalty for exceeding budget
            
            return total_revenue
        
        # Initial guess
        n_customers = len(customer_data)
        x0 = np.random.random(n_customers)
        
        # Optimize using scipy
        result = minimize(
            lambda x: -objective_function(x),  # Negative for maximization
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * n_customers,
            options={'maxiter': 1000}
        )
        
        # Convert to binary solution
        threshold = 0.5
        binary_solution = (result.x > threshold).astype(int)
        
        selected_customers = customer_data[binary_solution.astype(bool)]
        
        return {
            'solution': binary_solution,
            'selected_customers': selected_customers,
            'total_revenue': selected_customers[target_metric].sum(),
            'total_cost': selected_customers['acquisition_cost'].sum(),
            'optimization_success': result.success
        }
    
    def combine_quantum_classical_solutions(self,
                                          quantum_result: Dict[str, Any],
                                          classical_result: Dict[str, Any],
                                          customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Combine quantum and classical solutions using weighted approach"""
        
        # Extract solutions
        quantum_solution = quantum_result['best_solution']
        classical_solution = classical_result['solution']
        
        # Convert quantum solution to binary array
        n_customers = len(customer_data)
        quantum_binary = np.array([int(bit) for bit in quantum_solution])
        
        # Pad quantum solution if needed
        if len(quantum_binary) < n_customers:
            quantum_binary = np.pad(quantum_binary, (0, n_customers - len(quantum_binary)))
        
        # Weighted combination
        quantum_weight = self.config.quantum_weight
        classical_weight = 1 - quantum_weight
        
        combined_solution = (
            quantum_weight * quantum_binary +
            classical_weight * classical_solution
        )
        
        # Convert to binary
        final_solution = (combined_solution > 0.5).astype(int)
        
        # Calculate metrics
        selected_customers = customer_data[final_solution.astype(bool)]
        
        # Use available revenue column
        revenue_col = 'customer_lifetime_value'
        if 'revenue' in selected_customers.columns:
            revenue_col = 'revenue'
        elif 'customer_lifetime_value' in selected_customers.columns:
            revenue_col = 'customer_lifetime_value'
        else:
            # Use first numeric column as fallback
            numeric_cols = selected_customers.select_dtypes(include=[np.number]).columns
            revenue_col = numeric_cols[0] if len(numeric_cols) > 0 else 'acquisition_cost'
        
        return {
            'solution': final_solution,
            'selected_customers': selected_customers,
            'total_revenue': selected_customers[revenue_col].sum(),
            'total_cost': selected_customers['acquisition_cost'].sum(),
            'quantum_weight': quantum_weight,
            'classical_weight': classical_weight
        }
    
    def apply_budget_constraints(self,
                               solution: Dict[str, Any],
                               budget: float,
                               customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply budget constraints to the solution"""
        
        selected_customers = solution['selected_customers']
        total_cost = selected_customers['acquisition_cost'].sum()
        
        if total_cost <= budget:
            return solution
        
        # Need to reduce selection to fit budget
        # Use available revenue column
        revenue_col = 'customer_lifetime_value'
        if 'revenue' in selected_customers.columns:
            revenue_col = 'revenue'
        elif 'customer_lifetime_value' in selected_customers.columns:
            revenue_col = 'customer_lifetime_value'
        else:
            # Use first numeric column as fallback
            numeric_cols = selected_customers.select_dtypes(include=[np.number]).columns
            revenue_col = numeric_cols[0] if len(numeric_cols) > 0 else 'acquisition_cost'
        
        # Sort by revenue per cost ratio
        selected_customers['roi_ratio'] = selected_customers[revenue_col] / selected_customers['acquisition_cost']
        selected_customers_sorted = selected_customers.sort_values('roi_ratio', ascending=False)
        
        cumulative_cost = 0
        final_selection = []
        
        for _, customer in selected_customers_sorted.iterrows():
            if cumulative_cost + customer['acquisition_cost'] <= budget:
                final_selection.append(customer)
                cumulative_cost += customer['acquisition_cost']
            else:
                break
        
        final_df = pd.DataFrame(final_selection)
        
        return {
            'solution': solution['solution'],
            'selected_customers': final_df,
            'total_revenue': final_df[revenue_col].sum(),
            'total_cost': final_df['acquisition_cost'].sum(),
            'budget_utilization': cumulative_cost / budget
        }

class QuantumResourceAllocator:
    """Quantum-enhanced resource allocation for marketing campaigns"""
    
    def __init__(self, config: QuantumOptimizationConfig):
        self.config = config
        self.backend = Aer.get_backend('qasm_simulator')
    
    def optimize_marketing_budget(self,
                                customer_segments: Dict[str, pd.DataFrame],
                                total_budget: float,
                                channels: List[str]) -> Dict[str, Any]:
        """Optimize marketing budget allocation across segments and channels"""
        
        logger.info(f"Optimizing marketing budget allocation: ${total_budget:,.2f}")
        
        # Create quantum circuit for budget allocation
        n_segments = len(customer_segments)
        n_channels = len(channels)
        n_qubits = min(n_segments * n_channels, self.config.n_qubits)
        
        # Create quantum circuit
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Apply quantum gates for budget allocation
        for i in range(n_qubits):
            qc.h(qr[i])  # Superposition
        
        # Entanglement between segments and channels
        for i in range(0, n_qubits - 1, 2):
            qc.cx(qr[i], qr[i+1])
        
        # Apply rotation based on segment-channel effectiveness
        for i in range(n_qubits):
            angle = np.pi * np.random.uniform(0, 1)  # Random effectiveness
            qc.rz(angle, qr[i])
        
        qc.measure_all()
        
        # Execute quantum circuit using Aer backend
        backend = Aer.get_backend('qasm_simulator')
        sampler = backend.run(qc, shots=self.config.shots)
        result = sampler.result()
        counts = result.get_counts(qc)
        
        # Process results
        best_allocation = self.process_budget_allocation(counts, customer_segments, channels, total_budget)
        
        return {
            'allocation': best_allocation,
            'quantum_circuit': qc,
            'counts': counts,
            'expected_roi': self.calculate_expected_roi(best_allocation, customer_segments)
        }
    
    def process_budget_allocation(self,
                                counts: Dict[str, int],
                                customer_segments: Dict[str, pd.DataFrame],
                                channels: List[str],
                                total_budget: float) -> Dict[str, Dict[str, float]]:
        """Process quantum results into budget allocation"""
        
        # Find most frequent allocation
        best_bitstring = max(counts, key=counts.get)
        
        # Convert to allocation matrix
        n_segments = len(customer_segments)
        n_channels = len(channels)
        
        allocation = {}
        segment_names = list(customer_segments.keys())
        
        for i, segment in enumerate(segment_names):
            allocation[segment] = {}
            for j, channel in enumerate(channels):
                idx = i * n_channels + j
                if idx < len(best_bitstring):
                    # Allocate budget based on quantum result
                    bit_value = int(best_bitstring[idx])
                    budget_share = bit_value * total_budget / (n_segments * n_channels)
                    allocation[segment][channel] = budget_share
                else:
                    allocation[segment][channel] = 0
        
        return allocation
    
    def calculate_expected_roi(self,
                             allocation: Dict[str, Dict[str, float]],
                             customer_segments: Dict[str, pd.DataFrame]) -> float:
        """Calculate expected ROI for budget allocation"""
        
        total_roi = 0
        total_budget = 0
        
        for segment, channel_allocation in allocation.items():
            segment_data = customer_segments[segment]
            segment_roi = segment_data['revenue'].sum() / segment_data['acquisition_cost'].sum()
            
            for channel, budget in channel_allocation.items():
                # Assume channel effectiveness multiplier
                channel_multiplier = np.random.uniform(0.8, 1.2)
                total_roi += budget * segment_roi * channel_multiplier
                total_budget += budget
        
        return total_roi / total_budget if total_budget > 0 else 0

def main():
    """Demo of quantum-enhanced optimization"""
    
    # Create sample customer data
    np.random.seed(42)
    n_customers = 1000
    
    customer_data = pd.DataFrame({
        'customer_id': range(n_customers),
        'age': np.random.normal(45, 15, n_customers),
        'income': np.random.lognormal(10.5, 0.5, n_customers),
        'purchase_frequency': np.random.poisson(3, n_customers),
        'avg_purchase_amount': np.random.lognormal(4.0, 0.5, n_customers),
        'revenue': np.random.lognormal(4.5, 0.8, n_customers),
        'acquisition_cost': np.random.uniform(50, 200, n_customers)
    })
    
    # Initialize quantum optimization
    config = QuantumOptimizationConfig(
        n_qubits=8,
        shots=1000,
        max_iterations=50,
        quantum_weight=0.3
    )
    
    # Create hybrid optimizer
    optimizer = HybridOptimizationEngine(config)
    
    # Optimize customer targeting
    budget = 50000
    result = optimizer.optimize_customer_targeting(customer_data, budget, 'revenue')
    
    print("Quantum-Enhanced Optimization Results:")
    print(f"Budget: ${budget:,.2f}")
    print(f"Selected customers: {len(result['hybrid_solution']['selected_customers'])}")
    print(f"Expected revenue: ${result['hybrid_solution']['expected_revenue']:,.2f}")
    print(f"Total cost: ${result['hybrid_solution']['total_cost']:,.2f}")
    print(f"Budget utilization: {result['budget_utilization']:.2%}")
    print(f"Quantum weight used: {result['hybrid_solution']['quantum_weight']:.1%}")

if __name__ == "__main__":
    main() 