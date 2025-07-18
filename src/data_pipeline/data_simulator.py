"""
Advanced Customer Data Simulator with Quantum-Inspired Algorithms and Federated Learning
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Operator
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CustomerSegment:
    """Customer segment configuration with quantum-inspired parameters"""
    name: str
    size: int
    base_purchase_prob: float
    price_sensitivity: float
    promotion_response: float
    churn_rate: float
    quantum_entanglement: float  # Novel: quantum-inspired customer relationships
    network_influence: float     # Novel: social network influence
    temporal_volatility: float   # Novel: time-varying behavior patterns

class QuantumInspiredCustomerSimulator:
    """Novel: Quantum-inspired customer behavior simulation"""
    
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_quantum_circuit(self, customer_state: np.ndarray) -> QuantumCircuit:
        """Create quantum circuit for customer behavior simulation"""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        
        # Encode customer state into quantum circuit
        for i, bit in enumerate(customer_state[:self.n_qubits]):
            if bit:
                qc.x(i)
        
        # Apply quantum gates for behavior evolution
        qc.h(0)  # Hadamard gate for superposition
        qc.cx(0, 1)  # CNOT for entanglement
        qc.rz(np.pi/4, 2)  # Rotation for temporal dynamics
        qc.cx(1, 3)  # Additional entanglement
        
        qc.measure_all()
        return qc
    
    def simulate_quantum_behavior(self, customer_state: np.ndarray) -> Dict[str, float]:
        """Simulate quantum-inspired customer behavior"""
        qc = self.create_quantum_circuit(customer_state)
        backend = Aer.get_backend('qasm_simulator')
        job = backend.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Convert quantum results to behavioral probabilities
        total_shots = sum(counts.values())
        quantum_probs = {k: v/total_shots for k, v in counts.items()}
        
        return {
            'purchase_impulse': quantum_probs.get('0000', 0.1),
            'price_sensitivity': quantum_probs.get('0001', 0.2),
            'promotion_response': quantum_probs.get('0010', 0.3),
            'loyalty_factor': quantum_probs.get('0011', 0.4)
        }

class FederatedLearningSimulator:
    """Novel: Federated learning simulation for distributed customer insights"""
    
    def __init__(self, n_clients: int = 5):
        self.n_clients = n_clients
        self.client_models = {}
        self.global_model = None
        
    def initialize_client_models(self, feature_dim: int):
        """Initialize client-specific models"""
        for i in range(self.n_clients):
            self.client_models[f'client_{i}'] = {
                'weights': np.random.randn(feature_dim),
                'bias': np.random.randn(),
                'data_size': np.random.randint(100, 1000)
            }
    
    def federated_aggregation(self) -> Dict[str, np.ndarray]:
        """Simulate federated averaging"""
        total_size = sum(client['data_size'] for client in self.client_models.values())
        
        aggregated_weights = np.zeros_like(list(self.client_models.values())[0]['weights'])
        aggregated_bias = 0.0
        
        for client in self.client_models.values():
            weight = client['data_size'] / total_size
            aggregated_weights += weight * client['weights']
            aggregated_bias += weight * client['bias']
        
        return {
            'weights': aggregated_weights,
            'bias': aggregated_bias,
            'total_clients': self.n_clients
        }

class CustomerNetworkSimulator:
    """Novel: Graph-based customer network simulation"""
    
    def __init__(self, n_customers: int = 1000):
        self.n_customers = n_customers
        self.network = nx.Graph()
        
    def generate_customer_network(self, 
                                avg_degree: float = 8.0,
                                clustering_coeff: float = 0.3) -> nx.Graph:
        """Generate realistic customer social network"""
        # Use Watts-Strogatz model for small-world network
        self.network = nx.watts_strogatz_graph(
            self.n_customers, 
            int(avg_degree), 
            0.1
        )
        
        # Add customer attributes
        for node in self.network.nodes():
            self.network.nodes[node].update({
                'influence_score': np.random.beta(2, 5),
                'susceptibility': np.random.beta(3, 3),
                'segment': np.random.choice(['premium', 'regular', 'budget'])
            })
        
        return self.network
    
    def calculate_network_influence(self, customer_id: int) -> float:
        """Calculate network influence on customer behavior"""
        if customer_id not in self.network:
            return 0.0
        
        neighbors = list(self.network.neighbors(customer_id))
        if not neighbors:
            return 0.0
        
        # Calculate influence from neighbors
        neighbor_influence = sum(
            self.network.nodes[neighbor]['influence_score'] 
            for neighbor in neighbors
        )
        
        # Normalize by number of neighbors
        return neighbor_influence / len(neighbors)

class AdvancedCustomerDataSimulator:
    """
    Advanced Customer Data Simulator with Quantum-Inspired Algorithms,
    Federated Learning, and Graph-Based Networks
    """
    
    def __init__(self, 
                 n_customers: int = 10000,
                 n_days: int = 365,
                 seed: int = 42):
        self.n_customers = n_customers
        self.n_days = n_days
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize novel components
        self.quantum_simulator = QuantumInspiredCustomerSimulator()
        self.federated_simulator = FederatedLearningSimulator()
        self.network_simulator = CustomerNetworkSimulator(n_customers)
        
        # Customer segments with quantum-inspired parameters
        self.segments = {
            'premium': CustomerSegment(
                name='premium',
                size=int(n_customers * 0.15),
                base_purchase_prob=0.08,
                price_sensitivity=0.3,
                promotion_response=0.6,
                churn_rate=0.02,
                quantum_entanglement=0.8,
                network_influence=0.9,
                temporal_volatility=0.2
            ),
            'regular': CustomerSegment(
                name='regular',
                size=int(n_customers * 0.60),
                base_purchase_prob=0.05,
                price_sensitivity=0.6,
                promotion_response=0.4,
                churn_rate=0.05,
                quantum_entanglement=0.5,
                network_influence=0.6,
                temporal_volatility=0.4
            ),
            'budget': CustomerSegment(
                name='budget',
                size=int(n_customers * 0.25),
                base_purchase_prob=0.03,
                price_sensitivity=0.9,
                promotion_response=0.8,
                churn_rate=0.08,
                quantum_entanglement=0.3,
                network_influence=0.4,
                temporal_volatility=0.6
            )
        }
        
        logger.info(f"Initialized Advanced Customer Data Simulator with {n_customers} customers")
    
    def generate_customer_base(self) -> pd.DataFrame:
        """Generate customer base with advanced features"""
        customers = []
        
        for segment_name, segment in self.segments.items():
            segment_customers = []
            
            for i in range(segment.size):
                customer_id = len(customers) + i
                
                # Generate quantum-inspired customer state
                customer_state = np.random.rand(4)
                quantum_behavior = self.quantum_simulator.simulate_quantum_behavior(customer_state)
                
                customer = {
                    'customer_id': customer_id,
                    'segment': segment_name,
                    'age': np.random.normal(45, 15),
                    'income': np.random.lognormal(10.5, 0.5),
                    'location': np.random.choice(['urban', 'suburban', 'rural']),
                    'join_date': datetime.now() - timedelta(days=np.random.randint(30, 1000)),
                    'base_purchase_prob': segment.base_purchase_prob,
                    'price_sensitivity': segment.price_sensitivity,
                    'promotion_response': segment.promotion_response,
                    'churn_rate': segment.churn_rate,
                    'quantum_entanglement': segment.quantum_entanglement,
                    'network_influence': segment.network_influence,
                    'temporal_volatility': segment.temporal_volatility,
                    'quantum_purchase_impulse': quantum_behavior['purchase_impulse'],
                    'quantum_loyalty': quantum_behavior['loyalty_factor'],
                    'influence_score': np.random.beta(2, 5),
                    'susceptibility': np.random.beta(3, 3)
                }
                
                segment_customers.append(customer)
            
            customers.extend(segment_customers)
        
        return pd.DataFrame(customers)
    
    def generate_temporal_features(self, date: datetime) -> Dict[str, float]:
        """Generate temporal features with advanced patterns"""
        day_of_week = date.weekday()
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        # Seasonal patterns
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Weekly patterns
        weekly_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)
        
        # Holiday effects
        holiday_factor = 1.0
        if month == 12:  # December holidays
            holiday_factor = 1.5
        elif month == 11:  # Black Friday
            holiday_factor = 1.8
        
        # Economic cycle simulation
        economic_cycle = 1 + 0.1 * np.sin(2 * np.pi * day_of_year / (365 * 4))  # 4-year cycle
        
        return {
            'seasonal_factor': seasonal_factor,
            'weekly_factor': weekly_factor,
            'holiday_factor': holiday_factor,
            'economic_cycle': economic_cycle,
            'day_of_week': day_of_week,
            'month': month
        }
    
    def generate_promotion_events(self) -> List[Dict]:
        """Generate promotion events with quantum-inspired timing"""
        promotions = []
        current_date = datetime.now() - timedelta(days=self.n_days)
        
        for day in range(self.n_days):
            date = current_date + timedelta(days=day)
            
            # Quantum-inspired promotion probability
            quantum_state = np.random.rand(2)
            promotion_prob = 0.05 * (1 + 0.5 * np.sin(2 * np.pi * day / 30))
            
            if np.random.random() < promotion_prob:
                promotion = {
                    'date': date,
                    'promotion_id': f'promo_{day}',
                    'discount_rate': np.random.uniform(0.1, 0.4),
                    'duration_days': np.random.randint(1, 7),
                    'target_segment': np.random.choice(list(self.segments.keys())),
                    'quantum_effectiveness': np.random.beta(2, 3)
                }
                promotions.append(promotion)
        
        return promotions
    
    def simulate_customer_behavior(self, 
                                 customer: pd.Series, 
                                 date: datetime,
                                 temporal_features: Dict[str, float],
                                 active_promotions: List[Dict],
                                 network_influence: float = 0.0) -> Dict:
        """Simulate individual customer behavior with quantum and network effects"""
        
        # Base purchase probability
        base_prob = customer['base_purchase_prob']
        
        # Apply temporal effects
        temporal_multiplier = (
            temporal_features['seasonal_factor'] *
            temporal_features['weekly_factor'] *
            temporal_features['holiday_factor'] *
            temporal_features['economic_cycle']
        )
        
        # Apply quantum effects
        quantum_multiplier = (
            customer['quantum_purchase_impulse'] *
            customer['quantum_entanglement'] *
            customer['quantum_loyalty']
        )
        
        # Apply network influence
        network_multiplier = 1 + network_influence * customer['network_influence']
        
        # Apply promotion effects
        promotion_multiplier = 1.0
        for promo in active_promotions:
            if promo['target_segment'] == customer['segment']:
                promo_effect = (
                    promo['discount_rate'] *
                    customer['promotion_response'] *
                    promo['quantum_effectiveness']
                )
                promotion_multiplier += promo_effect
        
        # Calculate final purchase probability
        final_prob = base_prob * temporal_multiplier * quantum_multiplier * network_multiplier * promotion_multiplier
        
        # Simulate purchase decision
        purchase_made = np.random.random() < final_prob
        
        # Simulate purchase amount if purchase is made
        purchase_amount = 0
        if purchase_made:
            base_amount = np.random.lognormal(4.0, 0.5)  # $50-150 range
            segment_multiplier = {
                'premium': 2.0,
                'regular': 1.0,
                'budget': 0.6
            }[customer['segment']]
            
            purchase_amount = base_amount * segment_multiplier * temporal_multiplier
        
        # Simulate churn
        churn_prob = customer['churn_rate'] * (1 + customer['temporal_volatility'] * np.random.random())
        churned = np.random.random() < churn_prob
        
        return {
            'customer_id': customer['customer_id'],
            'date': date,
            'purchase_made': purchase_made,
            'purchase_amount': purchase_amount,
            'churned': churned,
            'temporal_multiplier': temporal_multiplier,
            'quantum_multiplier': quantum_multiplier,
            'network_multiplier': network_multiplier,
            'promotion_multiplier': promotion_multiplier,
            'final_probability': final_prob
        }
    
    def generate_interaction_data(self) -> pd.DataFrame:
        """Generate customer interaction data with quantum and network effects"""
        logger.info("Generating customer interaction data...")
        
        # Generate customer base
        customers_df = self.generate_customer_base()
        
        # Generate customer network
        customer_network = self.network_simulator.generate_customer_network()
        
        # Generate promotion events
        promotions = self.generate_promotion_events()
        
        # Initialize federated learning
        self.federated_simulator.initialize_client_models(feature_dim=10)
        
        interactions = []
        current_date = datetime.now() - timedelta(days=self.n_days)
        
        for day in range(self.n_days):
            date = current_date + timedelta(days=day)
            
            # Get temporal features
            temporal_features = self.generate_temporal_features(date)
            
            # Get active promotions
            active_promotions = [
                p for p in promotions 
                if p['date'] <= date <= p['date'] + timedelta(days=p['duration_days'])
            ]
            
            # Simulate behavior for each customer
            for _, customer in customers_df.iterrows():
                # Calculate network influence
                network_influence = self.network_simulator.calculate_network_influence(
                    customer['customer_id']
                )
                
                # Simulate behavior
                behavior = self.simulate_customer_behavior(
                    customer, date, temporal_features, active_promotions, network_influence
                )
                
                interactions.append(behavior)
            
            # Federated learning update (simulated)
            if day % 30 == 0:  # Monthly federated update
                federated_result = self.federated_simulator.federated_aggregation()
                logger.info(f"Federated learning update on day {day}: {federated_result['total_clients']} clients")
        
        return pd.DataFrame(interactions)
    
    def generate_customer_features(self) -> pd.DataFrame:
        """Generate customer features with quantum-inspired attributes"""
        customers_df = self.generate_customer_base()
        
        # Add derived features
        customers_df['age_group'] = pd.cut(
            customers_df['age'], 
            bins=[0, 25, 35, 50, 65, 100], 
            labels=['young', 'young_adult', 'adult', 'senior', 'elderly']
        )
        
        customers_df['income_group'] = pd.cut(
            customers_df['income'],
            bins=[0, 30000, 60000, 100000, 200000, float('inf')],
            labels=['low', 'medium', 'high', 'very_high', 'ultra_high']
        )
        
        # Quantum-inspired features
        customers_df['quantum_complexity'] = (
            customers_df['quantum_entanglement'] * 
            customers_df['temporal_volatility'] * 
            customers_df['network_influence']
        )
        
        customers_df['behavioral_entropy'] = (
            customers_df['price_sensitivity'] * 
            customers_df['promotion_response'] * 
            customers_df['churn_rate']
        )
        
        return customers_df
    
    def run_full_simulation(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Run complete simulation with all novel features"""
        logger.info("Starting full customer data simulation...")
        
        # Generate customer features
        customers_df = self.generate_customer_features()
        
        # Generate interaction data
        interactions_df = self.generate_interaction_data()
        
        # Generate summary statistics
        summary_stats = {
            'total_customers': len(customers_df),
            'total_interactions': len(interactions_df),
            'total_purchases': interactions_df['purchase_made'].sum(),
            'total_revenue': interactions_df['purchase_amount'].sum(),
            'avg_purchase_amount': interactions_df[interactions_df['purchase_made']]['purchase_amount'].mean(),
            'churn_rate': interactions_df['churned'].mean(),
            'quantum_entanglement_avg': customers_df['quantum_entanglement'].mean(),
            'network_influence_avg': customers_df['network_influence'].mean(),
            'federated_clients': self.federated_simulator.n_clients
        }
        
        logger.info(f"Simulation completed. Generated {len(interactions_df)} interactions")
        logger.info(f"Summary: {summary_stats}")
        
        return customers_df, interactions_df, summary_stats

def main():
    """Main function to run the advanced customer data simulation"""
    # Initialize simulator
    simulator = AdvancedCustomerDataSimulator(
        n_customers=5000,  # Reduced for faster execution
        n_days=90,         # 3 months of data
        seed=42
    )
    
    # Run simulation
    customers_df, interactions_df, summary_stats = simulator.run_full_simulation()
    
    # Save results
    customers_df.to_csv('data/simulated_customer_data.csv', index=False)
    interactions_df.to_csv('data/simulated_interaction_data.csv', index=False)
    
    print("Advanced Customer Data Simulation Completed!")
    print(f"Generated {len(customers_df)} customers and {len(interactions_df)} interactions")
    print(f"Total revenue: ${summary_stats['total_revenue']:,.2f}")
    print(f"Average purchase: ${summary_stats['avg_purchase_amount']:.2f}")

if __name__ == "__main__":
    main()
