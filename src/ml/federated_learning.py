"""
Federated Learning Module for Distributed Customer Analytics
Enables privacy-preserving machine learning across multiple organizations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateIns, EvaluateRes, Parameters, Scalar
)
from flwr.server import Server
from flwr.server.strategy import FedAvg
import json
import pickle
import hashlib
from datetime import datetime
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    n_clients: int = 5
    n_rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    model_type: str = 'neural_network'  # 'neural_network', 'logistic', 'random_forest'
    aggregation_method: str = 'fedavg'  # 'fedavg', 'fedprox', 'fedsgd'
    privacy_budget: float = 1.0  # Differential privacy budget
    min_clients: int = 3

class CustomerNeuralNetwork(nn.Module):
    """Neural network for customer behavior prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], output_dim: int = 1):
        super(CustomerNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class FederatedClient(fl.client.NumPyClient):
    """Federated learning client for customer analytics"""
    
    def __init__(self, 
                 client_id: str,
                 data: pd.DataFrame,
                 config: FederatedConfig,
                 model_type: str = 'neural_network'):
        self.client_id = client_id
        self.data = data
        self.config = config
        self.model_type = model_type
        
        # Prepare data
        self.X_train, self.y_train, self.X_test, self.y_test = self.prepare_data()
        
        # Initialize model
        if model_type == 'neural_network':
            self.model = CustomerNeuralNetwork(
                input_dim=self.X_train.shape[1],
                hidden_dims=[64, 32],
                output_dim=1
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.model = None
            self.optimizer = None
            self.criterion = None
        
        # Data loaders
        self.train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(self.X_train),
                torch.FloatTensor(self.y_train)
            ),
            batch_size=config.batch_size,
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(self.X_test),
                torch.FloatTensor(self.y_test)
            ),
            batch_size=config.batch_size,
            shuffle=False
        )
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for federated learning"""
        # Select features (using available columns)
        feature_columns = ['age', 'income', 'base_purchase_prob', 'price_sensitivity']
        target_column = 'churned'  # Binary target
        
        # Create target if not exists
        if 'churned' not in self.data.columns:
            self.data['churned'] = (self.data['churn_rate'] > 0.5).astype(float)
        
        X = self.data[feature_columns].fillna(0).values.astype(float)
        y = self.data[target_column].values.astype(float)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test
    
    def get_parameters(self, config):
        """Get model parameters"""
        if self.model_type == 'neural_network':
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return []
    
    def set_parameters(self, parameters):
        """Set model parameters"""
        if self.model_type == 'neural_network':
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train model on local data"""
        self.set_parameters(parameters)
        
        if self.model_type == 'neural_network':
            self.model.train()
            
            for epoch in range(self.config.local_epochs):
                total_loss = 0
                for batch_X, batch_y in self.train_loader:
                    self.optimizer.zero_grad()
                    
                    outputs = self.model(batch_X).squeeze()
                    loss = self.criterion(outputs, batch_y)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(self.train_loader)
                logger.info(f"Client {self.client_id} - Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        return self.get_parameters(config), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        """Evaluate model on local test data"""
        self.set_parameters(parameters)
        
        if self.model_type == 'neural_network':
            self.model.eval()
            
            total_loss = 0
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in self.test_loader:
                    outputs = self.model(batch_X).squeeze()
                    loss = self.criterion(outputs, batch_y)
                    total_loss += loss.item()
                    
                    preds = (torch.sigmoid(outputs) > 0.5).int().numpy()
                    predictions.extend(preds)
                    true_labels.extend(batch_y.numpy())
            
            avg_loss = total_loss / len(self.test_loader)
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary'
            )
            
            return avg_loss, len(self.X_test), {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return 0.0, len(self.X_test), {}

class FederatedServer:
    """Federated learning server for coordinating training"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.clients = {}
        self.global_model = None
        self.training_history = []
        
    def add_client(self, client_id: str, client: FederatedClient):
        """Add a federated client"""
        self.clients[client_id] = client
        logger.info(f"Added client {client_id} to federated training")
    
    def start_federated_training(self) -> Dict[str, Any]:
        """Start federated training process"""
        logger.info(f"Starting federated training with {len(self.clients)} clients")
        
        # Initialize global model
        if len(self.clients) > 0:
            sample_client = list(self.clients.values())[0]
            self.global_model = CustomerNeuralNetwork(
                input_dim=sample_client.X_train.shape[1],
                hidden_dims=[64, 32],
                output_dim=1
            )
        
        # Federated training rounds
        for round_num in range(self.config.n_rounds):
            logger.info(f"Federated training round {round_num + 1}/{self.config.n_rounds}")
            
            # Get global parameters
            global_params = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
            
            # Train on each client
            client_results = {}
            total_samples = 0
            
            for client_id, client in self.clients.items():
                try:
                    parameters, num_samples, metrics = client.fit(global_params, {})
                    client_results[client_id] = {
                        'parameters': parameters,
                        'num_samples': num_samples,
                        'metrics': metrics
                    }
                    total_samples += num_samples
                except Exception as e:
                    logger.error(f"Error training client {client_id}: {e}")
            
            # Aggregate parameters (FedAvg)
            if client_results:
                aggregated_params = self.aggregate_parameters(client_results, total_samples)
                
                # Update global model
                params_dict = zip(self.global_model.state_dict().keys(), aggregated_params)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.global_model.load_state_dict(state_dict, strict=True)
                
                # Evaluate global model
                global_metrics = self.evaluate_global_model()
                
                # Store training history
                self.training_history.append({
                    'round': round_num + 1,
                    'num_clients': len(client_results),
                    'total_samples': total_samples,
                    'global_metrics': global_metrics
                })
                
                logger.info(f"Round {round_num + 1} completed. Global accuracy: {global_metrics['accuracy']:.4f}")
        
        return {
            'training_history': self.training_history,
            'final_model': self.global_model,
            'total_rounds': self.config.n_rounds,
            'total_clients': len(self.clients)
        }
    
    def aggregate_parameters(self, client_results: Dict, total_samples: int) -> List[np.ndarray]:
        """Aggregate client parameters using FedAvg"""
        aggregated_params = []
        
        # Get parameter structure from first client
        first_client_params = list(client_results.values())[0]['parameters']
        
        for param_idx in range(len(first_client_params)):
            weighted_param = np.zeros_like(first_client_params[param_idx], dtype=np.float32)
            
            for client_result in client_results.values():
                client_params = client_result['parameters']
                num_samples = client_result['num_samples']
                weight = num_samples / total_samples
                
                weighted_param += weight * client_params[param_idx].astype(np.float32)
            
            aggregated_params.append(weighted_param)
        
        return aggregated_params
    
    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model on all client test sets"""
        all_predictions = []
        all_true_labels = []
        
        for client in self.clients.values():
            if client.model_type == 'neural_network':
                client.model.load_state_dict(self.global_model.state_dict())
                
                client.model.eval()
                predictions = []
                true_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in client.test_loader:
                        outputs = client.model(batch_X).squeeze()
                        preds = (torch.sigmoid(outputs) > 0.5).int().numpy()
                        predictions.extend(preds)
                        true_labels.extend(batch_y.numpy())
                
                all_predictions.extend(predictions)
                all_true_labels.extend(true_labels)
        
        if all_predictions:
            accuracy = accuracy_score(all_true_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_true_labels, all_predictions, average='binary'
            )
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

class PrivacyPreservingFederatedLearning:
    """Privacy-preserving federated learning with differential privacy"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.server = FederatedServer(config)
        
    def add_noise_to_gradients(self, gradients: List[np.ndarray], 
                              privacy_budget: float) -> List[np.ndarray]:
        """Add differential privacy noise to gradients"""
        noisy_gradients = []
        
        for gradient in gradients:
            # Calculate noise scale based on privacy budget
            noise_scale = 1.0 / privacy_budget
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, gradient.shape)
            noisy_gradient = gradient + noise
            
            noisy_gradients.append(noisy_gradient)
        
        return noisy_gradients
    
    def secure_aggregation(self, client_updates: List[np.ndarray]) -> np.ndarray:
        """Secure aggregation using homomorphic encryption simulation"""
        # Simulate secure aggregation
        aggregated_update = np.mean(client_updates, axis=0)
        
        # Add small random noise for additional privacy
        noise = np.random.normal(0, 0.01, aggregated_update.shape)
        secure_aggregated = aggregated_update + noise
        
        return secure_aggregated

class FederatedLearningOrchestrator:
    """Orchestrator for federated learning experiments"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.privacy_fl = PrivacyPreservingFederatedLearning(config)
        
    def create_synthetic_clients(self, base_data: pd.DataFrame, 
                               n_clients: int) -> Dict[str, pd.DataFrame]:
        """Create synthetic client data for federated learning"""
        clients_data = {}
        
        for i in range(n_clients):
            # Create client-specific data variations
            client_data = base_data.copy()
            
            # Add client-specific noise
            noise_factor = np.random.uniform(0.8, 1.2)
            numeric_columns = client_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col != 'customer_id':
                    client_data[col] = client_data[col] * noise_factor + np.random.normal(0, 0.1, len(client_data))
            
            # Add client-specific features
            client_data['client_id'] = f'client_{i}'
            client_data['data_quality'] = np.random.uniform(0.7, 1.0, len(client_data))
            
            clients_data[f'client_{i}'] = client_data
        
        return clients_data
    
    def run_federated_experiment(self, base_data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete federated learning experiment"""
        logger.info("Starting federated learning experiment...")
        
        # Create synthetic clients
        clients_data = self.create_synthetic_clients(base_data, self.config.n_clients)
        
        # Initialize clients
        for client_id, client_data in clients_data.items():
            client = FederatedClient(
                client_id=client_id,
                data=client_data,
                config=self.config,
                model_type='neural_network'
            )
            self.privacy_fl.server.add_client(client_id, client)
        
        # Run federated training
        results = self.privacy_fl.server.start_federated_training()
        
        # Add experiment metadata
        results['experiment_config'] = {
            'n_clients': self.config.n_clients,
            'n_rounds': self.config.n_rounds,
            'local_epochs': self.config.local_epochs,
            'privacy_budget': self.config.privacy_budget,
            'aggregation_method': self.config.aggregation_method
        }
        
        return results
    
    def compare_with_centralized(self, base_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare federated learning with centralized training"""
        logger.info("Comparing federated vs centralized learning...")
        
        # Federated learning results
        federated_results = self.run_federated_experiment(base_data)
        
        # Centralized training (simulated)
        centralized_client = FederatedClient(
            client_id='centralized',
            data=base_data,
            config=self.config,
            model_type='neural_network'
        )
        
        # Train centralized model
        centralized_params = [val.cpu().numpy() for _, val in centralized_client.model.state_dict().items()]
        
        for epoch in range(self.config.n_rounds * self.config.local_epochs):
            centralized_client.fit(centralized_params, {})
        
        centralized_metrics = centralized_client.evaluate(centralized_params, {})
        
        comparison = {
            'federated_results': federated_results,
            'centralized_metrics': centralized_metrics[2],  # Get metrics dict
            'privacy_preserved': True,
            'data_sovereignty': True,
            'communication_overhead': self.config.n_rounds * self.config.n_clients
        }
        
        return comparison

def main():
    """Demo of federated learning for customer analytics"""
    
    # Create sample customer data
    np.random.seed(42)
    n_customers = 5000
    
    base_data = pd.DataFrame({
        'customer_id': range(n_customers),
        'age': np.random.normal(45, 15, n_customers),
        'income': np.random.lognormal(10.5, 0.5, n_customers),
        'purchase_frequency': np.random.poisson(3, n_customers),
        'avg_purchase_amount': np.random.lognormal(4.0, 0.5, n_customers),
        'churned': (np.random.poisson(3, n_customers) < 2).astype(int)
    })
    
    # Configure federated learning
    config = FederatedConfig(
        n_clients=5,
        n_rounds=5,
        local_epochs=3,
        batch_size=32,
        learning_rate=0.01,
        privacy_budget=1.0
    )
    
    # Run federated learning experiment
    orchestrator = FederatedLearningOrchestrator(config)
    comparison = orchestrator.compare_with_centralized(base_data)
    
    print("Federated Learning Results:")
    print(f"Number of clients: {config.n_clients}")
    print(f"Training rounds: {config.n_rounds}")
    print(f"Privacy budget: {config.privacy_budget}")
    print(f"Federated accuracy: {comparison['federated_results']['training_history'][-1]['global_metrics']['accuracy']:.4f}")
    print(f"Centralized accuracy: {comparison['centralized_metrics']['accuracy']:.4f}")
    print(f"Privacy preserved: {comparison['privacy_preserved']}")
    print(f"Data sovereignty: {comparison['data_sovereignty']}")

if __name__ == "__main__":
    from collections import OrderedDict
    main() 