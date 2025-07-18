"""
Multi-Armed Bandit Algorithms
Basic implementations of common bandit algorithms for customer targeting.
"""

import numpy as np
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class EpsilonGreedy:
    """Epsilon-Greedy bandit algorithm."""
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.arm_counts = np.zeros(n_arms)
        self.arm_values = np.zeros(n_arms)
        
    def select_arm(self) -> int:
        """Select an arm using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            # Explore: choose random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: choose best arm
            return np.argmax(self.arm_values)
    
    def update(self, arm: int, reward: float):
        """Update the value estimate for the selected arm."""
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        # Incremental update
        self.arm_values[arm] = ((n - 1) * self.arm_values[arm] + reward) / n


class UCB:
    """Upper Confidence Bound bandit algorithm."""
    
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        self.arm_counts = np.zeros(n_arms)
        self.arm_values = np.zeros(n_arms)
        self.total_pulls = 0
        
    def select_arm(self) -> int:
        """Select an arm using UCB strategy."""
        if self.total_pulls < self.n_arms:
            # Pull each arm once initially
            return self.total_pulls
        
        # Calculate UCB values
        ucb_values = self.arm_values + self.c * np.sqrt(np.log(self.total_pulls) / self.arm_counts)
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """Update the value estimate for the selected arm."""
        self.arm_counts[arm] += 1
        self.total_pulls += 1
        n = self.arm_counts[arm]
        # Incremental update
        self.arm_values[arm] = ((n - 1) * self.arm_values[arm] + reward) / n


class ThompsonSampling:
    """Thompson Sampling bandit algorithm."""
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Beta distribution parameters
        self.beta = np.ones(n_arms)
        
    def select_arm(self) -> int:
        """Select an arm using Thompson sampling."""
        # Sample from beta distributions
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        """Update the beta distribution parameters."""
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


class LinUCB:
    """Linear UCB bandit algorithm for contextual bandits."""
    
    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Initialize parameters for each arm
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]
        
    def select_arm(self, context: np.ndarray) -> int:
        """Select an arm using LinUCB strategy."""
        ucb_values = []
        
        for arm in range(self.n_arms):
            # Calculate UCB value
            A_inv = np.linalg.inv(self.A[arm])
            self.theta[arm] = A_inv @ self.b[arm]
            
            # UCB term
            ucb_term = self.alpha * np.sqrt(context.T @ A_inv @ context)
            ucb_value = self.theta[arm].T @ context + ucb_term
            ucb_values.append(ucb_value)
        
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float, context: np.ndarray):
        """Update the parameters for the selected arm."""
        # Update A and b matrices
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context 