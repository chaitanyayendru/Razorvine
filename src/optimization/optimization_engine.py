"""
Advanced Optimization Engine
Implements sophisticated optimization algorithms for customer targeting and promotion strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Reinforcement learning
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Multi-armed bandits
# from bandits import EpsilonGreedy, UCB, ThompsonSampling, LinUCB  # Commented out - using contextualbandits instead

# Statistical libraries
from scipy.optimize import minimize, differential_evolution
from scipy.stats import beta, norm
import statsmodels.api as sm

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score


class MultiArmedBanditOptimizer:
    """
    Advanced multi-armed bandit optimizer for customer targeting.
    """
    
    def __init__(self, n_arms: int, customer_data: pd.DataFrame, 
                 reward_function: str = 'clv_increase'):
        """
        Initialize the multi-armed bandit optimizer.
        
        Args:
            n_arms: Number of arms (promotion strategies)
            customer_data: Customer data for simulation
            reward_function: Type of reward function
        """
        self.n_arms = n_arms
        self.customer_data = customer_data
        self.reward_function = reward_function
        
        # Initialize bandit algorithms
        self.bandits = {
            'epsilon_greedy': EpsilonGreedy(n_arms, epsilon=0.1),
            'ucb': UCB(n_arms),
            'thompson_sampling': ThompsonSampling(n_arms),
            'linucb': LinUCB(n_arms, context_dim=10)
        }
        
        # Store results
        self.results = {}
        self.regrets = {}
        
    def _calculate_reward(self, arm: int, customer_features: np.ndarray) -> float:
        """Calculate reward for a given arm and customer."""
        if self.reward_function == 'clv_increase':
            # Simulate CLV increase based on promotion
            base_clv = customer_features[0] if len(customer_features) > 0 else 1000
            promotion_effect = 0.1 + 0.2 * np.random.random()  # 10-30% increase
            return base_clv * promotion_effect
        elif self.reward_function == 'conversion_rate':
            # Simulate conversion probability
            base_conversion = 0.05 + 0.1 * np.random.random()
            return np.random.binomial(1, base_conversion)
        else:
            return np.random.normal(0, 1)
    
    def run_bandit_experiment(self, n_rounds: int = 1000, 
                            context_dim: int = 10) -> Dict:
        """
        Run multi-armed bandit experiment.
        
        Args:
            n_rounds: Number of rounds to run
            context_dim: Dimension of customer context
            
        Returns:
            Dictionary with experiment results
        """
        print("🎰 Running Multi-Armed Bandit Experiment...")
        
        results = {}
        
        for bandit_name, bandit in self.bandits.items():
            print(f"Running {bandit_name}...")
            
            cumulative_reward = 0
            cumulative_regret = 0
            rewards_history = []
            regrets_history = []
            
            for round_idx in range(n_rounds):
                # Generate customer context
                customer_context = np.random.normal(0, 1, context_dim)
                
                # Select arm
                if bandit_name == 'linucb':
                    arm = bandit.select_arm(customer_context)
                else:
                    arm = bandit.select_arm()
                
                # Get reward
                reward = self._calculate_reward(arm, customer_context)
                
                # Update bandit
                if bandit_name == 'linucb':
                    bandit.update(arm, reward, customer_context)
                else:
                    bandit.update(arm, reward)
                
                # Track metrics
                cumulative_reward += reward
                rewards_history.append(cumulative_reward)
                
                # Calculate regret (assuming arm 0 is optimal)
                optimal_reward = self._calculate_reward(0, customer_context)
                regret = optimal_reward - reward
                cumulative_regret += regret
                regrets_history.append(cumulative_regret)
            
            results[bandit_name] = {
                'cumulative_reward': cumulative_reward,
                'cumulative_regret': cumulative_regret,
                'rewards_history': rewards_history,
                'regrets_history': regrets_history,
                'final_arm_counts': bandit.arm_counts
            }
        
        self.results = results
        return results
    
    def plot_bandit_results(self, save_path: str = None):
        """Plot bandit experiment results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Rewards', 'Cumulative Regret',
                          'Arm Selection Distribution', 'Reward per Round'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (bandit_name, result) in enumerate(self.results.items()):
            color = colors[i % len(colors)]
            
            # Cumulative rewards
            fig.add_trace(
                go.Scatter(y=result['rewards_history'], name=f'{bandit_name} Rewards',
                          line=dict(color=color)),
                row=1, col=1
            )
            
            # Cumulative regret
            fig.add_trace(
                go.Scatter(y=result['regrets_history'], name=f'{bandit_name} Regret',
                          line=dict(color=color)),
                row=1, col=2
            )
            
            # Arm selection distribution
            arm_counts = result['final_arm_counts']
            fig.add_trace(
                go.Bar(x=list(range(len(arm_counts))), y=arm_counts,
                      name=f'{bandit_name} Arms', marker_color=color),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="Multi-Armed Bandit Results")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class ReinforcementLearningOptimizer:
    """
    Advanced reinforcement learning optimizer for customer targeting.
    """
    
    def __init__(self, customer_data: pd.DataFrame, 
                 state_features: List[str], action_space: int = 5):
        """
        Initialize the RL optimizer.
        
        Args:
            customer_data: Customer data for training
            state_features: Features to use as state
            action_space: Number of possible actions
        """
        self.customer_data = customer_data
        self.state_features = state_features
        self.action_space = action_space
        
        # Create environment
        self.env = self._create_environment()
        
        # Initialize models
        self.models = {
            'ppo': PPO('MlpPolicy', self.env, verbose=0),
            'a2c': A2C('MlpPolicy', self.env, verbose=0),
            'dqn': DQN('MlpPolicy', self.env, verbose=0)
        }
        
        self.results = {}
        
    def _create_environment(self):
        """Create custom RL environment for customer targeting."""
        class CustomerTargetingEnv(gym.Env):
            def __init__(self, data, state_features, action_space):
                super().__init__()
                self.data = data
                self.state_features = state_features
                self.action_space = action_space
                self.current_step = 0
                
                # Define action and observation spaces
                self.action_space = gym.spaces.Discrete(action_space)
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(len(state_features),), dtype=np.float32
                )
                
            def reset(self):
                self.current_step = 0
                return self._get_state()
            
            def step(self, action):
                # Get current customer
                customer = self.data.iloc[self.current_step % len(self.data)]
                
                # Calculate reward based on action
                reward = self._calculate_reward(action, customer)
                
                # Move to next step
                self.current_step += 1
                done = self.current_step >= len(self.data)
                
                return self._get_state(), reward, done, {}
            
            def _get_state(self):
                customer = self.data.iloc[self.current_step % len(self.data)]
                return customer[self.state_features].values.astype(np.float32)
            
            def _calculate_reward(self, action, customer):
                # Simulate reward based on action and customer characteristics
                base_reward = customer.get('customer_lifetime_value', 1000) / 1000
                action_effect = 0.1 * (action + 1)  # Different actions have different effects
                return base_reward * action_effect + np.random.normal(0, 0.1)
        
        return DummyVecEnv([lambda: CustomerTargetingEnv(
            self.customer_data, self.state_features, self.action_space
        )])
    
    def train_models(self, total_timesteps: int = 10000) -> Dict:
        """
        Train RL models.
        
        Args:
            total_timesteps: Number of timesteps for training
            
        Returns:
            Dictionary with training results
        """
        print("🤖 Training Reinforcement Learning Models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Create evaluation callback
            eval_env = self._create_environment()
            eval_callback = EvalCallback(
                eval_env, best_model_save_path=f"./best_{model_name}",
                log_path=f"./logs/{model_name}", eval_freq=1000,
                deterministic=True, render=False
            )
            
            # Train model
            model.learn(total_timesteps=total_timesteps, callback=eval_callback)
            
            # Evaluate model
            mean_reward = self._evaluate_model(model)
            
            results[model_name] = {
                'model': model,
                'mean_reward': mean_reward,
                'training_steps': total_timesteps
            }
        
        self.results = results
        return results
    
    def _evaluate_model(self, model, n_eval_episodes: int = 100) -> float:
        """Evaluate a trained model."""
        eval_env = self._create_environment()
        rewards = []
        
        for _ in range(n_eval_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)
    
    def optimize_policy(self, model_name: str = 'ppo') -> Dict:
        """
        Optimize the policy for the best model.
        
        Args:
            model_name: Name of the model to optimize
            
        Returns:
            Dictionary with optimization results
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        model = self.results[model_name]['model']
        
        # Create optimization environment
        opt_env = self._create_environment()
        
        # Optimize hyperparameters
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            n_steps = trial.suggest_int('n_steps', 64, 2048)
            batch_size = trial.suggest_int('batch_size', 32, 256)
            
            # Create model with suggested parameters
            if model_name == 'ppo':
                opt_model = PPO('MlpPolicy', opt_env, 
                               learning_rate=learning_rate,
                               n_steps=n_steps,
                               batch_size=batch_size,
                               verbose=0)
            elif model_name == 'a2c':
                opt_model = A2C('MlpPolicy', opt_env,
                               learning_rate=learning_rate,
                               n_steps=n_steps,
                               verbose=0)
            else:
                opt_model = DQN('MlpPolicy', opt_env,
                               learning_rate=learning_rate,
                               batch_size=batch_size,
                               verbose=0)
            
            # Train and evaluate
            opt_model.learn(total_timesteps=5000)
            mean_reward = self._evaluate_model(opt_model, n_eval_episodes=50)
            
            return mean_reward
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'optimization_history': study.trials_dataframe()
        }


class AdvancedOptimizationEngine:
    """
    Comprehensive optimization engine combining multiple approaches.
    """
    
    def __init__(self, customer_data: pd.DataFrame):
        """
        Initialize the optimization engine.
        
        Args:
            customer_data: Customer data for optimization
        """
        self.customer_data = customer_data
        
        # Initialize optimizers
        self.bandit_optimizer = MultiArmedBanditOptimizer(
            n_arms=5, customer_data=customer_data
        )
        
        state_features = ['age', 'income', 'loyalty_score', 'monthly_spend', 'online_activity']
        self.rl_optimizer = ReinforcementLearningOptimizer(
            customer_data=customer_data, state_features=state_features
        )
        
        # Store results
        self.results = {}
        
    def run_comprehensive_optimization(self, 
                                     bandit_rounds: int = 1000,
                                     rl_timesteps: int = 10000) -> Dict:
        """
        Run comprehensive optimization using all methods.
        
        Args:
            bandit_rounds: Number of rounds for bandit experiments
            rl_timesteps: Number of timesteps for RL training
            
        Returns:
            Dictionary with all optimization results
        """
        print("🚀 Running Comprehensive Optimization...")
        
        # Run bandit optimization
        print("1. Multi-Armed Bandit Optimization")
        bandit_results = self.bandit_optimizer.run_bandit_experiment(n_rounds=bandit_rounds)
        
        # Run RL optimization
        print("2. Reinforcement Learning Optimization")
        rl_results = self.rl_optimizer.train_models(total_timesteps=rl_timesteps)
        
        # Run policy optimization
        print("3. Policy Optimization")
        policy_results = self.rl_optimizer.optimize_policy('ppo')
        
        # Combine results
        self.results = {
            'bandit': bandit_results,
            'reinforcement_learning': rl_results,
            'policy_optimization': policy_results
        }
        
        return self.results
    
    def customer_targeting_strategy(self, customer_features: Dict) -> Dict:
        """
        Generate optimal targeting strategy for a customer.
        
        Args:
            customer_features: Customer features dictionary
            
        Returns:
            Dictionary with targeting recommendations
        """
        # Convert customer features to array
        feature_array = np.array([
            customer_features.get('age', 35),
            customer_features.get('income', 50000),
            customer_features.get('loyalty_score', 0.5),
            customer_features.get('monthly_spend', 500),
            customer_features.get('online_activity', 20)
        ])
        
        # Get recommendations from different methods
        recommendations = {}
        
        # Bandit recommendation (use best performing bandit)
        if 'bandit' in self.results:
            best_bandit = max(self.results['bandit'].keys(), 
                            key=lambda x: self.results['bandit'][x]['cumulative_reward'])
            bandit = self.bandit_optimizer.bandits[best_bandit]
            recommendations['bandit_action'] = bandit.select_arm()
            recommendations['bandit_confidence'] = bandit.arm_counts[recommendations['bandit_action']] / sum(bandit.arm_counts)
        
        # RL recommendation
        if 'reinforcement_learning' in self.results:
            best_rl_model = max(self.results['reinforcement_learning'].keys(),
                              key=lambda x: self.results['reinforcement_learning'][x]['mean_reward'])
            model = self.results['reinforcement_learning'][best_rl_model]['model']
            action, _ = model.predict(feature_array.reshape(1, -1), deterministic=True)
            recommendations['rl_action'] = int(action[0])
            recommendations['rl_model'] = best_rl_model
        
        # Calculate expected value
        expected_value = self._calculate_expected_value(customer_features, recommendations)
        recommendations['expected_value'] = expected_value
        
        return recommendations
    
    def _calculate_expected_value(self, customer_features: Dict, 
                                recommendations: Dict) -> float:
        """Calculate expected value of recommendations."""
        base_clv = customer_features.get('customer_lifetime_value', 1000)
        
        # Estimate improvement based on recommendations
        improvement = 0.1  # Base 10% improvement
        
        if 'bandit_action' in recommendations:
            # Different actions have different effects
            action_effect = 0.05 * (recommendations['bandit_action'] + 1)
            improvement += action_effect
        
        if 'rl_action' in recommendations:
            # RL action effect
            rl_effect = 0.03 * (recommendations['rl_action'] + 1)
            improvement += rl_effect
        
        return base_clv * (1 + improvement)
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report."""
        report = []
        report.append("=" * 60)
        report.append("🎯 ADVANCED OPTIMIZATION ENGINE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Bandit Results
        if 'bandit' in self.results:
            report.append("🎰 MULTI-ARMED BANDIT RESULTS")
            report.append("-" * 30)
            for bandit_name, result in self.results['bandit'].items():
                report.append(f"{bandit_name.upper()}:")
                report.append(f"  Cumulative Reward: {result['cumulative_reward']:.2f}")
                report.append(f"  Cumulative Regret: {result['cumulative_regret']:.2f}")
                report.append(f"  Best Arm: {np.argmax(result['final_arm_counts'])}")
            report.append("")
        
        # RL Results
        if 'reinforcement_learning' in self.results:
            report.append("🤖 REINFORCEMENT LEARNING RESULTS")
            report.append("-" * 35)
            for model_name, result in self.results['reinforcement_learning'].items():
                report.append(f"{model_name.upper()}:")
                report.append(f"  Mean Reward: {result['mean_reward']:.4f}")
                report.append(f"  Training Steps: {result['training_steps']:,}")
            report.append("")
        
        # Policy Optimization Results
        if 'policy_optimization' in self.results:
            report.append("⚙️ POLICY OPTIMIZATION RESULTS")
            report.append("-" * 30)
            po = self.results['policy_optimization']
            report.append(f"Best Parameters: {po['best_params']}")
            report.append(f"Best Value: {po['best_value']:.4f}")
            report.append("")
        
        return "\n".join(report)
    
    def plot_optimization_results(self, save_path: str = None):
        """Create comprehensive visualization of optimization results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bandit Cumulative Rewards', 'Bandit Arm Distribution',
                          'RL Model Performance', 'Policy Optimization History'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Bandit cumulative rewards
        if 'bandit' in self.results:
            colors = ['blue', 'red', 'green', 'orange']
            for i, (bandit_name, result) in enumerate(self.results['bandit'].items()):
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(y=result['rewards_history'], name=f'{bandit_name} Rewards',
                              line=dict(color=color)),
                    row=1, col=1
                )
        
        # Bandit arm distribution
        if 'bandit' in self.results:
            best_bandit = max(self.results['bandit'].keys(), 
                            key=lambda x: self.results['bandit'][x]['cumulative_reward'])
            arm_counts = self.results['bandit'][best_bandit]['final_arm_counts']
            fig.add_trace(
                go.Bar(x=list(range(len(arm_counts))), y=arm_counts,
                      name=f'{best_bandit} Arms'),
                row=1, col=2
            )
        
        # RL model performance
        if 'reinforcement_learning' in self.results:
            model_names = list(self.results['reinforcement_learning'].keys())
            mean_rewards = [self.results['reinforcement_learning'][name]['mean_reward'] 
                           for name in model_names]
            fig.add_trace(
                go.Bar(x=model_names, y=mean_rewards, name='Mean Rewards'),
                row=2, col=1
            )
        
        # Policy optimization history
        if 'policy_optimization' in self.results:
            opt_history = self.results['policy_optimization']['optimization_history']
            fig.add_trace(
                go.Scatter(x=opt_history.index, y=opt_history['value'],
                          mode='lines+markers', name='Optimization Progress'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Optimization Results")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def run_optimization_engine(data_path: str = "data/simulated_customer_data.csv") -> AdvancedOptimizationEngine:
    """
    Run comprehensive optimization engine.
    
    Args:
        data_path: Path to the customer data
        
    Returns:
        AdvancedOptimizationEngine instance with results
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Initialize optimization engine
    engine = AdvancedOptimizationEngine(customer_data=data)
    
    # Run comprehensive optimization
    engine.run_comprehensive_optimization()
    
    # Generate report
    report = engine.generate_optimization_report()
    print(report)
    
    return engine


if __name__ == "__main__":
    # Run optimization engine
    engine = run_optimization_engine()
    
    # Create visualizations
    engine.plot_optimization_results("optimization_results.html")
    
    # Test customer targeting
    sample_customer = {
        'age': 35,
        'income': 75000,
        'loyalty_score': 0.7,
        'monthly_spend': 800,
        'online_activity': 25,
        'customer_lifetime_value': 12000
    }
    
    targeting_strategy = engine.customer_targeting_strategy(sample_customer)
    print(f"\n🎯 Targeting Strategy for Sample Customer:")
    print(f"Bandit Action: {targeting_strategy.get('bandit_action', 'N/A')}")
    print(f"RL Action: {targeting_strategy.get('rl_action', 'N/A')}")
    print(f"Expected Value: ${targeting_strategy.get('expected_value', 0):,.2f}")
