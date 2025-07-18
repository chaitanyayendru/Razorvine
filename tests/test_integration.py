"""
Integration tests for RazorVine Analytics Platform
Tests the complete end-to-end functionality of the platform.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_pipeline.data_simulator import AdvancedCustomerSimulator
from causal_inference.causal_analysis import AdvancedCausalAnalyzer
from predictive_modeling.model_training import AdvancedPredictiveModeler
from optimization.optimization_engine import AdvancedOptimizationEngine


class TestRazorVineIntegration:
    """Integration tests for the complete RazorVine platform."""
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Generate sample data for testing."""
        simulator = AdvancedCustomerSimulator(seed=42)
        data = simulator.simulate_customer_data(n_customers=1000)
        return data
    
    def test_data_simulation(self, sample_data):
        """Test data simulation functionality."""
        assert len(sample_data) == 1000
        assert 'customer_id' in sample_data.columns
        assert 'customer_segment' in sample_data.columns
        assert 'customer_lifetime_value' in sample_data.columns
        assert 'promotion_response' in sample_data.columns
        
        # Check customer segments
        segments = sample_data['customer_segment'].value_counts()
        assert len(segments) == 4  # premium, regular, occasional, at_risk
        
        # Check data quality
        assert sample_data['customer_lifetime_value'].min() > 0
        assert sample_data['loyalty_score'].min() >= 0
        assert sample_data['loyalty_score'].max() <= 1
        
        print("✅ Data simulation test passed")
    
    def test_causal_inference(self, sample_data):
        """Test causal inference functionality."""
        analyzer = AdvancedCausalAnalyzer(
            data=sample_data,
            treatment_col='promotion_response',
            outcome_col='customer_lifetime_value'
        )
        
        # Test propensity score matching
        pm_results = analyzer.propensity_score_matching()
        assert 'ate' in pm_results
        assert 'ate_std' in pm_results
        assert 'ate_ci' in pm_results
        
        # Test ML-based causal inference
        ml_results = analyzer.ml_based_causal_inference()
        assert 'lr' in ml_results
        assert 'xgboost' in ml_results
        assert 'mlp' in ml_results
        
        # Test sensitivity analysis
        sa_results = analyzer.sensitivity_analysis()
        assert 'gamma_values' in sa_results
        assert 'p_values' in sa_results
        
        # Test report generation
        report = analyzer.generate_report()
        assert len(report) > 0
        assert "CAUSAL INFERENCE REPORT" in report
        
        print("✅ Causal inference test passed")
    
    def test_predictive_modeling(self, sample_data):
        """Test predictive modeling functionality."""
        modeler = AdvancedPredictiveModeler(
            data=sample_data,
            target_col='customer_lifetime_value',
            problem_type='regression'
        )
        
        # Test ensemble model training
        ensemble_results = modeler.train_ensemble_models()
        assert 'random_forest' in ensemble_results
        assert 'xgboost' in ensemble_results
        assert 'lightgbm' in ensemble_results
        
        # Test deep learning model
        dl_results = modeler.train_deep_learning_model()
        assert 'model' in dl_results
        assert 'metrics' in dl_results
        
        # Test ensemble creation
        ensemble = modeler.create_ensemble(['random_forest', 'xgboost'])
        assert 'model' in ensemble
        assert 'metrics' in ensemble
        
        # Test feature selection
        fs_results = modeler.feature_selection()
        assert 'selected_features' in fs_results
        assert 'metrics' in fs_results
        
        # Test model interpretation
        interpretation = modeler.model_interpretation('random_forest')
        assert 'shap_values' in interpretation
        assert 'lime_explanation' in interpretation
        
        # Test report generation
        report = modeler.generate_report()
        assert len(report) > 0
        assert "PREDICTIVE MODELING REPORT" in report
        
        print("✅ Predictive modeling test passed")
    
    def test_optimization_engine(self, sample_data):
        """Test optimization engine functionality."""
        engine = AdvancedOptimizationEngine(customer_data=sample_data)
        
        # Test comprehensive optimization
        results = engine.run_comprehensive_optimization(
            bandit_rounds=100,  # Reduced for testing
            rl_timesteps=1000   # Reduced for testing
        )
        
        assert 'bandit' in results
        assert 'reinforcement_learning' in results
        assert 'policy_optimization' in results
        
        # Test bandit results
        bandit_results = results['bandit']
        assert 'epsilon_greedy' in bandit_results
        assert 'ucb' in bandit_results
        assert 'thompson_sampling' in bandit_results
        
        # Test RL results
        rl_results = results['reinforcement_learning']
        assert 'ppo' in rl_results
        assert 'a2c' in rl_results
        assert 'dqn' in rl_results
        
        # Test customer targeting
        customer_features = {
            'age': 35,
            'income': 75000,
            'loyalty_score': 0.7,
            'monthly_spend': 800,
            'online_activity': 25,
            'customer_lifetime_value': 12000
        }
        
        strategy = engine.customer_targeting_strategy(customer_features)
        assert 'targeting_strategy' in strategy
        assert 'recommendations' in strategy
        
        # Test report generation
        report = engine.generate_optimization_report()
        assert len(report) > 0
        assert "OPTIMIZATION ENGINE REPORT" in report
        
        print("✅ Optimization engine test passed")
    
    def test_end_to_end_pipeline(self, sample_data):
        """Test complete end-to-end pipeline."""
        # Save sample data
        sample_data.to_csv('test_data.csv', index=False)
        
        # Test complete pipeline
        results = {}
        
        # 1. Causal Analysis
        causal_analyzer = AdvancedCausalAnalyzer(
            data=sample_data,
            treatment_col='promotion_response',
            outcome_col='customer_lifetime_value'
        )
        causal_analyzer.propensity_score_matching()
        results['causal'] = causal_analyzer.generate_report()
        
        # 2. Predictive Modeling
        predictive_modeler = AdvancedPredictiveModeler(
            data=sample_data,
            target_col='customer_lifetime_value',
            problem_type='regression'
        )
        predictive_modeler.train_ensemble_models()
        results['predictive'] = predictive_modeler.generate_report()
        
        # 3. Optimization
        optimization_engine = AdvancedOptimizationEngine(customer_data=sample_data)
        optimization_engine.run_comprehensive_optimization(
            bandit_rounds=50,   # Minimal for testing
            rl_timesteps=500    # Minimal for testing
        )
        results['optimization'] = optimization_engine.generate_optimization_report()
        
        # Verify all components worked
        assert len(results['causal']) > 0
        assert len(results['predictive']) > 0
        assert len(results['optimization']) > 0
        
        # Clean up
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
        
        print("✅ End-to-end pipeline test passed")
    
    def test_data_quality(self, sample_data):
        """Test data quality and consistency."""
        # Check for missing values
        missing_counts = sample_data.isnull().sum()
        assert missing_counts.sum() > 0  # Should have some missing values for realism
        
        # Check for outliers
        clv = sample_data['customer_lifetime_value']
        q1, q3 = clv.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = clv[(clv < q1 - 1.5 * iqr) | (clv > q3 + 1.5 * iqr)]
        assert len(outliers) > 0  # Should have some outliers for realism
        
        # Check logical constraints
        assert (sample_data['loyalty_score'] >= 0).all()
        assert (sample_data['loyalty_score'] <= 1).all()
        assert (sample_data['age'] >= 18).all()
        assert (sample_data['age'] <= 100).all()
        assert (sample_data['income'] > 0).all()
        
        # Check customer segments
        valid_segments = ['premium', 'regular', 'occasional', 'at_risk']
        assert sample_data['customer_segment'].isin(valid_segments).all()
        
        print("✅ Data quality test passed")
    
    def test_model_performance(self, sample_data):
        """Test model performance metrics."""
        modeler = AdvancedPredictiveModeler(
            data=sample_data,
            target_col='customer_lifetime_value',
            problem_type='regression'
        )
        
        # Train models
        ensemble_results = modeler.train_ensemble_models()
        
        # Check performance metrics
        for model_name, result in ensemble_results.items():
            metrics = result['metrics']
            
            # Regression metrics should be present
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert 'r2' in metrics
            
            # Metrics should be reasonable
            assert metrics['rmse'] > 0
            assert metrics['mae'] > 0
            assert metrics['r2'] >= 0  # R² can be negative for poor models
        
        print("✅ Model performance test passed")


def test_cli_imports():
    """Test that CLI can import all modules."""
    try:
        from cli import main
        assert main is not None
        print("✅ CLI imports test passed")
    except ImportError as e:
        pytest.fail(f"CLI import failed: {e}")


def test_api_imports():
    """Test that API can import all modules."""
    try:
        from api.main import app
        assert app is not None
        print("✅ API imports test passed")
    except ImportError as e:
        pytest.fail(f"API import failed: {e}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"]) 