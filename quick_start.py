#!/usr/bin/env python3
"""
RazorVine Quick Start Script
Demonstrates the platform capabilities with a complete example.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Run a complete demonstration of RazorVine capabilities."""
    print("🚀 RazorVine Analytics Platform - Quick Start")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n1️⃣ Generating sophisticated customer data...")
    from data_pipeline.data_simulator import AdvancedCustomerSimulator
    
    simulator = AdvancedCustomerSimulator(seed=42)
    data = simulator.simulate_customer_data(n_customers=5000)
    
    print(f"   ✅ Generated {len(data):,} customers")
    print(f"   📊 Customer segments: {data['customer_segment'].value_counts().to_dict()}")
    print(f"   💰 Average CLV: ${data['customer_lifetime_value'].mean():,.2f}")
    
    # Step 2: Causal Inference Analysis
    print("\n2️⃣ Running Causal Inference Analysis...")
    from causal_inference.causal_analysis import AdvancedCausalAnalyzer
    
    causal_analyzer = AdvancedCausalAnalyzer(
        data=data,
        treatment_col='promotion_response',
        outcome_col='customer_lifetime_value'
    )
    
    # Run key analyses
    pm_results = causal_analyzer.propensity_score_matching()
    ml_results = causal_analyzer.ml_based_causal_inference()
    
    print(f"   ✅ Propensity Score Matching ATE: {pm_results['ate']:.2f}")
    print(f"   ✅ ML-based ATE (XGBoost): {ml_results['xgboost']['ate']:.2f}")
    
    # Step 3: Predictive Modeling
    print("\n3️⃣ Running Predictive Modeling...")
    from predictive_modeling.model_training import AdvancedPredictiveModeler
    
    modeler = AdvancedPredictiveModeler(
        data=data,
        target_col='customer_lifetime_value',
        problem_type='regression'
    )
    
    # Train models
    ensemble_results = modeler.train_ensemble_models()
    modeler.train_deep_learning_model()
    
    # Show best model performance
    best_model = min(ensemble_results.keys(), 
                    key=lambda x: ensemble_results[x]['metrics']['rmse'])
    best_rmse = ensemble_results[best_model]['metrics']['rmse']
    print(f"   ✅ Best model ({best_model}) RMSE: {best_rmse:.2f}")
    
    # Step 4: Optimization
    print("\n4️⃣ Running Optimization Analysis...")
    from optimization.optimization_engine import AdvancedOptimizationEngine
    
    engine = AdvancedOptimizationEngine(customer_data=data)
    engine.run_comprehensive_optimization(
        bandit_rounds=500,   # Reduced for demo
        rl_timesteps=2000    # Reduced for demo
    )
    
    # Test customer targeting
    sample_customer = {
        'age': 35,
        'income': 75000,
        'loyalty_score': 0.7,
        'monthly_spend': 800,
        'online_activity': 25,
        'customer_lifetime_value': 12000
    }
    
    strategy = engine.customer_targeting_strategy(sample_customer)
    print(f"   ✅ Targeting strategy generated")
    print(f"   🎯 Expected value: ${strategy['expected_value']:,.2f}")
    
    # Step 5: Generate comprehensive report
    print("\n5️⃣ Generating Comprehensive Report...")
    
    report = f"""
{'='*80}
🚀 RAZORVINE QUICK START DEMO RESULTS
{'='*80}

📊 DATASET SUMMARY
{'-'*40}
Total customers: {len(data):,}
Customer segments: {dict(data['customer_segment'].value_counts())}
Average CLV: ${data['customer_lifetime_value'].mean():,.2f}
Average churn risk: {data['churn_risk'].mean():.3f}
Promotion response rate: {data['promotion_response'].mean():.3f}

🔬 CAUSAL INFERENCE RESULTS
{'-'*40}
Propensity Score Matching ATE: {pm_results['ate']:.2f}
ML-based ATE (XGBoost): {ml_results['xgboost']['ate']:.2f}
Treatment effect is {'positive' if pm_results['ate'] > 0 else 'negative'}

🤖 PREDICTIVE MODELING RESULTS
{'-'*40}
Best model: {best_model}
RMSE: {best_rmse:.2f}
R² Score: {ensemble_results[best_model]['metrics']['r2']:.3f}

🎯 OPTIMIZATION RESULTS
{'-'*40}
Sample customer targeting:
- Age: {sample_customer['age']}
- Income: ${sample_customer['income']:,}
- Loyalty Score: {sample_customer['loyalty_score']}
- Expected Value: ${strategy['expected_value']:,.2f}

{'='*80}
✅ DEMO COMPLETED SUCCESSFULLY!
{'='*80}

Next steps:
1. Start the API server: python src/api/main.py
2. Access the platform: http://localhost:8000
3. View interactive docs: http://localhost:8000/docs
4. Run full analysis: python src/cli.py analyze

For more information, see the README.md file.
"""
    
    # Save report
    with open('razorvine_demo_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print("📄 Full report saved to: razorvine_demo_report.txt")
    
    # Step 6: Show next steps
    print("\n🎉 Quick Start Demo Completed!")
    print("\n📚 Next Steps:")
    print("   1. Start the API server: python src/api/main.py")
    print("   2. Access the platform: http://localhost:8000")
    print("   3. View interactive docs: http://localhost:8000/docs")
    print("   4. Run full analysis: python src/cli.py analyze")
    print("   5. Explore Jupyter notebooks: jupyter lab")
    
    print("\n🔧 Available Commands:")
    print("   python src/cli.py simulate    - Generate more data")
    print("   python src/cli.py causal      - Run causal analysis")
    print("   python src/cli.py predict     - Run predictive modeling")
    print("   python src/cli.py optimize    - Run optimization")
    print("   python src/cli.py demo        - Run this demo again")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check that you're in the project root directory")
        print("   3. Ensure Python 3.8+ is being used")
        sys.exit(1) 