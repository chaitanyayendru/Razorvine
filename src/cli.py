"""
Command Line Interface for RazorVine Analytics Platform
Provides easy-to-use commands for data generation, analysis, and optimization.
"""

import click
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Optional
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_pipeline.data_simulator import AdvancedCustomerSimulator
from causal_inference.causal_analysis import AdvancedCausalAnalyzer
from predictive_modeling.model_training import AdvancedPredictiveModeler
from optimization.optimization_engine import AdvancedOptimizationEngine


@click.group()
@click.version_option(version="2.0.0")
def main():
    """
    🚀 RazorVine Analytics Platform CLI
    
    Advanced customer analytics with causal inference, predictive modeling, and optimization.
    """
    pass


@main.command()
@click.option('--customers', '-n', default=10000, help='Number of customers to simulate')
@click.option('--output', '-o', default='data/simulated_customer_data.csv', help='Output file path')
@click.option('--seed', '-s', default=42, help='Random seed for reproducibility')
def simulate(customers: int, output: str, seed: int):
    """Generate sophisticated customer data with realistic patterns."""
    click.echo(f"🎯 Generating {customers:,} customers with seed {seed}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # Initialize simulator
    simulator = AdvancedCustomerSimulator(seed=seed)
    
    # Generate data
    data = simulator.simulate_customer_data(n_customers=customers, save_path=output)
    
    click.echo(f"✅ Generated {len(data):,} customers")
    click.echo(f"📊 Data saved to: {output}")
    click.echo(f"🎯 Customer segments: {data['customer_segment'].value_counts().to_dict()}")


@main.command()
@click.option('--data', '-d', default='data/simulated_customer_data.csv', help='Input data file')
@click.option('--treatment', '-t', default='promotion_response', help='Treatment column')
@click.option('--outcome', '-o', default='customer_lifetime_value', help='Outcome column')
@click.option('--output', default='causal_analysis_results.html', help='Output visualization file')
def causal(data: str, treatment: str, outcome: str, output: str):
    """Run comprehensive causal inference analysis."""
    click.echo("🔬 Running Causal Inference Analysis...")
    
    if not os.path.exists(data):
        click.echo(f"❌ Data file not found: {data}")
        return
    
    # Load data
    customer_data = pd.read_csv(data)
    
    # Initialize analyzer
    analyzer = AdvancedCausalAnalyzer(
        data=customer_data,
        treatment_col=treatment,
        outcome_col=outcome
    )
    
    # Run analyses
    click.echo("  📊 Running propensity score matching...")
    analyzer.propensity_score_matching()
    
    click.echo("  🤖 Running ML-based causal inference...")
    analyzer.ml_based_causal_inference()
    
    click.echo("  🔬 Running sensitivity analysis...")
    analyzer.sensitivity_analysis()
    
    # Generate report
    report = analyzer.generate_report()
    click.echo("\n" + report)
    
    # Save visualizations
    click.echo(f"📈 Saving visualizations to: {output}")
    analyzer.plot_results(output)
    
    click.echo("✅ Causal analysis completed!")


@main.command()
@click.option('--data', '-d', default='data/simulated_customer_data.csv', help='Input data file')
@click.option('--target', '-t', default='customer_lifetime_value', help='Target variable')
@click.option('--problem-type', '-p', default='regression', type=click.Choice(['regression', 'classification']), help='Problem type')
@click.option('--output', default='predictive_modeling_results.html', help='Output visualization file')
def predict(data: str, target: str, problem_type: str, output: str):
    """Run advanced predictive modeling analysis."""
    click.echo(f"🤖 Running Predictive Modeling ({problem_type})...")
    
    if not os.path.exists(data):
        click.echo(f"❌ Data file not found: {data}")
        return
    
    # Load data
    customer_data = pd.read_csv(data)
    
    # Initialize modeler
    modeler = AdvancedPredictiveModeler(
        data=customer_data,
        target_col=target,
        problem_type=problem_type
    )
    
    # Train models
    click.echo("  🚀 Training ensemble models...")
    modeler.train_ensemble_models()
    
    click.echo("  🧠 Training deep learning model...")
    modeler.train_deep_learning_model()
    
    click.echo("  🎯 Creating model ensemble...")
    modeler.create_ensemble(['random_forest', 'xgboost', 'lightgbm'])
    
    click.echo("  🔍 Performing feature selection...")
    modeler.feature_selection()
    
    click.echo("  🔍 Interpreting models...")
    modeler.model_interpretation('random_forest')
    
    # Generate report
    report = modeler.generate_report()
    click.echo("\n" + report)
    
    # Save visualizations
    click.echo(f"📈 Saving visualizations to: {output}")
    modeler.plot_results(output)
    
    click.echo("✅ Predictive modeling completed!")


@main.command()
@click.option('--data', '-d', default='data/simulated_customer_data.csv', help='Input data file')
@click.option('--bandit-rounds', default=1000, help='Number of bandit rounds')
@click.option('--rl-timesteps', default=10000, help='Number of RL timesteps')
@click.option('--output', default='optimization_results.html', help='Output visualization file')
def optimize(data: str, bandit_rounds: int, rl_timesteps: int, output: str):
    """Run comprehensive optimization analysis."""
    click.echo("🎯 Running Optimization Analysis...")
    
    if not os.path.exists(data):
        click.echo(f"❌ Data file not found: {data}")
        return
    
    # Load data
    customer_data = pd.read_csv(data)
    
    # Initialize optimization engine
    engine = AdvancedOptimizationEngine(customer_data=customer_data)
    
    # Run optimization
    click.echo("  🎰 Running multi-armed bandit optimization...")
    click.echo("  🤖 Training reinforcement learning models...")
    click.echo("  ⚙️ Optimizing policies...")
    
    engine.run_comprehensive_optimization(
        bandit_rounds=bandit_rounds,
        rl_timesteps=rl_timesteps
    )
    
    # Generate report
    report = engine.generate_optimization_report()
    click.echo("\n" + report)
    
    # Save visualizations
    click.echo(f"📈 Saving visualizations to: {output}")
    engine.plot_optimization_results(output)
    
    click.echo("✅ Optimization analysis completed!")


@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--workers', default=4, help='Number of worker processes')
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the RazorVine API server."""
    click.echo(f"🚀 Starting RazorVine API server on {host}:{port}")
    
    if reload:
        click.echo("🔄 Development mode with auto-reload enabled")
        uvicorn.run("src.api.main:app", host=host, port=port, reload=True)
    else:
        click.echo(f"⚡ Production mode with {workers} workers")
        uvicorn.run("src.api.main:app", host=host, port=port, workers=workers)


@main.command()
@click.option('--data', '-d', default='data/simulated_customer_data.csv', help='Input data file')
@click.option('--output', '-o', default='analysis_report.txt', help='Output report file')
def analyze(data: str, output: str):
    """Run complete end-to-end analysis pipeline."""
    click.echo("🔬 Running Complete Analysis Pipeline...")
    
    if not os.path.exists(data):
        click.echo(f"❌ Data file not found: {data}")
        return
    
    # Load data
    customer_data = pd.read_csv(data)
    
    # Run all analyses
    results = {}
    
    # 1. Causal Analysis
    click.echo("1️⃣ Running Causal Inference...")
    causal_analyzer = AdvancedCausalAnalyzer(
        data=customer_data,
        treatment_col='promotion_response',
        outcome_col='customer_lifetime_value'
    )
    causal_analyzer.propensity_score_matching()
    causal_analyzer.ml_based_causal_inference()
    results['causal'] = causal_analyzer.generate_report()
    
    # 2. Predictive Modeling
    click.echo("2️⃣ Running Predictive Modeling...")
    predictive_modeler = AdvancedPredictiveModeler(
        data=customer_data,
        target_col='customer_lifetime_value',
        problem_type='regression'
    )
    predictive_modeler.train_ensemble_models()
    predictive_modeler.train_deep_learning_model()
    results['predictive'] = predictive_modeler.generate_report()
    
    # 3. Optimization
    click.echo("3️⃣ Running Optimization...")
    optimization_engine = AdvancedOptimizationEngine(customer_data=customer_data)
    optimization_engine.run_comprehensive_optimization()
    results['optimization'] = optimization_engine.generate_optimization_report()
    
    # Generate comprehensive report
    click.echo("📝 Generating comprehensive report...")
    with open(output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("🚀 RAZORVINE COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📊 DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total customers: {len(customer_data):,}\n")
        f.write(f"Features: {len(customer_data.columns)}\n")
        f.write(f"Missing values: {customer_data.isnull().sum().sum()}\n\n")
        
        f.write("🔬 CAUSAL INFERENCE RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(results['causal'] + "\n\n")
        
        f.write("🤖 PREDICTIVE MODELING RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(results['predictive'] + "\n\n")
        
        f.write("🎯 OPTIMIZATION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(results['optimization'] + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("✅ Analysis completed successfully!\n")
        f.write("=" * 80 + "\n")
    
    click.echo(f"✅ Complete analysis saved to: {output}")


@main.command()
@click.option('--data', '-d', default='data/simulated_customer_data.csv', help='Input data file')
def demo(data: str):
    """Run a quick demo of the platform capabilities."""
    click.echo("🎪 Running RazorVine Demo...")
    
    if not os.path.exists(data):
        click.echo("📊 Generating sample data for demo...")
        simulator = AdvancedCustomerSimulator()
        simulator.simulate_customer_data(n_customers=1000)
    
    # Load data
    customer_data = pd.read_csv(data)
    
    # Quick demo of each component
    click.echo("\n🔬 Causal Inference Demo:")
    causal_analyzer = AdvancedCausalAnalyzer(
        data=customer_data.head(1000),  # Use subset for speed
        treatment_col='promotion_response',
        outcome_col='customer_lifetime_value'
    )
    causal_analyzer.propensity_score_matching()
    click.echo("  ✅ Propensity score matching completed")
    
    click.echo("\n🤖 Predictive Modeling Demo:")
    predictive_modeler = AdvancedPredictiveModeler(
        data=customer_data.head(1000),  # Use subset for speed
        target_col='customer_lifetime_value',
        problem_type='regression'
    )
    predictive_modeler.train_ensemble_models()
    click.echo("  ✅ Ensemble models trained")
    
    click.echo("\n🎯 Optimization Demo:")
    optimization_engine = AdvancedOptimizationEngine(
        customer_data=customer_data.head(1000)  # Use subset for speed
    )
    optimization_engine.run_comprehensive_optimization(
        bandit_rounds=100,  # Reduced for demo
        rl_timesteps=1000   # Reduced for demo
    )
    click.echo("  ✅ Optimization completed")
    
    click.echo("\n🎉 Demo completed successfully!")
    click.echo("🚀 Start the API server with: razorvine serve")
    click.echo("📊 Access the platform at: http://localhost:8000")


@main.command()
def info():
    """Display platform information and status."""
    click.echo("🚀 RazorVine Analytics Platform v2.0.0")
    click.echo("=" * 50)
    
    # Check data availability
    data_file = 'data/simulated_customer_data.csv'
    if os.path.exists(data_file):
        data = pd.read_csv(data_file)
        click.echo(f"📊 Data: {len(data):,} customers available")
    else:
        click.echo("📊 Data: Not available (run 'razorvine simulate' to generate)")
    
    # Check Redis connection
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        click.echo("🔴 Redis: Connected")
    except:
        click.echo("🔴 Redis: Not connected")
    
    # Check MLflow
    try:
        import mlflow
        click.echo("📈 MLflow: Available")
    except ImportError:
        click.echo("📈 MLflow: Not installed")
    
    click.echo("\n📚 Available Commands:")
    click.echo("  razorvine simulate    - Generate customer data")
    click.echo("  razorvine causal      - Run causal inference")
    click.echo("  razorvine predict     - Run predictive modeling")
    click.echo("  razorvine optimize    - Run optimization")
    click.echo("  razorvine analyze     - Run complete analysis")
    click.echo("  razorvine serve       - Start API server")
    click.echo("  razorvine demo        - Run quick demo")
    click.echo("  razorvine info        - Show this information")


if __name__ == '__main__':
    main() 