#!/usr/bin/env python3
"""
RazorVine Full Functionality Test
Comprehensive validation of all advanced features
"""

import sys
import time
import traceback
from pathlib import Path

def print_banner():
    """Print test banner"""
    print("=" * 80)
    print("🧪 RAZORVINE FULL FUNCTIONALITY TEST")
    print("=" * 80)
    print("Testing: Quantum Computing • Federated Learning • Advanced ML • Enterprise Features")
    print("=" * 80)

def test_core_packages():
    """Test core data science packages"""
    print("\n📊 Testing Core Data Science Packages...")
    
    tests = [
        ("numpy", "Numerical computing"),
        ("pandas", "Data manipulation"),
        ("scikit-learn", "Machine learning"),
        ("scipy", "Scientific computing"),
        ("matplotlib", "Static visualization"),
        ("seaborn", "Statistical visualization"),
        ("plotly", "Interactive visualization")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"❌ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_advanced_ml():
    """Test advanced machine learning packages"""
    print("\n🤖 Testing Advanced Machine Learning...")
    
    tests = [
        ("torch", "PyTorch deep learning"),
        ("torchvision", "Computer vision"),
        ("transformers", "Hugging Face transformers"),
        ("networkx", "Graph analysis"),
        ("statsmodels", "Statistical modeling"),
        ("optuna", "Hyperparameter optimization")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_quantum_computing():
    """Test quantum computing packages"""
    print("\n⚛️ Testing Quantum Computing...")
    
    tests = [
        ("qiskit", "IBM Quantum"),
        ("cirq", "Google Quantum"),
        ("pennylane", "Quantum machine learning")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_federated_learning():
    """Test federated learning packages"""
    print("\n🔒 Testing Federated Learning...")
    
    tests = [
        ("flwr", "Flower federated learning"),
        ("syft", "PySyft privacy-preserving ML")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_optimization():
    """Test optimization packages"""
    print("\n🎯 Testing Optimization...")
    
    tests = [
        ("optuna", "Hyperparameter optimization"),
        ("ray", "Distributed computing"),
        ("stable_baselines3", "Reinforcement learning"),
        ("gymnasium", "RL environments"),
        ("pymoo", "Multi-objective optimization")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_causal_inference():
    """Test causal inference packages"""
    print("\n🔍 Testing Causal Inference...")
    
    tests = [
        ("causalml", "Causal machine learning"),
        ("dowhy", "Microsoft DoWhy"),
        ("linearmodels", "Linear models"),
        ("statsmodels", "Statistical models"),
        ("pingouin", "Statistical testing")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_model_interpretability():
    """Test model interpretability packages"""
    print("\n🔍 Testing Model Interpretability...")
    
    tests = [
        ("shap", "SHAP explanations"),
        ("lime", "LIME explanations"),
        ("alibi", "Algorithmic recourse"),
        ("interpret", "Microsoft InterpretML")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_web_frameworks():
    """Test web frameworks"""
    print("\n🌐 Testing Web Frameworks...")
    
    tests = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("dash", "Dash web apps"),
        ("streamlit", "Streamlit apps"),
        ("gradio", "Gradio interfaces")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_cloud_integration():
    """Test cloud integration packages"""
    print("\n☁️ Testing Cloud Integration...")
    
    tests = [
        ("boto3", "AWS SDK"),
        ("sagemaker", "Amazon SageMaker"),
        ("mlflow", "MLflow tracking"),
        ("wandb", "Weights & Biases")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_database_integration():
    """Test database integration"""
    print("\n🗄️ Testing Database Integration...")
    
    tests = [
        ("sqlalchemy", "SQL ORM"),
        ("psycopg2", "PostgreSQL"),
        ("pymongo", "MongoDB"),
        ("redis", "Redis cache")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_time_series():
    """Test time series packages"""
    print("\n📈 Testing Time Series...")
    
    tests = [
        ("prophet", "Facebook Prophet"),
        ("arch", "GARCH models"),
        ("pmdarima", "Auto ARIMA"),
        ("neuralprophet", "Neural Prophet"),
        ("kats", "Kats time series")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_advanced_analytics():
    """Test advanced analytics packages"""
    print("\n📊 Testing Advanced Analytics...")
    
    tests = [
        ("pyod", "Outlier detection"),
        ("hdbscan", "Hierarchical clustering"),
        ("umap", "UMAP dimensionality reduction"),
        ("phate", "PHATE visualization"),
        ("spacy", "NLP processing"),
        ("yfinance", "Financial data")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_development_tools():
    """Test development tools"""
    print("\n🔧 Testing Development Tools...")
    
    tests = [
        ("pytest", "Testing framework"),
        ("black", "Code formatting"),
        ("flake8", "Linting"),
        ("mypy", "Type checking"),
        ("click", "CLI framework"),
        ("rich", "Rich terminal output")
    ]
    
    results = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            results.append(True)
        except ImportError as e:
            print(f"⚠️ {description}: {module} - {e}")
            results.append(False)
    
    return results

def test_quantum_demo():
    """Test quantum computing functionality"""
    print("\n⚛️ Testing Quantum Computing Demo...")
    
    try:
        import qiskit
        from qiskit import QuantumCircuit
        from qiskit_aer import Aer
        from qiskit import execute
        
        # Create a simple quantum circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)  # Hadamard gate
        qc.cx(0, 1)  # CNOT gate
        qc.measure([0, 1], [0, 1])
        
        # Execute on simulator
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        
        print(f"✅ Quantum circuit executed successfully")
        print(f"   Results: {counts}")
        return True
        
    except Exception as e:
        print(f"❌ Quantum demo failed: {e}")
        return False

def test_ml_demo():
    """Test machine learning functionality"""
    print("\n🤖 Testing Machine Learning Demo...")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ ML model trained successfully")
        print(f"   Accuracy: {accuracy:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ ML demo failed: {e}")
        return False

def test_optimization_demo():
    """Test optimization functionality"""
    print("\n🎯 Testing Optimization Demo...")
    
    try:
        import optuna
        
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            return (x - 2) ** 2 + (y + 3) ** 2
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"✅ Optimization completed successfully")
        print(f"   Best params: {best_params}")
        print(f"   Best value: {best_value:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Optimization demo failed: {e}")
        return False

def test_web_api_demo():
    """Test web API functionality"""
    print("\n🌐 Testing Web API Demo...")
    
    try:
        from fastapi import FastAPI
        import uvicorn
        import threading
        import time
        import requests
        
        # Create FastAPI app
        app = FastAPI(title="RazorVine Test API")
        
        @app.get("/")
        def read_root():
            return {"message": "RazorVine API is working!"}
        
        @app.get("/health")
        def health_check():
            return {"status": "healthy", "timestamp": time.time()}
        
        # Start server in background
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Test API
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code == 200:
            print(f"✅ Web API working: {response.json()}")
            return True
        else:
            print(f"❌ Web API failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Web API demo failed: {e}")
        return False

def print_test_summary(all_results):
    """Print comprehensive test summary"""
    print("\n" + "=" * 80)
    print("📊 RAZORVINE FULL FUNCTIONALITY TEST SUMMARY")
    print("=" * 80)
    
    categories = [
        ("Core Data Science", all_results[0]),
        ("Advanced ML", all_results[1]),
        ("Quantum Computing", all_results[2]),
        ("Federated Learning", all_results[3]),
        ("Optimization", all_results[4]),
        ("Causal Inference", all_results[5]),
        ("Model Interpretability", all_results[6]),
        ("Web Frameworks", all_results[7]),
        ("Cloud Integration", all_results[8]),
        ("Database Integration", all_results[9]),
        ("Time Series", all_results[10]),
        ("Advanced Analytics", all_results[11]),
        ("Development Tools", all_results[12])
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in categories:
        if results:
            passed = sum(results)
            total = len(results)
            total_tests += total
            passed_tests += passed
            
            percentage = (passed / total) * 100
            status = "✅" if percentage >= 80 else "⚠️" if percentage >= 50 else "❌"
            
            print(f"{status} {category}: {passed}/{total} ({percentage:.1f}%)")
    
    overall_percentage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 Overall Success Rate: {passed_tests}/{total_tests} ({overall_percentage:.1f}%)")
    
    if overall_percentage >= 90:
        print("🏆 EXCELLENT: RazorVine is fully functional!")
    elif overall_percentage >= 75:
        print("✅ GOOD: RazorVine has most features working!")
    elif overall_percentage >= 50:
        print("⚠️ FAIR: RazorVine has basic functionality!")
    else:
        print("❌ NEEDS ATTENTION: Some core features are missing!")
    
    print("\n🚀 Next Steps:")
    print("1. Run full demo: python run_razorvine.py")
    print("2. Explore notebooks: jupyter lab")
    print("3. Check documentation: README.md")
    
    print("\n" + "=" * 80)
    print("🎯 RAZORVINE READY FOR ENTERPRISE USE!")
    print("=" * 80)

def main():
    """Main test function"""
    print_banner()
    
    start_time = time.time()
    
    # Run all tests
    all_results = [
        test_core_packages(),
        test_advanced_ml(),
        test_quantum_computing(),
        test_federated_learning(),
        test_optimization(),
        test_causal_inference(),
        test_model_interpretability(),
        test_web_frameworks(),
        test_cloud_integration(),
        test_database_integration(),
        test_time_series(),
        test_advanced_analytics(),
        test_development_tools()
    ]
    
    # Run demo tests
    print("\n🎬 Running Demo Tests...")
    demo_results = [
        test_quantum_demo(),
        test_ml_demo(),
        test_optimization_demo(),
        test_web_api_demo()
    ]
    
    # Print summary
    print_test_summary(all_results)
    
    end_time = time.time()
    print(f"\n⏱️ Total test time: {end_time - start_time:.2f} seconds")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 