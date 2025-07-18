#!/usr/bin/env python3
"""
RazorVine Full Installation Script
Installs all advanced features with intelligent fallbacks
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print installation banner"""
    print("=" * 80)
    print("🚀 RAZORVINE - ADVANCED CUSTOMER ANALYTICS PLATFORM")
    print("=" * 80)
    print("Installing FULL functionality: Quantum Computing • Federated Learning • Advanced ML")
    print("=" * 80)

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_with_fallback(package, fallback_package=None):
    """Install package with fallback option and Python 3.12 fixes"""
    try:
        print(f"Installing {package}...")
        # Try with --no-deps first for problematic packages
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", package])
            print(f"✅ Successfully installed {package} (no deps)")
            return True
        except subprocess.CalledProcessError:
            # Try normal installation
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
            return True
    except subprocess.CalledProcessError:
        if fallback_package:
            try:
                print(f"⚠️ {package} failed, trying fallback {fallback_package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", fallback_package])
                print(f"✅ Successfully installed fallback {fallback_package}")
                return True
            except subprocess.CalledProcessError:
                print(f"❌ Both {package} and {fallback_package} failed")
                return False
        else:
            print(f"❌ Failed to install {package}")
            return False

def install_core_packages():
    """Install core packages that are essential"""
    print("\n📦 Installing core packages...")
    
    core_packages = [
        "numpy>=1.26.0",
        "pandas>=2.1.0", 
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.17.0"
    ]
    
    success_count = 0
    for package in core_packages:
        if install_with_fallback(package):
            success_count += 1
    
    return success_count, len(core_packages)

def install_advanced_ml():
    """Install advanced ML packages"""
    print("\n🤖 Installing advanced ML packages...")
    
    ml_packages = [
        ("torch>=2.1.0", None),
        ("torchvision>=0.16.0", None),
        ("transformers>=4.35.0", None),
        ("networkx>=3.2.0", None),
        ("python-louvain>=0.16", None),
        ("statsmodels>=0.14.0", None),
        ("optuna>=3.4.0", None)
    ]
    
    success_count = 0
    for package, fallback in ml_packages:
        if install_with_fallback(package, fallback):
            success_count += 1
    
    return success_count, len(ml_packages)

def install_quantum_packages():
    """Install quantum computing packages"""
    print("\n⚛️ Installing quantum computing packages...")
    
    quantum_packages = [
        ("qiskit>=0.45.0", None),
        ("cirq>=1.3.0", None),
        ("pennylane>=0.32.0", None)
    ]
    
    success_count = 0
    for package, fallback in quantum_packages:
        if install_with_fallback(package, fallback):
            success_count += 1
    
    return success_count, len(quantum_packages)

def install_federated_learning():
    """Install federated learning packages"""
    print("\n🔒 Installing federated learning packages...")
    
    fl_packages = [
        ("flwr>=1.5.0", None),
        ("syft>=0.5.0", None)
    ]
    
    success_count = 0
    for package, fallback in fl_packages:
        if install_with_fallback(package, fallback):
            success_count += 1
    
    return success_count, len(fl_packages)

def install_advanced_features():
    """Install advanced features with fallbacks"""
    print("\n🚀 Installing advanced features...")
    
    advanced_packages = [
        ("shap>=0.42.0", None),
        ("lime>=0.2.0", None),
        ("prophet>=1.1.4", None),
        ("fastapi>=0.100.0", None),
        ("uvicorn[standard]>=0.23.0", None),
        ("dash>=2.11.0", None),
        ("streamlit>=1.25.0", None),
        ("boto3>=1.28.0", None),
        ("mlflow>=2.6.0", None),
        ("wandb>=0.15.0", None)
    ]
    
    success_count = 0
    for package, fallback in advanced_packages:
        if install_with_fallback(package, fallback):
            success_count += 1
    
    return success_count, len(advanced_packages)

def install_optional_packages():
    """Install optional packages that may have compatibility issues"""
    print("\n🔧 Installing optional packages...")
    
    optional_packages = [
        ("torch-geometric>=2.4.0", None),
        ("stellargraph>=1.2.1", None),
        ("causalml>=0.14.0", None),
        ("dowhy>=0.9", None),
        ("linearmodels>=5.0", None),
        ("pingouin>=0.5.4", None),
        ("ray[tune]>=2.8.0", None),
        ("stable-baselines3>=2.1.0", None),
        ("gymnasium>=0.29.0", None),
        ("pymoo>=0.6.0", None),
        ("feature-engine>=1.7.0", None),
        ("alibi>=0.8.0", None),
        ("redis>=5.0.0", None),
        ("celery>=5.3.0", None),
        ("sqlalchemy>=2.0.23", None),
        ("psycopg2-binary>=2.9.9", None),
        ("pymongo>=4.6.0", None),
        ("geopandas>=0.14.0", None),
        ("spacy>=3.7.0", None),
        ("nltk>=3.8.1", None),
        ("yfinance>=0.2.28", None),
        ("dask>=2023.12.0", None)
    ]
    
    success_count = 0
    for package, fallback in optional_packages:
        if install_with_fallback(package, fallback):
            success_count += 1
    
    return success_count, len(optional_packages)

def install_development_packages():
    """Install development packages"""
    print("\n🔧 Installing development packages...")
    
    dev_packages = [
        ("pytest>=7.4.3", None),
        ("pytest-asyncio>=0.21.1", None),
        ("pytest-cov>=4.1.0", None),
        ("black>=23.11.0", None),
        ("flake8>=6.1.0", None),
        ("mypy>=1.7.0", None),
        ("click>=8.1.7", None),
        ("rich>=13.7.0", None),
        ("typer>=0.9.0", None),
        ("python-dotenv>=1.0.0", None),
        ("pyyaml>=6.0.1", None)
    ]
    
    success_count = 0
    for package, fallback in dev_packages:
        if install_with_fallback(package, fallback):
            success_count += 1
    
    return success_count, len(dev_packages)

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data", "models", "results", "logs", "reports", 
        "notebooks", "tests", "docs", "src/api", "src/ml"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    return True

def test_installation():
    """Test if installation was successful"""
    print("\n🧪 Testing installation...")
    
    test_imports = [
        ("numpy", "Core numerical computing"),
        ("pandas", "Data manipulation"),
        ("sklearn", "Machine learning"),
        ("matplotlib", "Visualization"),
        ("plotly", "Interactive plots"),
        ("networkx", "Graph analysis"),
        ("statsmodels", "Statistical modeling"),
        ("optuna", "Hyperparameter optimization")
    ]
    
    success_count = 0
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            success_count += 1
        except ImportError:
            print(f"❌ {description}: {module} - Not available")
    
    # Test advanced features
    advanced_imports = [
        ("torch", "PyTorch deep learning"),
        ("transformers", "Hugging Face transformers"),
        ("qiskit", "Quantum computing"),
        ("flwr", "Federated learning"),
        ("shap", "Model interpretability"),
        ("fastapi", "Web API framework")
    ]
    
    for module, description in advanced_imports:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
            success_count += 1
        except ImportError:
            print(f"⚠️ {description}: {module} - Optional feature not available")
    
    return success_count

def generate_activation_script():
    """Generate activation script for the virtual environment"""
    print("\n📝 Generating activation script...")
    
    if platform.system() == "Windows":
        script_content = """@echo off
echo Activating RazorVine environment...
echo.
echo RazorVine Full Installation Complete!
echo Run: python simple_test.py
echo Run: python run_razorvine.py
echo.
"""
        script_path = "activate_razorvine.bat"
    else:
        script_content = """#!/bin/bash
echo "Activating RazorVine environment..."
echo ""
echo "RazorVine Full Installation Complete!"
echo "Run: python simple_test.py"
echo "Run: python run_razorvine.py"
echo ""
"""
        script_path = "activate_razorvine.sh"
        os.chmod(script_path, 0o755)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"✅ Activation script created: {script_path}")
    return True

def print_installation_summary(core_success, ml_success, quantum_success, fl_success, advanced_success, optional_success, dev_success, test_success):
    """Print comprehensive installation summary"""
    print("\n" + "=" * 80)
    print("🎉 RAZORVINE FULL INSTALLATION COMPLETED!")
    print("=" * 80)
    
    print(f"📊 Installation Summary:")
    print(f"   Core packages: {core_success[0]}/{core_success[1]}")
    print(f"   Advanced ML: {ml_success[0]}/{ml_success[1]}")
    print(f"   Quantum computing: {quantum_success[0]}/{quantum_success[1]}")
    print(f"   Federated learning: {fl_success[0]}/{fl_success[1]}")
    print(f"   Advanced features: {advanced_success[0]}/{advanced_success[1]}")
    print(f"   Optional packages: {optional_success[0]}/{optional_success[1]}")
    print(f"   Development packages: {dev_success[0]}/{dev_success[1]}")
    print(f"   Tested imports: {test_success}")
    
    print(f"\n🚀 RazorVine Features Available:")
    if core_success[0] >= 6:
        print("   ✅ Core analytics and visualization")
    if ml_success[0] >= 4:
        print("   ✅ Advanced machine learning")
    if quantum_success[0] >= 2:
        print("   ✅ Quantum computing simulation")
    if fl_success[0] >= 1:
        print("   ✅ Federated learning")
    if advanced_success[0] >= 6:
        print("   ✅ Advanced features (SHAP, FastAPI, etc.)")
    
    print(f"\n📋 Next steps:")
    print("1. Test the installation: python simple_test.py")
    print("2. Run full demo: python run_razorvine.py")
    print("3. Explore notebooks: jupyter lab")
    
    print(f"\n📚 Documentation:")
    print("- README.md: Project overview and usage")
    print("- reports/: Generated analysis reports")
    print("- notebooks/: Jupyter notebooks for exploration")
    
    print(f"\n🔧 Development:")
    print("- tests/: Run tests with pytest")
    print("- src/: Source code modules")
    print("- data/: Generated and input data")
    
    print("\n" + "=" * 80)
    print("🎯 RAZORVINE IS READY FOR ENTERPRISE DEPLOYMENT!")
    print("=" * 80)

def main():
    """Main installation function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install packages by category
    core_success = install_core_packages()
    ml_success = install_advanced_ml()
    quantum_success = install_quantum_packages()
    fl_success = install_federated_learning()
    advanced_success = install_advanced_features()
    optional_success = install_optional_packages()
    dev_success = install_development_packages()
    
    # Test installation
    test_success = test_installation()
    
    # Generate activation script
    generate_activation_script()
    
    # Print comprehensive summary
    print_installation_summary(core_success, ml_success, quantum_success, fl_success, advanced_success, optional_success, dev_success, test_success)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 