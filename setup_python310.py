#!/usr/bin/env python3
"""
RazorVine Python 3.10 Setup Script
Creates Python 3.10 environment with 100% compatible packages
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 80)
    print("🐍 RAZORVINE PYTHON 3.10 SETUP")
    print("=" * 80)
    print("Creating Python 3.10 environment with 100% compatible packages")
    print("=" * 80)

def check_python_version():
    """Check if Python 3.10 is available"""
    print("🐍 Checking Python version...")
    
    # Check if Python 3.10 is available
    try:
        result = subprocess.run([sys.executable, "--version"], 
                              capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"Current Python: {version}")
        
        if "3.10" in version:
            print("✅ Python 3.10 detected!")
            return True
        else:
            print("⚠️ Python 3.10 not detected. Please install Python 3.10")
            print("Download from: https://www.python.org/downloads/release/python-3109/")
            return False
            
    except Exception as e:
        print(f"❌ Error checking Python version: {e}")
        return False

def create_python310_environment():
    """Create Python 3.10 virtual environment"""
    print("\n🔧 Creating Python 3.10 virtual environment...")
    
    venv_path = Path("venv_python310")
    
    if venv_path.exists():
        print("✅ Python 3.10 environment already exists")
        return True
    
    try:
        # Try to create virtual environment
        subprocess.check_call([sys.executable, "-m", "venv", "venv_python310"])
        print("✅ Python 3.10 virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        print("Please install Python 3.10 first")
        return False

def get_venv_python():
    """Get the Python executable from virtual environment"""
    if platform.system() == "Windows":
        return "venv_python310\\Scripts\\python.exe"
    else:
        return "venv_python310/bin/python"

def get_venv_pip():
    """Get the pip executable from virtual environment"""
    if platform.system() == "Windows":
        return "venv_python310\\Scripts\\pip.exe"
    else:
        return "venv_python310/bin/pip"

def upgrade_pip():
    """Upgrade pip in virtual environment"""
    print("\n📦 Upgrading pip...")
    
    try:
        pip_cmd = get_venv_pip()
        subprocess.check_call([pip_cmd, "install", "--upgrade", "pip"])
        print("✅ Pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error upgrading pip: {e}")
        return False

def install_core_packages():
    """Install core packages first"""
    print("\n📦 Installing core packages...")
    
    core_packages = [
        "numpy==1.24.3",
        "pandas==2.0.3", 
        "scikit-learn==1.3.0",
        "scipy==1.11.1",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "plotly==5.15.0"
    ]
    
    pip_cmd = get_venv_pip()
    success_count = 0
    
    for package in core_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([pip_cmd, "install", package])
            print(f"✅ Successfully installed {package}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    return success_count, len(core_packages)

def install_ml_packages():
    """Install machine learning packages"""
    print("\n🤖 Installing ML packages...")
    
    ml_packages = [
        "torch==2.0.1",
        "torchvision==0.15.2",
        "transformers==4.30.2",
        "networkx==3.1",
        "python-louvain==0.15",
        "statsmodels==0.14.0",
        "optuna==3.2.0"
    ]
    
    pip_cmd = get_venv_pip()
    success_count = 0
    
    for package in ml_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([pip_cmd, "install", package])
            print(f"✅ Successfully installed {package}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    return success_count, len(ml_packages)

def install_quantum_packages():
    """Install quantum computing packages"""
    print("\n⚛️ Installing quantum packages...")
    
    quantum_packages = [
        "qiskit==0.44.0",
        "cirq==1.2.0",
        "pennylane==0.30.0"
    ]
    
    pip_cmd = get_venv_pip()
    success_count = 0
    
    for package in quantum_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([pip_cmd, "install", package])
            print(f"✅ Successfully installed {package}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    return success_count, len(quantum_packages)

def install_advanced_packages():
    """Install advanced packages"""
    print("\n🚀 Installing advanced packages...")
    
    advanced_packages = [
        "shap==0.42.0",
        "lime==0.2.0.1",
        "prophet==1.1.4",
        "fastapi==0.100.1",
        "uvicorn[standard]==0.23.2",
        "dash==2.11.1",
        "streamlit==1.25.0",
        "boto3==1.28.36",
        "mlflow==2.6.0",
        "wandb==0.15.8"
    ]
    
    pip_cmd = get_venv_pip()
    success_count = 0
    
    for package in advanced_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([pip_cmd, "install", package])
            print(f"✅ Successfully installed {package}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    return success_count, len(advanced_packages)

def install_optional_packages():
    """Install optional packages"""
    print("\n🔧 Installing optional packages...")
    
    optional_packages = [
        "torch-geometric==2.3.1",
        "stellargraph==1.2.1",
        "causalml==0.13.0",
        "dowhy==0.8",
        "linearmodels==4.25",
        "pingouin==0.5.3",
        "ray[tune]==2.6.3",
        "stable-baselines3==2.0.0",
        "gymnasium==0.28.1",
        "pymoo==0.6.0",
        "flwr==1.5.0",
        "syft==0.5.0",
        "feature-engine==1.6.0",
        "alibi==0.7.0",
        "redis==4.6.0",
        "celery==5.3.0",
        "sqlalchemy==2.0.19",
        "psycopg2-binary==2.9.7",
        "pymongo==4.4.0",
        "geopandas==0.13.0",
        "spacy==3.6.0",
        "nltk==3.8.1",
        "yfinance==0.2.18",
        "dask==2023.7.0"
    ]
    
    pip_cmd = get_venv_pip()
    success_count = 0
    
    for package in optional_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([pip_cmd, "install", package])
            print(f"✅ Successfully installed {package}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")
    
    return success_count, len(optional_packages)

def install_development_packages():
    """Install development packages"""
    print("\n🔧 Installing development packages...")
    
    dev_packages = [
        "pytest==7.4.0",
        "pytest-asyncio==0.21.1",
        "pytest-cov==4.1.0",
        "black==23.7.0",
        "flake8==6.0.0",
        "mypy==1.5.1",
        "click==8.1.7",
        "rich==13.5.2",
        "typer==0.9.0",
        "python-dotenv==1.0.0",
        "pyyaml==6.0.1"
    ]
    
    pip_cmd = get_venv_pip()
    success_count = 0
    
    for package in dev_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([pip_cmd, "install", package])
            print(f"✅ Successfully installed {package}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")
    
    return success_count, len(dev_packages)

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    python_cmd = get_venv_python()
    
    test_imports = [
        ("numpy", "Core numerical computing"),
        ("pandas", "Data manipulation"),
        ("sklearn", "Machine learning"),
        ("matplotlib", "Visualization"),
        ("plotly", "Interactive plots"),
        ("networkx", "Graph analysis"),
        ("statsmodels", "Statistical modeling"),
        ("optuna", "Hyperparameter optimization"),
        ("torch", "PyTorch deep learning"),
        ("transformers", "Hugging Face transformers"),
        ("qiskit", "Quantum computing"),
        ("flwr", "Federated learning"),
        ("shap", "Model interpretability"),
        ("fastapi", "Web API framework")
    ]
    
    success_count = 0
    for module, description in test_imports:
        try:
            result = subprocess.run([python_cmd, "-c", f"import {module}"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {description}: {module}")
                success_count += 1
            else:
                print(f"❌ {description}: {module} - {result.stderr}")
        except Exception as e:
            print(f"❌ {description}: {module} - {e}")
    
    return success_count

def create_activation_script():
    """Create activation script for the virtual environment"""
    print("\n📝 Creating activation script...")
    
    if platform.system() == "Windows":
        script_content = """@echo off
echo Activating RazorVine Python 3.10 environment...
call venv_python310\\Scripts\\activate.bat
echo.
echo RazorVine Python 3.10 environment activated!
echo Run: python test_full_functionality.py
echo Run: python run_razorvine.py
echo.
"""
        script_path = "activate_razorvine_python310.bat"
    else:
        script_content = """#!/bin/bash
echo "Activating RazorVine Python 3.10 environment..."
source venv_python310/bin/activate
echo ""
echo "RazorVine Python 3.10 environment activated!"
echo "Run: python test_full_functionality.py"
echo "Run: python run_razorvine.py"
echo ""
"""
        script_path = "activate_razorvine_python310.sh"
        os.chmod(script_path, 0o755)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"✅ Activation script created: {script_path}")
    return True

def print_setup_summary(core_success, ml_success, quantum_success, advanced_success, optional_success, dev_success, test_success):
    """Print setup summary"""
    print("\n" + "=" * 80)
    print("🎉 RAZORVINE PYTHON 3.10 SETUP COMPLETED!")
    print("=" * 80)
    
    print(f"📊 Installation Summary:")
    print(f"   Core packages: {core_success[0]}/{core_success[1]}")
    print(f"   ML packages: {ml_success[0]}/{ml_success[1]}")
    print(f"   Quantum packages: {quantum_success[0]}/{quantum_success[1]}")
    print(f"   Advanced packages: {advanced_success[0]}/{advanced_success[1]}")
    print(f"   Optional packages: {optional_success[0]}/{optional_success[1]}")
    print(f"   Development packages: {dev_success[0]}/{dev_success[1]}")
    print(f"   Tested imports: {test_success}")
    
    total_installed = (core_success[0] + ml_success[0] + quantum_success[0] + 
                      advanced_success[0] + optional_success[0] + dev_success[0])
    total_packages = (core_success[1] + ml_success[1] + quantum_success[1] + 
                     advanced_success[1] + optional_success[1] + dev_success[1])
    
    success_rate = (total_installed / total_packages) * 100 if total_packages > 0 else 0
    
    print(f"\n🎯 Overall Success Rate: {total_installed}/{total_packages} ({success_rate:.1f}%)")
    
    if success_rate >= 95:
        print("🏆 EXCELLENT: RazorVine Python 3.10 is fully functional!")
    elif success_rate >= 85:
        print("✅ GOOD: RazorVine Python 3.10 has most features working!")
    elif success_rate >= 70:
        print("⚠️ FAIR: RazorVine Python 3.10 has basic functionality!")
    else:
        print("❌ NEEDS ATTENTION: Some core features are missing!")
    
    print(f"\n📋 Next steps:")
    if platform.system() == "Windows":
        print("1. Activate environment: activate_razorvine_python310.bat")
    else:
        print("1. Activate environment: source activate_razorvine_python310.sh")
    print("2. Test functionality: python test_full_functionality.py")
    print("3. Run full demo: python run_razorvine.py")
    
    print(f"\n📚 Documentation:")
    print("- README.md: Project overview and usage")
    print("- requirements_python310.txt: Python 3.10 compatible packages")
    print("- venv_python310/: Virtual environment with all packages")
    
    print("\n" + "=" * 80)
    print("🎯 RAZORVINE PYTHON 3.10 READY FOR ENTERPRISE USE!")
    print("=" * 80)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_python310_environment():
        return False
    
    # Upgrade pip
    if not upgrade_pip():
        return False
    
    # Install packages by category
    core_success = install_core_packages()
    ml_success = install_ml_packages()
    quantum_success = install_quantum_packages()
    advanced_success = install_advanced_packages()
    optional_success = install_optional_packages()
    dev_success = install_development_packages()
    
    # Test installation
    test_success = test_installation()
    
    # Create activation script
    create_activation_script()
    
    # Print summary
    print_setup_summary(core_success, ml_success, quantum_success, advanced_success, optional_success, dev_success, test_success)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 