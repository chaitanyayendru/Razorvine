"""
Setup script for RazorVine - Advanced Customer Analytics Platform
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="razorvine",
    version="2.0.0",
    author="Chaitanya Sai Chandu",
    author_email="chaitanya@razorvine.com",
    description="Advanced Customer Analytics Platform with Causal Inference and Real-time Optimization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/razorvine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "mypy>=1.8.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "deploy": [
            "gunicorn>=21.2.0",
            "uvicorn[standard]>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "razorvine=src.cli:main",
            "razorvine-api=src.api.main:main",
            "razorvine-simulate=src.data_pipeline.data_simulator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "razorvine": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "customer analytics",
        "causal inference",
        "machine learning",
        "optimization",
        "reinforcement learning",
        "multi-armed bandits",
        "predictive modeling",
        "real-time analytics",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/razorvine/issues",
        "Source": "https://github.com/yourusername/razorvine",
        "Documentation": "https://razorvine.readthedocs.io/",
        "Changelog": "https://github.com/yourusername/razorvine/blob/main/CHANGELOG.md",
    },
)
