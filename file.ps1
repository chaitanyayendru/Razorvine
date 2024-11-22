# Define directories and files
$dirs = @(
    "C:/Users/gayat/Razorvine/data/raw",
    "C:/Users/gayat/Razorvine/data/processed",
    "C:/Users/gayat/Razorvine/notebooks",
    "C:/Users/gayat/Razorvine/src",
    "C:/Users/gayat/Razorvine/src/data_pipeline",
    "C:/Users/gayat/Razorvine/src/causal_inference",
    "C:/Users/gayat/Razorvine/src/predictive_modeling",
    "C:/Users/gayat/Razorvine/src/optimization",
    "C:/Users/gayat/Razorvine/src/utils",
    "C:/Users/gayat/Razorvine/tests",
    "C:/Users/gayat/Razorvine/aws",
    "C:/Users/gayat/Razorvine/aws/lambda_functions"
)

$files = @(
    "C:/Users/gayat/Razorvine/data/simulated_customer_data.csv",
    "C:/Users/gayat/Razorvine/notebooks/01_data_exploration.ipynb",
    "C:/Users/gayat/Razorvine/notebooks/02_causal_inference.ipynb",
    "C:/Users/gayat/Razorvine/notebooks/03_predictive_modelling.ipynb",
    "C:/Users/gayat/Razorvine/notebooks/04_optimization.ipynb",
    "C:/Users/gayat/Razorvine/src/__init__.py",
    "C:/Users/gayat/Razorvine/src/data_pipeline/__init__.py",
    "C:/Users/gayat/Razorvine/src/data_pipeline/data_ingestion.py",
    "C:/Users/gayat/Razorvine/src/data_pipeline/data_preprocessing.py",
    "C:/Users/gayat/Razorvine/src/causal_inference/__init__.py",
    "C:/Users/gayat/Razorvine/src/causal_inference/causal_analysis.py",
    "C:/Users/gayat/Razorvine/src/predictive_modeling/__init__.py",
    "C:/Users/gayat/Razorvine/src/predictive_modeling/model_training.py",
    "C:/Users/gayat/Razorvine/src/predictive_modeling/model_evaluation.py",
    "C:/Users/gayat/Razorvine/src/optimization/__init__.py",
    "C:/Users/gayat/Razorvine/src/optimization/optimization_engine.py",
    "C:/Users/gayat/Razorvine/src/optimization/reinforcement_learning.py",
    "C:/Users/gayat/Razorvine/src/utils/__init__.py",
    "C:/Users/gayat/Razorvine/src/utils/data_utils.py",
    "C:/Users/gayat/Razorvine/src/utils/visualization_utils.py",
    "C:/Users/gayat/Razorvine/tests/__init__.py",
    "C:/Users/gayat/Razorvine/tests/test_data_pipeline.py",
    "C:/Users/gayat/Razorvine/tests/test_causal_inference.py",
    "C:/Users/gayat/Razorvine/tests/test_predictive_modeling.py",
    "C:/Users/gayat/Razorvine/tests/test_optimization.py",
    "C:/Users/gayat/Razorvine/aws/s3_upload.py",
    "C:/Users/gayat/Razorvine/aws/sagemaker_training.py",
    "C:/Users/gayat/Razorvine/requirements.txt",
    "C:/Users/gayat/Razorvine/.gitignore",
    "C:/Users/gayat/Razorvine/README.md",
    "C:/Users/gayat/Razorvine/setup.py"
)

# Create directories
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force
}

# Create files
foreach ($file in $files) {
    New-Item -ItemType File -Path $file -Force
}

Write-Host "File structure created successfully."
