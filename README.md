# RazorVine - Customer Analytics Project

## Overview
This project simulates customer behavior and evaluates the impact of marketing promotions using **causal inference**, **predictive modeling**, and **optimization techniques**. The project also integrates AWS services for scalable data processing and model deployment.

## Key Features
1. **Simulated Real-World Data**:
   - Demographic, behavioral, and external factors.
   - Temporal dynamics and multiple treatment effects.
2. **Causal Inference**:
   - Understand the causal effect of promotions on customer behavior.
3. **Predictive Modeling**:
   - Predict customer churn risk and lifetime value.
4. **Optimization**:
   - Use reinforcement learning to optimize promotion strategies.
5. **AWS Integration**:
   - Scalable model training and deployment using AWS SageMaker and Lambda.

---

## Folder Structure
- `data/`: Raw and processed datasets.
- `notebooks/`: Jupyter notebooks for analysis.
- `src/`: Python source code for all major components.
  - `data_pipeline/`: Data ingestion and preprocessing scripts.
  - `causal_inference/`: Scripts for causal analysis.
  - `predictive_modeling/`: Machine learning models and evaluation.
  - `optimization/`: Optimization algorithms and strategies.
  - `utils/`: Helper functions for data processing and visualization.
- `tests/`: Unit tests for all components.
- `aws/`: Scripts for AWS integration.

---

## Getting Started

### Prerequisites
1. **Python 3.8+** installed on your system.
2. Install project dependencies:
   ```bash
   pip install -r requirements.txt

(Optional) AWS CLI configured for your account if integrating with AWS.
Dataset
A simulated dataset is provided in data/simulated_customer_data.csv.
To generate a new dataset, run:

    python src/data_pipeline/data_ingestion.py

Running the Project
Data Exploration:
    Use the notebook: notebooks/01_data_exploration.ipynb.

Causal Analysis:
    Scripts in src/causal_inference/ analyze the impact of promotions.
Run:
    python src/causal_inference/causal_analysis.py


Predictive Modeling:
Train and evaluate models using:
    python src/predictive_modeling/model_training.py

Optimization:
Apply reinforcement learning strategies:
    python src/optimization/optimization_engine.py

AWS Integration
    Upload Data to S3:
    Use aws/s3_upload.py to upload datasets to your S3 bucket.
Model Training on SageMaker:
Train models on AWS SageMaker using aws/sagemaker_training.py.

Testing
    Run unit tests:
    pytest tests/


## Future Scope
Real-world dataset integration.
Extend optimization to multi-armed bandits or advanced RL algorithms.
Expand AWS integration with auto-scaling features.
Contributors
Chaitanya Sai Chandu (Project Lead)
License
This project is licensed under the MIT License.



