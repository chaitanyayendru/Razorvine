import os
import pandas as pd
from .data_simulator import AdvancedCustomerDataSimulator

def load_data(file_path="data/simulated_customer_data.csv", regenerate=False):
    """
    Load the customer dataset from the specified file.
    If the file does not exist or `regenerate` is True, it will create a new dataset.
    
    Parameters:
        file_path (str): Path to the data file.
        regenerate (bool): Whether to regenerate the data if the file exists.

    Returns:
        pd.DataFrame: The loaded customer data.
    """
    if not os.path.exists(file_path) or regenerate:
        print(f"File not found or regenerate=True. Generating new dataset at {file_path}...")
        simulator = AdvancedCustomerDataSimulator(n_customers=2000, n_days=60, seed=42)
        customers_df, interactions_df, summary_stats = simulator.run_full_simulation()
        customers_df.to_csv(file_path, index=False)
    
    print(f"Loading dataset from {file_path}...")
    return pd.read_csv(file_path)


if __name__ == "__main__":
    # Example usage
    data = load_data()
    print(data.head())
