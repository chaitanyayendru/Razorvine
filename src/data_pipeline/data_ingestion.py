import os
import pandas as pd
from .data_simulator import simulate_customer_data

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
        simulate_customer_data(save_path=file_path)
    
    print(f"Loading dataset from {file_path}...")
    return pd.read_csv(file_path)


if __name__ == "__main__":
    # Example usage
    data = load_data()
    print(data.head())
