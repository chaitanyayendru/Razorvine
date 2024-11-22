import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def preprocess_data(data):
    """
    Preprocess the customer dataset:
    - Handle missing values
    - Normalize numerical data
    - One-hot encode categorical features
    
    Parameters:
        data (pd.DataFrame): Raw customer data.
    
    Returns:
        pd.DataFrame: Preprocessed data ready for analysis.
    """
    # Handle missing values
    imputer = SimpleImputer(strategy="most_frequent")
    data["income"] = imputer.fit_transform(data[["income"]])
    data["education_level"] = imputer.fit_transform(data[["education_level"]])

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_columns = ["age", "income", "monthly_spend", "loyalty_score", "online_activity"]
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # One-hot encode categorical features
    categorical_columns = ["education_level", "employment_status", "marital_status", "region", "promotion_type"]
    encoder = OneHotEncoder(sparse=False, drop="first")
    encoded = pd.DataFrame(
        encoder.fit_transform(data[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns),
        index=data.index
    )

    # Combine numerical and encoded features
    data = pd.concat([data.drop(columns=categorical_columns), encoded], axis=1)

    return data


if __name__ == "__main__":
    # Example usage
    from .data_ingestion import load_data
    
    raw_data = load_data()
    preprocessed_data = preprocess_data(raw_data)
    print(preprocessed_data.head())
