import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


def random_date(start, end):
    """Generate a random date between `start` and `end`."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def simulate_customer_data(n_customers=1000, save_path="data/simulated_customer_data.csv"):
    """Simulate customer data with demographics, behaviors, and external factors."""
    np.random.seed(42)

    data = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "age": np.random.randint(18, 70, size=n_customers),
        "income": np.random.normal(50000, 15000, size=n_customers).round(2),
        "education_level": np.random.choice(
            ["High School", "Bachelor's", "Master's", "PhD"], size=n_customers
        ),
        "employment_status": np.random.choice(
            ["Employed", "Unemployed", "Self-Employed"], size=n_customers
        ),
        "marital_status": np.random.choice(
            ["Single", "Married", "Divorced"], size=n_customers
        ),
    })

    data["monthly_spend"] = np.random.normal(500, 150, size=n_customers).round(2)
    data["loyalty_score"] = np.random.uniform(0, 1, size=n_customers).round(2)
    data["online_activity"] = np.random.randint(1, 50, size=n_customers)

    data["region"] = np.random.choice(["Urban", "Suburban", "Rural"], size=n_customers)
    data["economic_trend"] = np.random.uniform(-1, 1, size=n_customers).round(2)

    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    data["last_purchase_date"] = [
        random_date(start_date, end_date) for _ in range(n_customers)
    ]
    data["time_since_last_purchase"] = [
        (datetime.now() - last_date).days for last_date in data["last_purchase_date"]
    ]
    data["promotion_timeframe"] = [
        random.randint(5, 30) for _ in range(n_customers)
    ]

    data["promotion_type"] = np.random.choice(
        ["Discount", "Cashback", "Free Trial"], size=n_customers
    )
    data["promotion_duration"] = np.random.randint(1, 30, size=n_customers)
    data["multiple_treatments"] = np.random.choice([0, 1], size=n_customers)

    data["adoption_level"] = np.random.choice(
        ["None", "Partial", "Full"], size=n_customers
    )
    data["churn_risk"] = np.random.uniform(0, 1, size=n_customers).round(2)
    data["customer_lifetime_value"] = np.random.normal(5000, 2000, size=n_customers).round(2)

    data["income"] += np.random.normal(0, 2000, size=n_customers)
    data["monthly_spend"] += np.random.normal(0, 50, size=n_customers)

    missing_indices = np.random.choice(data.index, size=int(0.1 * n_customers), replace=False)
    data.loc[missing_indices, "income"] = np.nan
    data.loc[missing_indices, "education_level"] = np.nan

    data.to_csv(save_path, index=False)
    print(f"Dataset created and saved as '{save_path}'.")


if __name__ == "__main__":
    simulate_customer_data()
