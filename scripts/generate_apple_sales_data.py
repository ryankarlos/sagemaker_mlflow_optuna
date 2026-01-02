"""
Apple product sales data generator for multi-model Optuna tuning.
Generates per-product datasets with strong feature-target correlations.
Optimized for fast training with meaningful hyperparameter sensitivity.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# Product configurations with distinct demand patterns
PRODUCT_CONFIGS = {
    "iphone": {
        "base_demand": 800,
        "price_range": (799, 1199),
        "price_sensitivity": -0.8,  # High price sensitivity
        "promo_boost": 250,
        "weekend_boost": 150,
        "seasonality_amplitude": 0.3,  # Holiday peaks
        "peak_months": [11, 12],  # Holiday season
    },
    "macbook": {
        "base_demand": 200,
        "price_range": (1299, 2499),
        "price_sensitivity": -0.3,  # Less price sensitive
        "promo_boost": 100,
        "weekend_boost": 50,
        "seasonality_amplitude": 0.4,
        "peak_months": [8, 9],  # Back to school
    },
    "airpods": {
        "base_demand": 1200,
        "price_range": (129, 249),
        "price_sensitivity": -0.5,
        "promo_boost": 400,  # Very promo-driven
        "weekend_boost": 200,
        "seasonality_amplitude": 0.5,
        "peak_months": [11, 12],  # Gift season
    },
    "ipad": {
        "base_demand": 400,
        "price_range": (449, 1099),
        "price_sensitivity": -0.4,
        "promo_boost": 150,
        "weekend_boost": 80,
        "seasonality_amplitude": 0.35,
        "peak_months": [8, 9],  # Education
    },
    "watch": {
        "base_demand": 500,
        "price_range": (399, 799),
        "price_sensitivity": -0.45,
        "promo_boost": 180,
        "weekend_boost": 120,
        "seasonality_amplitude": 0.4,
        "peak_months": [1, 12],  # New Year fitness + holidays
    },
}


def generate_product_sales_data(
    product: str,
    n_rows: int = 750,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate sales data for a single Apple product.
    
    Optimized for fast XGBoost training with strong feature correlations.
    ~750 rows trains in <0.5s per Optuna trial.
    
    Args:
        product: Product name (must be in PRODUCT_CONFIGS)
        n_rows: Number of rows to generate (default 750 for fast training)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with features and 'demand' target
    """
    if product not in PRODUCT_CONFIGS:
        raise ValueError(f"Unknown product: {product}. Choose from {list(PRODUCT_CONFIGS.keys())}")
    
    config = PRODUCT_CONFIGS[product]
    
    if seed is not None:
        np.random.seed(seed)
    
    # Date range
    dates = pd.date_range(end=datetime.now(), periods=n_rows, freq="D")
    
    # Base features
    price_min, price_max = config["price_range"]
    price = np.random.uniform(price_min, price_max, n_rows)
    
    # Seasonality based on month
    months = dates.month
    is_peak = np.isin(months, config["peak_months"])
    seasonality = np.where(is_peak, config["seasonality_amplitude"], 0)
    seasonality += np.sin(2 * np.pi * months / 12) * 0.1  # Gentle wave
    
    # Binary features
    weekend = (dates.dayofweek >= 5).astype(int)
    holiday = np.random.choice([0, 1], n_rows, p=[0.95, 0.05])
    
    # Promo - more likely during peak months
    promo_prob = np.where(is_peak, 0.4, 0.15)
    promo = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in promo_prob])
    
    # Competitor price (relative to our price)
    competitor_price_ratio = np.random.uniform(0.85, 1.15, n_rows)
    competitor_cheaper = (competitor_price_ratio < 1.0).astype(int)
    
    # Calculate demand with clear feature relationships
    base = config["base_demand"]
    
    # Price effect (normalized to 0-1 range, then scaled)
    price_normalized = (price - price_min) / (price_max - price_min)
    price_effect = config["price_sensitivity"] * price_normalized * base
    
    # Other effects
    promo_effect = promo * config["promo_boost"]
    weekend_effect = weekend * config["weekend_boost"]
    holiday_effect = holiday * config["weekend_boost"] * 1.5
    seasonality_effect = seasonality * base
    competitor_effect = competitor_cheaper * (-50)  # Lose sales to cheaper competitor
    
    # Combine with noise
    noise = np.random.normal(0, base * 0.08, n_rows)
    
    demand = (
        base
        + price_effect
        + promo_effect
        + weekend_effect
        + holiday_effect
        + seasonality_effect
        + competitor_effect
        + noise
    )
    
    # Ensure positive demand
    demand = np.maximum(demand, base * 0.1)
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "product": product,
        "price": np.round(price, 2),
        "promo": promo,
        "weekend": weekend,
        "holiday": holiday,
        "competitor_cheaper": competitor_cheaper,
        "demand": np.round(demand, 0),
    })
    
    # Add lag feature (previous day demand)
    df["prev_demand"] = df["demand"].shift(1).bfill()
    
    return df


def generate_all_products_data(
    n_rows_per_product: int = 750,
    products: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Generate separate datasets for each product.
    
    Args:
        n_rows_per_product: Rows per product (750 recommended for fast training)
        products: List of products to generate (default: all)
        seed: Base random seed
    
    Returns:
        Dict mapping product name to DataFrame
    """
    if products is None:
        products = list(PRODUCT_CONFIGS.keys())
    
    datasets = {}
    for i, product in enumerate(products):
        datasets[product] = generate_product_sales_data(
            product=product,
            n_rows=n_rows_per_product,
            seed=seed + i,
        )
    
    return datasets


def save_product_datasets(
    output_dir: str = "data",
    n_rows_per_product: int = 750,
    products: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Generate and save per-product CSV files.
    
    Returns:
        Dict mapping product name to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    datasets = generate_all_products_data(n_rows_per_product, products, seed)
    
    paths = {}
    for product, df in datasets.items():
        file_path = output_path / f"{product}_sales.csv"
        df.to_csv(file_path, index=False)
        paths[product] = str(file_path)
        print(f"Saved {product}: {len(df)} rows -> {file_path}")
    
    return paths


def prepare_train_valid_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare train/validation split from product sales DataFrame.
    
    Returns:
        train_x, valid_x, train_y, valid_y
    """
    from sklearn.model_selection import train_test_split
    
    # Drop non-feature columns
    feature_cols = ["price", "promo", "weekend", "holiday", "competitor_cheaper", "prev_demand"]
    X = df[feature_cols]
    y = df["demand"]
    
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def get_product_list() -> List[str]:
    """Return list of available products."""
    return list(PRODUCT_CONFIGS.keys())


if __name__ == "__main__":
    print("Generating Apple product sales datasets...\n")
    
    # Generate all products
    datasets = generate_all_products_data(n_rows_per_product=750, seed=42)
    
    for product, df in datasets.items():
        config = PRODUCT_CONFIGS[product]
        print(f"\n{product.upper()}")
        print(f"  Rows: {len(df)}")
        print(f"  Price range: ${config['price_range'][0]} - ${config['price_range'][1]}")
        print(f"  Demand: mean={df['demand'].mean():.0f}, std={df['demand'].std():.0f}")
        
        # Show correlations
        numeric_cols = ["price", "promo", "weekend", "holiday", "competitor_cheaper", "prev_demand"]
        corr = df[numeric_cols + ["demand"]].corr()["demand"].drop("demand")
        print(f"  Top correlations:")
        for feat, val in corr.abs().sort_values(ascending=False).head(3).items():
            print(f"    {feat}: {corr[feat]:.3f}")
    
    # Save to files
    print("\n" + "="*50)
    save_product_datasets(output_dir="data", n_rows_per_product=750, seed=42)
