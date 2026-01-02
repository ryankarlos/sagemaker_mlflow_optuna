"""
Electronics sales data generator for multi-model Optuna tuning.
Generates compact datasets per product category with strong feature correlations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional


PRODUCT_CATEGORIES = {
    "smartphones": {
        "base_demand": 500,
        "price_range": (299, 1299),
        "price_sensitivity": -0.3,
        "promo_boost": 150,
        "weekend_boost": 80,
    },
    "laptops": {
        "base_demand": 200,
        "price_range": (499, 2499),
        "price_sensitivity": -0.15,
        "promo_boost": 100,
        "weekend_boost": 40,
    },
    "tablets": {
        "base_demand": 300,
        "price_range": (199, 999),
        "price_sensitivity": -0.25,
        "promo_boost": 120,
        "weekend_boost": 60,
    },
    "accessories": {
        "base_demand": 400,
        "price_range": (9, 149),
        "price_sensitivity": -0.2,
        "promo_boost": 150,
        "weekend_boost": 80,
    },
}


def generate_category_data(
    category: str,
    n_rows: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate sales data for a single product category.
    
    Compact feature set optimized for fast Optuna trials:
    - 8 numeric features with meaningful correlations
    - ~1000 rows default (fast training, good signal)
    """
    np.random.seed(seed)
    config = PRODUCT_CATEGORIES[category]
    
    # Date range
    dates = pd.date_range(end=datetime.now(), periods=n_rows, freq="D")
    
    # Core features
    price = np.random.uniform(*config["price_range"], n_rows)
    competitor_price = price * np.random.uniform(0.85, 1.15, n_rows)
    discount_pct = np.random.choice([0, 5, 10, 15, 20, 25], n_rows, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
    
    # Time features
    day_of_week = dates.dayofweek.values
    is_weekend = (day_of_week >= 5).astype(int)
    month = dates.month.values
    
    # Seasonality (Q4 holiday boost, back-to-school in Aug-Sep)
    seasonality = np.ones(n_rows)
    seasonality = np.where(month >= 11, 1.3, seasonality)
    seasonality = np.where((month == 8) | (month == 9), 1.15, seasonality)
    
    # Promo flag (more likely during peak seasons)
    promo_prob = np.where(seasonality > 1, 0.4, 0.15)
    is_promo = np.random.binomial(1, promo_prob)
    
    # Marketing spend (normalized 0-1)
    marketing_spend = np.random.beta(2, 5, n_rows)
    marketing_spend = np.where(is_promo == 1, marketing_spend + 0.3, marketing_spend)
    marketing_spend = np.clip(marketing_spend, 0, 1)
    
    # Customer rating (affects demand)
    avg_rating = np.random.uniform(3.5, 5.0, n_rows)
    
    # Stock level (low stock = urgency boost)
    stock_level = np.random.choice(["low", "medium", "high"], n_rows, p=[0.15, 0.5, 0.35])
    stock_multiplier = np.where(stock_level == "low", 1.1, np.where(stock_level == "high", 0.95, 1.0))
    
    # Calculate demand with clear feature relationships
    effective_price = price * (1 - discount_pct / 100)
    price_effect = config["price_sensitivity"] * (effective_price / config["price_range"][0])
    competitor_effect = 0.1 * (competitor_price - effective_price) / config["price_range"][0]
    
    demand = (
        config["base_demand"]
        + config["base_demand"] * price_effect
        + config["base_demand"] * competitor_effect
        + is_weekend * config["weekend_boost"]
        + is_promo * config["promo_boost"]
        + marketing_spend * 100
        + (avg_rating - 4.0) * 50
        + np.random.normal(0, config["base_demand"] * 0.1, n_rows)
    )
    demand = demand * seasonality * stock_multiplier
    demand = np.maximum(demand, 0).astype(int)
    
    df = pd.DataFrame({
        "date": dates,
        "category": category,
        "price": np.round(price, 2),
        "competitor_price": np.round(competitor_price, 2),
        "discount_pct": discount_pct,
        "is_weekend": is_weekend,
        "is_promo": is_promo,
        "marketing_spend": np.round(marketing_spend, 3),
        "avg_rating": np.round(avg_rating, 2),
        "stock_level": stock_level,
        "units_sold": demand,
    })
    
    return df


def generate_all_categories(
    n_rows_per_category: int = 1000,
    categories: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate combined dataset for all (or specified) categories."""
    if categories is None:
        categories = list(PRODUCT_CATEGORIES.keys())
    
    dfs = []
    for i, cat in enumerate(categories):
        df = generate_category_data(cat, n_rows_per_category, seed=seed + i)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def get_category_datasets(
    n_rows_per_category: int = 1000,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Return dict of DataFrames, one per category (for multi-model training)."""
    return {
        cat: generate_category_data(cat, n_rows_per_category, seed=seed + i)
        for i, cat in enumerate(PRODUCT_CATEGORIES.keys())
    }


def prepare_features(
    df: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare train/valid split with encoded features."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    
    # Encode stock_level
    le = LabelEncoder()
    df_encoded["stock_level"] = le.fit_transform(df_encoded["stock_level"])
    
    # Drop non-feature columns
    X = df_encoded.drop(columns=["date", "category", "units_sold"])
    y = df_encoded["units_sold"]
    
    return train_test_split(X, y, test_size=test_size, random_state=seed)


if __name__ == "__main__":
    # Demo: generate and show stats per category
    datasets = get_category_datasets(n_rows_per_category=1000)
    
    print("Electronics Sales Dataset Generator")
    print("=" * 50)
    
    for cat, df in datasets.items():
        print(f"\n{cat.upper()}")
        print(f"  Rows: {len(df)}")
        print(f"  Features: {len(df.columns) - 3}")  # exclude date, category, target
        print(f"  Units sold - Mean: {df['units_sold'].mean():.0f}, Std: {df['units_sold'].std():.0f}")
        
        # Show correlations
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()["units_sold"].drop("units_sold").sort_values()
        print(f"  Top correlations:")
        for feat in list(corr.index[-3:])[::-1]:
            print(f"    {feat}: {corr[feat]:.3f}")
