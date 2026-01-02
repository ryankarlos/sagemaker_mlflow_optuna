"""
Simulate a gambling dataset for AWS Factorization Machines.

This module generates synthetic gambling data with users, games, and betting behavior
suitable for training recommendation models using AWS SageMaker Factorization Machines.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


def generate_users(n_users: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic user data with demographics and behavior patterns."""
    np.random.seed(seed)

    users = pd.DataFrame({
        "user_id": [f"user_{i:06d}" for i in range(n_users)],
        "age_group": np.random.choice(["18-25", "26-35", "36-45", "46-55", "55+"], n_users, p=[0.15, 0.30, 0.25, 0.20, 0.10]),
        "region": np.random.choice(["north", "south", "east", "west", "central"], n_users),
        "account_age_days": np.random.exponential(365, n_users).astype(int),
        "vip_tier": np.random.choice(["bronze", "silver", "gold", "platinum"], n_users, p=[0.50, 0.30, 0.15, 0.05]),
    })
    return users


def generate_games(n_games: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic game catalog."""
    np.random.seed(seed)

    game_types = ["slots", "table", "live_dealer", "poker", "sports"]
    providers = ["provider_a", "provider_b", "provider_c", "provider_d"]

    games = pd.DataFrame({
        "game_id": [f"game_{i:04d}" for i in range(n_games)],
        "game_type": np.random.choice(game_types, n_games, p=[0.50, 0.20, 0.15, 0.10, 0.05]),
        "provider": np.random.choice(providers, n_games),
        "rtp": np.random.uniform(0.92, 0.98, n_games),
        "volatility": np.random.choice(["low", "medium", "high"], n_games, p=[0.30, 0.45, 0.25]),
        "min_bet": np.random.choice([0.10, 0.25, 0.50, 1.0, 5.0], n_games),
        "popularity_score": np.random.exponential(1.0, n_games),
    })
    return games


def generate_interactions(
    users: pd.DataFrame,
    games: pd.DataFrame,
    n_days: int = 365,
    avg_sessions_per_user: float = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic user-game interactions (betting sessions)."""
    np.random.seed(seed)

    n_users = len(users)
    n_games = len(games)
    n_interactions = int(n_users * avg_sessions_per_user)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)

    # Power law distributions for activity
    user_activity = np.random.exponential(1.0, n_users)
    user_probs = user_activity / user_activity.sum()

    game_popularity = games["popularity_score"].values
    game_probs = game_popularity / game_popularity.sum()

    user_indices = np.random.choice(n_users, n_interactions, p=user_probs)
    game_indices = np.random.choice(n_games, n_interactions, p=game_probs)

    random_days = np.random.randint(0, n_days, n_interactions)
    timestamps = [start_date + timedelta(days=int(d)) for d in random_days]

    vip_multipliers = {"bronze": 1.0, "silver": 2.0, "gold": 4.0, "platinum": 8.0}
    user_vip = users.iloc[user_indices]["vip_tier"].map(vip_multipliers).values

    bet_qty = np.random.exponential(10, n_interactions) * user_vip
    bet_amount = bet_qty * np.random.uniform(0.5, 5.0, n_interactions)
    session_duration = np.random.exponential(15, n_interactions)

    interactions = pd.DataFrame({
        "user_id": users.iloc[user_indices]["user_id"].values,
        "game_id": games.iloc[game_indices]["game_id"].values,
        "summary_date": timestamps,
        "bet_qty": bet_qty.astype(int) + 1,
        "bet_amount": np.round(bet_amount, 2),
        "session_duration_mins": np.round(session_duration, 1),
        "win_amount": np.round(bet_amount * np.random.uniform(0.8, 1.2, n_interactions), 2),
    })

    return interactions.sort_values("summary_date").reset_index(drop=True)


def create_fm_features(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Create feature matrix for Factorization Machines."""
    df = interactions.merge(users, on="user_id", how="left")
    df = df.merge(games, on="game_id", how="left")
    return df


def generate_gambling_dataset(
    n_users: int = 10000,
    n_games: int = 200,
    n_days: int = 365,
    avg_sessions_per_user: float = 50,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate complete gambling dataset."""
    users = generate_users(n_users, seed)
    games = generate_games(n_games, seed)
    interactions = generate_interactions(users, games, n_days, avg_sessions_per_user, seed)
    fm_features = create_fm_features(interactions, users, games)
    return users, games, interactions, fm_features


def ingest_to_feature_store(
    users: pd.DataFrame,
    games: pd.DataFrame,
    interactions: pd.DataFrame,
    project_name: str = "fm-gambling-recommender",
) -> None:
    """
    Ingest generated data into SageMaker Feature Store.
    
    Parameters
    ----------
    users : pd.DataFrame
        User features
    games : pd.DataFrame
        Game features
    interactions : pd.DataFrame
        Interaction features
    project_name : str
        Project name for feature group naming
    """
    from scripts.utils.feature_store import FeatureStoreManager
    
    fs_manager = FeatureStoreManager(project_name=project_name)
    
    print(f"AWS Account: {fs_manager.account_id}")
    print(f"Region: {fs_manager.region}")
    print(f"Feature Groups: {list(fs_manager.describe_feature_groups().keys())}")
    
    fs_manager.ingest_all_features(users, games, interactions)
    print("Feature Store ingestion complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate gambling dataset")
    parser.add_argument("--n_users", type=int, default=1000)
    parser.add_argument("--n_games", type=int, default=50)
    parser.add_argument("--n_days", type=int, default=90)
    parser.add_argument("--avg_sessions", type=float, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ingest", action="store_true", help="Ingest to Feature Store")
    parser.add_argument("--project", type=str, default="fm-gambling-recommender")
    args = parser.parse_args()
    
    users, games, interactions, fm_features = generate_gambling_dataset(
        n_users=args.n_users,
        n_games=args.n_games,
        n_days=args.n_days,
        avg_sessions_per_user=args.avg_sessions,
        seed=args.seed,
    )
    print(f"Users: {len(users)}, Games: {len(games)}, Interactions: {len(interactions)}")
    print(interactions.head())
    
    if args.ingest:
        ingest_to_feature_store(users, games, interactions, args.project)
