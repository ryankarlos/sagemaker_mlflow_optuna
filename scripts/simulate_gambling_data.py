"""
Simplified gambling dataset for SageMaker Factorization Machines demo.
Generates sparse user-game interaction matrices similar to MNIST FM example.
"""

import numpy as np
import scipy.sparse as sparse
from typing import Tuple

# Available brands
BRANDS = ["betmax", "luckyspin", "royalbet"]


def generate_demo_data(
    n_users: int = 500,
    n_games: int = 50,
    density: float = 0.1,
    brand: str = "betmax",
    seed: int = 42,
) -> Tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray]:
    """
    Generate simple sparse interaction data for FM demo.
    
    Returns train/test sparse matrices and labels (bet amounts).
    Similar to MNIST FM example format.
    """
    np.random.seed(seed)
    
    # Total features = n_users + n_games (one-hot encoded)
    n_features = n_users + n_games
    n_samples = int(n_users * n_games * density)
    
    # Generate random user-game pairs
    user_indices = np.random.randint(0, n_users, n_samples)
    game_indices = np.random.randint(0, n_games, n_samples)
    
    # Create sparse feature matrix (user one-hot + game one-hot)
    row_indices = np.arange(n_samples)
    
    # User features (columns 0 to n_users-1)
    user_data = np.ones(n_samples)
    user_cols = user_indices
    
    # Game features (columns n_users to n_users+n_games-1)  
    game_data = np.ones(n_samples)
    game_cols = n_users + game_indices
    
    # Combine into sparse matrix
    rows = np.concatenate([row_indices, row_indices])
    cols = np.concatenate([user_cols, game_cols])
    data = np.concatenate([user_data, game_data])
    
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))
    
    # Generate labels (bet amounts) - influenced by user/game indices
    # Higher user index = higher spender, some games more popular
    user_factor = (user_indices / n_users) * 10
    game_factor = np.sin(game_indices / n_games * np.pi) * 5
    noise = np.random.normal(0, 2, n_samples)
    y = (user_factor + game_factor + noise + 10).astype(np.float32)
    y = np.clip(y, 1, 50)  # Bet amounts between 1-50
    
    # Split 80/20
    split_idx = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    return X_train, y_train, X_test, y_test


def generate_multi_brand_demo(
    brands: list = None,
    n_users: int = 300,
    n_games: int = 30,
    seed: int = 42,
) -> dict:
    """Generate demo data for multiple brands."""
    if brands is None:
        brands = BRANDS
    
    results = {}
    for i, brand in enumerate(brands):
        X_train, y_train, X_test, y_test = generate_demo_data(
            n_users=n_users,
            n_games=n_games,
            brand=brand,
            seed=seed + i * 100,
        )
        results[brand] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "n_features": X_train.shape[1],
        }
    return results


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_demo_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Labels range: {y_train.min():.1f} - {y_train.max():.1f}")
