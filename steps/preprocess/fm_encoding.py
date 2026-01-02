"""
Feature encoding for AWS SageMaker Factorization Machines.

FM requires sparse one-hot encoded features in LibSVM format.
"""

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, lil_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class FMEncoder:
    """Encoder for Factorization Machines features."""

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.game_encoder = LabelEncoder()
        self.categorical_encoders: Dict[str, OneHotEncoder] = {}
        self.n_users = 0
        self.n_games = 0

    def fit(self, df: pd.DataFrame, categorical_cols: List[str] = None) -> "FMEncoder":
        """Fit encoders on training data."""
        self.user_encoder.fit(df["user_id"])
        self.game_encoder.fit(df["game_id"])
        self.n_users = len(self.user_encoder.classes_)
        self.n_games = len(self.game_encoder.classes_)

        if categorical_cols:
            for col in categorical_cols:
                encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
                encoder.fit(df[[col]])
                self.categorical_encoders[col] = encoder

        return self

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "bet_qty",
        categorical_cols: List[str] = None,
    ) -> Tuple[csr_matrix, np.ndarray]:
        """Transform data to FM-compatible sparse format."""
        n_samples = len(df)

        # User one-hot encoding
        user_indices = self.user_encoder.transform(df["user_id"])
        user_matrix = lil_matrix((n_samples, self.n_users))
        user_matrix[np.arange(n_samples), user_indices] = 1

        # Game one-hot encoding
        game_indices = self.game_encoder.transform(df["game_id"])
        game_matrix = lil_matrix((n_samples, self.n_games))
        game_matrix[np.arange(n_samples), game_indices] = 1

        matrices = [user_matrix.tocsr(), game_matrix.tocsr()]

        if categorical_cols:
            for col in categorical_cols:
                if col in self.categorical_encoders:
                    cat_matrix = self.categorical_encoders[col].transform(df[[col]])
                    matrices.append(cat_matrix)

        X = hstack(matrices).tocsr()
        y = df[target_col].values.astype(np.float32)

        return X, y

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "bet_qty",
        categorical_cols: List[str] = None,
    ) -> Tuple[csr_matrix, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(df, categorical_cols)
        return self.transform(df, target_col, categorical_cols)


def to_libsvm_format(X: csr_matrix, y: np.ndarray) -> str:
    """Convert sparse matrix and labels to LibSVM format string."""
    lines = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        parts = [str(y[i])]
        for idx, val in zip(row.indices, row.data):
            parts.append(f"{idx}:{val}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def write_libsvm_file(X: csr_matrix, y: np.ndarray, filepath: str) -> None:
    """Write sparse data to LibSVM format file."""
    with open(filepath, "w") as f:
        f.write(to_libsvm_format(X, y))


def to_protobuf_format(X: csr_matrix, y: np.ndarray) -> bytes:
    """Convert to SageMaker RecordIO-Protobuf format."""
    try:
        import sagemaker.amazon.common as smac

        buf = io.BytesIO()
        smac.write_spmatrix_to_sparse_tensor(buf, X, y)
        buf.seek(0)
        return buf.read()
    except ImportError:
        raise ImportError("sagemaker package required for protobuf format")


def create_user_item_matrix(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "game_id",
    value_col: str = "bet_qty",
    agg_func: str = "sum",
) -> Tuple[csr_matrix, LabelEncoder, LabelEncoder]:
    """Create user-item interaction matrix."""
    agg_df = df.groupby([user_col, item_col])[value_col].agg(agg_func).reset_index()

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_indices = user_encoder.fit_transform(agg_df[user_col])
    item_indices = item_encoder.fit_transform(agg_df[item_col])
    values = agg_df[value_col].values

    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)

    matrix = csr_matrix(
        (values, (user_indices, item_indices)),
        shape=(n_users, n_items),
    )

    return matrix, user_encoder, item_encoder
