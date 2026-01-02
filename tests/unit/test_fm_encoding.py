"""Tests for FM encoding utilities."""

import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from steps.preprocess.fm_encoding import (
    FMEncoder,
    to_libsvm_format,
    create_user_item_matrix,
)


@pytest.fixture
def sample_data():
    """Create sample interaction data."""
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "game_id": ["g1", "g2", "g1", "g3", "g2"],
        "bet_qty": [10, 20, 15, 25, 30],
        "game_type": ["slots", "table", "slots", "live", "table"],
        "region": ["north", "north", "south", "south", "east"],
    })


class TestFMEncoder:
    def test_fit_creates_encoders(self, sample_data):
        encoder = FMEncoder()
        encoder.fit(sample_data)

        assert encoder.n_users == 3
        assert encoder.n_games == 3

    def test_transform_returns_sparse_matrix(self, sample_data):
        encoder = FMEncoder()
        X, y = encoder.fit_transform(sample_data, target_col="bet_qty")

        assert isinstance(X, csr_matrix)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == len(sample_data)
        assert len(y) == len(sample_data)

    def test_transform_with_categorical_cols(self, sample_data):
        encoder = FMEncoder()
        X, y = encoder.fit_transform(
            sample_data,
            target_col="bet_qty",
            categorical_cols=["game_type", "region"],
        )

        # Should have more features with categorical encoding
        assert X.shape[1] > encoder.n_users + encoder.n_games

    def test_target_values_correct(self, sample_data):
        encoder = FMEncoder()
        X, y = encoder.fit_transform(sample_data, target_col="bet_qty")

        np.testing.assert_array_equal(y, sample_data["bet_qty"].values.astype(np.float32))


class TestLibSVMFormat:
    def test_converts_to_libsvm_string(self):
        X = csr_matrix([[1, 0, 1], [0, 1, 0]])
        y = np.array([1.0, 2.0])

        result = to_libsvm_format(X, y)

        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("1.0")
        assert lines[1].startswith("2.0")


class TestCreateUserItemMatrix:
    def test_creates_sparse_matrix(self, sample_data):
        matrix, user_enc, item_enc = create_user_item_matrix(sample_data)

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape[0] == 3  # 3 users
        assert matrix.shape[1] == 3  # 3 games

    def test_aggregates_values(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u1"],
            "game_id": ["g1", "g1", "g2"],
            "bet_qty": [10, 20, 15],
        })

        matrix, _, _ = create_user_item_matrix(df)

        # u1-g1 should be 30 (10+20)
        assert matrix.sum() == 45  # 30 + 15
