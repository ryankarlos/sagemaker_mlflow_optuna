"""Tests for gambling data simulation."""

import pytest
import pandas as pd
import numpy as np

from data.simulate_gambling_data import (
    generate_users,
    generate_games,
    generate_interactions,
    generate_gambling_dataset,
)


class TestGenerateUsers:
    def test_generates_correct_number_of_users(self):
        users = generate_users(100)
        assert len(users) == 100

    def test_has_required_columns(self):
        users = generate_users(10)
        required_cols = ["user_id", "age_group", "region", "account_age_days", "vip_tier"]
        assert all(col in users.columns for col in required_cols)

    def test_user_ids_are_unique(self):
        users = generate_users(100)
        assert users["user_id"].nunique() == 100

    def test_reproducible_with_seed(self):
        users1 = generate_users(50, seed=42)
        users2 = generate_users(50, seed=42)
        pd.testing.assert_frame_equal(users1, users2)


class TestGenerateGames:
    def test_generates_correct_number_of_games(self):
        games = generate_games(50)
        assert len(games) == 50

    def test_has_required_columns(self):
        games = generate_games(10)
        required_cols = ["game_id", "game_type", "provider", "rtp", "volatility"]
        assert all(col in games.columns for col in required_cols)

    def test_rtp_in_valid_range(self):
        games = generate_games(100)
        assert games["rtp"].min() >= 0.92
        assert games["rtp"].max() <= 0.98


class TestGenerateInteractions:
    def test_generates_interactions(self):
        users = generate_users(100)
        games = generate_games(20)
        interactions = generate_interactions(users, games, n_days=30, avg_sessions_per_user=10)
        assert len(interactions) > 0

    def test_has_required_columns(self):
        users = generate_users(50)
        games = generate_games(10)
        interactions = generate_interactions(users, games, n_days=30)
        required_cols = ["user_id", "game_id", "summary_date", "bet_qty", "bet_amount"]
        assert all(col in interactions.columns for col in required_cols)

    def test_bet_qty_positive(self):
        users = generate_users(50)
        games = generate_games(10)
        interactions = generate_interactions(users, games, n_days=30)
        assert (interactions["bet_qty"] > 0).all()


class TestGenerateGamblingDataset:
    def test_returns_all_dataframes(self):
        users, games, interactions, fm_features = generate_gambling_dataset(
            n_users=100, n_games=20, n_days=30, avg_sessions_per_user=5
        )
        assert isinstance(users, pd.DataFrame)
        assert isinstance(games, pd.DataFrame)
        assert isinstance(interactions, pd.DataFrame)
        assert isinstance(fm_features, pd.DataFrame)

    def test_fm_features_has_merged_data(self):
        users, games, interactions, fm_features = generate_gambling_dataset(
            n_users=50, n_games=10, n_days=30
        )
        # FM features should have columns from all sources
        assert "user_id" in fm_features.columns
        assert "game_id" in fm_features.columns
        assert "vip_tier" in fm_features.columns  # from users
        assert "game_type" in fm_features.columns  # from games
