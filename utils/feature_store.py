"""
Feature Store utilities for FM Gambling Recommender.

Provides helpers to ingest user, game, and interaction features
into SageMaker Feature Store.
"""

import logging
import time
from typing import Optional

import boto3
import pandas as pd
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

logger = logging.getLogger(__name__)


def get_aws_account_id() -> str:
    """Get AWS account ID using STS."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def get_aws_region() -> str:
    """Get current AWS region."""
    session = boto3.session.Session()
    return session.region_name or "us-east-1"


class FeatureStoreManager:
    """
    Manage Feature Store operations for FM pipeline.
    
    Handles ingestion of user, game, and interaction features
    into SageMaker Feature Store.
    """
    
    def __init__(
        self,
        project_name: str = "fm-gambling-recommender",
        region: Optional[str] = None,
    ):
        """
        Initialize Feature Store manager.
        
        Parameters
        ----------
        project_name : str
            Project name prefix for feature groups
        region : str, optional
            AWS region (auto-detected if not provided)
        """
        self.project_name = project_name
        self.region = region or get_aws_region()
        self.account_id = get_aws_account_id()
        
        self.boto_session = boto3.Session(region_name=self.region)
        self.sagemaker_client = self.boto_session.client("sagemaker")
        self.sagemaker_session = Session(
            boto_session=self.boto_session,
            sagemaker_client=self.sagemaker_client,
        )
        
        # Feature group names
        self.user_fg_name = f"{project_name}-user-features"
        self.game_fg_name = f"{project_name}-game-features"
        self.interaction_fg_name = f"{project_name}-interaction-features"
        
        logger.info(f"FeatureStoreManager initialized for account {self.account_id} in {self.region}")
    
    def _add_event_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event_time column if not present."""
        if "event_time" not in df.columns:
            df = df.copy()
            df["event_time"] = time.time()
        return df
    
    def _get_feature_group(self, name: str) -> FeatureGroup:
        """Get a FeatureGroup object."""
        return FeatureGroup(name=name, sagemaker_session=self.sagemaker_session)
    
    def ingest_user_features(
        self,
        users_df: pd.DataFrame,
        wait: bool = True,
        max_workers: int = 4,
    ) -> None:
        """
        Ingest user features into Feature Store.
        
        Parameters
        ----------
        users_df : pd.DataFrame
            DataFrame with columns: user_id, vip_tier, region, age_group, etc.
        wait : bool
            Wait for ingestion to complete
        max_workers : int
            Number of parallel workers
        """
        df = self._add_event_time(users_df)
        
        # Ensure required columns
        required = ["user_id", "event_time"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert types
        df["user_id"] = df["user_id"].astype(str)
        df["event_time"] = df["event_time"].astype(float)
        
        # Optional columns with defaults
        if "vip_tier" in df.columns:
            df["vip_tier"] = df["vip_tier"].astype(str)
        if "region" in df.columns:
            df["region"] = df["region"].astype(str)
        if "age_group" in df.columns:
            df["age_group"] = df["age_group"].astype(str)
        if "total_sessions" in df.columns:
            df["total_sessions"] = df["total_sessions"].astype(int)
        if "avg_bet_amount" in df.columns:
            df["avg_bet_amount"] = df["avg_bet_amount"].astype(float)
        if "favorite_game_type" in df.columns:
            df["favorite_game_type"] = df["favorite_game_type"].astype(str)
        
        fg = self._get_feature_group(self.user_fg_name)
        logger.info(f"Ingesting {len(df)} user records to {self.user_fg_name}")
        
        fg.ingest(data_frame=df, max_workers=max_workers, wait=wait)
        logger.info("User features ingestion complete")
    
    def ingest_game_features(
        self,
        games_df: pd.DataFrame,
        wait: bool = True,
        max_workers: int = 4,
    ) -> None:
        """
        Ingest game features into Feature Store.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            DataFrame with columns: game_id, game_type, rtp, volatility, etc.
        wait : bool
            Wait for ingestion to complete
        max_workers : int
            Number of parallel workers
        """
        df = self._add_event_time(games_df)
        
        # Ensure required columns
        required = ["game_id", "event_time"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert types
        df["game_id"] = df["game_id"].astype(str)
        df["event_time"] = df["event_time"].astype(float)
        
        if "game_type" in df.columns:
            df["game_type"] = df["game_type"].astype(str)
        if "rtp" in df.columns:
            df["rtp"] = df["rtp"].astype(float)
        if "volatility" in df.columns:
            df["volatility"] = df["volatility"].astype(str)
        if "min_bet" in df.columns:
            df["min_bet"] = df["min_bet"].astype(float)
        if "max_bet" in df.columns:
            df["max_bet"] = df["max_bet"].astype(float)
        
        fg = self._get_feature_group(self.game_fg_name)
        logger.info(f"Ingesting {len(df)} game records to {self.game_fg_name}")
        
        fg.ingest(data_frame=df, max_workers=max_workers, wait=wait)
        logger.info("Game features ingestion complete")
    
    def ingest_interaction_features(
        self,
        interactions_df: pd.DataFrame,
        wait: bool = True,
        max_workers: int = 4,
    ) -> None:
        """
        Ingest interaction features into Feature Store.
        
        Parameters
        ----------
        interactions_df : pd.DataFrame
            DataFrame with columns: user_id, game_id, bet_qty, total_stake, etc.
        wait : bool
            Wait for ingestion to complete
        max_workers : int
            Number of parallel workers
        """
        df = self._add_event_time(interactions_df)
        
        # Create interaction_id if not present
        if "interaction_id" not in df.columns:
            df = df.copy()
            df["interaction_id"] = df.apply(
                lambda r: f"{r['user_id']}_{r['game_id']}_{int(r['event_time']*1000)}", 
                axis=1
            )
        
        # Ensure required columns
        required = ["interaction_id", "event_time", "user_id", "game_id"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert types
        df["interaction_id"] = df["interaction_id"].astype(str)
        df["event_time"] = df["event_time"].astype(float)
        df["user_id"] = df["user_id"].astype(str)
        df["game_id"] = df["game_id"].astype(str)
        
        if "bet_qty" in df.columns:
            df["bet_qty"] = df["bet_qty"].astype(int)
        if "total_stake" in df.columns:
            df["total_stake"] = df["total_stake"].astype(float)
        
        fg = self._get_feature_group(self.interaction_fg_name)
        logger.info(f"Ingesting {len(df)} interaction records to {self.interaction_fg_name}")
        
        fg.ingest(data_frame=df, max_workers=max_workers, wait=wait)
        logger.info("Interaction features ingestion complete")
    
    def ingest_all_features(
        self,
        users_df: pd.DataFrame,
        games_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        wait: bool = True,
    ) -> None:
        """
        Ingest all feature types into Feature Store.
        
        Parameters
        ----------
        users_df : pd.DataFrame
            User features DataFrame
        games_df : pd.DataFrame
            Game features DataFrame
        interactions_df : pd.DataFrame
            Interaction features DataFrame
        wait : bool
            Wait for ingestion to complete
        """
        logger.info("Starting full feature ingestion...")
        
        self.ingest_user_features(users_df, wait=wait)
        self.ingest_game_features(games_df, wait=wait)
        self.ingest_interaction_features(interactions_df, wait=wait)
        
        logger.info("All features ingested successfully")
    
    def get_user_features(self, user_ids: list) -> pd.DataFrame:
        """
        Get user features from online store.
        
        Parameters
        ----------
        user_ids : list
            List of user IDs to fetch
            
        Returns
        -------
        pd.DataFrame
            User features
        """
        fg = self._get_feature_group(self.user_fg_name)
        records = []
        
        for user_id in user_ids:
            try:
                record = fg.get_record(record_identifier_value_as_string=str(user_id))
                if record:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Failed to get record for user {user_id}: {e}")
        
        if not records:
            return pd.DataFrame()
        
        return pd.DataFrame(records)
    
    def get_game_features(self, game_ids: list) -> pd.DataFrame:
        """
        Get game features from online store.
        
        Parameters
        ----------
        game_ids : list
            List of game IDs to fetch
            
        Returns
        -------
        pd.DataFrame
            Game features
        """
        fg = self._get_feature_group(self.game_fg_name)
        records = []
        
        for game_id in game_ids:
            try:
                record = fg.get_record(record_identifier_value_as_string=str(game_id))
                if record:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Failed to get record for game {game_id}: {e}")
        
        if not records:
            return pd.DataFrame()
        
        return pd.DataFrame(records)
    
    def describe_feature_groups(self) -> dict:
        """Get status of all feature groups."""
        result = {}
        for name in [self.user_fg_name, self.game_fg_name, self.interaction_fg_name]:
            try:
                response = self.sagemaker_client.describe_feature_group(
                    FeatureGroupName=name
                )
                result[name] = {
                    "status": response["FeatureGroupStatus"],
                    "creation_time": str(response.get("CreationTime", "")),
                    "record_identifier": response["RecordIdentifierFeatureName"],
                }
            except self.sagemaker_client.exceptions.ResourceNotFound:
                result[name] = {"status": "NOT_FOUND"}
            except Exception as e:
                result[name] = {"status": "ERROR", "error": str(e)}
        
        return result
