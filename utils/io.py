import glob
import logging
import os
import pickle
import shutil
from pathlib import Path
import boto3
import polars as pl
import optuna
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def read_visits_parquet(path):
    df_all = pl.read_parquet(path).to_pandas()
    df_all.columns = df_all.columns.str.lower()
    df_all["orig_bet_qty"] = df_all["bet_qty"]  # for validation purposes
    if "summary_date" in df_all.columns:
        df_all["last_active_day"] = df_all["summary_date"]
    return df_all


# https://github.com/optuna/optuna/discussions/4355
def save_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    joblib.dump(
        study,
        f"Data/backup/{study.study_name}.pkl",
    )


def trial_backup_exists(study_name) -> bool:
    return os.path.exists(f"Data/backup/{study_name}.pkl")


def load_study(study_name: str) -> optuna.Study:
    return joblib.load(f"Data/backup/{study_name}.pkl")


def cleanup_local_path(destination_dir, substring=None):
    # Delete only files and folders in the destination directory that match the substring
    destination_path = Path(destination_dir)

    files = destination_path.glob("**/*")
    # Handle deletion logic based on substring
    if destination_path.exists() and destination_path.is_dir():

        if substring is None:
            # Delete the entire directory if no substring is provided
            logger.info(f"removing {destination_path}")
            shutil.rmtree(destination_path)
        else:
            # Delete only files and directories matching the substring
            for item in files:
                if item.is_file() and substring in item.name:
                    item.unlink()  # Delete file
                    logger.debug(f"removing {item}")
                elif item.is_dir() and substring in str(item):
                    shutil.rmtree(item)  # Delete directory
                    logger.debug(f"removing folder {item}")


def load_pickle(object_name, folder):
    object_full_path = f"{folder}/{object_name}.pickle"
    if os.path.exists(object_full_path):
        with open(object_full_path, "rb") as handle:
            obj = pickle.load(handle)
        logger.info(f"loaded {object_name}")
    else:
        logger.info(f"could not load object {object_name}")
        obj = None
    return obj


def s3_upload(bucket_name, local_file_path, s3_prefix):
    s3 = boto3.resource("s3")
    s3.Bucket(bucket_name).upload_file(local_file_path, s3_prefix)


def output_to_pickle(object_to_pickle, pickle_path):
    """
    Saves an object in a pickle format.
    Args:
        object_to_pickle (object): any python object that can be serialized
        pickle_path (str): path to the output pickle file

    Returns:

    """
    with open(pickle_path, "wb") as handle:
        pickle.dump(object_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
