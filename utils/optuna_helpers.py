import optuna
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def copy_optuna_study_to_db(source_db_file, target_db_file):
    study_name = optuna.study.get_all_study_names(
        storage=f"sqlite:///{source_db_file}"
    )[0]
    optuna.copy_study(
        from_study_name=study_name,
        from_storage=f"sqlite:///{source_db_file}",
        to_storage=f"sqlite:///{target_db_file}",
    )


def copy_all_studies_to_new_file(target_db_file, source_db_dir):
    if Path(target_db_file).exists():
        logger.info(f"Removing existing optuna target db {target_db_file}")
        Path(target_db_file).unlink()
    for source_file in Path(source_db_dir).iterdir():
        if source_file.suffix == ".db":
            copy_optuna_study_to_db(source_file, target_db_file)
