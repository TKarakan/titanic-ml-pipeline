from pathlib import Path
from datetime import date
from src.utils.config_parser import load_config
from src.utils.logger import get_logger
import src.utils.io_helper as io



logger = get_logger(__name__)

_config = load_config("config/config.yaml")


def _today():
    return date.today().strftime("%Y%m%d")


def _get_project_name():
    return _config["project_name"]["name"]

def export_model(model, model_name: str):
    """
    Model'i models/ altına kaydeder. models/titanic_random_forest_20260419.pkl
    """
   
    project = _get_project_name()
    filename  = f"{project}_{model_name}_{_today()}.pkl"
    path = Path(_config["paths"]["model_export_dir"]) / filename

    io.save_model(model, path)
    logger.info(f"Model export edildi: {path}")

def export_processed_data(df, dataset_name:str, fmt: str ="csv"):
    """
    İşlenmiş veriyi data/processed/ altına kaydeder. data/processed/titanic_survival_train_processed_20260419.csv 
    """
    project = _get_project_name() 
    filename= f"{project}_{dataset_name}_{_today()}.{fmt}"
    path = Path(_config["paths"]["processed_data_dir"]) / filename

    io.save_data(df, path, format= fmt)
    logger.info(f"İşlenmiş veri export edildi: {path}")
    return path


def export_submission(df, experiment_name:str):
    """
    Kagle submission CSV'sini outputs/inputs altına kaydeder. data/processed/titanic_survival_submission_baseline_20260419.csv
    """
    project = _get_project_name()
    filename= f"{project}_submission_{experiment_name}_{_today()}.csv"
    path = Path(_config["paths"]["submission_dir"]) / filename

    io.save_data(df,path, format="csv")
    logger.info(f"Submission export edildi: {path}")
    return path