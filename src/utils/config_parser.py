import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_config(config_path):

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
            return config

     
    except FileNotFoundError:
        logger.error(f"Config dosyası bulunamadı! Yol: {config_path}")
        raise

    except yaml.YAMLError as e:
        logger.error(f"YAML formatı bozuk: {e}")
        raise 


