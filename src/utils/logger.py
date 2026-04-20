import logging
import sys
from pathlib import Path
from src.utils.config_parser import load_config

_configured = False

def _setup_logger(config):

    global _configured
    if _configured:
        return
    
    log_file = Path(config["logging"]["log_file_path"])

    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [
            
            logging.FileHandler(
                log_file,
                encoding="utf-8",
                mode='a'
            ),

             logging.StreamHandler() 
        ],
        force = True
    )
    _configured = True

def get_logger(name):

    config = load_config("config/config.yaml")
    _setup_logger(config)
    logger = logging.getLogger(name)
    return logger



   






