"""
Logging configuration for Gene Variant Analysis project.
"""

import logging
import logging.config
import os
from pathlib import Path


def setup_logging(config_path: str = "config/config.yaml") -> None:
    """
    Set up logging configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Basic logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
            }
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'INFO',
                'formatter': 'detailed',
                'class': 'logging.FileHandler',
                'filename': 'logs/gene_variant_analysis.log',
                'mode': 'a',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'src': {
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Logging configuration loaded successfully")