"""
Logging setup for Study Buddy AI Agent.
"""

import logging
import sys
from pathlib import Path
from config import BASE_DIR, LOG_LEVEL

def setup_logger(name: str = "StudyBuddyAI") -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    
    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)
    
    # File Handler
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    f_handler = logging.FileHandler(log_dir / "study_buddy.log", encoding="utf-8")
    f_handler.setFormatter(c_format)
    logger.addHandler(f_handler)
    
    return logger

logger = setup_logger()
