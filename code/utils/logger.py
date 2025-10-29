import logging
from logging.handlers import RotatingFileHandler


# Logger setup function
def setup_logger(name, log_file, level=logging.INFO):
    """Sets up a logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add file handler (write to log file)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    return logger
