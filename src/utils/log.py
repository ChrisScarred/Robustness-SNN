import logging
from functools import lru_cache
from typing import Callable

@lru_cache
def get_logger(name: str = "default", handler: Callable = logging.StreamHandler(), level = logging.DEBUG, format: str = "%(asctime)s [%(levelname)s] at %(name)s: %(message)s") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
