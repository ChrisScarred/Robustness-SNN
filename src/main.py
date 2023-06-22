"""App run loop.
"""
from src.data.load import get_data
from src.utils.project_config import ProjectConfig
from src.core.pipe import pipeline
import time
from src.utils.log import get_logger
logger = get_logger(name="main")

def main(config_source: str) -> None:
    """Load config from a file, obtain data, and run the pipeline.

    Args:
        config_source (str): The path to the configuration file.
    """
    start = time.time()
    config = ProjectConfig(config_source)
    data = get_data(config)
    pipeline(data, config)
    logger.info(f"Time elapsed: {time.time() - start:.2f} s")


if __name__ == "__main__":
    """Load the path to the configuration file from .env and run main.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()
    path = os.getenv("CONFIG_PATH")
    logger.debug(f"Config path is {path}.")
    main(path)
