"""App run loop.
"""
from src.data.load import get_data
from src.utils.config import Config
from src.core.pipe import pipeline
import time


def main(config_source: str) -> None:
    """Load config from a file, obtain data, and run the pipeline.

    Args:
        config_source (str): The path to the configuration file.
    """
    start = time.time()
    config = Config(config_source)
    data = get_data(config)
    pipeline(data, config)
    print(f"Time elapsed: {time.time() - start:.2f} s")


if __name__ == "__main__":
    """Load the path to the configuration file from .env and run main.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()
    main(os.getenv("CONFIG_PATH"))
