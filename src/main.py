from src.data.load import get_data
from src.utils.config import Config
from src.core.pipe import pipeline
import time


def main(config_source: str) -> None:
    start = time.time()
    config = Config(config_source)
    data = get_data(config)
    pipeline(data, config)
    print(f"Time elapsed: {time.time() - start:.2f} s")


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    main(os.getenv("CONFIG_PATH"))
