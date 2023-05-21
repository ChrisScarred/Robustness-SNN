from src.data.load import get_data
from src.utils.config import Config
from src.core.pipe import pipeline


def main(config_source: str) -> None:
    config = Config(config_source)
    data = get_data(config)
    out_data = pipeline(data, config)
    print(out_data)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    main(os.getenv("CONFIG_PATH"))
