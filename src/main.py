from src.data.load import get_data
from src.utils.config import Config


def main(config_source: str) -> None:
    config = Config(config_source)
    data = get_data(config)
    print(data)
    # mfscs = extract_from_config(config, data)
    # TODO: input layer, convolutional layer, output layer, svm


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    main(os.getenv("CONFIG_PATH"))
