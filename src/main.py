from src.utils.load import load_wavs
from src.utils.config import Config
from src.utils.split import train_test_validation


def main(config_source: str) -> None:
    config = Config(config_source)
    wavs = load_wavs(config)
    data = train_test_validation(config, wavs)
    # TODO: preprocess, input layer, convolutional layer, output layer, svm


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    main(os.getenv("CONFIG_PATH"))
