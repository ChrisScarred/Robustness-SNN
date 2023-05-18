from src.utils.load import load_wavs
from src.utils.config import Config
from src.utils.split import ttv_from_config
from src.core.preprocess import extract_from_config

def main(config_source: str) -> None:
    config = Config(config_source)
    wavs = load_wavs(config)
    data = ttv_from_config(config, wavs)
    mfscs = extract_from_config(config, data)
    # TODO: input layer, convolutional layer, output layer, svm


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    main(os.getenv("CONFIG_PATH"))
