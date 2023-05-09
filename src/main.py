from src.utils.load import load_wavs
from src.utils.config import Config


def main(config_source: str) -> None:
    config = Config(config_source)
    wavs = load_wavs(config)
    print(len(wavs))
    print(wavs[0])


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    main(os.getenv("CONFIG_PATH"))
