from src.utils.load import load_wavs
from src.utils.config import Config
from src.utils.split import train_test_validation


def main(config_source: str) -> None:
    config = Config(config_source)
    wavs = load_wavs(config)
    data = train_test_validation(config, wavs)
    i = []
    for subdata in data:
        i.append([x[0] for x in subdata])
    i = tuple(i)
    print(i)


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    main(os.getenv("CONFIG_PATH"))
