from src.data.load import get_data
from src.utils.config import Config
from src.core.pipe import pipeline
import matplotlib.pyplot as plt
import time


def main(config_source: str) -> None:
    n = time.time()
    config = Config(config_source)
    data = get_data(config)
    out_data = pipeline(data, config)
    print(time.time()-n)
    for i in range(min(len(out_data), config.get("dev_n", 3))):
        plt.imshow(out_data[i])
        plt.show()


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    main(os.getenv("CONFIG_PATH"))

