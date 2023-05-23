import os
import random
from functools import cache, lru_cache

from scipy.io.wavfile import read as read_wav

from src.data.split import train_test_validation
from src.utils.config import Config
from src.utils.custom_types import Data, DataPoint, Recording
from src.utils.parsing import label_from_fname


@lru_cache
def _dp_constructor(index: int, f_path: str, dir_path: str) -> DataPoint:
    sr, wav = read_wav(os.path.join(dir_path, f_path))
    return DataPoint(
        index=index,
        sampling_rate=sr,
        recording=Recording(content=wav),
        label=label_from_fname(f_path),
    )


@cache
def load_recordings(dir_path: str, choose_random: bool, choose_n: int) -> Data:
    files = [
        x
        for x in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, x))
        and os.path.splitext(os.path.join(dir_path, x))[-1] == ".wav"
    ]
    if choose_random:
        files = random.choices(files, k=choose_n)
    return Data(data=[_dp_constructor(i, x, dir_path) for i, x in enumerate(files)])


@cache
def get_data(config: Config) -> Data:
    (
        dir_path,
        ratios,
        seed,
        stratified,
    ) = config.get_data_loading_vars()
    random.seed(seed)
    data = load_recordings(dir_path, config.get("dev", False), config.get("dev_n", 3))
    data = train_test_validation(data, ratios, seed, stratified)

    return data
