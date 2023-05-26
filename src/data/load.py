import os
import random

from scipy.io.wavfile import read as read_wav

from src.data.split import train_test_validation
from src.utils.config import Config
from src.utils.custom_types import Data, DataPoint, Recording
from src.utils.parsing import label_from_fname
from src.utils.caching import region


@region.cache_on_arguments()
def _dp_constructor(index: int, f_path: str, dir_path: str) -> DataPoint:
    sr, wav = read_wav(os.path.join(dir_path, f_path))
    return DataPoint(
        index=index,
        recording=Recording(content=wav, sampling_rate=sr),
        label=label_from_fname(f_path),
    )


@region.cache_on_arguments()
def load_recordings(dir_path: str) -> Data:
    files = [
        x
        for x in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, x))
        and os.path.splitext(os.path.join(dir_path, x))[-1] == ".wav"
    ]
    return Data(data=[_dp_constructor(i, x, dir_path) for i, x in enumerate(files)])


@region.cache_on_arguments()
def get_data(config: Config) -> Data:
    (
        dir_path,
        ratios,
        seed,
        stratified,
    ) = config.get_data_loading_vars()
    random.seed(seed)
    data = load_recordings(dir_path)
    data = train_test_validation(data, ratios, seed, stratified)
    if config.get("dev", False):
        d = random.choices(
            [x for x in data.data if x.type_ == "train"], k=config.get("dev_n", 3)
        )
        data = Data(data=d)
    return data
