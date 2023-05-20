import os
import pickle
from numpy.typing import NDArray

from scipy.io.wavfile import read as read_wav

from src.utils.custom_types import Data, DataPoint
from src.utils.config import Config
from src.utils.parsing import label_from_fname
from src.data.split import train_test_validation


def _dp_constructor(i: int, sr: int, wav: NDArray, l: int) -> DataPoint:
    return DataPoint(index=i, sampling_rate=sr, wav=wav, label=l)


def load_wavs(dir_path: str) -> Data:
    wavs = Data(data=[
        _dp_constructor(i, *read_wav(os.path.join(dir_path, x)), label_from_fname(x))
        for i, x in enumerate(os.listdir(dir_path))
        if os.path.isfile(os.path.join(dir_path, x))
        and os.path.splitext(os.path.join(dir_path, x))[-1] == ".wav"
    ])
    return wavs


def get_data(config: Config) -> Data:
    to_pickle, pickle_path, dir_path, ratios, seed, stratified = config.get_data_loading_vars()
    data = None

    if to_pickle and os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
    
    if not data:
        data = load_wavs(dir_path)
        data = train_test_validation(data, ratios, seed, stratified)

    if to_pickle:
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

    return data
