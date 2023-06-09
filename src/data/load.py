"""Data loader."""

import os
import pickle
import random
from typing import Optional

from scipy.io.wavfile import read as read_wav

from src.data.split import train_test_validation
from src.utils.config import Config
from src.utils.custom_types import Data, DataPoint, Recording
from src.utils.defaults import DEV_MODE, DEV_SAMPLES
from src.utils.misc import label_from_fname


def _dp_constructor(index: int, f_path: str, dir_path: str, get_labels: bool) -> DataPoint:
    """Construct a DataPoint.

    Args:
        index (int): An index of the DataPoint.
        f_path (str): The path to the file containing the audio recording of this DataPoint.
        dir_path (str): The path to the directory containing the file of the audio recording.

    Returns:
        DataPoint: A DataPoint with index `index`, content of the recording at the specified path, and a label infered from the file name.
    """
    sr, wav = read_wav(os.path.join(dir_path, f_path))
    label = None
    if get_labels:
        label = label_from_fname(f_path)
    return DataPoint(
        index=index,
        recording=Recording(content=wav, sampling_rate=sr),
        label=label,
    )


def load_recordings(dir_path: str, pickle_path: Optional[str] = None, get_labels: bool = True) -> Data:
    """Read all audio recordings in the `.wav` format from the specified directory.

    Args:
        dir_path (str): The path to the directory that contains the recordings to load.
        pickle_path (str): The path to the pickled Data object of the recordings.

    Returns:
        Data: A list of all recordings from the supplied directory represented as DataPoint.
    """
    if pickle_path:
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                if data:
                    return data
    files = [
        x
        for x in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, x))
        and os.path.splitext(os.path.join(dir_path, x))[-1] == ".wav"
    ]
    data = Data(data=[_dp_constructor(i, x, dir_path, get_labels) for i, x in enumerate(files)])

    if pickle_path:
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

    return data


def get_data(config: Config) -> Data:
    """Read audio recordings according to the configuration parameters contained in the supplied Config object.

    Args:
        config (Config): A Config object containing at least the directory path to recordings, train/test/validation ratios, a seed, and an indication whether stratified split is desired, which can be accessed in this order by calling its class method `get_data_loading_vars()`.

    Returns:
        Data: A list of all recordings from the supplied directory represented as DataPoint, split into train/test/validation sets.
    """
    (dir_path, ratios, seed, stratified, pickle_path) = config.get_data_loading_vars()
    random.seed(seed)
    data = load_recordings(dir_path, pickle_path)
    data = train_test_validation(data, ratios, seed, stratified)
    if config.get("modes.dev.enabled", DEV_MODE):
        d = random.choices(
            [x for x in data.data if x.cat == "train"],
            k=config.get("modes.dev.samples", DEV_SAMPLES),
        )
        data = Data(data=d)
    return data
