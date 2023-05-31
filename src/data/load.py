"""Data loader."""

import os
import random

from scipy.io.wavfile import read as read_wav

from src.data.split import train_test_validation
from src.utils.caching import region
from src.utils.config import Config
from src.utils.custom_types import Data, DataPoint, Recording
from src.utils.misc import label_from_fname


@region.cache_on_arguments()
def _dp_constructor(index: int, f_path: str, dir_path: str) -> DataPoint:
    """Construct a DataPoint.

    Args:
        index (int): An index of the DataPoint.
        f_path (str): The path to the file containing the audio recording of this DataPoint.
        dir_path (str): The path to the directory containing the file of the audio recording.

    Returns:
        DataPoint: A DataPoint with index `index`, content of the recording at the specified path, and a label infered from the file name.
    """
    sr, wav = read_wav(os.path.join(dir_path, f_path))
    return DataPoint(
        index=index,
        recording=Recording(content=wav, sampling_rate=sr),
        label=label_from_fname(f_path),
    )


@region.cache_on_arguments()
def load_recordings(dir_path: str) -> Data:
    """Read all audio recordings in the `.wav` format from the specified directory.

    Args:
        dir_path (str): The path to the directory that contains the recordings to load.

    Returns:
        Data: A list of all recordings from the supplied directory represented as DataPoint.
    """
    files = [
        x
        for x in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, x))
        and os.path.splitext(os.path.join(dir_path, x))[-1] == ".wav"
    ]
    return Data(data=[_dp_constructor(i, x, dir_path) for i, x in enumerate(files)])


@region.cache_on_arguments()
def get_data(config: Config) -> Data:
    """Read audio recordings according to the configuration parameters contained in the supplied Config object.

    Args:
        config (Config): A Config object containing at least the directory path to recordings, train/test/validation ratios, a seed, and an indication whether stratified split is desired, which can be accessed in this order by calling its class method `get_data_loading_vars()`.

    Returns:
        Data: A list of all recordings from the supplied directory represented as DataPoint, split into train/test/validation sets.
    """
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
            [x for x in data.data if x.cat == "train"], k=config.get("dev_n", 3)
        )
        data = Data(data=d)
    return data
