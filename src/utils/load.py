import os
import pickle
from typing import Callable, List, Tuple

from numpy.typing import NDArray
from scipy.io.wavfile import read as read_wav

from src.utils.parsing import label_from_fname
from src.utils.config import Config


def load_wavs(config: Config) -> List[Tuple[Tuple[int, NDArray],int]]:
    if config.get("data.tidigits.pickle"):
        outf = config.get("data.tidigits.pickle_path")
        if os.path.isfile(outf):
            with open(outf, "rb") as f:
                return pickle.load(f)
    
    dir_path = config.get("data.tidigits.dir_path")
    wavs = [
        (read_wav(os.path.join(dir_path, x)), label_from_fname(x))
        for x in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, x))
        and os.path.splitext(os.path.join(dir_path, x))[-1] == ".wav"
    ]
    
    if config.get("data.tidigits.pickle"):
        with open(config.get("data.tidigits.pickle_path"), "wb") as f:
            pickle.dump(wavs, f)
    
    return wavs