import os
import pickle

from scipy.io.wavfile import read as read_wav

from src.utils.custom_types import Data
from src.utils.config import Config
from src.utils.parsing import label_from_fname


def load_wavs(config: Config) -> Data:
    if config.get("data.tidigits.pickle"):
        outf = config.get("data.tidigits.pickle_path")
        if os.path.isfile(outf):
            with open(outf, "rb") as f:
                return pickle.load(f)
    
    dir_path = config.get("data.tidigits.dir_path")
    wavs = [
        (i, read_wav(os.path.join(dir_path, x)), label_from_fname(x))
        for i, x in enumerate(os.listdir(dir_path))
        if os.path.isfile(os.path.join(dir_path, x))
        and os.path.splitext(os.path.join(dir_path, x))[-1] == ".wav"
    ]
    
    if config.get("data.tidigits.pickle"):
        with open(config.get("data.tidigits.pickle_path"), "wb") as f:
            pickle.dump(wavs, f)
    
    return wavs