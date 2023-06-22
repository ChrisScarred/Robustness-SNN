import math
import os
import pickle
import random
import re
from itertools import product
from math import ceil
from typing import Any, Dict, List, Tuple, Union

import librosa
import numpy as np
import requests
from numpy.typing import NDArray
from scipy.io.wavfile import read as read_wav
from src.utils.log import get_logger
from src.utils.project_config import ProjectConfig

logger = get_logger(name="noise")

API_URL = "https://freesound.org/apiv2/"
MAX_PAGE_SIZE = 150
PG_REGEX = r"Page (\d+)\: \[([^\n]+)\]"


class NoiseHandler:
    def __init__(self, config: ProjectConfig, api_url: str = API_URL) -> None:
        token = config._freesound_key()
        assert token, "Freesond API key required but not supplied in the config file"
        self.token = f"token={token}"
        self.domain = api_url
        self.dir_path = config._noise_dir()
        self.pickle_path = config._noise_pickle_path()
        self.seed = config._seed()
        self.data = []

    def get_request(
        self, endpoint: str, params: Dict[str, Union[str, Dict[str, str]]]
    ) -> Any:
        url = f"{self.domain}{endpoint}/?{self.token}"
        for k, v in params.items():
            value = ""
            if isinstance(v, Dict):
                for i, (field, val) in enumerate(v.items()):
                    if " " in val:
                        val = f'"{val}"'
                    value += f"{field}:{val}"
                    if i < len(v.items()) - 1:
                        value += " "
            else:
                value = v
            url += f"&{k}={value}"
        return requests.get(url)
    
    def _read_temp_db(self) -> Tuple[int, List[int]]:
        indices = []
        page = 1
        path = os.path.join(self.dir_path, "pg_out.txt")
        if os.path.isfile(path):
            pages = []
            regex = re.compile(PG_REGEX)
            with open(path, "r") as f:
                for line in f.readlines():
                    res = regex.findall(line)
                    if res:
                        if len(res[0]) == 2:
                            page, i = res[0]
                            pages.append(int(page))
                            indices.extend([int(x) for x in i.split(", ")])
                page = max(pages)
        return page, indices

    def _query_indices(self, allowed_licenses: List[str], allowed_formats: List[str], page_size: int, page: int) -> None:
        with open(os.path.join(self.dir_path, "pg_out.txt"), "a") as f:
            for license, format in product(allowed_licenses, allowed_formats):
                params = {
                    "filter": {"type": format, "license": license},
                    "page_size": page_size,
                    "fields": "id",
                    "page": page
                }            
                res = self.get_request("search/text", params).json()
                count = res.get("count")
                if count:
                    pages = ceil(count/page_size)
                    for p in range(page, pages+1):
                        params["page"] = p
                        res = self.get_request("search/text", params).json()
                        results = res.get("results")
                        if results:
                            i = [r.get("id") for r in results]
                            f.write(f"Page {p}: {i}\n")
                            logger.info(f"ID retrieval progress: {p / pages * 100 :.2f}%")
                else:
                    logger.warning(f"An error has occured. Response received: {res}")

    def _get_indices(self, allowed_licenses: List[str], allowed_formats: List[str], page_size: int) -> None:
        page, _ = self._read_temp_db()
        self._query_indices(allowed_licenses, allowed_formats, page_size, page)
        _, indices = self._read_temp_db()
        return indices

    def _select_indices(self, indices: List[int], n: int) -> List[int]:
        indices.sort()
        if n > len(indices):
            n = len(indices)
        random.seed(self.seed)
        random.shuffle(indices)
        return indices[:n]
    
    def _download_recordings(self, indices: List[int]) -> None:
        # NOTE: cannot automate easily as downloading requires OAuth2 authentication
        import webbrowser
        for i in indices:
            webbrowser.open_new_tab(f"{self.domain}sounds/{i}/download")

    def get_db(
        self,
        samples: int,
        allowed_licenses: List[str],
        allowed_formats: List[str],
        page_size: int = 15
    ) -> None:
        if page_size > MAX_PAGE_SIZE:
            page_size = MAX_PAGE_SIZE
        indices = self._get_indices(allowed_licenses, allowed_formats, page_size)
        selected_indices = self._select_indices(indices, samples)
        self._download_recordings(selected_indices)

    def load_data(self, target_sr: int, save: bool) -> List[NDArray]:
        if os.path.isfile(self.pickle_path):
            with open(self.pickle_path, "rb") as f:
                try:
                    self.data = pickle.load(f)
                    logger.info(f"Loaded {len(self.data)} processed noise recordings from pickled database at {self.pickle_path}.")
                    return self.data
                except AttributeError:
                    logger.warning("Invalid pickle object, data will be loaded again.")
        
        raw_data = [read_wav(os.path.join(self.dir_path, x)) for x in os.listdir(self.dir_path) if x.endswith("wav")]

        for sr, content in raw_data:
            content = content.astype(float)
            mono_content = librosa.to_mono(content.T)            
            downsampled = librosa.resample(mono_content, orig_sr=sr, target_sr=target_sr)
            self.data.append(downsampled.astype(np.int16))
        
        logger.info(f"Loaded {len(self.data)} processed noise recordings from raw files at {self.dir_path}.")

        if save:
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.data, f)
                logger.info(f"Saved {len(self.data)} processed noise recordings to {self.pickle_path}.")
        
        return self.data
    
    def get_random_noise(self) -> NDArray:
        return random.choice(self.data)
    
def _equalise_lengths(signal: NDArray, noise: NDArray) -> NDArray:
    """Pad or cut a noise signal such that it spans over the entire signal of interest."""
    sig_ln = signal.size
    noi_ln = noise.size
    max_delay = noi_ln - sig_ln
    if max_delay > 0:
        delay = random.randrange(0, max_delay)
        noise = noise[delay:delay+signal.size]
    elif max_delay < 0:
        max_delay = abs(max_delay)
        delay = random.randrange(0, max_delay)
        noise = np.pad(noise,(delay, max_delay-delay))
    return noise

def _get_scaling_factor(snr: float, signal: NDArray, noise: NDArray) -> float:
    pow2 = np.vectorize(lambda y: y ** 2)
    rms = lambda x: math.sqrt(np.mean(pow2(x)))
    required_rms_noise = math.sqrt(rms(signal)/(10**(snr/10)))
    return required_rms_noise / rms(noise)

def mix_signal_noise(snr: float, signal: NDArray, noise: NDArray) -> NDArray:
    """Mix signal with noise with the given signal-to-noise ratio in dB. Signal and noise must have the same sampling rate and the same number of channels."""
    noise = _equalise_lengths(signal, noise)
    scale = _get_scaling_factor(snr, signal, noise)
    scaled_noise = np.multiply(scale, noise)
    mix = np.add(signal, scaled_noise)
    return mix
