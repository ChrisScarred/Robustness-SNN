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
from src.utils.audio import rms, snr_db_to_a, norm_loudness, equalise_lengths


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

def _get_scaling_factor(snr: float, signal: NDArray, noise: NDArray) -> float:
    """Given a requested SNR in the power of amplitude, a signal, and a noise, calculate the scaling factor necessary to apply to the noise to achieve the SNR."""
    rms_s = rms(signal)
    rms_noise = rms(noise)
    snr_unedited = rms_s / rms_noise
    return snr_unedited / snr

def mix_signal_noise(snr: float, signal: NDArray, noise: NDArray, req_rms: float, snr_in_db: bool = True) -> NDArray:
    """Mix signal with noise with the given signal-to-noise ratio, optionally in dB. Signal and noise must have the same sampling rate and the same number of channels."""
    if snr_in_db:
        snr_a = snr_db_to_a(snr)
    else:
        snr_a = snr
    noise = equalise_lengths(signal, noise)
    scale = _get_scaling_factor(snr_a, signal, noise)
    scaled_noise = np.multiply(scale, noise)
    mix = np.add(signal, scaled_noise)
    return norm_loudness(mix, req_rms)
