import os
import pickle
import random
import re
from itertools import product
from math import ceil
from typing import Any, Dict, List, Tuple, Union

import librosa
import requests

from src.utils.custom_types import Config, Data
from src.utils.defaults import NOISE_DIR, NOISE_PICKLE, SEED
from src.data.load import load_recordings

API_URL = "https://freesound.org/apiv2/"
MAX_PAGE_SIZE = 150
PG_REGEX = r"Page (\d+)\: \[([^\n]+)\]"


class NoiseData:
    def __init__(self, config: Config, api_url: str = API_URL) -> None:
        token = config.get("data.noise.freesound_api_key")
        assert token, "Freesond API key required but not supplied in the config file"
        self.token = f"token={token}"
        self.domain = api_url
        self.dir_path = config.get("data.noise.dir_path", NOISE_DIR)
        self.pickle_path = config.get("data.noise.pickle_path", NOISE_PICKLE)
        self.seed = config.get("seed", SEED)

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
                            print(f"ID retrieval progress: {p / pages * 100 :.2f}%")
                else:
                    print(f"An error has occured. Response received: {res}")

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

    def load_data(self, target_sr: int) -> Data:
        if os.path.isfile(self.pickle_path):
            with open(self.pickle_path, "rb") as f:
                return pickle.load(f)

        data = load_recordings(self.dir_path, pickle_path=None, get_labels=False)
        for dp in data:
            content = dp.recording.content.astype(float)
            sr = dp.recording.sampling_rate
            mono_content = librosa.to_mono(content.T)
            downsampled = librosa.resample(mono_content, orig_sr=sr, target_sr=target_sr)
            dp.recording.content = downsampled
            dp.recording.sampling_rate = target_sr
        
        with open(self.pickle_path, "wb") as f:
            pickle.dump(data, f)
        
        return data
