from typing import Any, Dict, Optional, Tuple, Union, List
import os

from src.utils.custom_types import Ratios
from src.utils.config import Config

# defaults of optional processes
LOAD_SE = SAVE_SE = STRATIFIED_SPLIT = TRAINING = SE_MFSC_COMPARE = DEV_MODE = DOWNLOAD_NOISE = GET_NOISE = PICKLE_PROC_NOISE = False

# model parameter defaults
IN_TH = CONV_TH = 10.0
A_PLUS = A_MINUS = 0.05
W_MEAN = 0.8
W_SD = 0.05
F_MAPS = 4
CONV_RF = POOL_RF = 3
CONV_STRIDE = 2
WSG = 4
TIME_FRAMES = 40
DFT_OVERLAP = DFT_PAD = 0
FREQ_BANDS = 48
DIFF_TH = 0.0000001
EPOCHS = 1
SR = 20000

# pipeline parameter defaults
DEV_MODE_N = 1
NOISE_N = 100

# paths and file defaults
MODEL_DIR = "models"
LOAD_FILE = "model"
SAVE_FILE = "new_model"
DATA_DIR = "data"
NOISE_DIR = "noise"
PICKLE_NAME = "pickle"
NOISE_PICKLE = f"noise/{PICKLE_NAME}"

# other defaults
SEED = "seed"
ALLOWED_LIC = ["Creative Commons 0"]
ALLOWED_FORMATS = ["wav"]
SPLIT_RATIOS = {"train": 7, "test": 2, "validation": 1}
COMP_PARTS = {
    "training": True,
    "testing": False,
    "validation": False
}

class ProjectConfig(Config):
    """Project-specific implementation of Config."""
    def __init__(self, conf_source: Optional[str]) -> None:
        super().__init__(conf_source)

    def _ratios(self) -> Dict:
        return self.get("split.ratios", SPLIT_RATIOS)

    def parse_ratios(self) -> Ratios:
        ratios = self._ratios()
        sm = sum(ratios.values())
        return Ratios(content={k: v/sm for k, v in ratios.items()})
    
    def _tidigits_dir(self) -> str:
        return self.get("data.tidigits.dir_path", DATA_DIR)
    
    def _tidigits_pickle_path(self) -> str:
        return self.get(
            "data.tidigits.pickle_path", os.path.join(self._tidigits_dir(), PICKLE_NAME)
        )

    def _seed(self) -> Any:
        return self.get("seed", SEED)
    
    def _stratified(self) -> bool:
        return self.get("split.stratified", STRATIFIED_SPLIT)
    
    def _n_frames(self) -> int:
        return self.get("model_params.time_frames", TIME_FRAMES)
    
    def _dft_overlap(self) -> int:
        return self.get("model_params.dtf.overlap", DFT_OVERLAP)

    def _dft_pad(self) -> int:
        return self.get("model_params.dtf.pad", DFT_PAD)
    
    def _freq_bands(self) -> int:
        return self.get("model_params.mfsc.freq_bands", FREQ_BANDS)
    
    def _diff_th(self) -> float:
        return self.get("processes.train_snn.diff_th", DIFF_TH)
    
    def _epochs(self) -> int:
        return self.get("processes.train_snn.epochs", EPOCHS)
    
    def _batch_size(self) -> int:
        return self.get("processes.train_snn.batch_size", self._epochs())
    
    def _conv_rf(self) -> int:
        return self.get("model_params.snn.conv.rec_field", CONV_RF)
    
    def _conv_stride(self) -> int:
        return self.get("model_params.snn.conv.stride", self._conv_rf() - 1)

    def _f_maps(self) -> int:
        return self.get("model_params.snn.conv.f_maps", F_MAPS)
    
    def _pool_rf(self) -> int:
        return self.get("model_params.snn.pool.rec_field", POOL_RF)
    
    def _pool_stride(self) -> int:
        return self.get("model_params.snn.pool.rec_field", self._pool_rf())

    def _n_wsg(self) -> int:
        return self.get("model_params.snn.conv.wsg", WSG)
    
    def _a_minus(self) -> float:
        return self.get("model_params.snn.conv.a_minus", A_MINUS)
    
    def _a_plus(self) -> float:
        return self.get("model_params.snn.conv.a_plus", A_PLUS)
    
    def _conv_th(self) -> float:
        return self.get("model_params.snn.conv.th", CONV_TH)
    
    def _in_th(self) -> float:
        return self.get("model_params.snn.in.th", IN_TH)
    
    def _se_load_file(self) -> str:
        return self.get(
            "model_params.snn.conv.serialisation.load_file", LOAD_FILE
        )
    
    def _load_se(self) -> bool:
        return self.get(
            "model_params.snn.conv.serialisation.load_speech_encoder", LOAD_SE
        )
    
    def _se_dir(self) -> str:
        return self.get(
            "model_params.snn.conv.serialisation.folder", MODEL_DIR
        )
    
    def _se_save_file(self) -> str:
        return self.get(
            "model_params.snn.conv.serialisation.save_file", SAVE_FILE
        )

    def _w_mean(self) -> float:
        return self.get("model_params.snn.conv.weights.mean", W_MEAN)
    
    def _w_sd(self) -> float:
        return self.get("model_params.snn.conv.weights.sd", W_SD)
    
    def _dev_mode(self) -> bool:
        return self.get("modes.dev.enabled", DEV_MODE)

    def _dev_mode_samples(self) -> int:
        return self.get("modes.dev.samples", DEV_MODE_N)

    def _get_noise(self) -> bool:
        return self.get("processes.obtain_noise_dataset.enabled", GET_NOISE)

    def _download_noise_db(self) -> bool:
        return self.get("processes.obtain_noise_dataset.download", DOWNLOAD_NOISE)
    
    def _n_noise(self) -> int:
        return self.get("processes.obtain_noise_dataset.samples", NOISE_N)
    
    def _lic_whitelist(self) -> List[str]:
        return self.get("processes.obtain_noise_dataset.allowed_licenses", ALLOWED_LIC)
    
    def _format_whitelist(self) -> List[str]:
        return self.get("processes.obtain_noise_dataset.allowed_formats", ALLOWED_FORMATS)
    
    def _noise_dir(self) -> str:
        return self.get("data.noise.dir_path", NOISE_DIR)
    
    def _noise_pickle_path(self) -> str:
        return self.get("data.noise.pickle_path", NOISE_PICKLE)

    def _target_sr(self) -> int:
        return self.get("processes.obtain_noise_dataset.target_sr", SR)
    
    def _train_se(self) -> bool:
        return self.get("processes.train_snn.enabled", TRAINING)

    def _compare_se_mfsc(self) -> bool:
        return self.get("processes.compare_snn_mfsc.enabled", SE_MFSC_COMPARE)
    
    def _comp_parts(self) -> Dict[str, bool]:
        return self.get("processes.compare_snn_mfsc", COMP_PARTS)
    
    def _freesound_key(self) -> str:
        return self.get("data.noise.freesound_api_key")
    
    def _pickle_processed_noise(self) -> bool:
        return self.get("processes.obtain_noise_dataset.pickle_processed", PICKLE_PROC_NOISE)

    def get_noise_db_params(self) -> Tuple[int, List[str], List[str]]:
        return self._n_noise(), self._lic_whitelist(), self._format_whitelist()

    def get_snn_params(self) -> Dict[str, Union[int, float, bool]]:
        return {
            # architecture params
            "conv_rf": self._conv_rf(),
            "conv_stride": self._conv_stride(),
            "f_maps": self._f_maps(),
            "freq_maps": self._freq_bands(),
            "pool_rf": self._pool_rf(),
            "pool_stride": self._pool_stride(),
            "t_frames": self._n_frames(),
            "wsg": self._n_wsg(),
            # performance params
            "a_minus": self._a_minus(),
            "a_plus": self._a_plus(),
            "conv_th": self._conv_th(),
            "in_th": self._in_th(),
            # other params
            "load_file": self._se_load_file(),
            "load_speech_encoder": self._load_se(),
            "model_folder": self._se_dir(),
            "save_file": self._se_save_file(),
            "weight_mean": self._w_mean(),
            "weight_sd": self._w_sd(),
        }

    def get_training_params(self) -> Tuple[float, int, int]:
        return self._diff_th(), self._epochs(), self._batch_size

    def get_data_loading_vars(self) -> Tuple[str, Ratios, Any, bool, str]:
        return self._tidigits_dir(), self.parse_ratios(), self._seed(), self._stratified(), self._tidigits_pickle_path()

    def get_prep_vars(self) -> Tuple[int, int, int, int]:
        return self._n_frames(), self._dft_overlap(), self._dft_pad(), self._freq_bands()


