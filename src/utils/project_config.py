from typing import Any, Dict, Optional, Tuple, Union
import os

from src.utils.custom_types import Ratios
from src.utils.defaults import (
    A_MINUS,
    A_PLUS,
    CONV_RF,
    CONV_TH,
    DATA_DIR,
    DFT_OVERLAP,
    DFT_PAD,
    F_MAPS,
    FREQ_BANDS,
    IN_TH,
    LOAD_FILE,
    LOAD_SE,
    MODEL_DIR,
    POOL_RF,
    SAVE_FILE,
    SEED,
    SPLIT_RATIOS,
    STRATIFIED_SPLIT,
    TIME_FRAMES,
    WSG,
    W_MEAN,
    W_SD,
)
from src.utils.config import Config


class ProjectConfig(Config):
    """Project-specific implementation of Config."""
    def __init__(self, conf_source: Optional[str]) -> None:
        super().__init__(conf_source)

    def get_ratios(self) -> Ratios:
        ratios_temp = {}
        ratio_sum = 0
        for type_ in self.get("split.ratios", SPLIT_RATIOS).keys():
            r = self.get(f"split.ratios.{type_}", SPLIT_RATIOS.get(type_))
            ratios_temp[type_] = r
            ratio_sum += r
        ratios = {}
        for k, v in ratios_temp.items():
            ratios[k] = v / ratio_sum
        return ratios

    def get_data_loading_vars(self) -> Tuple[str, Ratios, Any, bool]:
        dir_path = self.get("data.tidigits.dir_path", DATA_DIR)
        ratios = Ratios(content=self.get_ratios())
        seed = self.get("seed", SEED)
        stratified = self.get("split.stratified", STRATIFIED_SPLIT)
        pickle_path = self.get(
            "data.tidigits.pickle_path", os.path.join(dir_path, "pickle")
        )
        return dir_path, ratios, seed, stratified, pickle_path

    def get_prep_vars(self) -> Tuple[int, int, int, int]:
        n_frames = self.get("model_params.time_frames", TIME_FRAMES)
        overlap = self.get("model_params.dtf.overlap", DFT_OVERLAP)
        pad = self.get("model_params.dtf.pad", DFT_PAD)
        freq_bands = self.get("model_params.mfsc.freq_bands", FREQ_BANDS)
        return n_frames, overlap, pad, freq_bands

    def get_snn_params(self) -> Dict[str, Union[int, float, bool]]:
        return {
            # architecture params
            "conv_rf": self.get("model_params.snn.conv.rec_field", CONV_RF),
            "conv_stride": self.get(
                "model_params.snn.conv.stride",
                self.get("model_params.snn.conv.rec_field", CONV_RF) - 1,
            ),
            "f_maps": self.get("model_params.snn.conv.f_maps", F_MAPS),
            "freq_maps": self.get("model_params.mfsc.freq_bands", FREQ_BANDS),
            "pool_rf": self.get("model_params.snn.pool.rec_field", POOL_RF),
            "pool_stride": self.get(
                "model_params.snn.pool.rec_field",
                self.get("model_params.snn.pool.rec_field", POOL_RF),
            ),
            "t_frames": self.get("model_params.time_frames", TIME_FRAMES),
            "wsg": self.get("model_params.snn.conv.wsg", WSG),
            # performance params
            "a_minus": self.get("model_params.snn.conv.a_minus", A_MINUS),
            "a_plus": self.get("model_params.snn.conv.a_plus", A_PLUS),
            "conv_th": self.get("model_params.snn.conv.th", CONV_TH),
            "in_th": self.get("model_params.snn.in.th", IN_TH),
            # other params
            "load_file": self.get(
                "model_params.snn.conv.serialisation.load_file", LOAD_FILE
            ),
            "load_speech_encoder": self.get(
                "model_params.snn.conv.serialisation.load_speech_encoder", LOAD_SE
            ),
            "model_folder": self.get(
                "model_params.snn.conv.serialisation.folder", MODEL_DIR
            ),
            "save_file": self.get(
                "model_params.snn.conv.serialisation.save_file", SAVE_FILE
            ),
            "weight_mean": self.get("model_params.snn.conv.weights.mean", W_MEAN),
            "weight_sd": self.get("model_params.snn.conv.weights.sd", W_SD),
        }

    def get_training_params(self) -> Tuple[float, int, int]:
        diff_th = self.get("processes.train_snn.diff_th")
        epochs = self.get("processes.train_snn.epochs")
        batch_size = self.get("processes.train_snn.batch_size")
        return diff_th, epochs, batch_size