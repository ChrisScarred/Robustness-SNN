"""Configuration loading module.
"""
from typing import Any, Optional, Tuple, Dict, Union

from yaml import load

from src.utils.custom_types import Ratios

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class Config:
    """Load, store, and query configuration settings.

    Attributes:
        settings (Dict): the configuration settings as a dictionary of string keys, optionally nested, and values of any YAML-supported type.
        conf_source (Optional[str]): the path to the source file of the configuration settings.
        loaded (bool): True if the settings are loaded, False otherwise.
    """

    def __init__(self, conf_source: Optional[str]) -> None:
        """Initialise the config handler."""
        self.conf_source = conf_source
        self.loaded = False
        self.settings = {}

    def load_config(self, path: Optional[str] = None) -> None:
        """Load the configuration settings from the supplied path or the conf_source."""
        if path:
            with open(path, "r") as f:
                self.settings.update(load(f, Loader=Loader))

        else:
            if self.conf_source:
                with open(self.conf_source, "r") as f:
                    self.settings.update(load(f, Loader=Loader))
                self.loaded = True

    def get(self, var: str, def_val: Any = None) -> Any:
        """Get the value of the configuration variable `var`, defaulting to `def_val`."""
        if not self.loaded:
            self.load_config()

        keys = var.split(".")
        n = len(keys)

        if n == 1:
            return self.settings.get(var, def_val)

        val = self.settings
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, {})
            else:
                return def_val

        return val

    def add(self, key: str, val: Any) -> None:
        """Add a configuration variable `key` with a value `val`."""
        if not self.loaded:
            self.load_config()
        self.settings[key] = val

    def get_ratios(self) -> Ratios:
        ratios_temp = {}
        ratio_sum = 0
        for type_ in self.get("split.ratios").keys():
            r = self.get(f"split.ratios.{type_}")
            ratios_temp[type_] = r
            ratio_sum += r
        ratios = {}
        for k, v in ratios_temp.items():
            ratios[k] = v / ratio_sum
        return ratios

    def get_data_loading_vars(self) -> Tuple[str, Ratios, Any, bool]:
        dir_path = self.get("data.tidigits.dir_path")
        ratios = Ratios(content=self.get_ratios())
        seed = self.get("seed", "seed")
        stratified = self.get("split.stratified", False)
        return dir_path, ratios, seed, stratified

    def get_prep_vars(self) -> Tuple[int, int, int, int]:
        n_frames = self.get("model_params.time_frames", 40)
        overlap = self.get("model_params.dtf.overlap", 0)
        pad = self.get("model_params.dtf.pad", 0)
        freq_bands = self.get("model_params.mfsc.freq_bands", 40)
        return n_frames, overlap, pad, freq_bands

    def get_snn_params(self) -> Dict[str, Union[int, float, bool]]:
        return {
            "in_th": self.get("model_params.snn.in.th", 10.0), 
            "conv_th": self.get("model_params.snn.conv.th", 10.0), 
            "f_maps": self.get("model_params.snn.conv.f_maps", 4), 
            "conv_rf": self.get("model_params.snn.conv.rec_field", 3), 
            "conv_stride": self.get("model_params.snn.conv.stride", self.get("model_params.snn.conv.rec_field", 3)-1), 
            "wsg": self.get("model_params.snn.conv.wsg", 4), 
            "pool_rf": self.get("model_params.snn.pool.rec_field", 3), 
            "pool_stride": self.get("model_params.snn.pool.rec_field", self.get("model_params.snn.pool.rec_field", 3)),
            "weights_folder": self.get("model_params.snn.conv.weights.folder", "model"),
            "weights_file": self.get("model_params.snn.conv.weights.file", "model"),
            "load_weights": self.get("model_params.snn.conv.weights.load_weights", False)
        }
