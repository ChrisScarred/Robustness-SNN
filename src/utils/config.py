"""Configuration loading module.
"""
from typing import Any, Optional, Tuple

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
        seed = self.get("seed")
        stratified = self.get("split.stratified")
        return dir_path, ratios, seed, stratified

    def get_prep_vars(self) -> Tuple[int, int, int, int]:
        n_frames = self.get("encoder.frames.num")
        overlap_t = self.get("encoder.frames.overlap")
        pad_t = self.get("encoder.frames.padding")
        freq_bands = self.get("encoder.bands")
        return n_frames, overlap_t, pad_t, freq_bands
