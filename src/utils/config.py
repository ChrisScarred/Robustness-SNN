"""Configuration loading module.
"""
from typing import Any, Optional

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src.utils.log import get_logger
logger = get_logger(name = "config")

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
        logger.info("Config loaded.")

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
