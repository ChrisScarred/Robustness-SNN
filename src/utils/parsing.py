import os
from typing import Any, Dict

from src.utils.custom_types import Config, Ratios, SamplingRate, Data


def nested_getter(dct: Dict, var: str, def_val: Any = None) -> Any:
    """Obtain a value of a variable in a nested dict.

    Args:
        var (str): The variable to obtain.
        def_val (Any, optional): The default value to use if the variable is not set. Defaults to None.
    Returns:
        Any: The value of the queried variable or the default value if the variable is unset and the default is supplied.
    """
    keys = var.split(".")
    n = len(keys)

    if n == 1:
        return dct.get(var, def_val)

    val = dct
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, {})
        else:
            return def_val

    return val


def label_from_fname(fname: str) -> int:
    main_name = os.path.splitext(fname)[0]
    return int(main_name.split("_")[-1])


def get_ratios(config: Config) -> Ratios:
    ratios_temp = {}
    ratio_sum = 0
    for type_ in ["train", "test", "validation"]:
        r = config.get(f"split.ratios.{type_}")
        ratios_temp[type_] = r
        ratio_sum += r
    ratios = {}
    for k, v in ratios_temp.items():
        ratios[k] = v / ratio_sum
    return ratios
