import os

from src.utils.custom_types import SamplingRate, Data


def label_from_fname(fname: str) -> int:
    main_name = os.path.splitext(fname)[0]
    return int(main_name.split("_")[-1])


def get_sr(data: Data) -> SamplingRate:
    sr = set([x.sampling_rate for x in data.data])
    if len(sr) != 1:
        raise ValueError("Multiple or no sampling rates detected.")
    return sr.pop()
