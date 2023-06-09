"""Miscellaneous utility functions."""
import os
from math import floor
from typing import Dict
from src.utils.custom_types import Lengths, Type_, Ratios, TiData
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from functools import lru_cache


square = np.vectorize(lambda x: x**2)


@lru_cache
def hann_window(ln: int) -> NDArray:
    """Get the Hann window of the supplied length.

    NOTE: Dong et al. do not mention which windowing function they used in the Fourier trasform step, so I opted for Hann, as it is the most widely used one.
    """
    return get_window("hann", ln)


def plot_signal(signal: NDArray) -> None:
    times = np.asarray(range(0, signal.size))
    plt.plot(times, signal)
    plt.show()


def label_from_fname(fname: str) -> int:
    """Infer TIDIGITS label from the file name."""
    main_name = os.path.splitext(fname)[0]
    return int(main_name.split("_")[-1])


def lengths_from_ratios(ratios: Ratios, data_len: int) -> Lengths:
    """Given split ratios and the length of data to split into those ratios, compute the lengths in data points of each split category."""
    ratios = ratios.content
    lengths = {}
    max_ln = len(list(ratios.values()))
    for i, (cat, ratio) in enumerate(ratios.items()):
        if i == max_ln - 1:
            sum_ln = sum(lengths.values())
            lengths[cat] = data_len - sum_ln
        else:
            len_ = floor(ratio * data_len)
            lengths[cat] = len_
    return lengths


def cat_from_lengths(i: int, lengths: Lengths) -> Type_:
    """Given split lengths and an index of a data point, return the split category in which the index belongs."""
    sum_ = 0
    for cat, len_ in lengths.items():
        sum_ += len_
        if i < sum_:
            return cat


def split_data(data: TiData) -> Dict[str, TiData]:
    res = {}
    for c in list(set([x.cat for x in data])):
        res[c] = TiData(data=[x for x in data if x.cat == c])
    return res
