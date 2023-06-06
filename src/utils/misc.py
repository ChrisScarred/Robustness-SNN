"""Miscellaneous utility functions."""
import os
from math import floor
from typing import Dict
from src.utils.custom_types import Lengths, Type_, Ratios, Data


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


def split_data(data: Data) -> Dict[str, Data]:
    res = {}
    for c in list(set([x.cat for x in data])):
        res[c] = Data(data=[x for x in data if x.cat==c])
    return res