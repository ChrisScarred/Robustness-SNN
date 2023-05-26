import os
from math import ceil, floor

from src.utils.custom_types import Lengths, Type_, Ratios


def label_from_fname(fname: str) -> int:
    main_name = os.path.splitext(fname)[0]
    return int(main_name.split("_")[-1])


def ms_to_samples(val: float, sr: int) -> int:
    return ceil((val / 1000) * sr)


def lengths_from_ratios(ratios: Ratios, data_len: int) -> Lengths:
    ratios = ratios.content
    lengths = {}
    max_ln = len(list(ratios.values()))
    for i, (type_, ratio) in enumerate(ratios.items()):
        if i == max_ln - 1:
            sum_ln = sum(lengths.values())
            lengths[type_] = data_len - sum_ln
        else:
            len_ = floor(ratio * data_len)
            lengths[type_] = len_
    return lengths


def type_from_lengths(i: int, lengths: Lengths) -> Type_:
    sum_ = 0
    for type_, len_ in lengths.items():
        sum_ += len_
        if i < sum_:
            return type_
