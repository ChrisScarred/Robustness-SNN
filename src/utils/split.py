import math
import random
from itertools import groupby
from typing import Tuple

from src.utils.config import Config
from src.utils.custom_types import Data, Lengths, Ratios, SplitData


def _get_ratios(config: Config) -> Ratios:
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


def _shuffling(data: Data) -> Data:
    data.sort(key=lambda x: x[0])
    indices = list(range(len(data)))
    random.shuffle(indices)
    return [data[i] for i in indices]


def _get_lengths(ratios: Ratios, data_len: int) -> Lengths:
    lengths = []
    max_ln = len(list(ratios.values()))
    for i, ratio in enumerate(ratios.values()):
        if i == max_ln - 1:
            sum_ln = sum(lengths)
            lengths.append(data_len - sum_ln)
        else:
            len_ = math.floor(ratio * data_len)
            lengths.append(len_)
    return lengths


def _get_indices(lengths: Lengths) -> Tuple[int, int, int, int]:
    a = lengths[0] - 1
    b = a + lengths[1]
    return a, a + 1, b, b + 1


def _stratified_split(ratios: Ratios, data: Data) -> SplitData:
    groups = {}
    for k, group in groupby(data, lambda x: x[-1]):
        if groups.get(k):
            groups[k].extend(list(group))
        else:
            groups[k] = list(group)
    results = {}
    for label, files in groups.items():
        print(f"Label {label} has {len(files)} data points.")
        split = _naive_split(ratios, files)
        for type_, content in zip(ratios.keys(), split):
            if results.get(type_):
                results[type_].extend(content)
            else:
                results[type_] = content
    return tuple(results.values())


def _naive_split(ratios: Ratios, data: Data) -> SplitData:
    shuffled = _shuffling(data)
    a, b, c, d = _get_indices(_get_lengths(ratios, len(data)))
    return (shuffled[:a], shuffled[b:c], shuffled[d:])


def train_test_validation(config: Config, data: Data) -> SplitData:
    ratios = _get_ratios(config)
    seed = config.get("seed")
    random.seed(seed)
    if config.get("split.stratified"):
        return _stratified_split(ratios, data)
    return _naive_split(ratios, data)
