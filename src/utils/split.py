import math
import random
from itertools import groupby
from typing import Any

from src.utils.custom_types import Data, Lengths, Ratios, Config, Type_
from src.utils.parsing import get_ratios


def _shuffling(data: Data) -> Data:
    data.data.sort(key=lambda x: x.index)
    indices = list(range(len(data.data)))
    random.shuffle(indices)
    data.data = [data.data[i] for i in indices]
    return data


def _stratified_split(ratios: Ratios, data: Data) -> Data:
    groups = {}
    for k, group in groupby(data.data, lambda x: x.label):
        if groups.get(k):
            groups[k].data.extend(list(group))
        else:
            groups[k] = Data(data=list(group))
    results = Data(data=[])
    for label, files in groups.items():
        print(f"Label {label} has {len(files)} data points.")
        split = _naive_split(ratios, files)
        results.data.extend(split.data)
    return results


def _get_lengths(ratios: Ratios, data_len: int) -> Lengths:
    lengths = {}
    max_ln = len(list(ratios.values()))
    for i, (type_, ratio) in enumerate(ratios.items()):
        if i == max_ln - 1:
            sum_ln = sum(lengths.values())
            lengths[type_] = data_len - sum_ln
        else:
            len_ = math.floor(ratio * data_len)
            lengths[type_] = len_
    return lengths


def _get_group(i: int, lengths: Lengths) -> Type_:
    sum_ = 0
    for type_, len_ in lengths.items():
        sum_ += len_
        if i < sum_:
            return type_


def _naive_split(ratios: Ratios, data: Data) -> Data:
    lengths = _get_lengths(ratios, len(data))
    shuffled_data = _shuffling(data)
    for i, x in enumerate(shuffled_data.data):
        x.type_ = _get_group(i, lengths)
    return shuffled_data


def train_test_validation(data: Data, ratios: Ratios, seed: Any, stratified: bool) -> Data:
    random.seed(seed)
    if stratified:
        return _stratified_split(ratios, data)
    return _naive_split(ratios, data)


def ttv_from_config(config: Config, data: Data) -> Data:
    ratios = get_ratios(config)
    seed = config.get("seed")
    stratified = config.get("split.stratified")
    return train_test_validation(data, ratios, seed, stratified)
