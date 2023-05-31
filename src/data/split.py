"""Optionally stratified train/test/vaslidation splitting functions."""
import random
from itertools import groupby
from typing import Any

from src.utils.custom_types import Data, Ratios
from src.utils.misc import lengths_from_ratios, cat_from_lengths
from src.utils.caching import region


def _shuffling(data: Data) -> Data:
    """Reproducibly shuffle the input data.
    """
    data.data.sort(key=lambda x: x.index)
    indices = list(range(len(data.data)))
    random.shuffle(indices)
    data.data = [data.data[i] for i in indices]
    return data


def _stratified_split(ratios: Ratios, data: Data) -> Data:
    """Perform a stratified split on the supplied data with the supplied train/test/validation ratios.

    Args:
        ratios (Ratios): The requested train/test/validation ratios.
        data (Data): The data to split.

    Returns:
        Data: The split data.
    """
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


def _naive_split(ratios: Ratios, data: Data) -> Data:
    """Perform a naive (non-stratified) split on the supplied data with the supplied train/test/validation ratios.

    Args:
        ratios (Ratios): The requested train/test/validation ratios.
        data (Data): The data to split.

    Returns:
        Data: The split data.
    """
    lengths = lengths_from_ratios(ratios, len(data))
    shuffled_data = _shuffling(data)
    for i, x in enumerate(shuffled_data.data):
        x.cat = cat_from_lengths(i, lengths)
    return shuffled_data


@region.cache_on_arguments()
def train_test_validation(
    data: Data, ratios: Ratios, seed: Any, stratified: bool
) -> Data:
    """Cached, randomised, and reproducible train/test/validation split of the supplied data with the requested ratios.

    Args:
        data (Data): The data to split.
        ratios (Ratios): The train/test/validation ratios requested.
        seed (Any): The seed to use to ensure reproducibility.
        stratified (bool): `True` if the split is stratified, `False` otherwise.

    Returns:
        Data: The split data.
    """
    random.seed(seed)
    if stratified:
        return _stratified_split(ratios, data)
    return _naive_split(ratios, data)
