import random
from src.utils.config import Config
from src.custom_types import Ratios, Data, SplitData, Lengths
import math

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


def _stratified_split(ratios: Ratios, data: Data) -> SplitData:
    # TODO: Implement
    pass


def _naive_split(ratios: Ratios, data: Data) -> SplitData:
    shuffled = _shuffling(data)
    lengths = _get_lengths(ratios, len(data))
    a = lengths[0]-1
    b = a + lengths[1]
    return (shuffled[:a], shuffled[a+1:b], shuffled[b+1:])

    

def train_test_validation(config: Config, data: Data) -> SplitData:
    ratios = _get_ratios(config)
    seed = config.get("seed")
    random.seed(seed)
    if config.get("split.stratified"):
        return _stratified_split(ratios, data)
    return _naive_split(ratios, data)
