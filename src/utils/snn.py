from functools import partial
from itertools import product
from math import ceil, floor
from typing import Any, Dict, List, Tuple

import numpy as np

from src.utils.custom_types import (
    Index,
    ModelWeights,
    Neuron,
    NeuronBuilder,
    Neurons,
    Weights,
)
from src.utils.defaults import (
    CONV_RF,
    F_MAPS,
    FREQ_BANDS,
    LOAD_FILE,
    LOAD_SE,
    MODEL_DIR,
    TIME_FRAMES,
    WSG,
)


def get_rf(
    neuron_index: int, conv_stride: int, conv_rf: int, time_frames: int, conv_size: int
) -> List[Index]:
    start = int((neuron_index % conv_size) * conv_stride)
    return [_ for _ in range(start, min(start + conv_rf, time_frames))]


def get_build_params(
    params: Dict[str, Any]
) -> Tuple[int, int, int, int, int, bool, int, str, str]:
    return (
        params.get("t_frames", TIME_FRAMES),
        params.get("conv_rf", CONV_RF),
        params.get("conv_stride", params.get("conv_rf", CONV_RF) - 1),
        params.get("f_maps", F_MAPS),
        params.get("freq_bands", FREQ_BANDS),
        params.get("load_speech_encoder", LOAD_SE),
        params.get("wsg", WSG),
        params.get("model_folder", MODEL_DIR),
        params.get("load_file", LOAD_FILE),
    )


def get_neuron_index(n: int, weights: Weights, sizes: List[int]) -> Index:
    i = weights.index % len(sizes)
    return n + sum(sizes[:i]) + weights.f_map * sum(sizes)


def get_neuron_builder(
    wsg_sizes: List[int],
    conv_stride: int,
    conv_rf: int,
    time_frames: int,
    conv_size: int,
) -> NeuronBuilder:
    rf_getter = partial(
        get_rf,
        conv_stride=conv_stride,
        conv_rf=conv_rf,
        time_frames=time_frames,
        conv_size=conv_size,
    )
    get_neuron_i = partial(get_neuron_index, sizes=wsg_sizes)
    return lambda n, w: Neuron(
        index=get_neuron_i(n, w),
        weights_index=w.index,
        f_map=w.f_map,
        rec_field=rf_getter(get_neuron_i(n, w)),
    )


def _get_wsg_sizes(conv_size: int, ws_count: int) -> List[int]:
    naive = conv_size / ws_count
    if naive.is_integer():
        return [naive] * ws_count
    floored = floor(naive)
    ceiled = ceil(naive)
    last = conv_size - floored * (ws_count - 1)
    diff = last - floored
    sizes = [ceiled] * diff
    sizes.extend([floored] * (ws_count - diff))
    return sizes


def get_model_weights(
    conv_size: int, ws_count: int, f_maps: int, conv_rf: int, freq_bands: int
) -> Tuple[ModelWeights, List[int]]:
    sizes = _get_wsg_sizes(conv_size, ws_count)
    return [
        Weights(
            index=i,
            n_members=size,
            f_map=mapi,
            content=np.random.normal(size=(conv_rf, freq_bands)),
        )
        for i, (mapi, size) in enumerate(product(range(f_maps), sizes))
    ], sizes


def get_neurons(weights: ModelWeights, neuron_builder: NeuronBuilder) -> Neurons:
    return [neuron_builder(n, w) for w in weights for n in range(w.n_members)]
