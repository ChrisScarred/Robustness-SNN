"""Speech encoder-related utility functions."""
from functools import partial
from itertools import product
from math import ceil, floor
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

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
    """Get the receptive field of a convolutional layer neuron as a list of row indices of the input layer neurons to which the convolutional layer neuron is receptive.

    Args:
        neuron_index (int): The 1D index of the convolutional layer neuron.
        conv_stride (int): The stride of receptive fields of convolutional layer neurons.
        conv_rf (int): The size of receptive fields of convolutional layer neurons.
        time_frames (int): The number of time frames in the input layer.
        conv_size (int): The number of neurons in one feature map of the convolutional layer.

    Returns:
        List[Index]: The receptive field of a convolutional layer neuron with 1D index of `neuron_index` as a list of row indices of the input layer neurons to which the convolutional layer neuron is receptive.
    """
    start = int((neuron_index % conv_size) * conv_stride)
    return [_ for _ in range(start, min(start + conv_rf, time_frames))]


def get_build_params(
    kwargs: Dict[str, Any]
) -> Tuple[int, int, int, int, int, bool, int, str, str]:
    """Obtain the build parameters from the kwargs passed to the SpeechEncoder object, using the defaults defined in src.utils.defaults for fallback options.

    Args:
        kwargs (Dict[str, Any]): The kwargs passed to the SpeechEncoder object at initialisation.

    Returns:
        The following parameters of the SpeechEncoder:
            number of time frames,
            receptive field of the convolutional layer neurons,
            stride of the receptive fields of the convolutional layer neurons,
            the number of feature maps in the convolutional layer,
            the number of frequency bands in the input layer,
            indicator whether the SpeechEncoder object should be loaded from a file,
            the number of weight-sharing groups,
            the path to the model folder,
            the path to the load file.
    """
    return (
        kwargs.get("t_frames", TIME_FRAMES),
        kwargs.get("conv_rf", CONV_RF),
        kwargs.get("conv_stride", kwargs.get("conv_rf", CONV_RF) - 1),
        kwargs.get("f_maps", F_MAPS),
        kwargs.get("freq_bands", FREQ_BANDS),
        kwargs.get("load_speech_encoder", LOAD_SE),
        kwargs.get("wsg", WSG),
        kwargs.get("model_folder", MODEL_DIR),
        kwargs.get("load_file", LOAD_FILE),
    )


def get_neuron_index(n: int, weights: Weights, sizes: List[int]) -> Index:
    """Given an index of a convolutional layer neuron within its weigh-sharing group, and its weights object, obtain the 1D neuron index wrt the entire convolutional layer.

    Args:
        n (int): The index of the neuron in its weight-sharing group.
        weights (Weights): The Weights object of this neuron.
        sizes (List[int]): The sizes of the weight-sharing groups in the number of neurons.

    Returns:
        Index: The 1D index of this neuron wrt the entire convolutional layer.
    """
    i = weights.index % len(sizes)
    return n + sum(sizes[:i]) + weights.f_map * sum(sizes)


def get_neuron_builder(
    wsg_sizes: List[int],
    conv_stride: int,
    conv_rf: int,
    time_frames: int,
    conv_size: int,
) -> NeuronBuilder:
    """Obtain a NeuronBuilder object that builds a Neuron object from the neuron's index in its weight-sharing group and the neuron's Weight object.

    Args:
        wsg_sizes (List[int]): The sizes of the weight-sharing groups.
        conv_stride (int): The stride of the receptive fields of the convolutional layer neurons.
        conv_rf (int): The size of the receptive fields of the convolutional layer neurons.
        time_frames (int): The number of time frames in the input layer.
        conv_size (int): The number of neurons in one feature map of the convolutional layer.

    Returns:
        NeuronBuilder: A function that builds a Neuron object from the neuron's index in its weight-sharing group and the neuron's Weight object.
    """
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
        time_index=get_neuron_i(n, w) % conv_size,
        f_map=w.f_map,
        rec_field=rf_getter(get_neuron_i(n, w)),
    )


def _get_wsg_sizes(conv_size: int, ws_count: int) -> List[int]:
    """Calculate the sizes in the number of neurons for the weight-sharing groups such that they are reasonably balanced.

    Args:
        conv_size (int): The size of one feature map of the convolutional layer in the number of neurons.
        ws_count (int): The number of weight-sharing groups.

    Returns:
        List[int]: The sizes of the weight-sharing groups.
    """
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


def get_input_spikes_at_t(spike_times: NDArray, t: int) -> NDArray:
    mask = np.asarray(spike_times == t)
    return np.where(mask, 1, 0)


def get_model_weights(
    conv_size: int, ws_count: int, f_maps: int, conv_rf: int, freq_bands: int
) -> Tuple[ModelWeights, List[int]]:
    """Obtain model weights represented by Weights objects.

    Args:
        conv_size (int): The size of the covolutional layer in the number of neurons per one feature map.
        ws_count (int): The number of weight-sharing groups.
        f_maps (int): The number of feature maps.
        conv_rf (int): The receptive fields of convolutional layer neurons in the number of input layer neurons to which the concolutional layer neurons are sensitive.
        freq_bands (int): The number of frequency bands in the input layer.

    Returns:
        Tuple[ModelWeights, List[int]]: The list of model's weight represented by Weights objects, the sizes of weight-sharing groups.
    """
    sizes = _get_wsg_sizes(conv_size, ws_count)
    return [
        Weights(
            index=i,
            n_members=size,
            f_map=mapi,
            content=np.random.normal(size=(conv_rf * freq_bands,)),
        )
        for i, (mapi, size) in enumerate(product(range(f_maps), sizes))
    ], sizes


def get_neurons(weights: ModelWeights, neuron_builder: NeuronBuilder) -> Neurons:
    """Given model's weights and a neuron builder, obtain the convolutional layer neurons of this model.

    Args:
        weights (ModelWeights): The list of model's weight represented by Weights objects.
        neuron_builder (NeuronBuilder): A function that builds a Neuron object from the neuron's index in its weight-sharing group and the neuron's Weight object.

    Returns:
        Neurons: A list of neurons in the convolutional layer of this model represented by a Neuron object.
    """
    return [neuron_builder(n, w) for w in weights for n in range(w.n_members)]
