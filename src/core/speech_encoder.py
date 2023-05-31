"""SNN-based Speech Encoder, an implementation of Dong et al. design described in 'Unsupervised speech recognition through spike-timing-dependent plasticity in a convolutional spiking neural network' (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204596)."""
import os
from math import ceil
from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from src.utils.custom_types import Data, PrepLayer, Recording
from src.utils.defaults import *
from src.utils.se import (
    get_build_params,
    get_model_weights,
    get_neuron_builder,
    get_neurons,
)


class SpeechEncoder:
    def __init__(self, prep_func: PrepLayer, **kwargs) -> None:
        self.prep_func = prep_func
        self.params = kwargs
        self.neurons = None
        self.weights = None
        self.build()

    def build(self) -> None:
        """Initialises or loads (depending on the params passed at initiation) the weights of the model and the neuron 2D index to weight 1D index mapping.

        TODO: Implement model loading
        """
        (
            time_frames,
            conv_rf,
            conv_stride,
            fm_count,
            freq_bands,
            load_model,
            ws_count,
            dir_,
            load_path,
        ) = get_build_params(self.params)

        if load_model:
            path = os.path.join(dir_, load_path)
            print(f"Placeholder for loading the model from the file {path}")

        if not self.neurons or not self.weights:
            conv_size = ceil(time_frames / conv_stride)
            self.weights, sizes = get_model_weights(
                conv_size, ws_count, fm_count, conv_rf, freq_bands
            )
            neuron_builder = get_neuron_builder(
                sizes, conv_stride, conv_rf, time_frames, conv_size
            )
            self.neurons = get_neurons(self.weights, neuron_builder)
            for n in self.neurons:
                print(n)

    def process(self, rec: Recording) -> Any:
        r = self.prep_func(rec)
        r = r.content
        r = self.input_layer(r)
        r = self.conv_layer(r)
        r = self.pool_layer(r)
        return r

    def input_layer(self, mfsc: NDArray) -> NDArray:
        spike_times = np.ceil(self._get_param("in_th", 10.0) / mfsc)
        spike_times[spike_times < 0] = -1
        return spike_times

    def conv_layer(self, spike_times: NDArray) -> NDArray:
        """
        TODO: Calculate the potential of neuron n as: A_nt = np.sum(I_t-1n [elem-wise mult] W_n) where A_nt is the potential of n at t, I_t-1n is the binary activation of the slice of input n is receptive to at time t-1 and W_n is the weight matrix n shares with the rest of its weight sharing group.
        """
        for t in range(1, np.max(spike_times) + 1):
            pass
        return spike_times

    def pool_layer(self, activations: NDArray) -> NDArray:
        # TODO: Implement pooling layer
        return activations

    def batch_process(self, data: Data) -> List[NDArray]:
        return [self.process(dp.recording) for dp in data]
