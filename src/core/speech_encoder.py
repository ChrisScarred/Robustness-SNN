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
from src.utils.defaults import IN_TH, CONV_TH, POOL_RF


class SpeechEncoder:
    def __init__(self, prep_func: PrepLayer, **kwargs) -> None:
        self.prep_func = prep_func
        self.params = kwargs
        self.neurons = None
        self.weights = None
        self.build()

    def _get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

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
            self.conv_th = self._get_param("conv_th", default=CONV_TH)
            self.in_th = self._get_param("in_th", default=IN_TH)
            self.pool_stride = self._get_param("pool_stride", default=POOL_RF)
            self.pool_rf = self._get_param("pool_rf", default=POOL_RF)
            self.conv_size = ceil(time_frames / conv_stride)
            self.fm_count = fm_count
            self.weights, sizes = get_model_weights(
                self.conv_size, ws_count, fm_count, conv_rf, freq_bands
            )
            neuron_builder = get_neuron_builder(
                sizes, conv_stride, conv_rf, time_frames, self.conv_size
            )
            
            self.neurons = get_neurons(self.weights, neuron_builder)

    def reset_neurons(self):
        for neuron in self.neurons:
            neuron.potential = 0
            neuron.ttfs = -1

    def process(self, rec: Recording) -> Any:
        r = self.prep_func(rec)        
        r = r.content
        r = self.input_layer(r)
        r = self.conv_layer(r)
        r = self.pool_layer(r)
        return r

    def input_layer(self, mfsc: NDArray) -> NDArray:
        spike_times = np.ceil(self.in_th / mfsc)
        spike_times[spike_times < 0] = -1
        return spike_times

    def conv_layer(self, spike_times: NDArray) -> NDArray:
        """
        TODO: Calculate the potential of neuron n as: A_nt = np.sum(I_t-1n [elem-wise mult] W_n) where A_nt is the potential of n at t, I_t-1n is the binary activation of the slice of input n is receptive to at time t-1 and W_n is the weight matrix n shares with the rest of its weight sharing group.
        """
        self.reset_neurons()

        for t in range(1, int(np.max(spike_times) + 1)):
            mask = np.asarray(spike_times/t==1)
            in_p = np.where(mask, 1, 0)
            for neuron in self.neurons:
                # if neuron has not spiked yet
                if neuron.ttfs == -1:
                    p = neuron.potential
                    w = self.weights[neuron.weights_index].content
                    rf = neuron.rec_field
                    if len(w) > len(rf):
                        w_slice = w[0:len(rf)-1]
                    else:
                        w_slice = w
                    p += np.sum(np.multiply(w_slice, in_p[rf, :]))
                    if p >= self.conv_th:
                        neuron.ttfs = t + 1
                        neuron.potential = 0

        ttfs = np.full((self.conv_size * self.fm_count), -1)
        for neuron in self.neurons:
            val = neuron.ttfs
            if val > 0:
                ttfs[neuron.index] = val
        ttfs = np.reshape(ttfs, (self.conv_size, self.fm_count))

        return ttfs

    def pool_layer(self, ttfs: NDArray) -> NDArray:
        # TODO: Implement pooling
        return ttfs

    def batch_process(self, data: Data) -> List[NDArray]:
        return [self.process(dp.recording) for dp in data]
