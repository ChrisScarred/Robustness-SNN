"""SNN-based Speech Encoder, an implementation of Dong et al. design described in 'Unsupervised speech recognition through spike-timing-dependent plasticity in a convolutional spiking neural network' (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204596)."""
import os
from math import ceil
from typing import Any, List, Optional
from itertools import product
import numpy as np
from numpy.typing import NDArray
from functools import lru_cache

from src.utils.custom_types import Data, PrepLayer, Recording, Neuron, Weights
from src.utils.defaults import *
from src.utils.se import (
    get_build_params,
    get_model_weights,
    get_neuron_builder,
    get_neurons,
    get_input_spikes_at_t
)
from src.utils.defaults import IN_TH, CONV_TH, POOL_RF


class SpeechEncoder:
    def __init__(self, prep_func: PrepLayer, **kwargs) -> None:
        self.prep_func = prep_func
        self.params = kwargs
        self.neurons = None
        self.weights = None
        self.training = True
        self.build()

    def set_training(self, to_train: bool) -> None:
        self.training = to_train

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
            # TODO: learning rate from config
            self.a_plus = self._get_param("a_plus", 0.05)
            self.a_minus = self._get_param("a_minus", 0.05)
            self.neurons = get_neurons(self.weights, neuron_builder)

    def reset_neurons(self):
        for neuron in self.neurons:
            neuron.potential = 0
            neuron.ttfs = None
            neuron.inhibited = False

    def process(self, rec: Recording) -> Any:
        r = self.prep_func(rec)
        r = r.mfsc_features
        r = self.input_layer(r)
        r = self.conv_layer(r)
        r = self.pool_layer(r)
        return r

    def input_layer(self, mfsc: NDArray) -> NDArray:
        spike_times = np.ceil(self.in_th / mfsc)
        spike_times[spike_times < 0] = -1
        return spike_times
    
    @lru_cache
    def _find_neuron(self, t: int, f: int) -> Neuron:
        for n in self.neurons:
            if n.f_map==f:
                if n.time_index==t:
                    return n

    def _inhibit(self, neuron: Neuron) -> None:
        t = neuron.time_index
        f = neuron.f_map
        to_inhibit = [(t, x) for x in range(self.fm_count) if x!=f]
        # NOTE: Dong et al. do not specify which 'neighbourhood' is inhibited
        to_inhibit.extend([(t-1, x) for x in range(f-1, f+2)])
        to_inhibit.extend([(t+1, x) for x in range(f-1, f+2)])
        for i, j in to_inhibit:
            n = self._find_neuron(i, j)
            if n:
                n.inhibited = True

    @lru_cache
    def _get_weights(self, index: int) -> Weights:
        for w in self.weights:
            if w.index == index:
                return w
            
    def _weight_update(self, s: NDArray, w: NDArray) -> NDArray:
        if s == 1:
            return w + self.a_plus * w * (1 - w)
        if s == 0:
            return w - self.a_minus * w * (1 - w)
        return w

    def _stdp(self, neuron: Neuron, rf_spikes: NDArray, weights: Weights) -> None:
        wc = weights.content
        s = rf_spikes
        if not neuron.ttfs:
            s = rf_spikes.fill(0)
        weights.content = np.vectorize(self._weight_update)(s, wc)

    def _potential_at_t(self, neuron: Neuron, in_spikes: NDArray, t: int) -> None:
        if not neuron.inhibited and not neuron.ttfs:
            p = neuron.potential
            w = self._get_weights(neuron.weights_index)
            if w:
                wc = w.content
                rf_spikes = in_spikes[neuron.rec_field, :]
                rf_spikes = rf_spikes.flatten()
                w_slice = wc
                if wc.size > rf_spikes.size:
                    w_slice = wc[0:rf_spikes.size]
                p += w_slice.T @ rf_spikes
                if self.training:
                    self._stdp(neuron, rf_spikes, w)
                if p >= self.conv_th:
                    neuron.ttfs = t + 1
                    neuron.potential = 0
                    self._inhibit(neuron)

    def _get_ttfs(self) -> NDArray:
        self.neurons.sort(key=lambda x: x.index)
        return np.array([neuron.ttfs if neuron.ttfs else -1 for neuron in self.neurons])

    def conv_layer(self, spike_times: NDArray) -> NDArray:
        self.reset_neurons()

        # starts at t=1 because a spike cannot have happened at t=0
        t_range = range(1, int(np.max(spike_times) + 1))
        in_spikes = [get_input_spikes_at_t(spike_times, t) for t in t_range]
        for t, neuron in product(t_range, self.neurons):
            self._potential_at_t(neuron, in_spikes[t-1], t)
        
        return self._get_ttfs()

    def pool_layer(self, ttfs: NDArray) -> NDArray:
        conv_size = self.conv_size
        rf = self.pool_rf
        st = self.pool_stride
        n = ceil((conv_size + 1) / st)

        ttfs = np.reshape(ttfs, (conv_size, self.fm_count))
        
        # NOTE: Dong et al. used weights of 1, but in the case that the last pooling section is smaller than the rest, this introduces a scaling into the pooling layer, hence I used weights equal to 1/pooling section size. Given no further processing is performed, this is functionally the same operation
        return np.array([np.around(np.mean(ttfs[st * p:min(st * p + rf, conv_size - 1), :], axis=0)) for p in range(n)])

    def batch_process(self, data: Data) -> List[NDArray]:
        for dp in data:
            result = self.process(dp.recording)
            dp.recording.encoded_features = result
        return data

    def train(self, data: Data, convergence: Optional[float] = None, epochs: int = 1, batch_size: Optional[int] = None) -> List[NDArray]:
        if not convergence:
            convergence = self.a_plus / 10
        if not batch_size:
            batch_size = epochs
        for i in range(epochs):
            print(f"Epoch {i+1}/{epochs}")
            res = self.batch_process(data)
        return res
