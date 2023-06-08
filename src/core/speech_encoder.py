"""SNN-based Speech Encoder, an implementation of Dong et al. design described in 'Unsupervised speech recognition through spike-timing-dependent plasticity in a convolutional spiking neural network' (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204596)."""
import os
import pickle
from datetime import datetime
from functools import lru_cache
from itertools import product
from math import ceil, floor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.utils.custom_types import (Data, Neuron, Recording,
                                    SerialisedSpeechEncoder, Weights)
from src.utils.defaults import *
from src.utils.defaults import (A_MINUS, A_PLUS, CONV_RF, CONV_TH, F_MAPS,
                                FREQ_BANDS, IN_TH, LOAD_FILE, LOAD_SE,
                                MODEL_DIR, POOL_RF, SAVE_FILE, TIME_FRAMES,
                                TRAINING, WSG, W_MEAN, W_SD)


class SpeechEncoder:
    def __init__(self, **params) -> None:
        # cannot build or load a model without these prerequisites
        path, path_valid = self._prerequisites(params)
        
        if self.load_model and path_valid:
            with open(path, "rb") as f:
                self._load_model_from_sse(pickle.load(f), params)

        else:
            self._build_model(params)

    def _prerequisites(self, params: Dict[str, Any]) -> Tuple[str, bool]:
        self.prep_layer = params.get("prep_layer")
        self.load_model = params.get("load_speech_encoder", LOAD_SE)
        self.load_path = params.get("load_file", LOAD_FILE)
        self.model_dir = params.get("model_folder", MODEL_DIR)
        path = os.path.join(self.model_dir, self.load_path)
        path_valid = os.path.isfile(path)

        assert self.prep_layer or self.load_model, "Missing prep layer."
        assert self.prep_layer or (self.load_model and path_valid), "Invalid model load path."

        return path, path_valid
    
    def _load_arch_params(self, params: Dict[str, Any]) -> None:
        self.conv_rf = params.get("conv_rf", CONV_RF)
        self.conv_stride = params.get("conv_stride", params.get("conv_rf", CONV_RF) - 1)
        self.fm_count = params.get("f_maps", F_MAPS)
        self.freq_bands = params.get("freq_bands", FREQ_BANDS)        
        self.pool_rf = params.get("pool_rf", POOL_RF)
        self.pool_stride = params.get("pool_stride", POOL_RF)
        self.time_frames = params.get("t_frames", TIME_FRAMES)
        self.ws_count = params.get("wsg", WSG)
        self.weight_mean = params.get("weight_mean", W_MEAN)
        self.weight_sd = params.get("weight_sd", W_SD)
        self.conv_size = ceil(self.time_frames / self.conv_stride)

    def _load_other_params(self, params: Dict[str, Any]) -> None:
        self.training = params.get("training", TRAINING)
        self.a_minus = params.get("a_minus", A_MINUS)
        self.a_plus = params.get("a_plus", A_PLUS)
        self.conv_th = params.get("conv_th", CONV_TH)
        self.in_th = params.get("in_th", IN_TH)
        self.save_file = params.get("save_file", SAVE_FILE)
        self.diff_th = self.a_plus / 10**3

    def _get_wsg_sizes(self) -> None:
        naive = self.conv_size / self.ws_count
        if naive.is_integer():
            return [naive] * self.ws_count
        floored = floor(naive)
        ceiled = ceil(naive)
        last = self.conv_size - floored * (self.ws_count - 1)
        diff = last - floored
        sizes = [ceiled] * diff
        sizes.extend([floored] * (self.ws_count - diff))
        self.wsg_sizes = sizes

    def _init_weights(self) -> None:
        self.weights = [
            Weights(
                index=i,
                n_members=size,
                f_map=mapi,
                content=np.random.normal(size=(self.conv_rf * self.freq_bands,), loc=self.weight_mean, scale=self.weight_sd),
            )
            for i, (mapi, size) in enumerate(product(range(self.fm_count), self.wsg_sizes))
        ]

    def _build_neuron(self, n: int, weights: Weights) -> Neuron:
        i = weights.index % len(self.wsg_sizes)
        neuron_index = n + sum(self.wsg_sizes[:i]) + weights.f_map * sum(self.wsg_sizes)
        
        start = int((neuron_index % self.conv_size) * self.conv_stride)
        rf = [_ for _ in range(start, min(start + self.conv_rf, self.time_frames))]

        return Neuron(
            index=neuron_index,
            weights_index=weights.index,
            time_index=neuron_index % self.conv_size,
            f_map=weights.f_map,
            rec_field=rf,
        )

    def _build_model(self, params: Dict[str, Any]) -> None:
        self._load_arch_params(params)
        self._load_other_params(params)
        self._get_wsg_sizes()
        self._init_weights()
        self.neurons = [self._build_neuron(n, w) for w in self.weights for n in range(w.n_members)]

    def _load_model_from_sse(self, sse: SerialisedSpeechEncoder, params: Dict[str, Any]) -> None:
        # objects
        self.weights = sse.weights
        self.neurons = sse.neurons
        self.prep_layer = sse.prep_layer
        
        # architecture
        self.conv_rf = sse.conv_rf
        self.conv_size = sse.conv_size
        self.conv_stride = sse.conv_stride
        self.fm_count = sse.fm_count
        self.freq_bands = sse.freq_bands
        self.pool_rf = sse.pool_rf
        self.pool_stride = sse.pool_stride
        self.time_frames = sse.time_frames
        self.ws_count = sse.ws_count
        self.wsg_sizes = sse.wsg_sizes
        self.weight_mean = sse.w_mean
        self.weight_sd = sse.w_sd

        # parameters
        a_minus = params.get("a_minus")
        if a_minus:
            self.a_minus = a_minus
        else:
            self.a_minus = sse.a_minus
        
        a_plus = params.get("a_plus")
        if a_plus:
            self.a_plus = a_plus
        else:
            self.a_plus = sse.a_plus

        conv_th = params.get("conv_th")
        if conv_th:
            self.conv_th = conv_th
        else:
            self.conv_th = sse.conv_th

        in_th = params.get("in_th")
        if in_th:
            self.in_th = in_th
        else:
            self.in_th = sse.in_th
        
        self.training = params.get("training", TRAINING)
        self.save_file = params.get("save_file", SAVE_FILE)
        

    def set_training(self, to_train: bool) -> None:
        self.training = to_train

    def _reset_neurons(self):
        for neuron in self.neurons:
            neuron.potential = 0
            neuron.ttfs = None
            neuron.inhibited = False

    def process(self, rec: Recording) -> Any:
        r = self.prep_layer(rec)
        r = r.mfsc_features
        r = self._input_layer(r)
        r = self._conv_layer(r)
        r = self._pool_layer(r)
        return r

    def _input_layer(self, mfsc: NDArray) -> NDArray:
        spike_times = np.ceil(self.in_th / mfsc)
        spike_times[spike_times < 0] = -1
        return spike_times

    @lru_cache
    def _find_neuron(self, t: int, f: int) -> Neuron:
        for n in self.neurons:
            if n.f_map == f:
                if n.time_index == t:
                    return n

    def _inhibit(self, neuron: Neuron) -> None:
        t = neuron.time_index
        f = neuron.f_map
        to_inhibit = [(t, x) for x in range(self.fm_count) if x != f]
        # NOTE: Dong et al. do not specify which 'neighbourhood' is inhibited
        to_inhibit.extend([(t - 1, x) for x in range(f - 1, f + 2)])
        to_inhibit.extend([(t + 1, x) for x in range(f - 1, f + 2)])
        for i, j in to_inhibit:
            n = self._find_neuron(i, j)
            if n:
                n.inhibited = True

    @lru_cache
    def _get_weights(self, index: int) -> Weights:
        for w in self.weights:
            if w.index == index:
                return w

    def _stdp(self, neuron: Neuron, rf_spikes: NDArray, weights: Weights) -> None:
        w_arr = weights.content
        if not neuron.ttfs:
            rf_spikes.fill(0)
        if w_arr.size > rf_spikes.size:
            rf_spikes = np.append(rf_spikes, [-1] * (w_arr.size - rf_spikes.size))
        s_list = rf_spikes.tolist()
        w_list = w_arr.tolist()
        updates = []
        for s, w in zip(s_list, w_list):
            u = 0
            if s == 1:
                u = self.a_plus * w * (1 - w)
            if s == 0:
                u = -1 * self.a_minus * w * (1 - w)
            if abs(u) < self.diff_th:
                u = 0
            updates.append(u)
        updates = np.array(updates)
        weights.content = w_arr + updates

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
                    w_slice = wc[0 : rf_spikes.size]
                p += w_slice.T @ rf_spikes
                if self.training:
                    self._stdp(neuron, rf_spikes, w)
                    self.record_diff = True
                if p >= self.conv_th:
                    neuron.ttfs = t + 1
                    neuron.potential = 0
                    self._inhibit(neuron)

    def _get_ttfs(self) -> NDArray:
        self.neurons.sort(key=lambda x: x.index)
        return np.array([neuron.ttfs if neuron.ttfs else 0 for neuron in self.neurons])

    def _conv_layer(self, spike_times: NDArray) -> NDArray:
        self._reset_neurons()

        # starts at t=1 because a spike cannot have happened at t=0
        t_range = range(1, int(np.max(spike_times) + 1))
        in_spikes = [np.where(np.asarray(spike_times == t), 1, 0) for t in t_range]
        for t in t_range:
            old_weights = [w.content for w in self.weights]
            self.record_diff = False 
            for neuron in self.neurons:
                self._potential_at_t(neuron, in_spikes[t - 1], t)
            if self.record_diff:
                new_weights = [w.content for w in self.weights]
                self.weight_diff = max([np.max(np.abs(n - o)) for n, o in zip(new_weights, old_weights)])
        return self._get_ttfs()

    def _pool_layer(self, ttfs: NDArray) -> NDArray:
        conv_size = self.conv_size
        rf = self.pool_rf
        st = self.pool_stride
        n = ceil((conv_size + 1) / st)

        ttfs = np.reshape(ttfs, (conv_size, self.fm_count))

        return np.array(
            [ 
                np.sum(ttfs[st * p : min(st * p + rf, conv_size - 1), :], axis=0)
                for p in range(n)
            ]
        )

    def batch_process(self, data: Data) -> List[NDArray]:
        for dp in data:
            result = self.process(dp.recording)
            dp.recording.encoded_features = result
        return data

    def save(self, model_dir: Optional[str] = None, f_name: Optional[str] = None) -> None:
        if not model_dir:
            model_dir = self.model_dir
        if not f_name:
            f_name = self.save_file

        self._reset_neurons()

        now = datetime.now()
        sse = SerialisedSpeechEncoder(
            creation_time = now,
            weights = self.weights,
            neurons = self.neurons,
            prep_layer = self.prep_layer,
            conv_rf = self.conv_rf,
            conv_size = self.conv_size,
            conv_stride = self.conv_stride,
            fm_count = self.fm_count,
            freq_bands = self.freq_bands,
            pool_rf = self.pool_rf,
            pool_stride = self.pool_stride,
            time_frames = self.time_frames,
            ws_count = self.ws_count,
            wsg_sizes = self.wsg_sizes,
            w_mean = self.weight_mean,
            w_sd = self.weight_sd,
            a_minus = self.a_minus,
            a_plus = self.a_plus,
            conv_th = self.conv_th,
            in_th = self.in_th
        )
        f_name += "_" + now.strftime("%d-%m-%Y_%H-%M-%S")
        path = os.path.join(model_dir, f_name)
        with open(path, "wb") as f:
            pickle.dump(sse, f)
        print(f"Model saved at {path}.")

    def train(
        self,
        data: Data,
        diff_th: Optional[float] = None,
        epochs: int = 1,
        batch_size: Optional[int] = None,
    ) -> List[NDArray]:
        self.set_training(True)
        if diff_th:
            self.diff_th = diff_th
        if not batch_size:
            batch_size = epochs
        self.weight_diff = 10
        for i in range(epochs):
            print(f"Epoch {i+1}/{epochs}")
            res = self.batch_process(data)
            print(f"Current weight diff: {self.weight_diff}")
            if self.weight_diff < self.diff_th:
                break
            if i % batch_size == 0 and i != 0:
                self.save()
        self.save()
        return res
