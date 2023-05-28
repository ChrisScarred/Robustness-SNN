from typing import Any, List
from src.utils.custom_types import Recording, PrepLayer, Data
from numpy.typing import NDArray
import numpy as np
from math import floor
import pickle
import os
from functools import lru_cache


@lru_cache
def weight_index(i: int, w: int, j: int) -> int:
    return i*w + j


class SpeechEncoder:
    def __init__(self, prep_func: PrepLayer, **kwargs) -> None:
        self.prep_func = prep_func
        self.params = kwargs
        self.build()

    def build(self) -> None:
        """Initialises or loads (depending on the params passed at initiation) the weights of the model and the neuron 2D index to weight 1D index mapping.
        
        TODO: add loading of the mapping of convolutional neuron row index (conv_size, not f_map) to indices of input neurons the convolutional neural is receptive to. Calculating the potential of neuron n then is: A_nt = np.sum(I_t-1n [elem-wise mult] W_n) where A_nt is the potential of n at t, I_t-1n is the binary activation of the slice of input n is receptive to at time t-1 and W_n is the weight matrix n shares with the rest of its weight sharing group.
        """
        time_frames = self.params.get("t_frames", 40)
        conv_rf = self.params.get("conv_rf", 3)
        conv_stride = self.params.get("conv_stride", conv_rf - 1)
        f_maps = self.params.get("f_maps")
        freq_bands = self.params.get("freq_bands")
        load_weighs = self.params.get("load_weights")
        wsg = self.params.get("wsg")
        conv_size = floor(time_frames/conv_stride)+1

        if load_weighs:
            dir_ = self.params.get("weights_folder", "model")
            f_path = self.params.get("weights_file", "model")
            with open(os.path.join(dir_, f_path), "rb") as f:
                self.weights = pickle.load(f)
                
        else:
            weights = np.empty((f_maps*wsg), dtype=object)
            for i in range(f_maps):
                for j in range(wsg):
                    weights[weight_index(i, wsg, j)] = np.random.normal(size=(conv_rf, freq_bands))
            self.weights = weights

        wsg_size = floor(conv_size / wsg)
        neuron_weight_map = np.zeros((f_maps, conv_size), dtype=int)
        for i in range(f_maps):
            for j in range(wsg):
                group_len = wsg_size
                if j == wsg-1:
                    group_len = conv_size - wsg_size*(wsg-1)
                for k in range(group_len):
                    neuron_weight_map[i, j*wsg_size+k] = weight_index(i, wsg, j)
        
        self.neuron_weight_map = neuron_weight_map
        
    def process(self, rec: Recording) -> Any:
        r = self.prep_func(rec)
        r = r.content
        r = self.input_layer(r)
        r = self.conv_layer(r)
        r = self.pool_layer(r)
        return r

    def input_layer(self, mfsc: NDArray) -> NDArray:
        spike_times = np.ceil(self.params.get("in_th", 10.0) / mfsc)
        spike_times[spike_times < 0] = -1
        return spike_times

    def conv_layer(self, spike_times: NDArray) -> NDArray:
        activations = self.conv_neurons
        for t in range(1, np.max(spike_times)+1):
            # TODO
            pass
        return spike_times

    def pool_layer(self, activations: NDArray) -> NDArray:
        # TODO: Implement pooling layer
        return activations

    def batch_process(self, data: Data) -> List[NDArray]:
        return [self.process(dp.recording) for dp in data]
