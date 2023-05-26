from typing import Any, List
from src.utils.custom_types import Recording, PrepLayer, Data
from numpy.typing import NDArray
import numpy as np


class SpeechEncoder:
    def __init__(self, prep_func: PrepLayer, in_th: float) -> None:
        self.prep_func = prep_func
        self.in_th = in_th

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
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Qt5Agg")

        _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.imshow(mfsc)
        ax2.imshow(spike_times)
        plt.show()
        return spike_times

    def conv_layer(self, spike_times: NDArray) -> NDArray:
        # TODO: Implement convolution layer
        return spike_times

    def pool_layer(self, activations: NDArray) -> NDArray:
        # TODO: Implement pooling layer
        return activations

    def batch_process(self, data: Data) -> List[NDArray]:
        return [self.process(dp.recording) for dp in data]
