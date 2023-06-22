"""Custom data types. Annotation is used in place of docstrings, as I find it more human-friendly than the former. Annotations serve no other purpose."""
from typing import Generator, Callable, Dict, List, Optional

from pydantic import BaseModel
from pydantic_numpy import NDArray
import numpy as np
from datetime import datetime

Index = int  # min 0
SamplingRate = int  # as obtained from scipy.io.wavfile.read
Type_ = str  # train/test/validation
Label = int  # 0 to 11, boundaries included
PrepLayer = Callable  # a preprocessing layer outputting a fixed-lenght feature vector
Ratios = Dict[Type_, float]
Lengths = Dict[Type_, int]


class MyBaseModel(BaseModel):
    """A hashable version of BaseModel."""

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))


class Ratios(MyBaseModel):
    """A hashable dictionary of split ratios."""

    content: Dict

    def __hash__(self) -> int:
        return hash("".join([f"{k}{v}" for k, v in self.content.items()]))


class Recording(MyBaseModel):
    """A hashable model of a recording."""

    content: NDArray
    mfsc_features: Optional[NDArray] = None
    encoded_features: Optional[NDArray] = None
    sampling_rate: Optional[SamplingRate] = None

    def __len__(self) -> int:
        return self.content.size

    def __hash__(self) -> int:
        return hash(str(self.content))

    def __eq__(self, __value: object) -> bool:
        if (
            self.content.size == __value.content.size
            and self.sampling_rate == __value.sampling_rate
        ):
            return np.equal(self.content, __value.content).all()
        return False


class Tidigit(MyBaseModel):
    """A hashable model of a TIDIGITS data point."""

    index: Index
    recording: Recording
    label: Optional[Label] = None
    cat: Optional[Type_] = None

    def __hash__(self) -> int:
        return hash(self.recording)


class TiData(BaseModel):
    """A set of TIDIGITS data points."""

    data: List[Tidigit]

    def __len__(self) -> int:
        return len(self.data)

    def __setitem__(self, index: int, dp: Tidigit) -> None:
        self.data.insert(index, dp)

    def __getitem__(self, index: int) -> Tidigit:
        return self.data[index]

    def __delitem__(self, index: int) -> None:
        del self.data[index]

    def __iter__(self) -> Generator[Tidigit, None, None]:
        for x in self.data:
            yield x

    def __hash__(self) -> int:
        return hash(str([hash(x) for x in self.data]))


class Neuron(BaseModel):
    """A model of a convolutional layer neuron of the SpeechEncoder."""

    index: Index
    weights_index: Index
    time_index: Optional[Index] = None
    f_map: Index
    rec_field: List[Index]
    potential: float = 0
    ttfs: Optional[int] = None  # time to first spike
    inhibited: bool = False


Neurons = List[Neuron]


class Weights(BaseModel):
    """A model of input to convolutional layer weights of the SpeechEncoder."""

    index: Index
    n_members: int
    f_map: Index
    content: NDArray


# weights of the entire model
ModelWeights = List[Weights]


class SerialisedSpeechEncoder(BaseModel):
    # id
    creation_time: datetime

    # objects
    weights: ModelWeights
    neurons: Neurons
    prep_layer: PrepLayer

    # architecture
    conv_rf: int
    conv_size: int
    conv_stride: int
    fm_count: int
    freq_bands: int
    pool_rf: int
    pool_stride: int
    time_frames: int
    ws_count: int
    wsg_sizes: List[Index]
    w_mean: float
    w_sd: float

    # parameters
    a_minus: float
    a_plus: float
    conv_th: float
    in_th: float


# builds an instance of a Neuron from its index in its weigh-sharing group, the index of the feature map it belongs to, and the index of its weight-sharing group (other parameters fixed upon initiation of SpeechEncoder)
NeuronBuilder = Callable

# extract a predictor from a TIDIGITS data point
Predictor = Callable
