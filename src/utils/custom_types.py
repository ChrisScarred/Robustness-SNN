from typing import Annotated, Generator, Callable, Dict, List, Optional

from pydantic import BaseModel
from pydantic_numpy import NDArray
import numpy as np

Index = Annotated[int, "min 0"]
SamplingRate = Annotated[int, "scipy.io.wavfile.read"]
Type_ = Annotated[str, "train/test/validation"]
Label = Annotated[int, "0 to 11, boundaries included"]
Ratios = Dict[Type_, float]
Lengths = Dict[Type_, int]
Config = Annotated[Callable, "loaded configuration object from src.utils.config"]
PrepLayer = Annotated[Callable, "a preprocessing layer outputting fixed-lenght data"]


class MyBaseModel(BaseModel):
    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))


class Ratios(MyBaseModel):
    content: Dict

    def __hash__(self) -> int:
        return hash("".join([f"{k}{v}" for k, v in self.content.items()]))


class Recording(MyBaseModel):
    content: NDArray
    sampling_rate: Optional[SamplingRate] = None

    def __len__(self) -> int:
        return self.content.size

    def __hash__(self) -> int:
        return hash(str(self.content))

    def __eq__(self, __value: object) -> bool:
        if self.content.size == __value.content.size:
            return np.equal(self.content, __value.content).all()
        return False


class DataPoint(MyBaseModel):
    index: Index
    recording: Recording
    label: Label
    type_: Optional[Type_] = None

    def __hash__(self) -> int:
        return hash(self.recording)


class Data(BaseModel):
    data: List[DataPoint]

    def __len__(self) -> int:
        return len(self.data)

    def __setitem__(self, index: int, dp: DataPoint) -> None:
        self.data.insert(index, dp)

    def __getitem__(self, index: int) -> DataPoint:
        return self.data[index]

    def __delitem__(self, index: int) -> None:
        del self.data[index]

    def __iter__(self) -> Generator[DataPoint, None, None]:
        for x in self.data:
            yield x

    def __hash__(self) -> int:
        return hash(str([hash(x) for x in self.data]))


class Neuron(BaseModel):
    index: Index
    weights_index: Index
    f_map: Index
    rec_field: List[Index]
    potential: float = 0


Neurons = List[Neuron]


class Weights(BaseModel):
    index: Index
    n_members: int
    f_map: Index
    content: NDArray


ModelWeights = Annotated[List[Weights], "weights of the entire model"]


class SavedSpeechEncoder(BaseModel):
    name: str
    weights: ModelWeights
    neurons: Neurons
    freq_bands: int
    time_frames: int


NeuronBuilder = Annotated[
    Callable,
    "builds an instance of a Neuron from its index in its weigh-sharing group, the index of the feature map it belongs to, and the index of its weight-sharing group (other parameters fixed upon initiation of SpeechEncoder)",
]
