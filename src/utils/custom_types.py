from typing import Tuple, List, Dict, Annotated
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.io.wavfile import read as read_wav


@dataclass
class MinVal:
    min: int


@dataclass
class Range:
    min: int
    max: int


Index = Annotated[int, MinVal(0)]
Label = Annotated[int, Range(0, 11)]
SamplingRate = Annotated[int, read_wav]
Recording = Annotated[NDArray, read_wav]

WavContent = Tuple[SamplingRate, Recording]
Datapoint = Tuple[Index, WavContent, Label]
Data = List[Datapoint]
SplitData = Tuple[Data, Data, Data]
Ratios = Dict[str, float]
Lengths = Tuple[int, int, int]
