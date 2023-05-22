from dataclasses import dataclass
from typing import Annotated, Generator, Callable, Dict, List, Optional

from pydantic import BaseModel
from pydantic_numpy import NDArray


@dataclass
class MinVal:
    min: int


@dataclass
class Range:
    min: int
    max: int


Index = Annotated[int, MinVal(0)]
SamplingRate = Annotated[int, "scipy.io.wavfile.read"]
Recording = Annotated[NDArray, "scipy.io.wavfile.read"]
Type_ = Annotated[str, "train/test/validation"]
Label = Annotated[int, Range(0, 11)]
Recordings = List[Recording]


class DataPoint(BaseModel):
    index: Index
    sampling_rate: SamplingRate
    wav: Recording
    label: Label
    type_: Optional[Type_] = None


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


Ratios = Dict[Type_, float]
Lengths = Dict[Type_, int]
Config = Annotated[Callable, "loaded configuration object from src.utils.config"]
PrepLayer = Annotated[Callable, "a preprocessing layer outputting fixed-lenght data"]
