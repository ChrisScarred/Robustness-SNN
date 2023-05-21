from typing import Any, List
from src.utils.custom_types import Recording, PrepLayer, Data

class DigitsClassifier:
    def __init__(self, prep_layer: PrepLayer) -> None:
        self.prep_layer = prep_layer

    def process(self, rec: Recording) -> Any:
        r = self.prep_layer(rec)
        # TODO: SNN here
        return r

    def batch_process(self, data: Data) -> List:
        return [self.process(dp.wav) for dp in data]
