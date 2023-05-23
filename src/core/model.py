from typing import Any, List
from src.utils.custom_types import Recording, PrepLayer, Data


class DigitsClassifier:
    def __init__(self, prep_func: PrepLayer) -> None:
        self.prep_func = prep_func

    def process(self, rec: Recording) -> Any:
        r = self.prep_func(rec)
        # TODO: SNN here
        return r

    def batch_process(self, data: Data) -> List:
        return [self.process(dp.recording) for dp in data]
