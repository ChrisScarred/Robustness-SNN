from sklearn.svm import SVC

from typing import List
from numpy.typing import NDArray
from src.utils.custom_types import TiData, Predictor


def get_features(data: TiData, predictor: Predictor) -> NDArray:
    features = [predictor(x) for x in data]
    return [f.flatten() for f in features]


def get_labels(data: TiData) -> List[int]:
    return [x.label for x in data]


class SupportVectorClassifier:
    def __init__(self) -> None:
        self.model = SVC(kernel="linear")

    def train(self, data: TiData, predictor: Predictor) -> None:
        self.model.fit(get_features(data, predictor), get_labels(data))

    def infer(self, data: TiData, predictor: Predictor) -> List[int]:
        return self.model.predict(get_features(data, predictor))

    def score(self, data: TiData, predictor: Predictor) -> float:
        return self.model.score(get_features(data, predictor), get_labels(data))
