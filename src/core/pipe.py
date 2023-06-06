"""The pipeline of the extracting, encoding and classificating processes."""
import random
from functools import partial
from typing import List, Tuple

from numpy.typing import NDArray

from src.core.feature_extractor import extract_mfscs
from src.core.speech_encoder import SpeechEncoder
from src.core.svc import SupportVectorClassifier
from src.utils.custom_types import Config, Data, DataPoint, PrepLayer
from src.utils.misc import split_data


def _get_prep_layer(config: Config) -> PrepLayer:
    n_frames, overlap_t, pad_t, freq_bands = config.get_prep_vars()
    return partial(
        extract_mfscs,
        n_frames=n_frames,
        frame_overlap=overlap_t,
        frame_padding=pad_t,
        n_filters=freq_bands,
    )


def get_speech_encoder(config: Config) -> SpeechEncoder:
    n_frames, _, _, freq_bands = config.get_prep_vars()
    prep_layer = _get_prep_layer(config)
    snn_cofig = config.get_snn_params()
    snn_cofig["t_frames"] = n_frames
    snn_cofig["freq_bands"] = freq_bands
    return SpeechEncoder(prep_layer, **snn_cofig)


def _prep_data(data: Data, config: Config) -> Tuple[Data, Data, Data]:
    s_data = split_data(data)
    if config.get("modes.dev", False):
        random.seed(config.get("seed"), "seed")
        n = config.get("modes.dev_n", 1)
        for key, value in s_data.items():
            d = value.data
            random.shuffle(d)
            s_data[key] = Data(data=value.data[:min(n, len(value)-1)])
    train = s_data.get("train")
    test = s_data.get("test")
    validation = s_data.get("validation")
    return train, test, validation


def encode(model: SpeechEncoder, train: Data, test: Data, validation: Data) -> Tuple[Data, Data, Data]:
    train_processed = model.batch_process(train)
    test_processed = model.batch_process(test)
    validation_processed = model.batch_process(validation)
    return train_processed, test_processed, validation_processed


def _get_modes(config: Config) -> List[bool]:
    modes = config.get("modes", {})
    return [modes.get("training", False), modes.get("testing", False), modes.get("validation", False)]


def _mfsc_predictor(x: DataPoint) -> NDArray:
    return x.recording.mfsc_features


def _enc_predictor(x: DataPoint) -> NDArray:
    return x.recording.encoded_features


def _compare_snn_mfsc(config: Config, train: Data, test: Data, validation: Data, ) -> None:
    modes = _get_modes(config)
    for predictor, predictor_name in zip([_mfsc_predictor, _enc_predictor], ["Feature Extractor (MFSC)", "Speech Encoder (SNN)"]):
        print(predictor_name)
        for mode_indication, mode_name, data_source in zip(modes, ["Training", "Testing", "Validation"], [train, test, validation]):
            if mode_indication:
                svc = SupportVectorClassifier()
                svc.train(train, predictor)
                score = svc.score(data_source, predictor)
                print(f"\t{mode_name}: {score:.2f} accurracy")

def processes(config: Config, train: Data, test: Data, validation: Data) -> None:
    p_conf = config.get("processes", {})
    if p_conf.get("compare_snn_mfsc", False):
        _compare_snn_mfsc(config, train, test, validation)


def pipeline(data: Data, config: Config) -> None:
    model = get_speech_encoder(config)
    processes(config, *encode(model, *_prep_data(data, config)))
