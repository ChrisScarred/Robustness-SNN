"""The pipeline of the extracting, encoding, classificating, and analysing processes."""
import random
from functools import partial
from typing import List, Tuple, Dict

from numpy.typing import NDArray

from src.core.feature_extractor import extract_mfscs
from src.core.speech_encoder import SpeechEncoder
from src.core.svc import SupportVectorClassifier
from src.utils.custom_types import  TiData, Tidigit, PrepLayer
from src.utils.misc import split_data
from src.data.noise import NoiseData
from src.utils.project_config import ProjectConfig
from src.utils.log import get_logger
logger = get_logger(name="pipe")


def _get_prep_layer(config: ProjectConfig) -> PrepLayer:
    n_frames, overlap_t, pad_t, freq_bands = config.get_prep_vars()
    return partial(
        extract_mfscs,
        n_frames=n_frames,
        frame_overlap=overlap_t,
        frame_padding=pad_t,
        n_filters=freq_bands,
    )


def get_speech_encoder(config: ProjectConfig) -> SpeechEncoder:
    prep_layer = _get_prep_layer(config)
    return SpeechEncoder(prep_layer=prep_layer, **config.get_snn_params())


def _prep_data(data: TiData, config: ProjectConfig) -> Tuple[TiData, TiData, TiData]:
    s_data = split_data(data)
    if config.get("modes.dev.enabled", False):
        random.seed(config.get("seed"), "seed")
        n = config.get("modes.dev.samples", 1)
        for key, value in s_data.items():
            d = value.data
            random.shuffle(d)
            s_data[key] = TiData(data=value.data[: min(n, len(value) - 1)])
    train = s_data.get("train")
    test = s_data.get("test")
    validation = s_data.get("validation")
    return train, test, validation


def encode(
    model: SpeechEncoder, train: TiData, test: TiData, validation: TiData
) -> Tuple[TiData, TiData, TiData]:
    model.set_training(False)
    if train:
        train = model.batch_process(train)
    if test:
        test = model.batch_process(test)
    if validation:
        validation = model.batch_process(validation)
    return train, test, validation


def _get_comparison_modes(modes: Dict[str, bool]) -> List[bool]:
    return [
        modes.get("training", False),
        modes.get("testing", False),
        modes.get("validation", False),
    ]


def _mfsc_predictor(x: Tidigit) -> NDArray:
    return x.recording.mfsc_features


def _enc_predictor(x: Tidigit) -> NDArray:
    return x.recording.encoded_features


def _compare_snn_mfsc(
    modes_dict: Dict[str, bool],
    train: TiData,
    test: TiData,
    validation: TiData,
) -> None:
    modes = _get_comparison_modes(modes_dict)
    for predictor, predictor_name in zip(
        [_mfsc_predictor, _enc_predictor],
        ["Feature Extractor (MFSC)", "Speech Encoder (SNN)"],
    ):
        logger.info(f"{predictor_name}")
        for mode_indication, mode_name, data_source in zip(
            modes, ["Training", "Testing", "Validation"], [train, test, validation]
        ):
            if mode_indication:
                svc = SupportVectorClassifier()
                svc.train(train, predictor)
                score = svc.score(data_source, predictor)
                logger.info(f"\t{mode_name}: {score:.2f} accurracy")


def noise_db_handler(config: ProjectConfig) -> None:
    noise = NoiseData(config)
    if config.get("processes.obtain_noise_dataset.download", False):
        samples = config.get("processes.obtain_noise_dataset.samples")
        allowed_licenses = config.get("processes.obtain_noise_dataset.allowed_licenses")
        allowed_formats = config.get("processes.obtain_noise_dataset.allowed_formats")
        noise.get_db(samples, allowed_licenses, allowed_formats)
    noise_data = noise.load_data(config.get("processes.obtain_noise_dataset.target_sr"))
    logger.info(f"Pickled {len(noise_data)} loaded and processed noise recordings into {config.get('data.noise.pickle_path')}.")


def processes(config: ProjectConfig, train: TiData, test: TiData, validation: TiData) -> None:
    model = None

    if config.get("processes.train_snn.enabled", False):
        if not model:
            model = get_speech_encoder(config)
        model.train(train, *config.get_training_params())

    if config.get("processes.compare_snn_mfsc.enabled", False):
        if not model:
            model = get_speech_encoder(config)
        _compare_snn_mfsc(
            config.get("processes.compare_snn_mfsc"),
            *encode(model, train, test, validation),
        )

    if config.get("processes.obtain_noise_dataset.enabled", False):
        noise_db_handler(config)

def pipeline(data: TiData, config: ProjectConfig) -> None:
    """Prepare data and run the extracting, encoding, classificating, and analysing processes according to the supplied configuration.
    """
    processes(config, *_prep_data(data, config))
