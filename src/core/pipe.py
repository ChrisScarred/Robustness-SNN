"""The pipeline of the extracting, encoding, classificating, and analysing processes."""
import random
from functools import partial
from typing import Dict, List, Tuple

from numpy.typing import NDArray
from src.core.feature_extractor import extract_mfscs
from src.core.speech_encoder import SpeechEncoder
from src.core.svc import SupportVectorClassifier
from src.data.noise import NoiseHandler, mix_signal_noise
from src.utils.custom_types import PrepLayer, TiData, Tidigit
from src.utils.log import get_logger
from src.utils.misc import split_data
from src.utils.audio import save_wav
from src.utils.project_config import ProjectConfig

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
    if config._dev_mode():
        random.seed(config._seed())
        n = config._dev_mode_samples()
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


def get_noise_db_handler(config: ProjectConfig) -> NoiseHandler:
    noise = NoiseHandler(config)
    if config._download_noise_db():
        samples, allowed_licenses, allowed_formats = config.get_noise_db_params()
        noise.get_db(samples, allowed_licenses, allowed_formats)
    noise.load_data(config._target_sr(), config._pickle_processed_noise())
    return noise


def snr_tests(noise: NoiseHandler, data: TiData) -> None:
    import numpy as np
    from src.utils.audio import rms

    data = [d.recording.content for d in data]
    avg_rms = np.mean(np.array([rms(d) for d in data]))
    signal = random.choice(data)
    noise_sample = noise.get_random_noise()
    save_wav(noise_sample, 20000, "tests/noise.wav")
    snrs = [20, 10, 0, -10, -20]
    for snr in snrs:
        noisy_signal = mix_signal_noise(snr, signal, noise_sample, avg_rms)
        save_wav(noisy_signal, 20000, f"tests/noisy{snr}.wav")
    save_wav(signal, 20000, "tests/og.wav")


def processes(config: ProjectConfig, train: TiData, test: TiData, validation: TiData) -> None:
    model = None

    if config._train_se():
        if not model:
            model = get_speech_encoder(config)
        model.train(train, *config.get_training_params())

    if config._compare_se_mfsc():
        if not model:
            model = get_speech_encoder(config)
        _compare_snn_mfsc(
            config._comp_parts(),
            *encode(model, train, test, validation),
        )

    if config._get_noise():
        noise = get_noise_db_handler(config)
        snr_tests(noise, train)

def pipeline(data: TiData, config: ProjectConfig) -> None:
    """Prepare data and run the extracting, encoding, classificating, and analysing processes according to the supplied configuration.
    """
    processes(config, *_prep_data(data, config))
