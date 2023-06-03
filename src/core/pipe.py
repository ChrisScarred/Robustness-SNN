"""The pipeline of the extracting, encoding and classificating processes."""
from functools import partial
from typing import List

from src.core.speech_encoder import SpeechEncoder
from src.core.feature_extractor import extract_mfscs
from src.utils.custom_types import Config, Data


def pipeline(data: Data, config: Config) -> List:
    n_frames, overlap_t, pad_t, freq_bands = config.get_prep_vars()
    prep_func = partial(
        extract_mfscs,
        n_frames=n_frames,
        frame_overlap=overlap_t,
        frame_padding=pad_t,
        n_filters=freq_bands,
    )
    snn_cofig = config.get_snn_params()
    snn_cofig["t_frames"] = n_frames
    snn_cofig["freq_bands"] = freq_bands
    model = SpeechEncoder(prep_func, **snn_cofig)
    model.batch_process(data)
    return []
