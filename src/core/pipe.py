from functools import partial
from typing import List

from src.core.snn_encoder import SpeechEncoder
from src.core.preprocess import extract_mfscs
from src.utils.custom_types import Config, Data
from src.utils.parsing import get_sr


def pipeline(data: Data, config: Config) -> List:
    sr = get_sr(data)
    n_frames, overlap_t, pad_t, freq_bands = config.get_prep_vars()
    prep_func = partial(
        extract_mfscs,
        n_frames=n_frames,
        frame_overlap=overlap_t,
        sampling_rate=sr,
        frame_padding=pad_t,
        n_filters=freq_bands,
    )
    model = SpeechEncoder(prep_func, config.get("models.snn_encoder.in_th", 10.0))
    return model.batch_process(data)
