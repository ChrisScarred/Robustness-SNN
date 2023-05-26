from functools import partial
from typing import List

from src.core.snn_encoder import SpeechEncoder
from src.core.preprocess import extract_mfscs
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
    model = SpeechEncoder(prep_func, config.get("model_params.snn.in.th", 10.0))
    return model.batch_process(data)
