"""Extract MFSC features from audio signals. Dong et al. describe their proprocessing as: 'we compute the Mel-scaled filter banks by applying triangular filters on a Mel-scale to the power spectrum, and take the logarithm of the result'.

This means mfsc = log(coeffs(mel_filterbank(power_spectrum(time_series)))). 

Dong et al. specify: 'we use different window length in the Fourier transform step during the MFSC feature extraction to get an input of fixed length', which means that the power_spectrum should be taken for N windowed frames of variable length.
"""
from functools import cache, lru_cache
from math import ceil, floor
from typing import Tuple

import numpy as np
from librosa.filters import mel
from numpy.typing import NDArray
from scipy.fft import fft
from scipy.signal import get_window

from src.utils.custom_types import Recording
from src.utils.parsing import ms_to_samples


@lru_cache
def compute_frame_parameters(
    audio: NDArray,
    n_frames: int,
    overlap_ms: int,
    sampling_rate: int,
    padding_ms: int,
) -> Tuple[int, int, int]:
    overlap_samples = ms_to_samples(overlap_ms, sampling_rate)
    padding_samples = ms_to_samples(padding_ms, sampling_rate)
    s = len(audio)

    samples_frame = (
        ceil(((n_frames - 1) * overlap_samples + s) / n_frames) + 2 * padding_samples
    )

    return samples_frame, overlap_samples, padding_samples


@cache
def split_audio_in_frames(
    rec: Recording,
    n_frames: int,
    samples_frame: int,
    samples_overlap: int,
    samples_pad: int,
) -> NDArray:
    rec = rec.content
    framed_audio = []
    previous_end = 0
    for n in range(n_frames):
        start = previous_end
        end = start + samples_frame
        if n > 0:
            start = previous_end - ceil(samples_overlap / 2)
        if n == n_frames - 1:
            start = rec.size - n_frames
            end = rec.size
        frame = rec[start:end]
        if frame.size != samples_frame:
            deviation = samples_frame - frame.size
            if n == n_frames - 1:
                frame = np.pad(frame, (samples_pad, samples_pad + deviation))
            else:
                frame = np.pad(
                    frame,
                    (
                        samples_pad + ceil(deviation / 2),
                        samples_pad + floor(deviation / 2),
                    ),
                )
        else:
            frame = np.pad(frame, (samples_pad,))
        previous_end = end
        framed_audio.append(frame)
    return np.array(framed_audio, dtype=object)


@lru_cache
def hann_window(ln: int) -> NDArray:
    # Dong et al. do not mention which windowing function they used in the Fourier trasform step, so I assume Hann
    return get_window("hann", ln)


@cache
def extract_mfscs(
    audio: Recording,
    n_frames: int,
    frame_overlap: float,
    sampling_rate: int,
    frame_padding: float,
    n_filters: int,
) -> NDArray:
    framed_audio = split_audio_in_frames(
        audio,
        n_frames,
        *compute_frame_parameters(
            audio, n_frames, frame_overlap, sampling_rate, frame_padding
        )
    )

    spectrum = np.apply_along_axis(
        lambda x: np.abs(fft(x * hann_window(x.size))), 1, framed_audio
    )

    mel_spectrum = np.apply_along_axis(
        lambda x: mel(
            sr=sampling_rate, n_fft=framed_audio.shape[1] * 2 - 1, n_mels=n_filters
        ).dot(x),
        1,
        spectrum,
    )

    return np.log(mel_spectrum)
