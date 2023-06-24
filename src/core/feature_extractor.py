"""MFSC-based feature extractor for audio signals, an implementation of Dong et al. design described in 'Unsupervised speech recognition through spike-timing-dependent plasticity in a convolutional spiking neural network' (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204596).
"""
from math import ceil
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft
from src.utils.audio import (hz_spectrum_to_mel, mel_filterbank, ms_to_samples,
                             pad_signal)
from src.utils.custom_types import Recording
from src.utils.misc import hann_window


def extract_mfscs(
    audio: Recording,
    n_frames: int,
    frame_overlap: float,
    frame_padding: float,
    n_filters: int,
) -> Recording:
    """Extracts MFSCs from audio signals.

    Dong et al. describe their proprocessing as: 'We compute the Mel-scaled filter banks by applying triangular filters on a Mel-scale to the power spectrum, and take the logarithm of the result'. This suggests the following feature extraction pipeline: `mfsc = log(coeffs(mel_filterbank(power_spectrum(time_series))))`.

    Dong et al. also specify: 'We use different window length in the Fourier transform step during the MFSC feature extraction to get an input of fixed length', which means that the power spectrum should be taken for N windowed frames of variable length. They do not specify whether the windowing is strided (an equivalent of convolution), so this function was developed to support strided, sliding, and standard windowing depending on the supplied parameters.

    Args:
        audio (Recording): The signal to process.
        n_frames (int): The requested number of frames.
        frame_overlap (float): The requested frame overlap in time in ms.
        frame_padding (float): Padding of each time frame in ms.
        n_filters (int): The number of mel frequency bands (filters).

    Returns:
        Recording: MFSCs of the input.
    """
    sampling_rate = audio.sampling_rate
    frame_samples, stride_samples, pad_samples, side_pad = compute_frame_parameters(
        len(audio), n_frames, frame_overlap, sampling_rate, frame_padding
    )
    side_padded_signal = pad_signal(audio.content, side_pad, multichannel=False)
    framed_audio = get_striding_windows(
        side_padded_signal,
        stride_samples,
        side_padded_signal.size,
        frame_samples,
        stride_samples,
    )
    padded_audio = pad_signal(framed_audio, pad_samples, split=False)
    spectrum = get_windowed_spectrum(padded_audio, fft, hann_window)
    mel_spectrum = hz_spectrum_to_mel(
        spectrum, mel_filterbank(sampling_rate, padded_audio.shape[1], n_filters)
    )
    log_mel = np.log(mel_spectrum)

    audio.mfsc_features = log_mel
    return audio


def get_windowed_spectrum(
    signal: NDArray,
    spectrum: Callable,
    window: Callable,
    abs: bool = True,
    multichannel: bool = True,
    axis: int = 1,
) -> NDArray:
    if multichannel:
        return np.apply_along_axis(
            lambda x: get_windowed_spectrum(x, spectrum, window, multichannel=False),
            axis,
            signal,
        )

    w = window(signal.size)
    s = spectrum(signal * w)
    if abs:
        s = np.abs(s)
    return s


def get_striding_windows(signal: NDArray, c: int, t: int, k: int, v: int) -> NDArray:
    """Slide an array of a time-series signal into `t/v` windows of `k` time frames sliding over the portion of interest of the input signal indicated by clearing time `c` with the stride of `v`.

    Automatically pads to zeroes in the case of out-of-bound indices.

    This function is adapted from https://gist.githubusercontent.com/syaffers/4b4c3b2c17c3f53cdf4d2d1df6ab4f9e/raw/e9ffceec1dcbb03a17be752e3ec54a58ef1c1d9c/vectorized_stride.py as presented in https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5.

    Args:
        signal (NDArray): The signal to window.
        c (int): The clearing time index specifying the last index of the portion of signal that preceeds the portion of signal that is of interest.
        t (int): The maximum time index specifting the last index of the portion of the signal that is of interest. Bound inclusive.
        k (int): The size of windows.
        v (int): The size of the stride (overlap) between windows.

    Returns:
        NDArray: A strided sliding windows representation of the input signal.
    """
    start = c + 1 - k + 1
    window_indices = (
        start
        + np.expand_dims(np.arange(k), 0)
        + np.expand_dims(np.arange(t + 1, step=v), 0).T
    )
    # get padding mask
    mask = np.asarray((window_indices < 0) | (window_indices > (signal.size - 1)))
    # perform the windowing
    window_indices = window_indices % signal.size
    signal = signal[window_indices]
    # perform the padding
    return np.where(mask, 0, signal)


def compute_frame_parameters(
    total_samples: int,
    n_frames: int,
    overlap_ms: int,
    sr: int,
    padding_ms: int,
) -> Tuple[int, int, int]:
    """Obtain necessary parameters for computing striding sliding windows of a signal.

    Args:
        total_samples (int): The number of total samples in the original signal.
        n_frames (int): The requested number of time frames.
        overlap_ms (int): The requested overlap of time frames in ms.
        sr (int): The sampling rate of the original signal.
        padding_ms (int): The requested padding of each time frame in ms.

    Returns:
        Tuple[int, int, int]: The number of samples in a time frame, the stride of the time frames in samples, the padding of the time frames in samples, and the padding of the signal in samples.
    """
    n_frames -= 1
    non_overlap_samples = ceil((total_samples - 1) / n_frames)
    side_pad = (non_overlap_samples * n_frames) - (total_samples - 1)
    overlap_samples = ms_to_samples(overlap_ms, sr)
    frame_samples = overlap_samples + non_overlap_samples

    # stride equals the number on non-overlapping samples
    return frame_samples, non_overlap_samples, ms_to_samples(padding_ms, sr), side_pad
