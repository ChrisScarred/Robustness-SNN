from functools import lru_cache
from math import ceil, floor
from typing import Callable, Tuple

import numpy as np
from librosa.filters import mel
from numpy.typing import NDArray
from scipy.signal import get_window


def pad_signal(
    signal: NDArray,
    pad: int,
    split: bool = True,
    multichannel: bool = True,
    axis: int = 1,
) -> NDArray:
    if multichannel:
        return np.apply_along_axis(
            lambda x: pad_signal(x, pad, split, multichannel=False), axis, signal
        )

    split_pad = (pad,)
    if split:
        p = pad / 2
        split_pad = (floor(p), ceil(p))

    return np.pad(signal, split_pad)


def ms_to_samples(val: float, sr: int) -> int:
    """Given a sampling rate `sr`, return the value in ms converted to number of samples."""
    return ceil((val / 1000) * sr)


@lru_cache
def hann_window(ln: int) -> NDArray:
    """Get the Hann window of the supplied length.

    NOTE: Dong et al. do not mention which windowing function they used in the Fourier trasform step, so I opted for Hann, as it is the most widely used one.
    """
    return get_window("hann", ln)


def get_mel(sr: int, ln: int, filters: int) -> NDArray:
    n = ln * 2 - 1
    return mel(sr=sr, n_fft=n, n_mels=filters)


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


def hz_spectrum_to_mel(
    signal: NDArray, mel: NDArray, multichannel: bool = True, axis: int = 1
) -> NDArray:
    if multichannel:
        return np.apply_along_axis(
            lambda x: hz_spectrum_to_mel(x, mel, multichannel=False), axis, signal
        )
    return mel.dot(signal)


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
