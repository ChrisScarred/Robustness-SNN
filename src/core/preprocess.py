"""Extract MFSC features from audio signals.
"""
from math import ceil
from typing import Tuple

from functools import lru_cache
import numpy as np
from librosa.filters import mel
from numpy.typing import NDArray
from scipy.fft import fft
from scipy.signal import get_window

from src.utils.custom_types import Recording
from src.utils.parsing import ms_to_samples
from src.utils.caching import region


def get_striding_windows(signal: NDArray, c: int, t: int, k: int, v: int) -> NDArray:
    """Slide an array of a time-series signal into t/v windows of k time frames sliding over the portion of interest of the input signal with the stride of v.

    Automatically pads to zeroes in the case of out-of-bound indices.

    This function is adapted from https://gist.githubusercontent.com/syaffers/4b4c3b2c17c3f53cdf4d2d1df6ab4f9e/raw/e9ffceec1dcbb03a17be752e3ec54a58ef1c1d9c/vectorized_stride.py as presented in https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5.

    Args:
        signal (NDArray): the signal to window.
        c (int): the clearing time index specifying the last index of the portion of signal that preceeds the portion of signal that is of interest.
        t (int): the maximum time index specifting the last index of the portion of the signal that is of interest. Bound inclusive.
        k (int): the size of windows.
        v (int): The size of the stride (overlap) between windows.

    Returns:
        NDArray: A strided sliding windows representation of the input signal.
    """
    # get indices
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
    signal = np.where(mask, 0, signal)
    return signal


def compute_frame_parameters(
    rec: Recording,
    n_frames: int,
    overlap_ms: int,
    sr: int,
    padding_ms: int,
) -> Tuple[int, int, int]:
    """Get the length of frames in number of samples, the stride of frames in number of samples, and the window padding in number of samples."""
    total_samples = len(rec)
    overlap_samples = ms_to_samples(overlap_ms, sr)

    frame_samples = ceil(overlap_samples + (total_samples - 1) / n_frames)
    stride_samples = frame_samples - overlap_samples

    return frame_samples, stride_samples, ms_to_samples(padding_ms, sr)


@lru_cache
def hann_window(ln: int) -> NDArray:
    """Get the Hann window of supplied length."""
    # Dong et al. do not mention which windowing function they used in the Fourier trasform step, so I assume Hann
    return get_window("hann", ln)


@region.cache_on_arguments()
def extract_mfscs(
    audio: Recording,
    n_frames: int,
    frame_overlap: float,
    frame_padding: float,
    n_filters: int,
) -> Recording:
    """Extracts MFSCs from audio signals.

    Dong et al. describe their proprocessing as: 'we compute the Mel-scaled filter banks by applying triangular filters on a Mel-scale to the power spectrum, and take the logarithm of the result'. This suggests the following preprocessing pipeline: mfsc = log(coeffs(mel_filterbank(power_spectrum(time_series)))).

    Dong et al. also specify: 'we use different window length in the Fourier transform step during the MFSC feature extraction to get an input of fixed length', which means that the power spectrum should be taken for N windowed frames of variable length. They do not specify whether the windowing is strided (an equivalent of convolution), so this function was developed to support strided, sliding, and standard windowing depending on the supplied parameters.

    Args:
        audio (Recording): The signal to process.
        n_frames (int): The requested number of frames.
        frame_overlap (float): The requested frame overlap in time in ms.
        frame_padding (float): Padding of each time frame in ms.
        n_filters (int): The number of mel frequency bands (filters).

    Returns:
        Recording: MFSCs of the supplied recording.
    """
    # get params
    sampling_rate = audio.sampling_rate
    frame_samples, stride_samples, pad_samples = compute_frame_parameters(
        audio, n_frames, frame_overlap, sampling_rate, frame_padding
    )
    # get striding windows
    framed_audio = get_striding_windows(
        audio.content, stride_samples, audio.content.size, frame_samples, stride_samples
    )
    # pad each window
    framed_audio = np.apply_along_axis(
        lambda x: np.pad(x, (pad_samples,)), 1, framed_audio
    )
    # compute the hann-windowed spectrum for each window
    spectrum = np.apply_along_axis(
        lambda x: np.abs(fft(x * hann_window(x.size))), 1, framed_audio
    )
    # compute the mel spectrum for each window
    mel_spectrum = np.apply_along_axis(
        lambda x: mel(
            sr=sampling_rate, n_fft=framed_audio.shape[1] * 2 - 1, n_mels=n_filters
        ).dot(x),
        1,
        spectrum,
    )
    # get the log
    return Recording(content=np.log(mel_spectrum))
