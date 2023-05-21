"""Extract MFSC features from audio signals. Code reused from https://github.com/ChrisScarred/spoken-digit-classification-SNN/blob/main/src/model/mfsc.py, a MFSC extraction module written solely by the Github user ChrisScarred.
"""
from math import ceil, floor
from typing import Tuple

import numpy as np
from numpy import inf
from numpy.typing import NDArray
from scipy.fft import fft
from scipy.signal import get_window

from src.utils.custom_types import Recording


def normalize_audio(audio: NDArray) -> NDArray:
    audio = audio / np.max(np.abs(audio))
    return audio


def _compute_frame_parameters(
    audio: NDArray,
    n_frames: int,
    time_overlap: int,
    sampling_rate: int,
    time_frame_padding: int,
) -> Tuple[int, int, int, Tuple[int, int]]:
    """Perform the computing task of audio framing.

    Args:
        audio (NDArray): The audio signal to split.
        n_frames (int): The number of frames to use
        time_overlap (int): The time for which frames overlap in ms.
        sampling_rate (int): The sampling rate of the signal.
        time_frame_padding (int): The time for which the frames are zero-padded from both sides in ms.

    Returns:
        Tuple[int, int, int, Tuple[int, int]]: The length of a frame in samples, the length of the overlap in samples, the length of the frame zero-padding in samples, and the padding sizes of the signal to assure equal length of frames.
    """
    len_overlap = ceil((time_overlap / 1000) * sampling_rate)
    len_frame_pad = ceil((time_frame_padding / 1000) * sampling_rate)
    len_frame = ceil((len(audio) + len_overlap * (n_frames + 1)) / n_frames)
    samples_needed = len_frame * n_frames - (n_frames - 1) * len_overlap
    padding = ceil(samples_needed / 2)
    if padding * 2 > samples_needed:
        padding = (padding, padding - 1)
    else:
        padding = (padding, padding)
    return len_frame, len_overlap, len_frame_pad, padding


def _split_audio_in_frames(
    audio: NDArray,
    n_frames: int,
    ln: int,
    len_frame: int,
    len_overlap: int,
    len_frame_pad: int,
) -> NDArray:
    """Perform the splitting part of audio framing.

    Args:
        audio (NDArray): The audio signal to split.
        n_frames (int): The number of frames to use.
        ln (int): The length of the audio signal in samples.
        len_frame (int): The length of a frame in samples.
        len_overlap (int): The length of the frame overlap in samples.
        len_frame_pad (int): The length of the frame zero-padding from both sides in samples.

    Returns:
        NDArray: The split audio signal.
    """
    framed_audio = []
    for n in range(n_frames):
        start = n * len_frame - (n - 1) * len_overlap
        if start < 0:
            start = 0
        end = (n + 1) * len_frame - (n - 1) * len_overlap
        if end > ln:
            end = ln - 1
        frame = audio[start:end]
        frame = np.pad(frame, (len_frame_pad, len_frame_pad))
        frame = frame.tolist()
        framed_audio.append(frame)

    return np.array(framed_audio)


def frame_audio(
    audio: NDArray,
    n_frames: int,
    time_overlap: int,
    sampling_rate: int,
    time_frame_padding: int,
) -> Tuple[NDArray, int]:
    """Split an audio signal into `n_frames` identically long frames which overlap for `time_overlap` ms and are padded with zeroes for `time_frame_padding` ms.

    Args:
        audio (NDArray): The normalised audio signal.
        n_frames (int): The number of frames to obtain.
        time_overlap (int): The amount of time in ms for which the frames overlap.
        sampling_rate (int): The sampling rate of the singal.
        time_frame_padding (int): The amount of time in ms for which the frames are padded with zeroes from both sides. Smaller values improve temporal resolution of the FFT, larger values improve frequency resolution (the time-frequency localization trade-off).

    Returns:
        Tuple[NDArray, int]: The audio signal split into the frames.
    """
    len_frame, len_overlap, len_frame_pad, padding = _compute_frame_parameters(
        audio, n_frames, time_overlap, sampling_rate, time_frame_padding
    )

    audio = np.pad(audio, padding)

    framed_audio = _split_audio_in_frames(
        audio, n_frames, len(audio), len_frame, len_overlap, len_frame_pad
    )

    return framed_audio, len_frame + 2 * len_frame_pad


def discrete_fourier_transform(framed_audio: NDArray) -> NDArray:
    """Get the signal in the frequency domain by computing the discrete Fourier transform via the FFT algorithm for every overlapping window in `framed_audio`. To ensure continuity, which is assumed by FFT, the frames are windowed before being transformed."""
    # Dong et al. do not mention which windowing function they used
    window = get_window("hann", len(framed_audio[0]), fftbins=True)
    windowed_audio = framed_audio * window
    fft_audio = []
    for w in range(len(windowed_audio)):
        window = windowed_audio[w, :]
        # Dong et al. do not mention which Fourier transform algorithm they used
        window_fft = fft(window)
        fft_audio.append(window_fft)
    return np.asarray(fft_audio)


def audio_power(fft_audio: NDArray) -> NDArray:
    """Get the power of the audio signal over frequencies. Power is defined here as the squared value of the signal, not the physical power."""
    return np.square(np.abs(fft_audio))


def freq_to_mel(frequency: float) -> float:
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def mel_to_freq(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def get_filter_points(
    min_freq: float, max_freq: float, n_filters: int, frame_len: int, sampling_rate: int
) -> Tuple[NDArray, NDArray]:
    """Get the starting and ending points in mels for traingular n_filters given a frame length in samples, including the frame padding, and the sampling rate."""
    min_mel = freq_to_mel(min_freq)
    max_mel = freq_to_mel(max_freq)
    mels = np.linspace(min_mel, max_mel, num=n_filters + 2)
    freqs = mel_to_freq(mels)
    return np.floor((frame_len + 1) / sampling_rate * freqs).astype(int), freqs


def get_filters(
    filter_mels: NDArray, filter_freqs: NDArray, n_filters: int, frame_len: int
) -> NDArray:
    """Obtain the matrix of filters.

    Args:
        filter_mels (NDArray): The starting and ending mels for each filter. Filter i (starting from 0) has the starting mel of filter_points[i] and the ending mel of filter_points[i+1].
        filter_freqs (NDArray): The filter points in frequencies instead of mels.
        n_filters (int): The number of filters.
        frame_len (int): The frame length in samples including the frame padding.

    Returns:
        NDArray: The matrix of triangular filters.
    """
    filters = np.zeros((len(filter_mels) - 2, frame_len))

    for n in range(len(filter_mels) - 2):
        filters[n, filter_mels[n] : filter_mels[n + 1]] = np.linspace(
            0, 1, filter_mels[n + 1] - filter_mels[n]
        )
        filters[n, filter_mels[n + 1] : filter_mels[n + 2]] = np.linspace(
            1, 0, filter_mels[n + 2] - filter_mels[n + 1]
        )

    # normalisation so that higher mel filters are not disproportionally noisy
    enorm = 2.0 / (filter_freqs[2 : n_filters + 2] - filter_freqs[:n_filters])
    filters *= enorm[:, np.newaxis]

    return filters


def to_nums(features: NDArray) -> NDArray:
    if np.isin(-inf, features):
        vals = list(set(list(features.flatten())))
        vals.sort()
        new_min = 0
        if len(vals) > 1:
            new_min = vals[1]
        features[features == -inf] = new_min
    if np.isin(inf, features):
        vals = list(set(list(features.flatten())))
        vals.sort()
        new_max = 0
        if len(vals) > 1:
            new_max = vals[1]
        features[features == inf] = new_max
    return features


def mel_space_filtering(
    fft_audio: NDArray, sampling_rate: int, n_filters: int, frame_len: int
) -> NDArray:
    """Obtain the MFSC feature matrix using n triangular filters.

    Inspired by https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial/notebook.

    Args:
        fft_audio (NDArray): Audio recording in the frequency domain obtained via Fast Fourier Transform.
        sampling_rate (int): The sampling rate of the audio recording.
        n_filters (int): The desired number of filters, i.e. frequency bands. Bands of higher frequencies have lower specificity.
        frame_len (int): The length of frames in number of samples (NOT in a temporal metric).

    Returns:
        NDArray: n_filters x n_frames MFSC feature matrix of the audio input. The first dimension can be interpreted as the frequency bands and the second dimension as the temporal frames.
    """
    # Dong et al. do not mention the minimal frequency they used
    min_freq = 0
    # Nyquist rate
    max_freq = floor(sampling_rate / 2)

    filter_points, mel_freqs = get_filter_points(
        min_freq, max_freq, n_filters, frame_len, sampling_rate
    )
    filters = get_filters(filter_points, mel_freqs, n_filters, frame_len)
    power_audio = audio_power(fft_audio)
    audio_filtered = np.dot(filters, power_audio.T)

    # Dong et al. do not mention the base of the logarithm they used
    features = np.log(audio_filtered)
    return to_nums(features)


def extract_mfscs(
    audio: Recording,
    n_frames: int,
    frame_overlap: float,
    sampling_rate: int,
    frame_padding: float,
    n_filters: int,
) -> NDArray:
    """Extract MFSC features following the description in Dong et al.

    Args:
        audio (NDArray): The recording whose features to extract.
        n_frames (int): The number of frames to divide the recording into for windowing.
        frame_overlap (float): The overlap of frames in ms.
        sampling_rate (int): The sampling rate of the audio.
        frame_padding (float): The padding of every frame from each side in ms. Smaller values improve temporal resolution of the FFT, larger values improve frequency resolution (the time-frequency localization trade-off).
        n_filters (int): The desired number of filters, i.e. frequency bands. Bands of higher frequencies have lower specificity.

    Returns:
        NDArray: n_filters x n_frames MFSC feature matrix of the audio input. The first dimension can be interpreted as the frequency bands and the second dimension as the temporal frames.
    """
    audio = normalize_audio(audio)
    framed_audio, frame_len = frame_audio(
        audio, n_frames, frame_overlap, sampling_rate, frame_padding
    )
    fft_audio = discrete_fourier_transform(framed_audio)
    mfsc = mel_space_filtering(fft_audio, sampling_rate, n_filters, frame_len)
    return mfsc
