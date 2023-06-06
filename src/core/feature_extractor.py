"""MFSC-based feature extractor for audio signals, an implementation of Dong et al. design described in 'Unsupervised speech recognition through spike-timing-dependent plasticity in a convolutional spiking neural network' (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204596).
"""
import numpy as np
from scipy.fft import fft

from src.utils.caching import region
from src.utils.custom_types import Recording
from src.utils.fe import (compute_frame_parameters, get_mel,
                          get_striding_windows, get_windowed_spectrum,
                          hann_window, hz_spectrum_to_mel, pad_signal)


#@region.cache_on_arguments()
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
        side_padded_signal, stride_samples, side_padded_signal.size, frame_samples, stride_samples
    )
    padded_audio = pad_signal(framed_audio, pad_samples, split=False)
    spectrum = get_windowed_spectrum(padded_audio, fft, hann_window)
    mel_spectrum = hz_spectrum_to_mel(spectrum, get_mel(sampling_rate, padded_audio.shape[1], n_filters))
    log_mel = np.log(mel_spectrum)

    audio.mfsc_features = log_mel
    return audio
