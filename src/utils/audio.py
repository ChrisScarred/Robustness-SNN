import random
from math import ceil, floor
import math
from typing import Tuple

import numpy as np
from librosa.filters import mel
from numpy.typing import NDArray
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_audio
from src.utils.misc import square


def rms(x: NDArray) -> float:
    """Root mean square of a signal.
    """
    sq = square(x)
    return math.sqrt(np.mean(sq))    

def snr_db_to_a(x: float) -> float:
    """Convert SNR in dB to SNR in the power of amplitude."""
    if x == 0:
        return 1
    return 10 ** (x/10)


def norm_loudness(x: NDArray, req_rms: float) -> NDArray:
    """Scales a signal to have a requested root mean square metric. Used to ensure that mixed recordings have similar 'loudness' to the original recordings by making req_rms equal to the average of rms of original recordings."""
    og_rms = rms(x)
    scale = req_rms / og_rms
    return scale * x


def equalise_lengths(signal: NDArray, noise: NDArray) -> NDArray:
    """Pad or cut a noise signal such that it spans over the entire signal of interest."""
    sig_ln = signal.size
    noi_ln = noise.size
    max_delay = noi_ln - sig_ln
    if max_delay > 0:
        delay = random.randrange(0, max_delay)
        noise = noise[delay:delay+signal.size]
    elif max_delay < 0:
        max_delay = abs(max_delay)
        delay = random.randrange(0, max_delay)
        noise = np.pad(noise,(delay, max_delay-delay))
    return noise


def load_wav(path: str) -> Tuple[int, NDArray]:
    return read_wav(path)


def save_wav(signal: NDArray, sr: int, fname: str) -> None:
    write_audio(fname, sr, signal.astype(np.int16))


def mel_filterbank(sr: int, ln: int, filters: int) -> NDArray:
    n = ln * 2 - 1
    return mel(sr=sr, n_fft=n, n_mels=filters)


def ms_to_samples(val: float, sr: int) -> int:
    """Given a sampling rate `sr`, return the value in ms converted to number of samples."""
    return ceil((val / 1000) * sr)


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

def hz_spectrum_to_mel(
    signal: NDArray, mel: NDArray, multichannel: bool = True, axis: int = 1
) -> NDArray:
    if multichannel:
        return np.apply_along_axis(
            lambda x: hz_spectrum_to_mel(x, mel, multichannel=False), axis, signal
        )
    return mel.dot(signal)
