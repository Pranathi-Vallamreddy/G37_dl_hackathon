from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(signal: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    if not 0 < lowcut < highcut < nyquist:
        raise ValueError("Expected 0 < lowcut < highcut < fs/2 for bandpass filtering")

    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, np.asarray(signal, dtype=np.float64))


def compute_envelope(signal: np.ndarray) -> np.ndarray:
    analytic = hilbert(np.asarray(signal, dtype=np.float64))
    return np.abs(analytic)


def compute_fft(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return np.asarray([], dtype=np.float32)
    return np.abs(np.fft.rfft(signal)) / signal.size


def compute_envelope_fft(signal: np.ndarray) -> np.ndarray:
    return compute_fft(signal)


def compute_mel_spectrogram(signal: np.ndarray, fs: float) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=np.asarray(signal, dtype=np.float32),
        sr=int(fs),
        n_fft=512,
        hop_length=128,
        n_mels=64,
        fmax=fs / 2.0,
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


def preprocess_signal(signal: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    nyquist = fs / 2.0
    filtered_signal = bandpass_filter(signal, fs, lowcut=10.0, highcut=0.4 * nyquist)
    envelope = compute_envelope(filtered_signal)
    fft = compute_fft(signal)
    envelope_fft = compute_envelope_fft(envelope)
    mel_spectrogram = compute_mel_spectrogram(filtered_signal, fs)

    return {
        "filtered_signal": filtered_signal,
        "envelope": envelope,
        "fft": fft,
        "envelope_fft": envelope_fft,
        "mel_spectrogram": mel_spectrogram,
    }