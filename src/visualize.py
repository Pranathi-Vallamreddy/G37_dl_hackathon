import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_signal(signal, fs, title="Signal", save_path=None, filename=None):
    t = np.arange(len(signal)) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    output_path = save_path
    if filename is not None:
        output_path = Path("outputs") / filename

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def plot_fft(signal, fs, rpm, fault_mult, title="FFT", save_path=None, filename=None):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    spectrum = np.abs(np.fft.rfft(signal))

    plt.figure(figsize=(12, 4))
    plt.plot(freqs, spectrum, linewidth=0.8)

    fr = rpm / 60
    fault_lines = [
        (fr * fault_mult["BPFI"], "BPFI"),
        (fr * fault_mult["BPFO"], "BPFO"),
        (fr * fault_mult["BSF"], "BSF"),
        (fr * fault_mult["FTF"], "FTF"),
    ]

    for frequency, name in fault_lines:
        plt.axvline(frequency, linestyle="--", linewidth=1, label=f"{name} ({frequency:.3f} Hz)")

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()

    output_path = save_path
    if filename is not None:
        output_path = Path("outputs") / filename

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved FFT -> {output_path}")

    plt.show()