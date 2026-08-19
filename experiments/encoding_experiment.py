#encoding_experiment.py

import os

import numpy as np
import matplotlib.pyplot as plt

from src.temporal_encoding import (
    state_to_frequency,
    frequency_to_state,
    oscillator_from_frequency,
    quantize_frequency,
    add_frequency_noise
)


def run():

    os.makedirs(
        "results",
        exist_ok=True
    )

    np.random.seed(42)

    time = np.linspace(
        0,
        10,
        2000
    )

    # Neural state
    state = (
        0.5
        + 0.4 * np.sin(time)
        + 0.1 * np.sin(4 * time)
    )

    # State -> frequency
    frequency = state_to_frequency(
        state,
        f_min=20,
        f_max=100
    )

    # Hardware-like imperfections
    frequency_quantized = (
        quantize_frequency(
            frequency,
            resolution=2.0
        )
    )

    frequency_noisy = (
        add_frequency_noise(
            frequency_quantized,
            noise_std=1.0
        )
    )

    # Frequency -> oscillator
    oscillator, phase = (
        oscillator_from_frequency(
            frequency_noisy,
            time
        )
    )

    # Decode
    reconstructed = frequency_to_state(
        frequency_noisy,
        np.min(state),
        np.max(state),
        f_min=20,
        f_max=100
    )

    mse = np.mean(
        (state - reconstructed) ** 2
    )

    print(
        f"Encoding MSE: {mse:.6f}"
    )

    # =================================
    # State
    # =================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        state
    )

    plt.xlabel("Time")
    plt.ylabel("State")

    plt.title(
        "Liquid State"
    )

    plt.tight_layout()

    plt.savefig(
        "results/encoding_state.png"
    )

    plt.close()

    # =================================
    # Frequency
    # =================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        frequency,
        label="Ideal frequency"
    )

    plt.plot(
        time,
        frequency_noisy,
        label="Quantized + noisy"
    )

    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")

    plt.title(
        "Temporal Frequency Encoding"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/frequency_encoding.png"
    )

    plt.close()

    # =================================
    # Oscillator
    # =================================

    plt.figure(figsize=(10, 4))

    plt.plot(
        time[:1000],
        oscillator[:1000]
    )

    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.title(
        "Oscillator Signal"
    )

    plt.tight_layout()

    plt.savefig(
        "results/oscillator_signal.png"
    )

    plt.close()

    # =================================
    # Reconstruction
    # =================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        state,
        label="Original"
    )

    plt.plot(
        time,
        reconstructed,
        "--",
        label="Reconstructed"
    )

    plt.xlabel("Time")
    plt.ylabel("State")

    plt.title(
        f"State Reconstruction — MSE={mse:.5f}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/state_reconstruction.png"
    )

    plt.close()

    print(
        "Encoding experiment completed."
    )