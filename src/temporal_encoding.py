#temporal_encoding.py
import numpy as np


def state_to_frequency(
    state,
    f_min=20.0,
    f_max=100.0
):

    state_min = np.min(state)
    state_max = np.max(state)

    normalized = (
        state - state_min
    ) / (
        state_max - state_min + 1e-8
    )

    frequency = (
        f_min
        + normalized
        * (f_max - f_min)
    )

    return frequency


def frequency_to_state(
    frequency,
    state_min,
    state_max,
    f_min=20.0,
    f_max=100.0
):

    normalized = (
        frequency - f_min
    ) / (
        f_max - f_min
    )

    state = (
        state_min
        + normalized
        * (state_max - state_min)
    )

    return state


def oscillator_from_frequency(
    frequency,
    time
):

    dt = time[1] - time[0]

    phase = (
        2
        * np.pi
        * np.cumsum(frequency)
        * dt
    )

    oscillator = np.sin(phase)

    return oscillator, phase


def quantize_frequency(
    frequency,
    resolution=2.0
):

    return (
        np.round(
            frequency / resolution
        )
        * resolution
    )


def add_frequency_noise(
    frequency,
    noise_std=1.0
):

    noise = np.random.normal(
        0,
        noise_std,
        size=frequency.shape
    )

    return frequency + noise