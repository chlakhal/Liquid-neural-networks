#memory_experiment.py
import os

import torch
import matplotlib.pyplot as plt

from src.liquid_dynamics import LiquidDynamics


def simulate_for_tau(tau_min):

    model = LiquidDynamics(
        input_dim=1,
        hidden_dim=16,
        tau_min=tau_min
    )

    # Pulse followed by silence
    sequence = torch.zeros(
        1,
        1000,
        1
    )

    sequence[
        0,
        100:300,
        0
    ] = 1.0

    with torch.no_grad():

        states, taus = model.simulate(
            sequence
        )

    state_norm = torch.norm(
        states[0],
        dim=1
    )

    return (
        state_norm.numpy(),
        taus.mean(
            dim=-1
        )[0].numpy()
    )


def run():

    os.makedirs(
        "results",
        exist_ok=True
    )

    short_memory, _ = simulate_for_tau(
        0.02
    )

    long_memory, _ = simulate_for_tau(
        0.2
    )

    time = torch.arange(
        1000
    ).numpy() * 0.01

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        short_memory,
        label="Fast dynamics"
    )

    plt.plot(
        time,
        long_memory,
        label="Slower dynamics"
    )

    plt.axvline(
        3.0,
        linestyle="--",
        label="Input removed"
    )

    plt.xlabel("Time")
    plt.ylabel("State norm")

    plt.title(
        "Liquid Temporal Memory"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/liquid_memory.png"
    )

    plt.close()

    print(
        "Memory experiment completed."
    )