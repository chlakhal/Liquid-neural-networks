#dynamics_experiment.py
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.liquid_dynamics import LiquidDynamics


def run():

    os.makedirs("results", exist_ok=True)

    torch.manual_seed(42)

    # Time
    time = np.linspace(
        0,
        10,
        1000
    )

    # Temporal input
    signal = (
        np.sin(1.5 * time)
        + 0.3 * np.sin(5 * time)
    )

    sequence = torch.tensor(
        signal,
        dtype=torch.float32
    ).view(1, -1, 1)

    # Liquid model
    model = LiquidDynamics(
        input_dim=1,
        hidden_dim=16
    )

    with torch.no_grad():

        states, taus = model.simulate(
            sequence
        )

    states = states[0].numpy()
    taus = taus[0].numpy()

    # ==================================
    # Input + liquid states
    # ==================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        signal,
        label="Input"
    )

    for i in range(4):

        plt.plot(
            time,
            states[:, i],
            label=f"Liquid neuron {i}"
        )

    plt.xlabel("Time")
    plt.ylabel("State")

    plt.title(
        "Liquid Neural Dynamics"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/liquid_dynamics.png"
    )

    plt.close()

    # ==================================
    # Adaptive tau
    # ==================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        taus.mean(axis=1)
    )

    plt.xlabel("Time")
    plt.ylabel("Mean time constant")

    plt.title(
        "Adaptive Time Constant"
    )

    plt.tight_layout()

    plt.savefig(
        "results/adaptive_tau.png"
    )

    plt.close()

    print(
        "Dynamics experiment completed."
    )