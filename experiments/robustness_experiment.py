#robustness_experiment.py
import os


import torch
import numpy as np
import matplotlib.pyplot as plt

from src.liquid_network import LiquidNetwork


def run():

    os.makedirs(
        "results",
        exist_ok=True
    )

    torch.manual_seed(42)

    # Generate temporal signal
    t = torch.linspace(
        0,
        20,
        1000
    )

    signal = (
        torch.sin(0.7 * t)
        + 0.5 * torch.sin(2.3 * t)
    )

    sequence = signal[:-1].view(
        1,
        -1,
        1
    )

    target = signal[1:].view(
        1,
        -1,
        1
    )

    # Train reference model
    model = LiquidNetwork(
        input_dim=1,
        hidden_dim=16,
        output_dim=1
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = torch.nn.MSELoss()

    for _ in range(300):

        optimizer.zero_grad()

        prediction, _, _ = model(
            sequence
        )

        loss = loss_fn(
            prediction,
            target
        )

        loss.backward()

        optimizer.step()

    # ======================================
    # Test parameter perturbations
    # ======================================

    mismatch_levels = [
        0.0,
        0.05,
        0.10,
        0.20,
        0.30
    ]

    errors = []

    for mismatch in mismatch_levels:

        perturbed_model = LiquidNetwork(
            input_dim=1,
            hidden_dim=16,
            output_dim=1
        )

        perturbed_model.load_state_dict(
            model.state_dict()
        )

        # Perturb parameters
        with torch.no_grad():

            for parameter in (
                perturbed_model.parameters()
            ):

                noise = torch.randn_like(
                    parameter
                ) * mismatch

                parameter.add_(noise)

        with torch.no_grad():

            prediction, _, _ = (
                perturbed_model(
                    sequence
                )
            )

        mse = torch.mean(
            (prediction - target) ** 2
        ).item()

        errors.append(mse)

        print(
            f"Mismatch={mismatch:.2f} "
            f"MSE={mse:.6f}"
        )

    # ======================================
    # Plot
    # ======================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        np.array(mismatch_levels) * 100,
        errors,
        marker="o"
    )

    plt.xlabel(
        "Parameter mismatch (%)"
    )

    plt.ylabel(
        "Prediction MSE"
    )

    plt.title(
        "Robustness to Parameter Variability"
    )

    plt.tight_layout()

    plt.savefig(
        "results/robustness.png"
    )

    plt.close()

    print(
        "Robustness experiment completed."
    )