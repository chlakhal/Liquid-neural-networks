#training_experiment.py

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.liquid_network import LiquidNetwork
from src.rnn_baseline import RNNBaseline


def generate_signal(n=1000):

    t = torch.linspace(
        0,
        20,
        n
    )

    signal = (
        torch.sin(0.7 * t)
        + 0.5 * torch.sin(2.3 * t)
        + 0.2 * torch.sin(4.1 * t)
    )

    return signal


def train_liquid(
    model,
    sequence,
    target,
    epochs=300
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        optimizer.zero_grad()

        prediction, _, _ = model(
            sequence
        )

        loss = loss_fn(
            prediction,
            target
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        if epoch % 50 == 0:

            print(
                f"Liquid epoch {epoch}: "
                f"{loss.item():.6f}"
            )


def train_rnn(
    model,
    sequence,
    target,
    epochs=300
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        optimizer.zero_grad()

        prediction = model(
            sequence
        )

        loss = loss_fn(
            prediction,
            target
        )

        loss.backward()

        optimizer.step()

        if epoch % 50 == 0:

            print(
                f"RNN epoch {epoch}: "
                f"{loss.item():.6f}"
            )


def run():

    os.makedirs(
        "results",
        exist_ok=True
    )

    torch.manual_seed(42)

    signal = generate_signal()

    sequence = signal[:-1].view(
        1, -1, 1
    )

    target = signal[1:].view(
        1, -1, 1
    )

    # ================================
    # Liquid
    # ================================

    liquid = LiquidNetwork(
        input_dim=1,
        hidden_dim=16,
        output_dim=1
    )

    train_liquid(
        liquid,
        sequence,
        target
    )

    # ================================
    # RNN
    # ================================

    rnn = RNNBaseline(
        input_dim=1,
        hidden_dim=16,
        output_dim=1
    )

    train_rnn(
        rnn,
        sequence,
        target
    )

    # ================================
    # Evaluation
    # ================================

    with torch.no_grad():

        liquid_prediction, _, _ = liquid(
            sequence
        )

        rnn_prediction = rnn(
            sequence
        )

    liquid_mse = torch.mean(
        (liquid_prediction - target) ** 2
    ).item()

    rnn_mse = torch.mean(
        (rnn_prediction - target) ** 2
    ).item()

    print()
    print("Final results")
    print("----------------")
    print(
        f"Liquid MSE: {liquid_mse:.6f}"
    )
    print(
        f"RNN MSE:    {rnn_mse:.6f}"
    )

    # ================================
    # Plot
    # ================================

    t = torch.linspace(
        0,
        20,
        len(target[0])
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        t.numpy(),
        target[0, :, 0].numpy(),
        label="Target"
    )

    plt.plot(
        t.numpy(),
        liquid_prediction[0, :, 0].numpy(),
        label="Liquid"
    )

    plt.plot(
        t.numpy(),
        rnn_prediction[0, :, 0].numpy(),
        label="RNN"
    )

    plt.xlabel("Time")
    plt.ylabel("Signal")

    plt.title(
        "Temporal Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/prediction_comparison.png"
    )

    plt.close()

    print(
        "Training experiment completed."
    )