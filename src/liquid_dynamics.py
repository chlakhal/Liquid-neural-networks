#liquid_dynamics.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidDynamics(nn.Module):

    def __init__(
        self,
        input_dim=1,
        hidden_dim=16,
        tau_min=0.05
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min

        # Input -> liquid neurons
        self.input_layer = nn.Linear(
            input_dim,
            hidden_dim
        )

        # Recurrent coupling between liquid neurons
        self.recurrent_layer = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        # Adaptive time constant
        self.tau_layer = nn.Linear(
            hidden_dim + input_dim,
            hidden_dim
        )

    def forward_step(
        self,
        u,
        x,
        dt=0.01
    ):

        # Input contribution
        input_drive = self.input_layer(u)

        # Recurrent contribution
        recurrent_drive = self.recurrent_layer(x)

        # Nonlinear neural drive
        drive = torch.tanh(
            input_drive + recurrent_drive
        )

        # State/input-dependent time constant
        tau_input = torch.cat(
            [x, u],
            dim=-1
        )

        tau = (
            self.tau_min
            + F.softplus(
                self.tau_layer(tau_input)
            )
        )

        # Continuous-time dynamics
        dx_dt = (
            -x + drive
        ) / tau

        # Euler integration
        x_next = (
            x + dt * dx_dt
        )

        return x_next, tau

    def simulate(
        self,
        sequence,
        dt=0.01
    ):

        batch_size = sequence.shape[0]

        x = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=sequence.device
        )

        states = []
        taus = []

        for t in range(sequence.shape[1]):

            u = sequence[:, t, :]

            x, tau = self.forward_step(
                u,
                x,
                dt
            )

            states.append(
                x.unsqueeze(1)
            )

            taus.append(
                tau.unsqueeze(1)
            )

        states = torch.cat(
            states,
            dim=1
        )

        taus = torch.cat(
            taus,
            dim=1
        )

        return states, taus