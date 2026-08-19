#liquid_network.py
import torch.nn as nn


from .liquid_dynamics import LiquidDynamics


class LiquidNetwork(nn.Module):

    def __init__(
        self,
        input_dim=1,
        hidden_dim=16,
        output_dim=1
    ):
        super().__init__()

        self.liquid = LiquidDynamics(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            output_dim
        )

    def forward(
        self,
        sequence,
        dt=0.01
    ):

        states, taus = self.liquid.simulate(
            sequence,
            dt=dt
        )

        output = self.output_layer(
            states
        )

        return output, states, taus