#rnn_baseline.py
import torch.nn as nn



class RNNBaseline(nn.Module):

    def __init__(
        self,
        input_dim=1,
        hidden_dim=16,
        output_dim=1
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            output_dim
        )

    def forward(self, sequence):

        hidden, _ = self.rnn(
            sequence
        )

        output = self.output_layer(
            hidden
        )

        return output