from torch import nn
import torch


class SimpleMLP(nn.Module):
    """
    Two-layer MLP for a *single* multilabel output.

    Args
    ----
    input_dim   : size of the encoder/embedding vector.
    output_dim  : number of labels (e.g. 48).
    hidden_dim  : width of the two hidden layers.
    dropout_prob: dropout applied after each hidden ReLU.
    """
    def __init__(
        self,
        input_dim    : int,
        output_dim   : int,):
        super().__init__()

        self.fc1     = nn.Linear(input_dim,  output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return  self.fc1(x)
