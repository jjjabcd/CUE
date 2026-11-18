import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple MLP for modality-specific regression and embedding extraction
    with automatic divisor-based dimension reduction.

    A configurable ``nn.Dropout`` layer follows every hidden ReLU activation.
    With ``dropout_rate=0.0`` (the default) these layers are no-ops, so the
    architecture is unchanged for the LTAU/Deep Ensemble paths. For MC Dropout,
    ``dropout_rate`` should be > 0 and dropout is kept active only during
    stochastic uncertainty inference (see ``Trainer.predict_mc_dropout``);
    ordinary training/eval forward passes use PyTorch's standard
    train()/eval() semantics for ``nn.Dropout``.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 1, dropout_rate: float = 0.0):
        """
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Last hidden embedding dimension
            output_dim (int): Output dimension
            dropout_rate (float): Dropout probability applied after each hidden ReLU
                activation. Use 0.0 to disable (LTAU/Deep Ensemble); use a value in
                (0, 1) to enable MC Dropout.
        """
        super().__init__()
        h1_dim = max(input_dim // 4, hidden_dim * 2)
        h2_dim = max(h1_dim // 2, hidden_dim)

        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout_rate = float(dropout_rate)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.dropout3 = nn.Dropout(p=self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, D)

        Returns:
            torch.Tensor: Output tensor of shape (N, output_dim)
        """
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.dropout3(self.relu(self.fc3(x)))
        return self.fc4(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, D)

        Returns:
            torch.Tensor: Hidden embedding of shape (N, hidden_dim)
        """
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        return self.dropout3(self.relu(self.fc3(x)))
