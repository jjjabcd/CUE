import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple MLP for modality-specific regression and embedding extraction
    with automatic divisor-based dimension reduction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 1):
        """
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Last hidden embedding dimension
            output_dim (int): Output dimension
        """
        super().__init__()
        h1_dim = max(input_dim // 4, hidden_dim * 2)
        h2_dim = max(h1_dim // 2, hidden_dim)

        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, D)

        Returns:
            torch.Tensor: Output tensor of shape (N, output_dim)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, D)

        Returns:
            torch.Tensor: Hidden embedding of shape (N, hidden_dim)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.relu(self.fc3(x))
