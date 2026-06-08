from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

@dataclass
class DownstreamMLPConfigs:
    """
    Args:
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        batch_size (int): Batch size
        save_every (int): Save frequency
        hidden_dim_1 (int): First hidden dimension
        hidden_dim_2 (int): Second hidden dimension
        device (str): Device string
        seed (int): Random seed
    """
    num_epochs: int = 500
    learning_rate: float = 1e-3
    batch_size: int = 256
    save_every: int = 1

    hidden_dim_1: int = 16
    hidden_dim_2: int = 8
    hidden_dim_3: int = 4
    
    device: str = "cuda:0"
    seed: int = 2025



class DownstreamMLP(nn.Module):
    """
    Simple MLP for modality-specific regression and embedding extraction
    with automatic divisor-based dimension reduction.
    """

    def __init__(self, input_dim: int = 32, hidden_dim_1: int = 16, hidden_dim_2: int = 8, hidden_dim_3: int = 4, output_dim: int = 1):
        """
        Args:
            input_dim (int): Input feature dimension
            hidden_dim_1 (int): First hidden embedding dimension
            hidden_dim_2 (int): Second hidden embedding dimension
            hidden_dim_3 (int): Third hidden embedding dimension
            output_dim (int): Output dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, output_dim)
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
        x = self.relu(self.fc3(x))
        return x