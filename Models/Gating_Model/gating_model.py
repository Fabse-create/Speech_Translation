from typing import Optional

import torch
from torch import nn

from utils.load_config import load_config


class GatingModel(nn.Module):
    def __init__(
        self,
        config_path: str = "Config/gating_model_config.json",
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_experts: Optional[int] = None,
    ) -> None:
        super().__init__()
        config = load_config(config_path)

        input_dim = input_dim if input_dim is not None else config.get("input_dim", 1280)
        hidden_dim = hidden_dim if hidden_dim is not None else config.get("hidden_dim", 512)
        num_experts = num_experts if num_experts is not None else config.get("num_experts", 8)

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        return self.linear2(x)