import torch
import torch.nn as nn


class Reparametrize(nn.Module):
    """Reparametrization trick layer"""

    def __init__(self) -> None:
        super(Reparametrize, self).__init__()

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(logvar * 0.5)
