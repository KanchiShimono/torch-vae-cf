from typing import Sequence, Tuple

import torch
import torch.nn as nn

from torch_vae_cf.nn.reparametrize import Reparametrize


class VAE(nn.Module):
    """Variational Autoencoder Model for collaborative filtering

    Args:
        dims (Sequence[int]): 1-D array like of encoder and decoder dimensions.
    """

    def __init__(self, dims: Sequence[int]) -> None:
        super(VAE, self).__init__()
        # q(z|x)
        self.encoder = Encoder(dims)
        self.reparametrize = Reparametrize()
        # p(x|z)
        self.decoder = Decoder(dims[::-1])

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(input)
        z = self.reparametrize(mu, logvar)
        logits: torch.Tensor = self.decoder(z)
        return logits, mu, logvar


class Encoder(nn.Module):
    """Encoder part model for Variational Autoencoder

    Inference network for q(z|x).

    Args:
        dims (Sequence[int]): 1-D array like of encoder dimensions.
            [dim1, dim2, ..., dimn]. Value of the dim1 is item vocabulary size.
            dimn is used for variational parameters of Gaussian distribution mean and log-variance.
            dimension.
    """

    def __init__(self, dims: Sequence[int]) -> None:
        super(Encoder, self).__init__()
        self.dims = dims
        layers = nn.ModuleList()
        for i in range(len(self.dims) - 2):
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.dims[-2], self.dims[-1] * 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder forward method

        Args:
            input (torch.Tensor): Input tensor. [batch_size, dims[0]]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensor mean and log-variance.
                Two tensors have same shape. ([batch_size, dims[-1]], [batch_size, dims[-1]]).
        """
        x = self.layers(input)
        return torch.chunk(x, 2, dim=1)  # type: ignore[return-value]


class Decoder(nn.Module):
    """Decoder part model for Variational Autoencoder

    Inference network for p(x|z).

    Args:
        dims (Sequence[int]): 1-D array like of decoder dimensions.
            [dim1, dim2, ..., dimn]. Value of the dim1 must be same as
            last dimension of encoder part. dimn also must be same as item
            vocabulary size.
    """

    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        self.dims = dims
        layers = nn.ModuleList()
        for i in range(len(self.dims) - 2):
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.dims[-2], self.dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Decoder forward method

        Args:
            input (torch.Tensor): Input tensor. [batch_size, dims[0]]

        Returns:
            torch.Tensor: Logits tensor. [batch_size, dims[-1]]
        """
        logits: torch.Tensor = self.layers(input)
        return logits
