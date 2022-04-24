import torch


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Caluculate KL divergence

    Args:
        mu (torch.Tensor): Mean of Gaussian distribution. [batch_size, dim]
        logvar (torch.Tensor): Log-variance of Gaussian distribution. [batch_size, dim].

    Returns:
        torch.Tensor: KL divergence.
    """

    return torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + torch.square(mu) - 1), dim=1))
