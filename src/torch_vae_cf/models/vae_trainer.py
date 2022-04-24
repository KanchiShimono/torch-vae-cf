from typing import Dict, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch_vae_cf.losses import kl_divergence_loss
from torch_vae_cf.models.vae import VAE
import torch.optim as optim


class VAETrainer(pl.LightningModule):
    def __init__(
        self,
        dims: Sequence[int],
        init_anneal: float = 0.0,
        anneal_upadte_rate: float = 0.001,
        anneal_cap: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 0.1,
    ) -> None:
        super(VAETrainer, self).__init__()
        self.dims = dims
        self.init_anneal = init_anneal
        self.anneal = init_anneal
        self.anneal_upadte_rate = anneal_upadte_rate
        self.anneal_cap = anneal_cap
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = VAE(self.dims)

    def update_anneal(self, diff: float) -> None:
        if self.anneal + diff >= self.anneal_cap:
            self.anneal = self.anneal_cap
            return
        self.anneal += diff

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(  # type: ignore[override]
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.model(input)
        return output

    def training_step(  # type: ignore[override]
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        output, mu, logvar = self.model(batch)
        # log-likilihood
        ll_loss = -torch.mean(torch.sum(F.log_softmax(output, dim=-1) * batch, dim=-1))
        # kl divergence
        kl_loss = kl_divergence_loss(mu, logvar)
        kl_loss = self.anneal * kl_loss
        loss = ll_loss + kl_loss
        self.update_anneal(self.anneal_upadte_rate)
        return {'loss': loss}
