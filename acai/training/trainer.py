from typing import Optional

import torch

from ..autoencoders.base import AutoencoderBase
from ..data.dataloader import DataLoaderBase
from ..wandb_logger import WandbLogger
from ..monitoring.factory import MonitoringFactory, MonitoringType


class Trainer:
    EVAL_EVERY = 500
    
    def __init__(
        self,
        autoencoder: AutoencoderBase,
        dataloader: DataLoaderBase,
        logger: Optional[WandbLogger] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.autoencoder = autoencoder
        self.dataloader = dataloader
        self.logger = logger or WandbLogger('debug', 'acai_karpathy')
        self.device = device
        self._configure_optimizers()
        self._configure_monitorings()

    def _configure_optimizers(self):
        raise NotImplementedError

    def _configure_monitorings(self):
        self.monitorings = [
            MonitoringFactory().build(MonitoringType.reconstruction, self.dataloader, n_images=4),
            MonitoringFactory().build(MonitoringType.interpolation, self.dataloader, n_images=5),
        ]
        
    def train(self, n_steps):
        eval_batch = self.dataloader.get_eval_batch()
        self.autoencoder = self.autoencoder.to(self.device)
        for step in range(n_steps):
            self._train_step()
            if step % self.EVAL_EVERY == 0:
                self._eval_step(eval_batch)
                self._monitor_step()
            self.logger.commit()

    def _train_step(self) -> None:
        raise NotImplementedError()
    
    def _eval_step(self, batch):
        raise NotImplementedError()

    def _monitor_step(self):
        for monitoring in self.monitorings:
            monitoring(self)