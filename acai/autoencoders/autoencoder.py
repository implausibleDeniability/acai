from typing import Any

import torch

from .base import AutoencoderBase


class AutoencoderDefault(AutoencoderBase):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        latent = self.encoder(images)
        reconstructed_images = self.decoder(latent)
        loss = self.compute_loss(reconstructed_images, images)
        return {
            "loss": loss,
            "reconstructed_images": reconstructed_images,
        }