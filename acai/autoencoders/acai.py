from typing import Any

import torch
from torch import nn

from .base import AutoencoderBase


class ACAI(AutoencoderBase):
    GAMMA = 0.2
    LAMBDA = 0.5 / 4
    
    def __init__(self, encoder, decoder, critic):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.compute_loss = torch.nn.MSELoss()
        
    def forward(self, images: torch.Tensor, images4interpolation: torch.Tensor) -> dict[str, Any]:
        latent = self.encoder(images)
        latent4interpolation = self.encoder(images4interpolation)
        interpolated_latent, alphas = self._mix_latents(latent, latent4interpolation)
        reconstructed_images = self.decoder(latent)
        interpolated_images = self.decoder(interpolated_latent)
        predicted_alphas = torch.mean(self.critic(interpolated_images), [1, 2, 3])
        
        reconstruction_loss = self.compute_loss(reconstructed_images, images)
        interpolation_loss = self.LAMBDA * (predicted_alphas ** 2).mean(axis=0)
        loss = reconstruction_loss + interpolation_loss
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "interpolation_loss": interpolation_loss,
            "reconstructed_images": reconstructed_images,
            "reconstructed_interpolated_images": interpolated_images,
            "reconstructed_interpolated_alphas": predicted_alphas,
        }
    
    def _mix_latents(self, latent, latent4interpolation) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param latent: torch.Tensor of shape [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
        :return: tuple of two tensors. 
            1. Mixed latents, torch.Tensor of shape [BATCH_SIZE, CHANNELS, HEGIHT, WIDTH]
            2. Alphas, torch.Tensor of shape [BATCH_SIZE, 1, 1, 1]
        """
        batch_size = latent.shape[0]
        alphas = 0.5 * torch.rand(batch_size).reshape(-1, 1, 1, 1).to(latent.device)
        interpolated_latent = (1 - alphas) * latent + alphas * latent4interpolation
        return interpolated_latent, alphas
    
    def forward_critic(self, images: torch.Tensor, images4interpolation: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            latent = self.encoder(images)
            latent4interpolation = self.encoder(images4interpolation)
            interpolated_latent, alphas = self._mix_latents(latent, latent4interpolation)
            reconstructed_images = self.decoder(latent)
            interpolated_images = self.decoder(interpolated_latent)
            regularization_images = self.GAMMA * images + (1 - self.GAMMA) * reconstructed_images
        predicted_alphas = torch.mean(self.critic(interpolated_images), [1, 2, 3])
        predicted_alphas_regularization = torch.mean(self.critic(regularization_images), [1, 2, 3])

        alpha_recovery_loss = torch.nn.functional.mse_loss(predicted_alphas, alphas.squeeze())
        regularization_loss = (predicted_alphas_regularization ** 2).mean(0)
        loss = alpha_recovery_loss + regularization_loss
        return {
            "loss": loss,
            "alpha_recovery_loss": alpha_recovery_loss,
            "regularization_loss": regularization_loss,
            "reconstructed_interpolated_images": interpolated_images,
            "reconstructed_images": reconstructed_images,
            "blended_non_interpolated_images": regularization_images,
        }
