import numpy as np
import torch

from .base import MonitoringCallbackBase
from ..image_utils import torch2numpy_image, collage_images
from ..autoencoders.acai import AutoencoderBase


class InterpolationMonitoring(MonitoringCallbackBase):
    def __init__(self, src_image, target_image, interpolation_steps: int = 7, key='monitoring/interpolation'):
        self.interpolation_steps = interpolation_steps
        self.key = key
        self.src_image = src_image
        self.target_image = target_image

    def __call__(self, trainer):
        trainer.autoencoder.eval()
        self._transfer_images_to_device(trainer.device)
        interpolated_images = self._interpolate(trainer.autoencoder)
        collage = self._combine_interpolated_with_original(interpolated_images)
        trainer.logger.log_images(self.key, collage)

    def _transfer_images_to_device(self, device):
        self.src_image = self.src_image.to(device)
        self.target_image = self.target_image.to(device)

    def _interpolate(self, model: AutoencoderBase) -> torch.Tensor:
        """
        :param model: autoencoder
        :param src_image: torch.Tensor of shape [N_CHANNELS, HEIGHT, WIDTH]
        :param target_image: torch.Tensor of shape [N_CHANNELS, HEIGHT, WIDTH]
        :return: list of interpolated images
        :rtype: list[torch.Tensor], tensors of shape [N_CHANNELS, HEIGHT, WIDTH]
        """
        src_latent = model.encoder(self.src_image.unsqueeze(0))
        target_latent = model.encoder(self.target_image.unsqueeze(0))
        latents_interpolation = [
            (1 - alpha) * src_latent + alpha * target_latent
            for alpha in np.linspace(0.0, 1.0, self.interpolation_steps)
        ]
        images_interpolation = [
            model.decoder(latent).detach()[0]
            for latent in latents_interpolation
        ]
        return images_interpolation

    def _combine_interpolated_with_original(self, interpolated_images):
        torch_image_sequence = torch.stack([self.src_image] + interpolated_images + [self.target_image])
        numpy_image_sequence = torch2numpy_image(torch_image_sequence)
        collaged_image = collage_images([numpy_image_sequence])
        return collaged_image