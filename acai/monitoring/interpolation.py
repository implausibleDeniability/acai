import numpy as np
import torch

from .base import MonitoringCallbackBase
from ..image_utils import torch2numpy_image, collage_images
from ..autoencoders.acai import AutoencoderBase


class InterpolationMonitoring(MonitoringCallbackBase):
    def __init__(self, src_images, target_images, interpolation_steps: int = 7, key='monitoring/interpolation'):
        self.interpolation_steps = interpolation_steps
        self.key = key
        self.src_images = src_images
        self.target_images = target_images

    def __call__(self, trainer):
        trainer.autoencoder.eval()
        self._transfer_images_to_device(trainer.device)
        interpolated_images = self._interpolate(trainer.autoencoder)
        collage = self._make_collage_image(interpolated_images)
        trainer.logger.log_images(self.key, collage)

    def _transfer_images_to_device(self, device):
        self.src_images = self.src_images.to(device)
        self.target_images = self.target_images.to(device)

    def _interpolate(self, model: AutoencoderBase) -> torch.Tensor:
        """
        :param model: autoencoder
        :return: list of interpolated images
        :rtype: list[torch.Tensor], tensors of shape [N_IMAGES, 2+N_INTERPOLATION_STEPS, N_CHANNELS, H, W]
        """
        src_latent = model.encoder(self.src_images)
        target_latent = model.encoder(self.target_images)
        latents_interpolation = [
            (1 - alpha) * src_latent + alpha * target_latent
            for alpha in np.linspace(0.0, 1.0, self.interpolation_steps)
        ]
        images_interpolation = [
            model.decoder(latent).detach()
            for latent in latents_interpolation
        ]
        # [2+N_INTERPOLATION_STEPS, N_IMAGES, N_CHANNELS, H, W]
        merged_original_and_interpolation = torch.stack([self.src_images] + images_interpolation + [self.target_images])
        return torch.transpose(merged_original_and_interpolation, 0, 1)

    def _make_collage_image(self, interpolated_images):
        numpy_images = torch2numpy_image(interpolated_images)
        collaged_image = collage_images([image for image in numpy_images])
        return collaged_image