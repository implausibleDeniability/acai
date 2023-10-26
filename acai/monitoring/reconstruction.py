from acai.monitoring.base import MonitoringCallbackBase
from acai.image_utils import torch2numpy_image, collage_images


class ReconstructionMonitoring(MonitoringCallbackBase):
    def __init__(self, images, key='monitoring/reconstruction'):
        self.images = images
        self.key = key

    def __call__(self, trainer):
        trainer.autoencoder.eval()
        latent = trainer.autoencoder.encoder(self.images.to(trainer.device))
        reconstructed_images = trainer.autoencoder.decoder(latent).detach().cpu()
        trainer.logger.log_images(
            self.key,
            collage_images([torch2numpy_image(self.images), torch2numpy_image(reconstructed_images)])
        )
