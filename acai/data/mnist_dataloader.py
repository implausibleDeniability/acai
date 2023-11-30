from typing import Optional

import cv2
import numpy as np
import torch
from sklearn.datasets import fetch_openml

from .dataloader import DataLoaderBase
    

class MNISTDataLoader(DataLoaderBase):
    IMAGE_SIZE = (32, 32)

    def __init__(self, device, batch_size: int = 64):
        self.batch_size = batch_size
        self.device = device
        original_images = self._load_images()
        resized_images = self._resize_images(original_images)
        images = torch.Tensor(resized_images).unsqueeze(1)
        self.eval_images = images[:batch_size]
        self.train_images = images[batch_size:]

    def _load_images(self):
        """
        :return: NDArray [70000, HEIGHT, WIDTH]
        """
        x, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False, data_home='mnist/')
        return x.reshape(70000, 28, 28) / 255

    def _resize_images(self, images):
        """
        :param images: NDArray [N, 28, 28]
        :return: NDArray [N, 32, 32]
        """
        resized_images = np.empty((len(images), *self.IMAGE_SIZE))
        for i, image in enumerate(images):
            resized = cv2.resize(image.reshape(28, 28, 1), dsize=self.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
            resized_images[i] = resized.squeeze()
        resized_images = np.clip(resized_images, 0, 1)
        return resized_images
    
    def get_train_batch(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        indeces = np.random.randint(len(self.train_images), size=batch_size)
        batch = self.train_images[indeces]
        return batch.to(self.device)

    def get_eval_batch(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        assert batch_size <= len(self.eval_images)
        return self.eval_images[:batch_size].to(self.device)