from typing import Optional

import numpy as np
import torch
from sklearn.datasets import fetch_openml

from .dataloader import DataLoaderBase
    

class MNISTDataLoader(DataLoaderBase):
    def __init__(self, device, batch_size: int = 64):
        images = torch.Tensor(self._load_images()).unsqueeze(1)
        self.eval_images = images[:batch_size]
        self.train_images = images[batch_size:]
        self.batch_size = batch_size
        self.device = device

    def _load_images(self):
        """
        :return: NDArray [70000, HEIGHT, WIDTH]
        """
        x, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False, data_home='mnist/')
        return x.reshape(70000, 28, 28)
    
    def get_train_batch(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        indeces = np.random.randint(len(self.train_images), size=batch_size)
        batch = self.train_images[indeces]
        return batch.to(self.device)

    def get_eval_batch(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        assert batch_size <= len(self.eval_images)
        return self.eval_images[:batch_size].to(self.device)