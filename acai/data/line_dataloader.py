from typing import Optional

import numpy as np

from .line_image_generation import generate_batch


class LineDataLoader:
    def __init__(self, device, batch_size: int = 64):
        self.batch_size = batch_size
        self.device = device

    def get_train_batch(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        return generate_batch(batch_size).to(self.device)

    def get_eval_batch(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        return generate_batch(batch_size).to(self.device)
