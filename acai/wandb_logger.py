import numpy as np
import wandb


class WandbLogger:
    def __init__(self, 
                 run: str = "debug", 
                 project: str = "untitled-project",
                 log_every: int = 1,
                 log_images_every: int = 1,
                ):
        wandb.init(project=project, name=run)
        self.step_counter = 0
        self.log_every = log_every
        self.log_images_every = log_images_every
    
    def log_images(self, key: str, image: np.ndarray):
        """
        Parameters
        ----------
        key : str
            path to the data in wandb
        images : np.ndarray
            ndarray of floats in range [0.0, 1.0), shape [HEIGHT, WIDTH, N_CHANNELS]
        """
        if self.step_counter % self.log_images_every == 0:
            assert image.ndim == 3
            image = wandb.Image(image)
            wandb.log({key: image}, commit=False)
    
    def log(self, data: dict):
        if self.step_counter % self.log_every == 0:
            wandb.log(data, commit=False)
        
    def commit(self):
        wandb.log({}, commit=True)
        self.step_counter += 1