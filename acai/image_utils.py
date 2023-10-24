from matplotlib import pyplot as plt
import numpy as np
import torch


def draw_images(images: np.ndarray):
    """Draws an array of grayscale images with pyplot.
    :param images: np.ndarray with shape [N_IMAGES, H, W]
    """
    fig, ax = plt.subplots(ncols=images.shape[0])
    for i, image in enumerate(images):
        if image.ndim == 2:
            ax[i].imshow(image, cmap='gray')
        else:
            raise f"Image dimensionality {image.shape} not supported"
        ax[i].axis("off")


def torch2numpy_image(image: torch.Tensor) -> np.ndarray:
    """Converts torch image[s] to numpy image[s]
    Parameters
    ----------
    image: torch.Tensor
        tensor of shape [N_CHANNELS, H, W] or [BATCH_SIZE, N_CHANNELS, H, W]
    
    Returns
    -------
    np.ndarray
        image or images with shape [H, W, N_CHANNELS] or [BATCH_SIZE, H, W, N_CHANNELS]
    """
    assert image.ndim in [3, 4]
    image = image.cpu().detach().numpy()
    image = np.moveaxis(image, -3, -1)
    return image


def collage_images(images: list[np.ndarray]) -> np.ndarray:
    """Puts puts together list of [batches of] images into single image
    Parameters
    ----------
    images: list[np.ndarray]
        ndarrays are expected to have the same shape of [H, W, N_CHANNELS] or [BATCH_SIZE, H, W, N_CHANNELS]
    
    Returns
    -------
    np.ndarray
        with shape [H * len(images), W * BATCH_SIZE, N_CHANNELS]
    """
    assert type(images) is list
    assert all([image.shape == images[0].shape for image in images]), "Shape mismatch among ndarrays"
    len_images = len(images)
    image_shape = images[0].shape
    batch_size = image_shape[0] if len(image_shape) == 4 else 1
    height = image_shape[-3]
    width = image_shape[-2]
    n_channels = image_shape[-1]
    
    images = np.stack(images).reshape(len_images, batch_size, height, width, n_channels)
    images = images.transpose(0, 2, 1, 3, 4) # result has shape [len, height, batch_size, width, n_channels]
    images = images.reshape(len_images * height, batch_size * width, n_channels) # target shape
    return images


def unravel_images(images: np.ndarray, max_rows=4) -> list[np.ndarray]:
    for n_rows in range(max_rows, 0, -1):
        if images.shape[0] % n_rows == 0 and images.shape[0] // n_rows >= n_rows:
            break
    list_of_images = [
        images[i::n_rows]
        for i in range(n_rows)
    ]
    return list_of_images