import cv2
import numpy as np
from matplotlib import pyplot as plt

def make_line_image(angle: float) -> np.ndarray:
    DRAWING_IMAGE_SIZE = 256
    IMAGE_SIZE = 32
    
    startpoint = (0.5, 0.5)
    endpoint = (startpoint[0] + np.sin(angle) * 0.5, startpoint[1] + np.cos(angle) * 0.5)
    absolute_startpoint = [int(DRAWING_IMAGE_SIZE * coordinate) for coordinate in startpoint]
    absolute_endpoint = [int(DRAWING_IMAGE_SIZE * coordinate) for coordinate in endpoint]
    
    image = np.zeros((DRAWING_IMAGE_SIZE, DRAWING_IMAGE_SIZE), dtype='uint8')
    image = cv2.line(image, absolute_startpoint, absolute_endpoint, color=255, thickness=12)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image