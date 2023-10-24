import numpy as np


class ConvolutionVisualizer:
    def extract_convolutions(self, model) -> dict[str, np.ndarray]:
        convolutions = dict()
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weights = next(module.parameters()).detach().cpu().numpy()
                input_dim, output_dim, kernel_height, kernel_width = weights.shape
                convolution_images = weights.reshape(input_dim*output_dim, kernel_height, kernel_width, 1)
                convolutions[name] = convolution_images
        return convolutions