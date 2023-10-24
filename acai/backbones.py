from torch import nn


def build_one_layer_encoder():
    encoder = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, padding='same'),
    )
    return encoder

def build_one_layer_decoder():
    decoder = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, padding='same')
    )
    return decoder




def build_simple_encoder(width_coef=1):
    encoder = nn.Sequential(
        build_simple_encoder_layer(1, 2*width_coef),
        build_simple_encoder_layer(2*width_coef, 4*width_coef),
        build_simple_encoder_layer(4*width_coef, 8*width_coef),
        build_simple_encoder_layer(8*width_coef, 16*width_coef),
        nn.Conv2d(16*width_coef, 16*width_coef, kernel_size=3, padding='same'),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(16*width_coef),
        nn.Conv2d(16*width_coef, 16*width_coef, kernel_size=3, padding='same'),
    )
    return encoder

def build_simple_decoder(width_coef=1):
    decoder = nn.Sequential(
        build_simple_decoder_layer(16*width_coef, 16*width_coef),
        build_simple_decoder_layer(16*width_coef, 8*width_coef),
        build_simple_decoder_layer(8*width_coef, 4*width_coef),
        build_simple_decoder_layer(4*width_coef, 2*width_coef),
        nn.Conv2d(2*width_coef, 2*width_coef, kernel_size=3, padding='same'),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(2*width_coef),
        nn.Conv2d(2*width_coef, 1, kernel_size=3, padding='same'),
        nn.Sigmoid(),
    )
    return decoder

def build_simple_encoder_layer(input_dim, output_dim):
    layer = nn.Sequential(
        nn.Conv2d(input_dim, input_dim, kernel_size=3, padding='same'),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(input_dim),
        nn.Conv2d(input_dim, output_dim, kernel_size=3, padding='same'),
        nn.LeakyReLU(0.2),
        nn.AvgPool2d(kernel_size=2),
    )
    return layer

def build_simple_decoder_layer(input_dim, output_dim):
    layer = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, padding='same'),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(output_dim),
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding='same'),
        nn.LeakyReLU(0.2),
        nn.Upsample(scale_factor=2),
    )
    return layer