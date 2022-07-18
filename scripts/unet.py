import torch
import torchvision

from torch.nn import functional, ModuleList

from torch import nn


class DoubleConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.convolve = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convolve(x)


class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.maxpool_convolve = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvolutionLayer(in_features, out_features)
        )

    def forward(self, x):
        return self.maxpool_convolve(x)


class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_features, in_features // 2, kernel_size=(2, 2), stride=(2, 2))
        self.convolve = DoubleConvolutionLayer(in_features, out_features)

    @staticmethod
    def crop(encoder_features, x):
        height, weight = x.shape[2], x.shape[3]

        # Returning new cropped matrix for input encoder features
        return torchvision.transforms.CenterCrop([height, weight])(encoder_features)

    def forward(self, x, x_pass):
        x_scaled = self.upsample(x)
        offset_y = x_pass.size()[2] - x_scaled.size()[2]
        offset_x = x_pass.size()[3] - x_scaled.size()[3]

        x_padded = functional.pad(x_scaled, [offset_x // 2, offset_x - offset_x // 2,
                                             offset_y // 2, offset_y - offset_y // 2])
        x = torch.cat([x_pass, x_padded], dim=1)
        return self.convolve(x)


class OutLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(OutLayer, self).__init__()

        self.convolve = nn.Conv2d(in_features, out_features, kernel_size=(1, 1))

    def forward(self, x):
        return self.convolve(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        encode_channels = (256, 512, 1024, 2048)
        decode_channels = (2048, 1024, 512, 256)

        self.input_layer = DoubleConvolutionLayer(n_channels, encode_channels[0])

        self.encoders = ModuleList([Encoder(encode_channels[i], encode_channels[i + 1]) for i in range(len(encode_channels) - 1)])
        self.decoders = ModuleList([Decoder(decode_channels[i], decode_channels[i + 1]) for i in range(len(decode_channels) - 1)])
        self.output_layer = OutLayer(decode_channels[-1], n_classes)

    def forward(self, x):
        x = self.input_layer(x)

        skip_layers = []

        for encoder in self.encoders:
            skip_layers.append(x)
            x = encoder(x)

        for idx, decoder in enumerate(self.decoders):
            x = decoder(x, skip_layers[len(skip_layers) - idx - 1])

        classes = self.output_layer(x)
        return classes
